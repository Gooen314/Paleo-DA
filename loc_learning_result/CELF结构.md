# CELF 结构（direct-K 版本）

> 本文说明本项目当前使用的 CELF（CNN-based Empirical Localization Function）的**结构、公式与训练流程**。
> 版本：**direct-K**（网络直接学习增益列，而非学习乘性因子 ρ）。
> 依据代码：`online_ml/loc_learning/train_celf_directK.py`、`eval_directK.py`。

---

## 1. 定位与总览

在 OSEE 古气候同化流程中，每一步的原始（无局地化）分析为

```
xa_raw = xf + K0 · d
```

其中 `xf` 为类比集合均值、`K0` 为样本 Kalman 增益、`d = y − H xf` 为新息。
**CELF（direct-K）用它学到的增益取代 `K0`**：

```
xa = xf + K_mod · d ,   K_mod = K0 + s · g_θ
```

即"CELF 学习后的分析"，无需 GC 局地化、无需 inflation。

```
类比集合 (K=100)  →  K0 (n×M)  ─┐
                               ├─→  K_mod = K0 + s·g_θ  →  xa = xf + K_mod d
逐观测特征 φ_j (4通道) → U-Net g_θ ┘
```

---

## 2. 记号

| 符号 | 含义 | 维度 |
|---|---|---|
| `n` | 状态维数（TAS 网格） | 6144 (64×96) |
| `M` | 当步活跃观测（gwb 伪代理）数 | 388–720（逐时变化） |
| `K0` | 无局地化、无 inflation 的样本增益 `C_xh(C_hh+R)⁻¹` | n×M |
| `d` | 新息 `y − H xf` | M |
| `xf` | 类比集合均值（先验） | n |
| `g_θ(φ_j)` | 网络对第 j 个观测输出的**增益残差列** | n |
| `s` | 固定尺度 `ks_scale` | 标量 |
| `K_mod[:,j]` | 修正后第 j 列增益 | n |

---

## 3. 参数化：直接学习增益列

```
K_mod[:,j] = K0[:,j] + s · g_θ(φ_j)          (j = 1..M)
xa = xf + Σ_j K_mod[:,j] · d_j
```

- `g_θ` 是**加性**的增益修正列 —— 其方向不受 `K0[:,j]` 约束（这是与"乘性因子 `ρ∘K0`"的本质区别，也是精度提升的来源）；
- 末层零初始化 ⇒ 训练起点 `g_θ ≡ 0` ⇒ `K_mod = K0`，即**起点精确等于 raw 分析**（与 baseline 同起点，可比）。

---

## 4. 输入特征（`make_input`）

对每个观测 `j`，构造一张 **4×64×96** 的输入图（状态网格 64×96，经度最快）：

| 通道 | 内容 | 说明 |
|---|---|---|
| 0 | `K0[:,j] / ks_scale` | 该观测的 raw 增益列（归一化） |
| 1 | `D[oi_j] / 8000` | 观测 j 到各格点的球面距离场（km / 8000） |
| 2 | `cos(lat)` | 纬度权重场 |
| 3 | `spread_j / spread_scale` | 该观测的先验离散度（归一化） |

- `D`：1276×6144 的"代理点→格点"球面距离矩阵（`haversine`）；
- `oi_j`：观测 j 在 1276 个代理中的索引；
- 全部特征**不含真值**，故无泄漏。

---

## 5. 网络结构（U-Net）

```
输入 4×64×96
 e1: ConvBlock(4  → 32)                      → 32×64×96
 e2: ConvBlock(32 → 64)   after MaxPool(2)   → 64×32×48
 e3: ConvBlock(64 → 64)   after MaxPool(2)   → 64×16×24
 b : ConvBlock(64 →128)   after MaxPool(2)   → 128×8×12
 u3: ConvBlock(128+64 →64)  (上采样 b ⧺ e3)  → 64×16×24
 u2: ConvBlock(64+64  →64)  (上采样 u3 ⧺ e2) → 64×32×48
 u1: ConvBlock(64+32  →32)  (上采样 u2 ⧺ e1) → 32×64×96
 out: Conv2d(32 → 1, 1×1)   **权重与偏置零初始化**
输出 1×64×96  → 展平为 6144 维 = g_θ(φ_j)
```
- `ConvBlock` = `Conv3×3 → LeakyReLU(0.1) → Conv3×3 → LeakyReLU(0.1)`；
- 解码器用 `F.interpolate(mode='nearest')` 上采样后与编码器特征**拼接**；
- **末层零初始化**是起点=raw 的关键；
- 参数量约 6.5×10⁵。

---

## 6. 尺度 `s = ks_scale`

网络输出 `g` 为 O(1)，而增益量级极小，故需尺度换算：

```
ks_scale     = mean over 采样时刻( mean(|K0|) )    ≈ 0.00911
spread_scale = mean over 采样时刻( mean(spread) )  ≈ 0.3204
```

- 实现：对训练窗内每约 60 步取样一帧，逐帧算 `mean(|K0|)`，再对帧求平均；
- **只用训练数据计算一次后固定**（非学习参数），保证 `s·g` 与 `K0` 同量级；
- 与输入通道的归一化口径一致。

---

## 7. 前向与增益合成（`field_all_for` + `analysis`）

```python
# 逐观测前向（按 chunk=128 分批，控制显存）
for chunk of obs:
    g[j, :] = UNet(make_input(K0[:, j], D[oi[j]], coslat, spread[j]))  # (M, 6144)

# 增益合成（在 (M, n) 布局上；K0.T 即 K0 的转置）
Kmod    = K0.T + ks_scale * g                # Kmod[j,:] = K0[:,j] + s·g_θ(φ_j)
contrib = (Kmod * d[:, None]).sum(axis=0)    # Σ_j K_mod[:,j] · d_j     → (6144,)
xa      = xf + contrib
```

## 8. 损失与优化

```
loss = mean_j ( W_j · (xa_j − xᵗ_j)² ) ,   W = coslat / mean(coslat)
```
- 因 `Σ_j (coslat_j / mean(coslat)) = 1`，故 `loss` **就是 cos-lat 加权 MSE（即 RMSE²）**，与评估口径一致；
- 优化：Adam，`lr = 1e-4`；每步梯度裁剪 `clip_grad_norm_(5.0)`；
  `ReduceLROnPlateau(factor=0.5, patience=3)` 按 val 调整 lr。

## 9. 训练协议

```
数据：pairs_unw_mpi（无加权类比选择；K0 / innov / xf / xt / obs_idx / spread）
划分：train 1–550 ｜ val 551–600 ｜ test 601–1100（盲测）
每 epoch：从 train 随机抽 200 步做 SGD；epoch 末在 val 全量评估；保存 val 最优 checkpoint
总轮数：30 epochs
```

---

## 10. 结果（MPI 真值、TAS-only、无加权；cos-lat 加权 TAS RMSE）

| 方案 | train 1-550 | val 551-600 | **test 601-1100** |
|---|---|---|---|
| xf（先验均值） | 0.2972 | 0.3427 | 0.6609 |
| raw K0（无局地化） | — | — | 0.5771 |
| CELF（学 ρ∘K0） | 0.2386 | 0.2633 | 0.5247 |
| **CELF direct-K** | **0.2206** | **0.2449** | **0.5098** |
| GC(10000, infl 1.0) | 0.2492 | 0.2664 | 0.5118 |
| CELF + 状态空间 UV（参照） | 0.1710 | 0.2190 | 0.4732 |
| GC(10000) + UV（参照） | — | — | 0.4520 |

**要点**：direct-K **全面优于** ρ∘K0 版 CELF（train −0.018、val −0.018、test −0.015），且是**首个在盲测窗超过 GC(10000) 的纯 CELF 族方案**（0.5098 < 0.5118）；best val MSE = 0.06095（⇒ val RMSE 0.2469）。

---

## 11. 与 Wang et al. (2023, JAMES) 的关系

- 该文的 **CLF**：`L* = argmin E‖L(K) − Kᵗ‖²_F`（学"大集合真增益"）；**CELF**：损失换成后验误差 `E‖xᶠ + L(K)(y−Hxᶠ) − xᵗ‖²`；
- **两者都是"增益矩阵 → 增益矩阵"的 CNN 回归**（把增益当含噪图像去噪）；文中明确把 **`L(K)=ρ∘K`（Schur 乘子）当作经典基线**（GC/LRLF/ELF）；
- ⇒ 我们的 **direct-K 才对应文章的正式设计**；此前的 `ρ∘K0` 实际落在其基线形式内。
- 差异：文章对"状态×观测"整体做 2D 卷积；我们因代理位置不规则、逐时缺测，改为**逐观测**（每观测一张 64×96 状态场）处理。

---

## 12. 代码与产物索引

| 文件 | 说明 |
|---|---|
| `train_celf_directK.py` | 训练（UNet、make_input、field_all_for、analysis、损失与协议） |
| `eval_directK.py` | 全窗（1100 步）评估，输出 train/val/test RMSE |
| `results/loc_celf_directK.pt` | 训练好的权重（val 最优） |
| `results/directK_rmse.npy` | 逐时（1100）RMSE |
| `results/figs/F54_directK.png` | 与 CELF/GC/UV 的对照图 |
| `results/REPORT29.md` | 结果报告 |
