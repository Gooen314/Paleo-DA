# 从论文到项目：CNN 自适应局地化的迁移与改进（技术记录）

**论文（PDF 1）**：Wang et al. (2023), *Convolutional Neural Network-Based Adaptive Localization for an Ensemble Kalman Filter*, JAMES.
**项目**：paleo/OSEE 古气候数据同化（TAS，n=6144，三先验模式，gwb 伪代理，AOEnKF）。
**代码**：`online_ml/loc_learning/`（Python + torch 1.12.1，4 GPU）。

本文记录三个阶段：①论文式迁移（batch CLF/CELF）→ ②串行一致 GC 残差局地化 → ③改进并成功的**低秩状态空间算子**。每阶段给出：数据、网络、训练公式、输出如何作用于同化、结果与诊断。

---

## 1. 论文方法回顾

论文用 CNN 学习**局地化函数**，两种监督方式：

- **CLF**（CNN Localization Function）：输入样本增益矩阵 `K`，输出局地化后的增益 `L(K)`，与大集合“真增益” `K_true` 配对：
  ```
  L_CLF = || L(K) − K_true ||_F^2
  ```
- **CELF**（CNN Ensemble Localization Function）：同架构，但损失为**后验误差**（需要真值 `x_t`）：
  ```
  L_CELF = Σ_j w_j ( x_a,j − x_t,j )^2 / Σ_j w_j ,   x_a = x_b + L(K)(y − H x_b)
  ```
- 论文网络：3 层 Conv2D（64@27×27 → 32@15×15 → 1@9×9）+ ReLU + 线性输出，作用于 state×obs 的**规则 2D 增益矩阵**。
- 关键经验：非对称位移依赖、自适应可 >1、patch 分解、用观测替代真值；主要陷阱是**训练（offline）与循环同化之间的 domain shift**。

**迁移到本项目的两个结构性差异**（决定了后面的设计）：
1. 本项目观测是**不规则代理点**，不能把 state×obs 当成规则 2D 图 → 改为**逐观测列的状态 2D 图**（96×64）去噪。
2. 本项目是**跨模式误差主导**（三先验 vs mpi 真值），而非论文的采样误差 → CLF 的“大集合真增益”目标会被模式误差污染，CELF 更合适。

---

## 2. 迁移总览

| 阶段 | 网络/参数化 | 作用位置 | 关键结果（评估窗 601–1100 TAS RMSE） |
|---|---|---|---|
| ① 论文式 batch CLF/CELF | U-Net（逐观测 2D 去噪器） | batch 更新 `x_a = x_f + (ρ∘K_s)d` | GC 0.5403 / CLF 0.5518 / CELF 0.5385 |
| ② 串行一致 GC 残差局地化 | MLP 标量场（GC 残差） | 串行 EnSRF 的局地化 `c_i` | 0.5023（≈基线 0.5021） |
| ③ 低秩状态空间算子（成功） | 低秩矩阵 `ρ=I+UVᵀ` | 串行分析增量 `z3` 之后 | **0.4724（r5）/ 0.4581（r20）** |

---

## 3. 阶段①：论文式迁移（batch CLF / CELF）

### 3.1 数据构造（`build_pairs.py :: main()`）

对训练窗 t=1..600，每个时刻：

```
观测：idx, y, R = o.select_obs(t, ...)
候选池：srch = o.search_range(t, tm)         # |pool_time − t| ≤ 250
类比选择：idx = o.select_analog(y, idx, srch, hx, plat)   # K=100，cos·Δ² 度量
E  = tas[:, idx]        (6144 × K)
Hc = hx[idx, obs_idx]   (K × M)
膨胀：Ed = infl·(E − Ē),  Hd = infl·(Hc − H̄c)     # infl=1.5
```

**样本全增益**（与 operational EnSRF 先验一致）：

```
C_hh = Hdᵀ Hd / (K−1)          (M×M)
C_xh = Ed Hd  / (K−1)          (6144×M)
K_s  = C_xh (C_hh + diag(R))⁻¹  (6144×M)          # 全矩阵增益
```

**大集合目标**（CLF 的“真增益”）：用全部候选（≈1500 个，不膨胀）

```
K_l = C_xh^l (C_hh^l + diag(R))⁻¹
```

另存：`xf`（先验均值）、`xt`（mpi 真值）、`spread`（先验离散度）、`innov = y − H x_f`。
存为 `/data1/wbgu/agentdata/loc_learning/pairs/t####.npz`。

### 3.2 网络（`train_loc.py :: UNet`）

逐观测列的状态 2D 去噪器（96×64 图）：

```
输入通道 (4, 64, 96)：[ K_s[:,j]/ks_scale, dist_j/8000, cos(lat), spread/spread_scale ]
架构：ConvBlock(4→32) → down → ConvBlock(32→64) → down → ConvBlock(64→64)
      → bottleneck ConvBlock(64→128) → 上采样 + skip → 输出 Conv2d(·,1,1)
输出：rho = 1.0 + out(...)          # 初始化为 ≈1（即“无局地化”起点）
```
签名：`UNet(cin=4, base=32)`；`make_input(ks_col, dist_col, coslat, spread, ks_scale, spread_scale) -> (4,64,96)`。

### 3.3 训练（`train_loc.py :: main()`，两损失）

**CLF**（随机抽 8 列/时刻）：

```
rho = model(x)[:,0]                        # (8,64,96)
L = weighted_mse( rho ⊙ ks , K_l/ks_scale , W )      # W = cos(lat) 归一
```

**CELF**（全部 M 列）：

```
rho_all (M,6144) = model(K_s 的每列特征)
contrib = Σ_j  K_s[:,j] ⊙ ( rho_j · innov_j )        # (6144,)
x_a     = x_f + contrib
L = weighted_mse( x_a , x_t , W )
```

优化：Adam lr=1e-4，30 epoch，训练 1–550、验证 551–600，ReduceLROnPlateau，按验证损失保存 `results/loc_{clf,celf}.pt`。

### 3.4 输出如何作用于同化

**batch（论文式）** —— `eval_batch.py`：

```
x_a = x_f + (K_s ∘ ρᵀ) innov
```

**串行（operational）** —— `eval_serial.py`：把串行 EnSRF 的局地化向量 `c_i` 的**状态部分**替换为网络输出 `ρ_i`（obs–obs 块保持 GC）：

```
xb   = [ E ; Hcᵀ ]                     # 增广集合
cmat = [ ρ ; GC_pp[obs_idx,obs_idx] ]
x_a  = o.serial_ensrf(xb, y, R, cmat)
```

### 3.5 结果与失败诊断

| 变体 | 全窗 | 评估窗 |
|---|---|---|
| prior（不更新） | 0.4617 | 0.6568 |
| unloc（无局地化） | 0.4231 | 0.5883 |
| GC（batch） | 0.3854 | 0.5403 |
| CLF（batch） | 0.3866 | 0.5518 |
| CELF（batch） | 0.3725 | 0.5385 |
| canonical 串行 AOEnKF | **0.3658** | **0.5021** |
| 串行 CLF（clip[0,1.2]） | — | 0.5204 |
| 串行 CELF（clip[0,1.2]） | — | 0.5355 |

**诊断**：
- CELF 在匹配的 batch 框架下全窗 0.3725 优于 batch GC 0.3854，但评估窗几乎持平；
- CLF 相对大集合增益的 Frobenius 误差更小（0.685 vs GC 0.752），但**不转化为 RMSE**；
- **串行部署发散**（无 clamp），说明 batch 训练与串行 EnSRF 存在**域不匹配**；
- 根因：局地化空间 headroom < 0.3%（oracle 扫描），且 batch/serial 更新方式不同。

---

## 4. 阶段②：串行一致 GC 残差局地化

思路：让训练目标与 operational **串行 EnSRF 完全一致**，并让局地化是 GC 的**有界残差**。

### 4.1 可微串行 EnSRF（`serial_torch.py :: serial_analysis`）

```
A = [X ; H]                     # 增广集合 (n+M, K)
Ā = mean_K(A) ;  A' = infl·(A − Ā)
for i = 1..M:
    h_i  = A'[n+i] ;  hb_i = Ā[n+i]
    ν_i  = (h_i·h_i)/(K−1) + R_i
    fact = 1/(1 + sqrt(R_i/ν_i))
    PHT_i = (A' h_i)/(K−1)
    c_i  = [ ρ_i ; GC_pp_i ]
    kg_i = c_i ⊙ PHT_i / ν_i
    Ā  ← Ā + kg_i (y_i − hb_i)
    A' ← A' − fact·outer(kg_i, h_i)
x_a = Ā[:n]
```

### 4.2 参数化与训练（`train_serial.py`）

```
ρ_ij = clip( ρ_GC,ij · exp( tanh( g_θ(φ_ij) ) ), 0, 2 )
φ_ij = [ d_ij/7500, cosΔlon, sinΔlon, Δlat, cos(lat_j), spread_j/σ̄, log1p(|K_s,ij|/K̄_s) ]
g_θ  = MLP(7→64→64→1)，末层零初始化 ⇒ 起点 ρ = ρ_GC
```
损失：
```
L = CELF + 0.1·CLF + 1e-3·mean(tanh²)
```
签名：`MLP(cin=7,h=64)`、`build_rho(model,d,glon,glat,ks_scale,sp_scale)`、`run_time(...)`；训练 1–550、验证 551–600。

### 4.3 结果

- 训练后 `exp(δ)` 近似**均匀 ≈1.08**，即退化为 `1.08×GC`；
- 全窗 0.3657、评估窗 0.5023（≈基线）；
- oracle 局地化扫描（半径/幅值）headroom <0.3% → **局地化不是瓶颈**。

---

## 5. 阶段③：改进版——低秩状态空间算子（成功）

思路：**不动同化任何设置**，只在 AOEnKF 输出的**分析增量**上学一个低秩状态空间线性算子。

### 5.1 定义与公式

```
z3 = xa_serial − xf                    # AOEnKF 分析增量（canonical GC，原封不动）
ρ  = I + U Vᵀ ,  U,V ∈ R^{6144×r}
x_a = xf + ρ z3 = xf + z3 + U (Vᵀ z3)
```

- `Vᵀ z3`：把增量投影到 r 个方向；
- `U(·)`：把 r 个系数线性组合回全状态并相加；
- 等价于**有效增益** `K_GC → (I+UVᵀ)K_GC` 的秩-r 修正。

### 5.2 训练（`train_matrix_rho.py` / `validate_t0.py`）

预计算 `z3`（`build_serial_analysis.py`），损失：

```
L(U,V) = Σ_t wᵀ (x_a(t) − x_t(t))² / Σ w  +  λ ( ||U||_F² + ||V||_F² )
w = cos(lat)
```

签名：
- `train_matrix_rho.py :: main()`（argparse：`--base {b1,b2,b3}`、`--r`、`--lr`、`--lam`、`--noclf`、`--diag`、`--tag`）；`rmse_t(xa,xt,w)`；
- `validate_t0.py :: train(Z, XF, XT, w, tr, va, r, lam, seed, epochs=5000)`、`main()`；`rmse_t(xa,xt,w)`；
- 训练 1–550 / 验证 551–600 / 测试 601–1100（Split A），反向 Split B（train 601–1100 / test 1–550）。

关键实现细节：`U,V` 必须**随机小初始化**（零初始化是鞍点，梯度恒为 0）；全批量训练，秒级/epoch。

### 5.3 结果（TAS RMSE）

| 口径 | AOEnKF | + r5 (λ=1e-3) | + r20 (λ=0) |
|---|---|---|---|
| Split A 测试 601–1100 | 0.5021 | **0.4724 ± 0.0001** | **0.4581** |
| Split B 测试 1–550 | 0.2510 | **0.2442** | **0.2404** |
| 无泄漏全窗（1–550 ∪ 601–1100） | 0.3706 | **0.3529** | **0.3441** |

### 5.4 机制（为什么它有效而局地化无效）

实测 `A = UVᵀ`（r=5）：

```
||A||_F = 1.577 ; ||diag(A)||_F = 0.0475 ; ||offdiag(A)||_F = 1.576
非对角能量占比 = 99.91%
测试窗修正量中对角部分占比 = 1.8e-5
```

- ρ **不是对角矩阵**：增益几乎全部来自**跨格点的非对角混合**；
- 逐格点对角缩放（0.5214）与全局标量缩放（0.5030）都**差于基线**；
- 因此有效的是“修正增益的**状态空间结构**（跨模式误差）”，而非“抑制伪相关的局地化”。

---

## 6. 三阶段对比与结论

| | ① batch CLF/CELF | ② 串行 GC 残差 | ③ 低秩状态算子 |
|---|---|---|---|
| 学什么 | 局地化因子 ρ（逐观测 2D 图） | 局地化因子 ρ（位移特征） | 状态空间算子 ρ=I+UVᵀ |
| 作用位置 | batch 更新 | 串行循环内 `c_i` | 循环后作用于 `z3` |
| 训练目标 | CLF / CELF | CELF+0.1CLF | CELF |
| 是否改同化设置 | 是（局地化） | 是（局地化） | **否** |
| 评估窗 RMSE | 0.5385–0.5518 | 0.5023 | **0.4581–0.4724** |
| 结论 | 无效（headroom/域差） | 无效（≈基线） | **有效（−6%~−8.8%）** |

**核心结论**：
1. 论文式“学习局地化”在本项目**无空间**（oracle headroom <0.3%），且 batch 训练与串行部署存在域差；
2. 真正瓶颈是**增益/协方差的状态空间结构（跨模式误差）**；
3. 用**低秩状态空间算子**修正 AOEnKF 分析增量，可在**完全保留同化设置**的前提下取得真实盲测增益。

---

## 7. 符号表

| 符号 | 含义 |
|---|---|
| n = 6144 | TAS 状态维数（96 lon × 64 lat，lon 最快） |
| K = 100 | 类比集合成员数 |
| M = M_t | t 时刻有效代理观测数（约 380–720） |
| infl = 1.5 | inflation |
| GC / ρ_GC | Gaspari–Cohn 局地化，cutoff 7500 km |
| E / Hc | 集合状态 (n×K) / 观测块 (K×M) |
| K_s | 样本全增益 (n×M) |
| K_l | 大集合增益 (n×M)，CLF 目标 |
| x_f, x_t | 先验均值、mpi 真值 |
| innov | 新息 `y − H x_f` |
| z3 | AOEnKF 分析增量 `xa_serial − xf` |
| ρ, U, V, r | 低秩状态算子、低秩因子、秩 |
| w_j | cos(lat_j) 面积权重 |

---

## 8. 代码与产物索引

**代码（`online_ml/loc_learning/`）**
- 数据/引擎：`osee_tas.py`、`serial_torch.py`、`build_pairs.py`、`build_matrix_cache.py`、`build_serial_analysis.py`
- 阶段①：`train_loc.py`（UNet/CLF/CELF）、`eval_batch.py`、`eval_serial.py`
- 阶段②：`train_serial.py`（MLP、可微串行）
- 阶段③：`train_matrix_rho.py`、`validate_t0.py`、`finalize_t0.py`
- 图件：`fig_precompute.py`、`fig_make.py`、`coastline.py`

**数据缓存（`/data1/wbgu/agentdata/loc_learning/`）**
- `pairs/`（阶段①配对增益）、`serial/`（串行张量）、`matrix_cache.npz`、`matrix_cache_serial.npz`

**结果/报告（`online_ml/loc_learning/results/`）**
- `REPORT.md`（阶段①）、`REPORT2.md`（阶段②）、`REPORT3.md`/`REPORT4.md`（阶段③）
- `FIGS.md`、`INTERNAL_REPORT.md`、`figs/*.png`、`mrho_b3_r5_noclf.npz`
- 基线：`baseline_aoenkf.npy`
