# REPORT 28：让 CELF 同时"看到 GC"（A: GC 输入通道；B: raw/GC 混合增益）

**日期**：2026-09-13
**动机**：GC 的固定距离先验在盲测窗比 CELF 更稳（0.5118 vs 0.5247）；REPORT24/25 显示"训练完再乘 GC"会伤 CELF，故改为**训练时就引入 GC**。

## 1. 两种实现
```
共同：U-Net 输入 5 通道 [K0/s_k, K_gc/s_k, D/8000, coslat, spread/s_s]，K_gc = ρ_GC(10000km) ⊙ K0
A) K_mod = ρ_θ ⊙ K0,          ρ_θ = clip(1+g, 0, 1.5)      （起点 = raw）
B) K_mod = K_gc + α ⊙ (K0−K_gc), α = sigmoid(g)            （末层 bias=−4 ⇒ α≈0.018，起点≈纯 GC）
xa = xf + K_mod·d；损失 CELF（cos-lat 状态 MSE）；train 1-550 / val 551-600（30 ep × 200 步）
```

## 2. 结果（test 601–1100，cos-lat TAS RMSE）

| 方案 | train | val | **test** |
|---|---|---|---|
| CELF（原始，4 通道） | 0.2386 | 0.2633 | **0.5247** |
| **A：+GC 输入通道** | 0.2383 | 0.2632 | **0.5324** |
| **B：raw/GC 混合（起点=纯 GC）** | 0.2462 | 0.2736 | **0.5310** |
| GC(10000) AOEnKF | 0.2492 | 0.2664 | **0.5118** |
| xf（先验均值） | 0.2972 | 0.3427 | 0.6609 |

## 3. 结论
1. **两种"GC-aware CELF"都更差**：A 0.5324、B 0.5310，均**劣于原始 CELF 0.5247**，更远差于 GC 0.5118。→ 让 CELF 在训练时"看到 GC"**没有**把 GC 的优势学进来。
2. **A 的 val 与 CELF 完全相同（0.2632 vs 0.2633），但 test 更差（+0.0077）**：GC 通道未带来 val 收益，却让盲测泛化变差（模型更依赖 GC 结构，而该结构在盲测段并非最优）。
3. **B 从"纯 GC"出发（val≈0.0710）却被训练推离**：最佳 val 0.0758（差于纯 GC 的 0.0710），test 0.5310（差于 GC 0.5118）→ 混合参数化在本数据上**优化方向错误/易过拟合**。
4. **根本矛盾（val 与 test 偏好相反）**：
   - **val(551–600)** 偏好 **CELF 式**解（0.2632/0.2633 < GC 的 0.2664）；
   - **test(601–1100)** 偏好 **GC 式**解（0.5118 < CELF 的 0.5247）。
   在 val 上做模型选择/早停，自然选出"CELF 式"解 → 盲测吃亏。这正是**非平稳**在选参层面的体现。
5. 因此：**"让 CELF 学习含 GC 的结果"在该协议下不奏效**；若要利用 GC 优势，更有效的是**在 GC 基上做学习型修正**（GC+UV 0.4520 已是当前最优），而非把 GC 塞进 CELF 的学习目标。

## 4. 产物
- 代码：`train_celf_gcaware.py`、`eval_gcaware.py`、`make_fig_F53.py`
- 模型/图：`results/loc_celf_gcA.pt`、`results/loc_celf_gcB.pt`、`results/gcaware_{A,B}_rmse.npy`、`results/figs/F53_gcaware_celf.png`
