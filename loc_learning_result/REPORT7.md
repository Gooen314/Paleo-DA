# REPORT 7：无 GC、无 inflation —— 用 CELF + PQ 取代局地化与膨胀

**日期**：2026-09-11
**目标**：去掉 canonical AOEnKF 的 GC 局地化与 inflation，改由
① CELF 网络（逐观测 2D U-Net，后验误差训练）+ ② 低秩算子 PQ `(I+UVᵀ)`
直接作用在 raw ensemble 上；判据：test 601–1100 **< AOEnKF_Opt 0.5021**。

---

## 1. 方案

```
raw ensemble（无 inflation）:
  Ed = E - Ebar ; Hd = H - Hbar
  K0 = C_xh (C_hh + diag(R))^{-1} ,  C_xh = Ed Hd/(K-1), C_hh = Hd^T Hd/(K-1)
  d  = y - Hbar

A1 CELF（逐观测 2D U-Net, batch, 无 GC/无 inflation）:
  rho = clip(1 + UNet(features), 0, 1.5)        # 可 >1（兼任 inflation）
  xa_C = xf + (rho .* K0) d ,  z_C = xa_C - xf
  loss = CELF = Σ_t Σ_j w_j (xa_C - xt)^2 / Σ_j w_j

A2 PQ（在冻结的 z_C 上）:
  xa = xf + z_C + U (V^T z_C) ,  rho_PQ = I + U V^T
  loss = CELF + lam (||U||^2 + ||V||^2)
```

- 网络：`UNet(cin=4, base=32)`，输入 4 通道 `[K0列, 距离场, coslat, 先验离散度]`，
  末层零初始化 → 初值 rho=1（无局地化、无膨胀），参数量 656,417。
- 划分：train 1–550 / val 551–600 / test 601–1100。
- PQ 主配置 r=5, lam=1e-3（事前默认），另做 r/lam 扫描（用 val 选择）。

## 2. 泄漏防护（严格）

| 环节 | 使用真值 | 说明 |
|---|---|---|
| CELF 输入/特征 | 无 | 仅先验 X,H 与观测 y,R |
| CELF 损失/早停 | 仅 train/val | test 不参与 |
| z_C 计算 | 无 | 冻结 CELF + 先验 + 观测 |
| PQ 损失/早停 | **仅 train 1–550 / val 551–600** | 见下"修正记录" |
| 超参数 | 事前固定 + val 选择 | 未用 test 选参 |

**修正记录（重要）**：`train_pq_on_zC.py` 初版误把损失写成对**全部 1100 步**求（含 test），
导致 test RMSE 虚假降到 0.14（相当于在 test 上拟合）。已修正为**只在 train 窗口求损失**后重跑，
下表为修正后的结果。此前的 `validate_t0.py / min_window*.py / z0_new_scheme.py` 均为 `xa[tr]-XT[tr]`，
**无此问题**，其历史结论不受影响。

## 3. 结果（test 601–1100 TAS RMSE）

| 方案 | test RMSE | 相对 AOEnKF_Opt |
|---|---|---|
| raw z0（无 loc/infl，rho=1, U=V=0） | 0.5805 | +0.0784 |
| **CELF only**（A1） | **0.5261** | +0.0240 |
| CELF+PQ（r=5, lam=1e-3，事前默认） | 0.4976 | **−0.0045** |
| **CELF+PQ（r=20, lam=0，val 选中）** | **0.4759** | **−0.0262** |
| AOEnKF_Opt（GC 7500 + infl 1.5） | 0.5021 | 基线 |
| GC+PQ（原方案，1–550） | 0.4724 | −0.0297 |

- **G1（不发散门）PASS**：CELF only 0.5261 ≤ 1.5×0.5021=0.7532，且训练稳定。
- **G2 PASS**：CELF+PQ 0.4759 < 0.5021（val 选中 r=20,lam=0；5 seed std 0.0010）。
- 事前默认 r=5,lam=1e-3 也通过（0.4976）。
- 秩扫描（lam=0）：r=5→0.4867, r=20→0.4759, r=50→0.4830（r=20 最优）。

## 4. 机制

- CELF 误差 `e_C = xa_C - xt` 在 test 上**高度低秩**：top-5 模态占 83.1%、top-20 占 91.4%、top-50 占 95.4% 能量；
- 因此固定秩-20/50 的低秩修正能去掉大部分误差 → CELF+PQ 有效；
- 对比：GC 之后的误差 `z3` 是高秩跨模式误差，PQ 只能改善 ~6%；raw/CELF 的误差则是低秩大尺度结构。

## 5. 结论

1. **CELF 可以替代 GC+inflation**：单独把 raw 从 0.5805 降到 0.5261（接近 AOEnKF_Opt）；
2. **CELF+PQ 可以超越 AOEnKF_Opt**：0.4759 < 0.5021（val 选中），达到"无 GC/无 inflation 下优于调参最优 GC+inflation"的目标；
3. 但**仍略逊于 GC+PQ（0.4724）**：即"学习式 CELF 替代 GC"的性价比不及"直接用 GC"；
4. 关键结构事实：**raw/CELF 的分析误差是低秩大尺度结构，而 GC 之后的残差是高秩跨模式结构**——这解释了为什么 PQ 在无 GC 时"看起来更有效"（去掉的是低秩部分），也说明 GC 的价值在于压制高秩远距噪声。

## 6. 产物
- 数据：`pairs_raw/t*.npz`（raw 增益，1100 步）、`matrix_cache_serial_zC.npz`（z_C）、
  `raw_ensemble_cache.npz`（raw batch/serial 结果，供复用）
- 模型/结果：`results/loc_celf_raw.pt`、`results/zC_pq.npz|json`、`results/zC_summary.json`
- 图：`results/figs/F26_new_scheme_celf_pq.png`
- 代码：`build_raw_pairs.py`、`build_raw_cache.py`、`train_celf_raw.py`、
  `eval_celf_raw.py`、`train_pq_on_zC.py`、`verify_zC_pq.py`、`make_fig_zC.py`
