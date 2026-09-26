# REPORT 15：无纬度加权的类比选择重跑（MPI / TRACE）

**日期**：2026-09-13
**改动**：`osee_tas.select_analog(..., weighted=False)`——**仅去掉类比选择的纬度加权**；
CELF 损失、UV 损失、评估 RMSE **仍为 cos-lat 加权**（按用户选择 (a)）。
**范围**：真值 = MPI、TRACE；池 = 其余 3 模式；划分 1–550 / 551–600 / 601–1100；无 loc、无 inflation；四件套 + CELF/UV。
原"加权版"结果全部保留。

---

## 1. 结果（test 601–1100 TAS RMSE）

| 真值 | 选择 | AOEnKF-raw | OEnKF-raw | CELF only | CELF+UV |
|---|---|---|---|---|---|
| MPI | 加权 | 0.5805 | 0.5751 | 0.5261 | 0.4759 |
| MPI | **无加权** | 0.5771 | 0.5751 | 0.5247 | **0.4737** |
| TRACE | 加权 | 0.5640 | 0.5187 | 0.6204 | 0.6188 |
| TRACE | **无加权** | 0.5644 | 0.5187 | 0.6197 | **0.6133** |

（无加权 UV 最优：两真值均为 r=20, λ=0。）

## 2. 结论

1. **去掉纬度加权只带来边际变化**（各件套差 ≤ 0.006）：
   - MPI：CELF+UV 0.4759 → 0.4737（略降）；
   - TRACE：CELF+UV 0.6188 → 0.6133（略降），但仍**远差于最优 raw 0.5187**。
2. **定性结论完全不变**：
   - MPI：CELF+UV 相对 raw 有效（0.577 → 0.474）；
   - TRACE：CELF+UV 失效（0.613 vs raw 0.519）。
3. 因此**纬度加权不是 TRACE 失败的原因**，也**不是上一轮"OEnKF 偶尔优于 AOEnKF"的原因**——后者是"最近邻逐成员选择 ≠ 集合均值最优"的尾部效应（交叉极少、幅度多为平局级、集中在晚段真值分叉处），且已在  中量化。

## 3. 产物
- 数据：`pairs_unw_mpi/`、`pairs_unw_trace/`
- 基线：`results/raw_unw_mpi.json|npz`、`raw_unw_trace.json|npz`
- 模型：`results/loc_celf_unw_{mpi,trace}.pt`、`results/uv_unw_{mpi,trace}.npz|json`、`zC_unw_*.npz`
- 汇总：`results/unw_summary.json`；图 `results/figs/F37_weighted_vs_unweighted.png`
- 代码：`build_pairs_unw.py`、`eval_unw.py`（+ `select_analog` 新增 `weighted` 参数、`oenkf_raw.py` 新增 `--unweighted`）
