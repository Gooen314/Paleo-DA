# REPORT 12：真值换成 TRACE —— CELF+UV 失效

**日期**：2026-09-13
**问题**：把评估真值从 MPI 换成 TRACE（池 = {had_g, had_h, mpi}，观测 = trace 伪代理），
检验 "CELF+UV" 是否仍能相对无 loc/inflation 的 OEnKF / AOEnKF 基线带来改善。

---

## 1. 设置
```
真值 = trace； 观测 = pseudo_proxies_trace； 池 = {had_g, had_h, mpi}
划分 = 1-550 / 551-600 / 601-1100
基线 = OEnKF-raw（随机 K=100, seed 42）、AOEnKF-raw（类比 K=100），均无 loc / 无 inflation
方案 = CELF（替代 GC+infl，逐观测 U-Net）+ UV（z_C 上低秩算子, r=20 λ=0）
```

## 2. 结果（test 601–1100 TAS RMSE）

| 方案 | truth=MPI | truth=TRACE |
|---|---|---|
| AOEnKF-raw（无 loc/infl） | 0.5805 | 0.5640 |
| OEnKF-raw（无 loc/infl） | 0.5751 | **0.5187** |
| CELF only | 0.5261 | 0.6204 |
| **CELF + UV** | **0.4759** | **0.6188** |
| CELF+UV 相对最优 raw | **−0.099（有效）** | **+0.100（失效）** |

**窗口明细（TRACE）**：CELF only 1-550 = 0.210 / 551-600 = 0.219 / 601-1100 = 0.620；
CELF+UV = 0.172 / 0.199 / 0.619（UV 只改善训练/验证，测试几乎不变）。

## 3. 结论

1. **CELF+UV 在 TRACE 真值下失效**：0.619 劣于两个 raw 基线（AOEnKF-raw 0.564、OEnKF-raw 0.519）；
   而在 MPI 真值下它从 0.581 改善到 0.476。
2. **原因是非平稳**：CELF 在 1–550 上把误差压到 0.21，但在 601–1100 退化到 0.62——
   训练窗学到的修正在评估窗不迁移（与之前反复观察到的跨窗非平稳一致）；
   而 raw 基线不依赖训练窗，故在评估窗反而更稳。
3. **副观察**：TRACE 设定下 **OEnKF（0.519）优于 AOEnKF（0.564）**；而在 MPI 设定下二者接近（0.575 vs 0.580）。
   说明类比选择在 trace 真值下反而有害（可能选到"拟合观测但与 trace 偏离"的成员）。

## 4. Caveat
- OEnKF 为单次随机实现（seed 42），存在运行间波动；
- 池含 mpi（2500 步），与 MPI 设定（池=trace,had_g,had_h）在成员构成上不完全对称；
- trace 相对 {had_g,had_h} 后段是离群模式（见诊断），加入 mpi 仍未能覆盖评估窗。

## 5. 产物
- 数据：`pairs_trace_B/`（1100 步）
- 基线：`results/raw_mpi.npz|json`、`results/raw_traceB.npz|json`
- 模型：`results/loc_celf_traceB.pt`、`results/uv_traceB.npz|json`、`zC_traceB.npz`
- 图：`results/figs/F31_truth_swap.png`
- 代码：`build_pairs_traceB.py`、`oenkf_raw.py`、`train_uv_generic.py`、`make_fig_traceB.py`
