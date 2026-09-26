# REPORT 10：xb 算子路线的收束（放弃）

**日期**：2026-09-13
**决定**：用户决定**放弃**"把算子放在 xb（先验均值）上 / 作为 CELF 残差修正"这一整条路线。本文档画上句号。

---

## 1. 路线回顾与全部结果（test 601–1100 TAS RMSE）

| 编号 | 形式 | test | 结论 |
|---|---|---|---|
| P1-a | `xa = xa_raw + U(Vᵀxb)`（raw 基础，不自洽） | 0.5556 ± 0.0009 | 未过原门限 |
| P1-c | `xa = xa_raw + (I-K0H)U(Vᵀxb)`（raw 基础，自洽） | **0.5419** | 用户降门槛(<0.55)后通过，但仍未超 AOEnKF_Opt |
| P2 | `corr = Net(xb,spread,coslat)`（动态） | 0.6536 ± 0.0188 | 过拟合，劣于 raw |
| P3 | 增量算子 + xb 算子叠加（raw 基础） | 0.5403（r5,λ1e-3）/ 0.7963（r20,λ0） | 与 P1-c 持平，无增益 |
| P0′ | 诊断：e_C 能否由 xb 预测？ | 从 xb 预测 0.807；从 zC 预测 0.486 | **不能由 xb 预测** |
| 确认 | `xa = xa_C + U(Vᵀxb)`（CELF 之上加性） | r5λ0 0.7522 / r5λ1e-3 **0.5173** / r20λ0 0.7529 | 最优仅略优于 CELF only（0.5261），仍差于 AOEnKF_Opt |

## 2. 结论（为什么放弃）

1. **xb 不能预测误差**：先验误差 `xb-xt`（P0）与 CELF 残差 `e_C`（P0′）都无法由 xb 线性预测（0.807/0.877 远差于基线）；直接相关 corr(e_C, xb) ≈ −0.1。
2. **非平稳是根本障碍**：先验误差从训练窗到评估窗近乎翻倍（train 0.296 → test 0.657；e_C: 0.238 → 0.526），任何以 xb 为输入的静态/动态修正都学不到可迁移结构。
3. **对比反证**：同样的低秩形式，输入换成**增量 z**（当前观测/集合自适应）就有效（GC+PQ 0.4724；CELF+PQ 0.4759），输入 zC 的预测诊断也与实际结果吻合（0.486 vs 0.4759）——**有效信息在增量里，不在先验里**。

## 3. 项目当前最优（评估窗 601–1100 TAS RMSE）

```
AOEnKF_Opt (GC7500+infl1.5 基线)   0.5021
CELF+PQ (无 GC/无 inflation)        0.4759
GC+PQ  (增量算子, 最好)             0.4724   ← 当前最优
```

## 4. 产物（本路线，归档）
- 代码：`build_H.py`、`build_H2.py`、`train_xb_operator.py`、`train_dyn_xb.py`、`train_combined.py`、`xb_on_celf.py`、`p0prime_diag.py`、`make_fig_xb.py`
- 结果：`results/xb_op_a|c.npz|json`、`results/dyn_xb.json`、`results/combined.json`、`results/P0PRIME.md`、`results/REPORT8.md`、`results/REPORT9.md`、`results/figs/F29_xb_operator.png`
- 观测算子：`H_refined.npz`（精确，RMSE 4.5e-7，可复用）
