# REPORT 13：窗口实验（truth=TRACE，train 300–699 / val 700–799 / test 1–299 ∪ 800–1100）

**日期**：2026-09-13
**设定**：truth=TRACE；观测=`pseudo_proxies_trace`；池=`{had_g,had_h,mpi}`
**划分**：训练 300–699（400步）｜验证 700–799（早停）｜测试 1–299 ∪ 800–1100（600步，盲测）

---

## 1. 结果（逐时 TAS RMSE 的均值）

| 窗口 | OEnKF-raw | AOEnKF-raw | CELF only | **CELF+UV** |
|---|---|---|---|---|
| 训练 300–699 | — | — | 0.2034 | 0.1805 |
| 验证 700–799 | — | — | 0.2861 | 0.2283 |
| **测试 1–299** | 0.2540 | 0.2589 | 0.2288 | **0.2169** |
| **测试 800–1100** | **0.6551** | 0.7312 | 0.8336 | 0.8146 |
| **测试 union** | **0.4552** | 0.4958 | 0.5322 | 0.5167 |

UV 最优：r=5, λ=0（按验证 700–799 择优，val 0.2283）。

## 2. 结论

1. **结果随时段分裂**：
   - **早段（1–299，训练之前）**：CELF+UV **0.217** 优于 OEnKF-raw 0.254 / AOEnKF-raw 0.259 —— **方法有效**；
   - **晚段（800–1100，验证之后）**：CELF+UV **0.815** **显著劣于** OEnKF-raw 0.655 —— **方法失效**；
2. **合并测试集上方法整体失效**（0.517 vs 最优 raw 0.455）；
3. **原因仍是跨段非平稳**：CELF+UV 在训练覆盖/可达的时段（早段、中段）有效，但在 trace 与池子发散的晚段（≈850 之后）把误差放大；
4. 逐时曲线显示：CELF+UV 在 1–800 基本优于 OEnKF-raw，之后急剧恶化（≥0.8）——与 trace 后段离群一致。

## 3. 与之前实验的关系
- truth=TRACE + 测试 601–1100（REPORT12）：CELF+UV 0.619 vs raw 0.519（失效）；
- 本实验把训练窗移到中部、测试分成两端：**早段成功、晚段失败**；
- 两实验共同说明：**CELF+UV 的增益只在"真值模型处于池子可达范围"的时段成立**，一旦真值轨迹偏离先验模式集合就失效。

## 4. 产物
- 模型：`results/loc_celf_traceB_win.pt`、`results/uv_traceB_win.npz|json`、`zC_traceB_win.npz`
- 结果：`results/traceB_win.json`、`results/figs/F32_window_trace.png`
- 代码：`train_celf_raw.py`（+`--train_start`）、`train_uv_generic.py`（+`--tr_start`）、`make_fig_traceB_win.py`
