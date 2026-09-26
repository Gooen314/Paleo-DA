# REPORT 14：真值换成 HAD_g（原划分 1–550/551–600/601–1100）

**日期**：2026-09-13
**设定**：truth=HAD_g；观测=`pseudo_proxies_had_g`；池=`{trace, had_h, mpi}`（排除 had_g）
**划分**：1–550（训练）/ 551–600（早停）/ 601–1100（盲测）；OEnKF 种子 42。

---

## 1. 结果（逐时 TAS RMSE 均值）

| 窗口 | AOEnKF-raw | OEnKF-raw | CELF only | CELF+UV |
|---|---|---|---|---|
| 1–550 | 0.2126 | 0.2218 | 0.1918 | 0.1808 |
| 551–600 | 0.2167 | 0.2270 | 0.1964 | 0.1942 |
| **601–1100** | **0.2462** | 0.2833 | **0.2401** | 0.2426 |

UV 最优：r=5, λ=0（val 0.1942）。

## 2. 结论（HAD_g）
1. **raw 基线本身很低**（AOEnKF-raw test 0.2462）——因为 had_g 与池 `{trace,had_h,mpi}` 距离很近（可达）；
2. **方法只有边际增益**：CELF-only 0.2401（−0.0061，约 −2.5%）好于最优 raw；CELF+UV 0.2426（−0.0036）；
3. **此设定下 UV 无益**（CELF+UV 略差于 CELF-only）——PQ 的增益是设定相关的。

## 3. 三种真值的横向对比（test 601–1100，同划分/同池构造法）

| 真值 | AOEnKF-raw | OEnKF-raw | CELF only | **CELF+UV** | 相对最优 raw |
|---|---|---|---|---|---|
| MPI | 0.5805 | 0.5751 | 0.5261 | **0.4759** | **−0.099（大增益）** |
| TRACE | 0.5640 | 0.5187 | 0.6204 | 0.6188 | **+0.100（失效）** |
| HAD_g | 0.2462 | 0.2833 | **0.2401** | 0.2426 | −0.006（边际增益） |

**解读**：
- 真值与先验池的"距离"决定了 **raw 基线水平**：had_g 最近（raw 0.246）→ 改善空间极小；MPI 居中（0.58）→ CELF+UV 增益最大；trace 后段发散 → 学习修正反而放大误差；
- 因此 **CELF+UV 的收益依赖真值相对先验模式的"可达性/headroom"**：太近（had_g）无空间、太远（trace）有害，MPI 恰在有效区间。
- 这也解释了为何"同一方法、同一划分、同一池构造法"在三种真值下结论截然不同。

## 4. 产物
- 数据：`pairs_hadg_B/`（1100 步）
- 基线：`results/raw_hadgB.npz|json`
- 模型：`results/loc_celf_hadgB.pt`、`results/uv_hadgB.npz|json`、`zC_hadgB.npz`
- 图：`results/figs/F33_truth_settings.png`（三真值对比）
- 代码：`build_pairs_hadgB.py`、`make_fig_hadgB.py`
