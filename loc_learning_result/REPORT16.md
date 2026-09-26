# REPORT 16：四候选逐时选择（观测空间残差门控）

**日期**：2026-09-13
**目的**：解决 truth=TRACE（尤其 601–1100）CELF+UV 发散；同时检验"逐时四选一"能否不劣于固定最优、逼近 oracle。

## 1. 设置

```
每时刻并行 4 个候选（先验集合相同：无纬度加权类比选择，池=其余3模式）：
  1 raw         : xa = xf + K0 d                    （无 loc / 无 inflation）
  2 CELF+UV     : xa = xf + z_C + U_C(V_C^T z_C)
  3 Loc8000     : xa = xf + z_loc，z_loc = 串行 EnSRF(GC 8000, infl=1.0)
  4 Loc8000+UV  : xa = xf + z_loc + U_L(V_L^T z_loc)  （UV_loc 单独在该 loc 增量上训练）
判据：逐时选观测空间残差最小者（cos-lat 加权）；另给 R 归一化 chi2 版对照
划分：1-550 / 551-600 / 601-1100；评估 TAS cos-lat 加权 RMSE
```

## 2. 结果（state RMSE）

| | raw | CELF+UV | Loc8000 | Loc8000+UV | **选择(obs-res)** | 选择(chi2) | oracle |
|---|---|---|---|---|---|---|---|
| **MPI** test | 0.5771 | 0.4737 | 0.5097 | **0.4546** | **0.4601** | 0.4991 | 0.4488 |
| **TRACE** test | 0.5644 | 0.6133 | 0.5131 | 0.6220 | **0.5018** | 0.5122 | 0.4916 |

被选比例（test，obs-res）：MPI `raw 0 / celf 19.2% / loc 25.0% / loc_uv 55.8%`；TRACE `raw 0 / celf 6.2% / loc 61.0% / loc_uv 32.8%`。

## 3. 结论

1. **加入 Loc 候选是关键增益**：
   - **MPI：`Loc8000+UV = 0.4546`，创下新最好**（超过阶段③ 0.4724、CELF+UV 0.4737）；
   - **TRACE：`Loc8000 = 0.5131`**，远好于 raw 0.5644 / CELF+UV 0.6133。
2. **逐时选择（obs-res）**：
   - **TRACE 0.5018**：**优于全部固定候选**、满足 <0.5187 的目标，逼近 oracle 0.4916——**晚段自动避开 CELF+UV/UV_loc 的发散**；
   - **MPI 0.4601**：优于此前最好（0.4724），但**略逊于固定最优 loc_uv 0.4546**（选择器并非对 MPI 最优）。
3. **chi2(R 归一化) 判据反而更差**（MPI 0.4991 / TRACE 0.5122）——本数据上朴素 cos-lat 加权残差更可靠。
4. oracle（用真值事后选，仅诊断）MPI 0.4488 / TRACE 0.4916；TRACE 选择器已捕获大部分空间。

## 4. Caveat
- **判据不独立**（4 个候选都用同一 y）：有轻微"过拟合观测"风险；MPI 下选择器离 oracle 仍有 0.011 差距，说明可改进；
- **逐时硬切换**（未加驻留/滞回，功能已预留）；
- oracle 用真值、不可实现；且口径敏感——**总体 RMSE 口径**下 MPI oracle 为 0.4873（≠逐时 min 的 0.4488）。

## 5. 产物
- 代码：`build_loc_candidate.py`、`train_uv_generic.py`（复用）、`eval_select.py`、`make_fig_select4.py`
- 数据：`loc8000_{mpi,trace}.npz`、`zloc_*.npz`、`results/uvloc_*.npz`
- 结果：`results/select4.json|npz`；图 `F38_select4_curves.png`、`F39_select4_bands.png`
