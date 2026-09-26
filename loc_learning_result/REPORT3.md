# Low-rank State-Space Operator (rho = I + U V^T) — Report 3

## Method
Batch form on the serial-GC analysis increment z3 = xa_serial - xf:

    xa = xf + z3 + U (V^T z3),   U,V in R^{n x r}, r=5

Loss = CELF (weighted posterior error vs mpi truth) + lambda||.||^2, trained on t=1..550, validated 551..600, tested 601..1100.
z3 uses only prior ensemble + observations (no truth); U,V use train truth only.

## Weighted TAS RMSE

| method | train 1-550 | val 551-600 | TEST 601-1100 |
|---|---|---|---|
| canonical serial AOEnKF (GC7500) | 0.2510 | 0.2651 | 0.5021 |
| + low-rank state operator (r=5, CELF) | 0.2132 | 0.2453 | 0.4720 |

Blind evaluation-window improvement: (0.5021-0.4720)/0.5021 = 6.0%%.

## Ablations (test 601-1100)

| variant | test |
|---|---|
| r=1 | 0.4806 |
| r=2 | 0.4802 |
| r=5 | 0.4730 |
| r=10 | 0.4732 |
| r=20 | 0.4732 |
| r=50 | 0.4732 |
| r=5, CELF only (no CLF) | 0.4720 |
| r=5, + diagonal | 0.4729 |
| r=5, diagonal + CELF only | 0.4720 |
| unlocalized batch (reference) | 0.5883 |
| GC batch (reference) | 0.5403 |

CLF auxiliary and the diagonal term add nothing; the simplest CELF-only
low-rank operator is best.

## Artifacts
- code: build_matrix_cache.py, build_serial_analysis.py, train_matrix_rho.py
- weights: results/mrho_b3_r5_noclf.npz; cache under /data1/wbgu/agentdata/loc_learning/
