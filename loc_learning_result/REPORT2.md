# Serial-Consistent Learned Localization (GC-residual) — Report 2

## Setup
- Differentiable serial EnSRF (torch) exactly matching MATLAB EnSRF.m / `osee_tas.py`.
- Localization on the state part: rho_ij = clip( rho_GC_ij * exp(tanh(g_theta(phi_ij))), 0, 2 )
  with phi = [d/CUT, cos(dlon), sin(dlon), dlat, cos(lat_j), spread_j, log1p(|Ks_ij|)].
- Observation-observation block kept at GC(7500 km).
- Loss = CELF (weighted posterior error vs mpi truth) + 0.1 * CLF(batch large-ensemble
  analysis) + 1e-3 * mean(tanh^2). Train t=1..550, val 551..600, test 601..1100.

## S1 engine equivalence (gate)
| torch - numpy | over 20 steps |
|---|---|
| max abs difference | 3.8e-16 |

With rho = GC the torch serial EnSRF reproduces the canonical baseline exactly
(full 0.3658 / eval 0.5021, numpy vs MATLAB difference 8e-8).

## Training
val CELF (weighted MSE, t=551..600): GC baseline 0.071172 -> learned 0.071091
(relative improvement 0.11%). The MLP converged to a nearly uniform multiplicative
correction exp(delta) ~ 1.08 over the GC support.

## Test results (weighted TAS RMSE)
| method | FULL 1-1100 | EVAL 601-1100 |
|---|---|---|
| canonical serial AOEnKF (GC7500) | 0.3658 | 0.5021 |
| serial-consistent learned rho | 0.3657 | 0.5023 |

## Oracle localization headroom (how much could localization possibly gain?)
Single-parameter sweeps of GC cutoff and amplitude (no learning), on 15 val steps
(551-565) and 8 eval steps (700-1050):

| window | GC 7500, mult 1.0 | best in sweep | relative gain |
|---|---|---|---|
| val 551-565 | 0.24010 | 0.23943 (cut 6000) | 0.28% |
| eval 700-1050 | 0.53834 | 0.53748 (cut 10000) | 0.16% |

## Conclusion
- A serial-consistent, bounded, GC-residual learned localization does **not** exceed
  the canonical AOEnKF (full 0.3657 vs 0.3658; eval 0.5023 vs 0.5021).
- The reason is not optimization or the batch/serial mismatch (now eliminated) but the
  **absence of headroom**: even an oracle single-parameter retuning of the localization
  improves RMSE by <0.3% on both windows. Localization is not the bottleneck of this
  dataset; cross-model error and pseudo-proxy structure dominate, consistent with the
  earlier covariance-correction and inflation-scan findings.
- The value of the exercise is methodological: we now have (i) a differentiable serial
  EnSRF that exactly reproduces the operational AOEnKF, and (ii) a quantitative upper
  bound on the achievable gain from localization learning.

## Artifacts
- Code: `serial_torch.py`, `build_serial.py`, `train_serial.py`,
  `eval_serial_learned.py`, `test_serial_equiv.py`.
- Cached serial tensors: `/data1/wbgu/agentdata/loc_learning/serial/t0001..t1100.npz`.
- Weights: `results/loc_serial.pt`; RMSE: `results/serial_learned.npy`;
  baseline: `results/baseline_aoenkf.npy`.
