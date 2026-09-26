# Learned Localization (CLF/CELF) Pilot Report

TAS-only, per-observation 2D denoiser (U-Net, 3 common modes not used).
Training t=1..550, val 551..600, test 601..1100. K=100 analog prior, GC7500.

## RMSE (weighted TAS)

| method | FULL 1-1100 | EVAL 601-1100 |
|---|---|---|
| canonical serial AOEnKF (GC7500) | 0.3658 | 0.5021 |
| batch prior (no update) | 0.4617 | 0.6568 |
| batch unlocalized | 0.4231 | 0.5883 |
| batch GC | 0.3854 | 0.5403 |
| batch CLF | 0.3866 | 0.5518 |
| batch CELF | 0.3725 | 0.5385 |
| serial CLF (clip[0,1.2]) | n/a (eval-only run) | 0.5204 |
| serial CELF (clip[0,1.2]) | n/a (eval-only run) | 0.5355 |

## Relative gain error vs large-ensemble gain (val 551-600)

| rho | rel. Frobenius gain error |
|---|---|
| unloc | 1.5593 |
| gc | 0.7522 |
| clf | 0.6851 |
| celf | 0.8031 |

## Findings

- CELF improves over the matched batch GC (full 0.3725 vs 0.3854, ~3.3%),
  but the eval-window improvement is marginal (0.5385 vs 0.5403) and both are
  worse than the canonical serial AOEnKF (0.3658 full / 0.5021 eval).
- CLF (sampling-error target) matches GC in batch and is worse on eval.
- Serial deployment of the learned rho (the operational EnSRF) is unstable
  without clipping (diverges at t~900); with clip [0,1.2] it is stable but
  worse than GC (CLF 0.5204, CELF 0.5355 vs GC 0.5021 eval).
- The batch/serial mismatch is the pilot's key limitation: the paper trains
  on the full gain matrix and applies L(K) directly; our operational AOEnKF is
  a serial EnSRF, where per-observation learned factors compound differently.

Artifacts: results/batch_eval.npz, serial_{clf_c,celf_c}.npy,
results/loc_clf.pt, results/loc_celf.pt, loc_learning_comparison.png
