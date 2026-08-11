# Frozen QC risk-coverage validation

Evidence status: `INVALID_FOR_FINAL_CANONICAL_CLAIM`. These QC diagnostics are
computed correctly from the saved predictions, but those R84 predictions depend
on a legacy `100x8` H1 prior mixed into the canonical `50x8` pipeline. They
cannot close the final canonical QC claim.

All comparators use the exact Phase-1 C5 post-hoc R84 predictions. Calibration
risk fields determine the frozen normalization scales. The analysis curve ranks
deployment-visible test risks without labels and retains identical counts;
target-test labels are used only after retention to compute error and event-
capture diagnostics. Q0 uses 1000 fixed-seed random draws per coverage. AURC
uses the same deterministic coverage grid and NumPy trapezoidal integration.

| Policy | NRMSE AURC | RMSE AURC |
|---|---:|---:|
| Q0 random | 0.099492 | 14.003338 |
| Q1 confidence | 0.039913 | 7.004838 |
| Q2 regression disagreement | 0.107931 | 13.437286 |
| Q3 frozen equal mean | 0.099131 | 12.315601 |

At both actual HC90 and HC95 counts, Q3 has lower RMSE than random retention but
higher RMSE than confidence-only. Decision:
`QC_RANKING_SUPPORTED__MULTISIGNAL_ADVANTAGE_NOT_SUPPORTED`. The manuscript may retain a calibration-locked QC ranking claim,
but the present C5 evidence does not support an advantage for the multi-signal
equal-mean mechanism over classifier confidence alone.
