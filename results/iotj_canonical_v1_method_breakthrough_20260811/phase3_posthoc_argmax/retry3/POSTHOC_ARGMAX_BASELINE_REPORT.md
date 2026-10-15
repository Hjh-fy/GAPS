# Posthoc Argmax R84 baseline

Selected identity: `I0+B20` by `simplest_effective_within_0.01_of_best_B20`.

C5 classification Accuracy/Macro-F1: 0.976471/0.976544.

| Scope | N | RMSE | MAE | NRMSE | R2 | Bias |
|---|---:|---:|---:|---:|---:|---:|
| S_ALL | 1360 | 28.057496 | 10.691171 | 0.199727 | 0.821260 | 1.167081 |
| S_CC | 1328 | 13.110113 | 7.553216 | 0.065601 | 0.960362 | 0.720222 |
| Oracle_ALL | 1360 | 14.448829 | 7.939890 | 0.070837 | 0.952599 | 0.288836 |
| Oracle_CC | 1328 | 13.110113 | 7.553216 | 0.065601 | 0.960362 | 0.720222 |

The classifier was not retrained. R84 used the unchanged H1 source pool and fixed C5 alpha table; C5 test opened only after the calibration lock.
