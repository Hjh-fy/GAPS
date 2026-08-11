# Phase 3 result analysis

## Registered identity and classification

The preregistered simplest-effective rule selected `I0+B20`: its C5 B20
Macro-F1 was within 0.01 of the best Phase 2 identity. The immutable post-hoc
classifier achieved Accuracy `0.976471` and Macro-F1 `0.976544` on the sealed
C5 test.

## Frozen R84 result

| Scope | N | RMSE | MAE | NRMSE_range | R2 | Bias |
|---|---:|---:|---:|---:|---:|---:|
| S_ALL | 1360 | 28.057496 | 10.691171 | 0.199727 | 0.821260 | 1.167081 |
| S_CC | 1328 | 13.110113 | 7.553216 | 0.065601 | 0.960362 | 0.720222 |
| Oracle_ALL | 1360 | 14.448829 | 7.939890 | 0.070837 | 0.952599 | 0.288836 |
| Oracle_CC | 1328 | 13.110113 | 7.553216 | 0.065601 | 0.960362 | 0.720222 |

The all-window argmax routing penalty is large: `RMSE(S_ALL)-RMSE(S_CC) =
14.947383 ppm`. Oracle routing reduces all-window RMSE by `13.608668 ppm`
relative to argmax S_ALL. The equality of S_CC and Oracle_CC is expected
because both contain the correctly classified subset using the true gas route.

Per-gas S_ALL RMSE is `20.2143` ppm (Ethanol), `21.5693` ppm (CO), `35.1339`
ppm (Ethylene), and `32.2591` ppm (Methane). Under oracle routing those values
fall to `4.7603`, `19.9001`, `6.1393`, and `19.4604` ppm, respectively.

## Interpretation for Phase 4

This result establishes the required immutable argmax baseline. It does not
claim that cost-aware routing is beneficial. It shows enough downstream error
heterogeneity to justify the already-preregistered Phase 4 direct test, whose
cost matrix must be built only from B20 calibration and locked before test
comparison.
