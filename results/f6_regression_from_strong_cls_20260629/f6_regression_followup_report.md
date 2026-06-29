# F6 Strong Fixed-DA Classification Backbone Regression Follow-up

## Summary Table

| profile | ALL RMSE | ALL NRMSE | C3 NRMSE | C4 NRMSE | C5 NRMSE | C3 CO | C4 CO | C5 CO | C4 high CO | C5 high CO | nonCO |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| old_B0_final_r25 | 27.34 | 0.1578 | 0.1108 | 0.1520 | 0.2272 | 33.70 | 56.59 | 46.12 |  |  |  |
| old_H2_3_current | 18.62 | 0.1326 | 0.1023 | 0.0713 | 0.2100 | 16.15 | 22.02 | 26.85 | 34.24 | 34.82 | 17.83 |
| F6_r25_adapted_B0 | 28.57 | 0.1899 | 0.1131 | 0.1396 | 0.3138 | 33.57 | 50.36 | 43.36 |  |  |  |
| F6_r25_H2_3_current | 22.94 | 0.1789 | 0.0978 | 0.0734 | 0.3207 | 15.19 | 22.99 | 25.23 | 36.05 | 30.80 | 23.79 |
| F6_r25_H2_mlp_rescue | 22.70 | 0.1784 | 0.0978 | 0.0781 | 0.3185 | 15.19 | 25.26 | 26.39 | 35.84 | 33.33 | 23.17 |
| F6_r25_H1_ridge_rescue | 23.48 | 0.1806 | 0.0966 | 0.0734 | 0.3252 | 18.39 | 22.99 | 20.50 | 36.05 | 20.85 | 24.49 |
| F6_r19_adapted_B0 | 27.09 | 0.1633 | 0.1101 | 0.1504 | 0.2436 | 33.70 | 55.95 | 43.77 |  |  |  |
| F6_r19_H2_3_current | 18.62 | 0.1357 | 0.0846 | 0.0610 | 0.2352 | 16.14 | 17.31 | 25.92 | 24.24 | 30.80 | 18.38 |
| F6_r19_H2_mlp_rescue | 18.26 | 0.1348 | 0.0846 | 0.0691 | 0.2309 | 16.14 | 17.77 | 26.98 | 22.94 | 33.33 | 17.71 |
| F6_r19_H1_ridge_rescue | 19.54 | 0.1427 | 0.0901 | 0.0610 | 0.2472 | 19.11 | 17.31 | 21.14 | 24.24 | 20.85 | 19.65 |

## Reading

- Official F6 final adapted r25 restores classification but does not improve full-set regression by itself: B0 ALL RMSE is 28.57 versus old B0 27.34.
- The gain is client/gas specific: F6 r25 improves C4 CO and C5 CO, but C5 nonCO becomes much worse, so C5 NRMSE rises to 0.3138.
- Target calibration still helps under F6 r25: H2.3 current reduces ALL RMSE from 28.57 to 22.94, but this is worse than old H2.3 18.62 because C5 nonCO remains weak.
- Best adapted round19 is a better regression diagnostic than final r25: B0 ALL RMSE returns to 27.09, and H2 MLP + C4 rescue reaches ALL RMSE 18.26 / ALL NRMSE 0.1348, slightly better than old H2.3 ALL RMSE 18.62.
- However, round19 should be reported as best-checkpoint/oracle analysis, while official 25-round final remains r25 adapted unless a validation-based checkpoint selector is formalized.

## C5 Diagnosis

The main regression degradation is not from C5 CO after F6. C5 CO improves:

- old B0 C5 CO RMSE: 46.12
- F6 r25 B0 C5 CO RMSE: 43.36
- F6 r25 H2.3 C5 CO RMSE: 25.23
- F6 r19 H2.3 C5 CO RMSE: 25.92

The full-set degradation comes from C5 nonCO, especially low-concentration Ethanol/Ethylene windows routed as CO and then mapped to high CO ppm by the old auto_v2 package:

- old B0 C5 Ethanol route accuracy: 0.9824; F6 r25: 0.9382
- old B0 C5 Ethylene route accuracy: 0.9706; F6 r25: 0.9412
- old B0 C5 Ethanol->CO wrong routes: 3; F6 r25: 13
- old B0 C5 Ethylene->CO wrong routes: 10; F6 r25: 20

These wrong-route nonCO samples often have true ppm 12.5-25 but final ppm near 240, which dominates C5 NRMSE. Round19 reduces this issue but does not remove it completely.

## Entrypoints

- F6 r25 ppm audit: `results/f6_c12_c345_strong_r25_r3ak16_auto_v2_eval/ppm_layer_co_audit/`
- F6 r25 H2.3: `results/f6_h2_3_no_b0_feature_ablation_20260629/c12_c345/`
- F6 r19 ppm audit: `results/f6r19_c12_c345_strong_r25_r3ak16_auto_v2_eval/ppm_layer_co_audit/`
- F6 r19 H2.3: `results/f6r19_h2_3_no_b0_feature_ablation_20260629/c12_c345/`
- Comparison CSV: `results/f6_regression_from_strong_cls_20260629/f6_regression_comparison.csv`
