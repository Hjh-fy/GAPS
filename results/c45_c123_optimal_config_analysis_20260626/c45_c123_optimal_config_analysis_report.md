# C45 -> C123 Optimal-Config Target Analysis

Scope: source clients C4/C5, target clients C1/C2/C3, target test, no-QC full-set.

- Calibration-selected client profile: `{"C1": "mlp", "C2": "mlp", "C3": "mlp"}`
- Best test-only diagnostic profile: `C1:ridge,C2:ridge,C3:ridge`

## Test Metrics

| mode | family | ALL | NRMSE | C1 CO | C2 CO | C3 CO | C1 high | C2 high | C3 high | nonCO |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A0_baseline_final | reference | 22.94 | 0.1473 | 37.68 | 22.00 | 32.31 | 51.97 | 25.56 | 38.16 | 19.34 |
| H1_target_Ridge_direct | formal direct head | 15.59 | 0.1123 | 23.77 | 15.55 | 14.68 | 38.30 | 17.09 | 16.37 | 14.50 |
| H2_target_MLP_direct | formal direct head | 16.46 | 0.1201 | 23.84 | 17.31 | 14.93 | 39.73 | 22.13 | 18.47 | 15.49 |
| H2b_source_aug_target_Ridge | source-augmented target Ridge | 16.01 | 0.1183 | 24.92 | 15.78 | 10.73 | 40.16 | 17.35 | 11.04 | 15.24 |
| H8_style_source_aug_CO_else_Ridge | CO specialist switch + Ridge fallback | 16.13 | 0.1192 | 24.70 | 15.78 | 10.73 | 39.70 | 17.35 | 11.04 | 15.44 |
| H3_calibration_client_hybrid | calibration-selected Ridge/MLP | 16.46 | 0.1201 | 23.84 | 17.31 | 14.93 | 39.73 | 22.13 | 18.47 | 15.49 |
| H4_test_oracle_client_hybrid | test-only diagnostic | 15.59 | 0.1123 | 23.77 | 15.55 | 14.68 | 38.30 | 17.09 | 16.37 | 14.50 |

## Test-Only Hybrid Grid

| combo | ALL RMSE |
| --- | ---: |
| C1:ridge,C2:ridge,C3:ridge | 15.59 |
| C1:ridge,C2:mlp,C3:ridge | 15.69 |
| C1:mlp,C2:ridge,C3:ridge | 15.84 |
| C1:mlp,C2:mlp,C3:ridge | 15.94 |
| C1:ridge,C2:ridge,C3:mlp | 16.12 |
| C1:ridge,C2:mlp,C3:mlp | 16.21 |
| C1:mlp,C2:ridge,C3:mlp | 16.36 |
| C1:mlp,C2:mlp,C3:mlp | 16.46 |

## Reading

- Reverse direction C45 -> C123 benefits strongly from target direct-head calibration.
- Formal target Ridge is the current best clean reverse-direction candidate among Ridge/MLP direct heads.
- Source-augmented target Ridge and H8-style CO switching improve C3 CO/high-CO, but they worsen ALL RMSE and nonCO, so they are diagnostic CO-specialist variants rather than the reverse mainline.
- Calibration selection prefers MLP on C1/C2/C3, but this overfits calibration badly; test ALL is worse than all-Ridge.
- The test-only oracle is diagnostic only. If it does not beat Ridge materially, there is little reason to build a more complex H2.3-style profile for this direction.
- C4 route-rescue is not relevant in this direction because C4 is a source client, not a target client.
