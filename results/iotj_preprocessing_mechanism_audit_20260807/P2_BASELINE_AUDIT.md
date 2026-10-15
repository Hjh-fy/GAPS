# P2 baseline G0 audit

`p2_baseline_g0_comparison.csv` gives legacy/current G0 for every file/channel. `p2_baseline_counterfactual_regression.csv` reports two fixed counterfactuals: (A) time-aware response with legacy G0 and (B) legacy response with time-aware G0. They are diagnostic-only oracle-route Ridge refits; no classifier was retrained and no target test item selected alpha or preprocessing.

| Variant | C3 RMSE | C4 RMSE | C5 RMSE |
|---|---:|---:|---:|
| A_timeaware_response_legacy_G0 | 9.4945 | 11.5930 | 10.7443 |
| B_legacy_response_timeaware_G0 | 10.5336 | 12.8923 | 12.8416 |
