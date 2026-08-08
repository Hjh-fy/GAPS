# Final claim-evidence matrix

| Claim | Evidence | Allowed wording | Status |
|---|---|---|---|
| Cross-target classification | `03_classification_final.csv` | A4 reaches 99.851%, 99.779%, 99.412% accuracy on C3/C4/C5 | SUPPORTED |
| Concentration regression | `04_regression_final.csv` | R84 S_ALL RMSE is 9.333/13.808/18.477 ppm | SUPPORTED |
| QC improves retained error over random retention | QC/random CSVs | At achieved HC90/HC95 accepted coverage, aggregate RMSE gain is 1.125/1.038 ppm | SUPPORTED_DESCRIPTIVELY |
| Edge runtime | `07_pi5_benchmark.csv` | P50/P95/P99 3.149/3.193/4.924 ms; 295.93 windows/s | SUPPORTED |
| 50% temporal input reduction | dataset manifest/model-size audit | 100x8 to 50x8 points, 3200 to 1600 raw FP32 bytes | SUPPORTED; do not infer 50% latency or FL communication |
| Equal-label superiority | `09_a0t_equal_label.csv` | No comparative claim permitted | BLOCKED |
| Independent calibration/test raw observations | `11_window_overlap.csv` | Exact window identities are disjoint, but raw-time overlap is present | CONTRADICTED/BLOCKED |
| Calibration-budget robustness | `CALIBRATION_BUDGET_GAP.md` | No canonical quantitative claim permitted | MISSING |
