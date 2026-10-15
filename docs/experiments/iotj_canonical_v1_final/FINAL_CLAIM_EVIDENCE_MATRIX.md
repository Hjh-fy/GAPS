# Final claim-evidence matrix

| Claim | Evidence | Allowed wording | Status |
|---|---|---|---|
| Cross-target classification | `03_classification_final.csv` | A4 reaches 99.851%, 99.779%, 99.412% accuracy on C3/C4/C5 | SUPPORTED |
| Concentration regression | `04_regression_final.csv` | R84 S_ALL RMSE is 9.333/13.808/18.477 ppm | SUPPORTED |
| QC improves retained error over random retention | QC/random CSVs | At achieved HC90/HC95 accepted coverage, aggregate RMSE gain is 1.125/1.038 ppm | SUPPORTED_DESCRIPTIVELY |
| Edge runtime | `07_pi5_benchmark.csv` | P50/P95/P99 3.149/3.193/4.924 ms; 295.93 windows/s | SUPPORTED |
| 50% temporal input reduction | dataset manifest/model-size audit | 100x8 to 50x8 points, 3200 to 1600 raw FP32 bytes | SUPPORTED; do not infer 50% latency or FL communication |
| Equal-label superiority | `09_a0t_equal_label.csv`, `canonical_classification_comparison.csv` | A4 and equal-label A0T are effectively tied at seed42; do not claim material classification superiority beyond label access | SUPPORTED_WITH_LIMITATION |
| Independent calibration/test raw observations | `strict_non_overlap_summary.csv`, `strict_non_overlap_deltas.csv` | Strict raw-file-disjoint sensitivity was completed; C5 triggers both preregistered collapse flags | COMPLETED/BLOCKED_C5_ROBUSTNESS |
| Calibration-budget robustness | `CALIBRATION_BUDGET_GAP.md` | No canonical quantitative claim permitted | MISSING |
