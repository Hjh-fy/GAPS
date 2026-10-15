# Final submission readiness

| Evidence item | Status | Main result | Artifact | Canonical? |
|---|---|---|---|---|
| Dataset | PASS | hash `2f810d7e93ca...`; 5 Hz, 50x8 | `01_dataset_manifest.json` | Yes |
| Classification | PASS | C3/C4/C5 accuracy 99.851/99.779/99.412% | `03_classification_final.csv` | Yes |
| Regression | PASS | RMSE 9.333/13.808/18.477 ppm | `04_regression_final.csv` | Yes |
| QC HC90 | PASS | coverage 85.90%, NRMSE 0.06424 | `05_qc_final.csv` | Yes |
| QC HC95 | PASS | coverage 90.77%, NRMSE 0.06484 | `05_qc_final.csv` | Yes |
| Random reference | PASS | aggregate RMSE gain 1.125/1.038 ppm | source index | Yes |
| Quality robustness | PASS/WARNING | methane225 repeat1 RMSE 70.969 ppm, retained | `06_quality_robustness.csv` | Yes |
| Equal-label fairness | PASS/LIMITED | A0T C3/C4/C5 Macro-F1 0.998505/0.994857/0.994139; A4 deltas +0.000001/+0.002937/-0.000013 | `09_a0t_equal_label.csv`, `canonical_classification_comparison.csv` | Yes |
| Comparator matrix | PASS/LIMITED | FedAvg/FedProx/SCAFFOLD/MMD/A0T/GAPS fixed endpoints complete; supervision and optimizer regimes disclosed | `canonical_classification_comparison.csv` | Yes |
| FedRidge ablation | PASS/MIXED | ALL RMSE 13.849->13.314 ppm (3.86%); C4 RMSE worsens | `10_fedridge_83d_84d.csv` | Yes |
| Strict non-overlap | BLOCKED_C5 | C3/C4 retain near-canonical performance; C5 delta Macro-F1 -0.300519 and delta S_ALL RMSE +54.461 ppm | `strict_non_overlap_deltas.csv` | Supplementary sensitivity |
| Deployment package | PASS | SHA256 `52328c9cd9f8...` | package manifest | Yes |
| Pi 5 | PASS | P50/P95/P99 3.149/3.193/4.924 ms; 295.93 windows/s; 258.92 MiB | `07_pi5_benchmark.csv` | Yes |
| Model size | PASS | 22,765 params; 91,060 FP32 bytes | `08_model_size.json` | Yes |
| Figure/table tracker | NEEDS_REGEN | canonical data mapped, final plots absent | tracker | Mixed |
| Manuscript consistency | BLOCKED | v7 source unavailable | consistency audit | N/A |

Decision: stop algorithm/preprocessing/R84/QC exploration. The experimental matrix is complete, but the project is **not yet submission-ready** because the strict C5 robustness claim is blocked, canonical figures remain to be regenerated, and v7 manuscript consistency remains unaudited. Calibration-budget evidence is an additional claim-dependent gap.
