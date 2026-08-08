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
| Equal-label fairness | BLOCKED | A0T not run | `09_a0t_equal_label.csv` | No result |
| FedRidge ablation | PASS/MIXED | ALL RMSE 13.849->13.314 ppm (3.86%); C4 RMSE worsens | `10_fedridge_83d_84d.csv` | Yes |
| Window overlap | BLOCKER | exact overlap 0; raw-time overlap present for C3/C4/C5 | `11_window_overlap.csv` | Yes |
| Deployment package | PASS | SHA256 `52328c9cd9f8...` | package manifest | Yes |
| Pi 5 | PASS | P50/P95/P99 3.149/3.193/4.924 ms; 295.93 windows/s; 258.92 MiB | `07_pi5_benchmark.csv` | Yes |
| Model size | PASS | 22,765 params; 91,060 FP32 bytes | `08_model_size.json` | Yes |
| Figure/table tracker | NEEDS_REGEN | canonical data mapped, final plots absent | tracker | Mixed |
| Manuscript consistency | BLOCKED | v7 source unavailable | consistency audit | N/A |

Decision: stop algorithm/preprocessing/R84/QC exploration. The project may enter final writing and canonical figure regeneration, but is **not yet submission-ready** until the A0T fairness gap, raw-time-overlap robustness, final figures, and v7 manuscript consistency are resolved. Calibration-budget evidence is an additional claim-dependent gap.
