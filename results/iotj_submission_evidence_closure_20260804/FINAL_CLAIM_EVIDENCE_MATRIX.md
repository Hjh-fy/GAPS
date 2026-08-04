# Final claim–evidence matrix

| Claim ID | Manuscript-ready claim | Evidence | Link type | Status |
|---|---|---|---|---|
| C-FIG3 | GAPS classification is compared at fixed seed-42 endpoints across C3/C4/C5. | Fig.3; classification_main_comparison.csv | direct | SUPPORTED_WITH_SINGLE_SEED_LIMIT |
| C-FIG5 | The final A4 router with R84_FED_H1 provides the registered C5 concentration estimates. | Fig.5; regression_main_summary.csv; A4 checkpoint hash | direct | SUPPORTED |
| C-FIG6 | Source-prior and calibration-budget results are separate protocols and are not pooled. | Fig.6 source CSVs | direct | SUPPORTED |
| C-FIG7 | Final QC uses equal mean of three calibration-p95-normalized components and sealed-test thresholds. | qc_threshold_lock.csv; Fig.7 CSVs | direct | SUPPORTED |
| C-FIG8 | The exact final runtime runs on Raspberry Pi 5 at the measured latency, throughput, and RSS. | pi5_final_deployed_runtime_benchmark.json; FINAL_DEPLOYMENT_MANIFEST.json | direct | SUPPORTED |
| C-PARAM | The former value 80 denotes state tensor entries, while the classifier has 22,765 parameters and 91,060 FP32 parameter bytes. | FINAL_DEPLOYMENT_MANIFEST.json | direct | SUPPORTED_CORRECTED_SEMANTICS |

Claim boundary: seed 42 only; no uncertainty claim is made for classification comparisons. Hardware numbers apply only to the audited Pi 5 environment and fixed 5,000-window protocol.
