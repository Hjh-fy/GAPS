# Final figure/table scientific map

This is the pre-plot evidence contract. `MISSING/NEEDS_REGEN` means the panel must not be presented as final canonical evidence yet.

| Item/panel | Scientific content | Data source -> CSV/JSON | Checkpoint/artifact hash | Producer script | Caption contract | Status |
|---|---|---|---|---|---|---|
| Fig.1 all | Cloud-edge-sensor lifecycle | canonical protocol -> `01_dataset_manifest.json`, deployment `package_manifest.json` | archive `52328c9c...c1c31` | illustration, no numerical producer | 5 Hz, 50x8, LE1; A4 -> R84_FED_H1 -> QC | NEEDS_REGEN |
| Fig.2 all | GAPS pipeline | classification/regression/QC manifests -> `02_final_experiment_state.json` | A4 C3/C4/C5 hashes; H1 `d32217a3...ecc`; QC `7da42eb5...04b` | illustration, no numerical producer | Distinguish source FL, labeled commissioning, routing, source sufficient statistics, R84 and QC | NEEDS_REGEN |
| Fig.3 all | Canonical device shift | `dataset/iotj_canonical_v1` -> canonical shift CSV not yet produced | dataset `2f810d7e...94f6` | MISSING | Descriptive sensor-space shift only; no causal claim | MISSING |
| Table I | FedAvg/FedProx/SCAFFOLD/MMD/A0T/GAPS classification | future comparator evaluation -> `canonical_classification_comparison.csv` | each round-25 endpoint to be indexed | frozen comparator runner/evaluator | Include optimizer and target-information regime; do not imply single-factor superiority across regimes | BLOCKED_BY_RUNS |
| Fig.4 a-c | A4 C3/C4/C5 classification | `03_classification_final.csv` | C3 `e2364290...4414`; C4 `422a49f2...99c3`; C5 `3965ec86...2b93` | `scripts/finalize_iotj_canonical_v1_evidence.py` | A4, round25, LE1, seed42, fixed endpoint | READY_FOR_REGEN |
| Fig.4 d | C5 A0-A6 ablation | only historical old-preprocessing artifact exists | legacy hashes only | historical final-classification scripts | Must be labeled historical/supplementary, never canonical-v1 | SUPPLEMENT_ONLY |
| Fig.5 a-c | S_ALL/S_CC/oracle regression by target/gas | regression predictions -> `04_regression_final.csv`, `routing_scope_summary.csv` | canonical A4 and R84 hashes above | `scripts/run_iotj_canonical_v1_r84.py`; `scripts/analyze_iotj_canonical_v1_scientific_claims.py` | S_ALL deployable; S_CC/oracle diagnostic populations | READY_FOR_REGEN |
| Table II | 83D versus 84D source-prior contribution | matched predictions -> `10_fedridge_83d_84d.csv`, `fedridge_bootstrap_summary.csv` | H1 `d32217a3...ecc`; R84 target hashes | `scripts/analyze_iotj_canonical_v1_scientific_claims.py` | Report grouped-bootstrap CI and C4 degradation; no significance claim | READY |
| Fig.6 | Strict non-overlap robustness | future strict split -> `strict_non_overlap_split_manifest.csv`, `strict_non_overlap_summary.csv` | future strict A4/R84 endpoints | strict builder/runner | Sensitivity analysis; does not replace canonical main protocol | BLOCKED_BY_RUNS |
| Fig.7 a | QC risk-coverage | QC predictions -> `qc_risk_coverage_final.csv` | QC `7da42eb5...04b` | `scripts/analyze_iotj_canonical_v1_scientific_claims.py` | Frozen equal-mean score; test used only for post-hoc curve | READY_FOR_REGEN |
| Fig.7 b | HC90/HC95 versus same-coverage random | `05_qc_final.csv`, `qc_error_capture_summary.csv` | QC `7da42eb5...04b` | evidence finalizer + scientific-claims analyzer | Random seed 20260804; report achieved coverage | READY_FOR_REGEN |
| Fig.8 a | Communication | model-size JSON plus analytical formulas/history -> `08_model_size.json` | canonical classifier hashes; H1 `d32217a3...ecc` | system audit | Separate analytical, historical measured, and reused-H1 quantities | READY_WITH_LIMITATION |
| Fig.8 b | Pi 5 deployment | package benchmark -> `07_pi5_benchmark.csv` | archive `52328c9c...c1c31` | `scripts/benchmark_iotj_canonical_v1_pi5.py` | 10,000 windows, batch1, four threads, `FINAL_DEPLOYED_RUNTIME` | READY |
| Fig.8 c | Physical/quality validation | `06_quality_robustness.csv` | canonical A4/R84/QC chain | evidence finalizer | Retain C5 methane 225 ppm repeat1; sparse Q1/Q2 and no Q3 limit robustness claim | READY_WITH_LIMITATION |

Calibration-budget sensitivity is deliberately absent. The manuscript may state only that a predefined approximately 20% commissioning calibration split was used.
