# Claim–Evidence matrix

| Claim | Section | Canonical value | Evidence | Status | Limitation |
|---|---|---|---|---|---|
| C1 | Abstract; 4.1 | Accuracy 0.989118 ± 0.005983; macro-F1 0.989134 ± 0.005960 | results/iotj_b5_multiseed_20260724/b5_classification_multiseed_summary.csv | approved | 仅覆盖 seeds 42–46 和 C1/C2→C5。 |
| C2 | Methods; 4.3 | 4/4 alpha agreement; max prediction difference 6.2532e-08 ppm | results/iotj_b5_c5_runtime_v5_candidate_20260724/federated_h1/equivalence_decision.json | approved | 数值实用等价，不是形式化安全性证明。 |
| C3 | Methods; Discussion | raw_source_rows_transmitted=false | results/iotj_b5_c5_runtime_v5_candidate_20260724/federated_h1/communication_payload_summary.json | approved | 统计量未由安全聚合或差分隐私保护。 |
| C4 | Abstract; 4.2 | paired relative S_CC degradation 0.981637% | results/iotj_b5_regression_multiseed_20260724/final_regression_decision.json | approved | RG2 在 5/5 seeds 的绝对 S_CC 更低。 |
| C5 | Abstract; 4.5 | 28737→844 params; 2,971,538→289,916 bytes; Pi p50 4.571→3.725 ms | results/iotj_final_system_benchmark_20260725/benchmarks/benchmark_summary.csv | approved | v5 core 不含 QC，比较需标注角色。 |
| C6 | 4.4; Discussion | HC95 yield 93.75%, accepted RMSE 13.9178 ppm; HC90 CO guard failed | results/iotj_b5_c5_runtime_v5_qc_20260725/decision_gate.json | approved | accepted RMSE 必须与 yield 同报。 |
| C7 | 4.4; 4.5 | FORMAL_BASELINE | results/iotj_final_system_benchmark_20260725/paper_tables/table_qc_overall.csv | approved | v5 regression core 与 v4 QC baseline 角色不同。 |
| C8 | 4.6; Discussion | group-aware S_CC 10.8724→23.9156 ppm when 320→160 | results/iotj_low_calibration_sensitivity_20260725/low_calibration_summary.csv | approved | 描述性 frozen-method sensitivity，不是新独立 confirmatory test。 |
| C9 | 4.7; Discussion | SENSITIVITY_PARTLY_PROTOCOL_DEPENDENT | results/iotj_calibration_protocol_harmonization_20260726/decision_gate.json | approved | 相同历史 test 被描述性复用。 |
| C10 | Protocol; Limitations | 61/61 validation filenames overlap fit | results/iotj_calibration_protocol_harmonization_20260726/historical_holdout_audit.json | approved | 这是 calibration-internal overlap，不是 test-label leakage。 |
