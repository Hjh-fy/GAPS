# Runtime Profile Benchmark

- data_root: `dataset/client_data_c12src_c345tgt_2080_timeaware_60_170_window_fullgrid`
- clients: `C3,C4,C5`
- limit per client: `300`
- repeats: `3`
- device: `cpu`

|profile|role|status|artifact_size_mb|model_file_size_mb|mean_latency_ms_per_window|p90_latency_ms_per_window|expected_full_all_rmse|benchmark_subset_rmse|auto_output_field_present|
|---|---|---|---|---|---|---|---|---|---|
|H2.3|balanced_mainline|ok|24.0419|21.238|2.73438|3.28685|18.62|20.380173|False|
|H8|co_specialist|ok|24.638|21.238|2.63191|2.84976|18.47|20.127612|False|
|H8+C4|guarded_co_priority|ok|24.6414|21.238|2.58565|2.72153|18.3|20.127612|True|
|L1|deployment_lite_candidate|missing_bundle|0.0|0.0|||22.6|||
|B0|baseline_reference|missing_bundle|0.0|0.0|||27.34|||

Notes:
- `benchmark_subset_rmse` is measured on the benchmark subset only; use `expected_full_all_rmse` for the fixed no-QC full-set model comparison.
- Profiles with `missing_bundle` are not deployment-validated yet and should not be claimed as runtime-ready.
- `auto_output_field_present` checks whether the public runtime row exposes the deployment-only accepted-output field.
