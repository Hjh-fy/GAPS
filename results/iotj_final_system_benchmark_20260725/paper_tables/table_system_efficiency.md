| runtime | classifier_params | regression_params | QC_params_assets | bundle_size_bytes | PC_p50_ms | PC_p95_ms | Pi_p50_ms | Pi_p95_ms | Pi_peak_RSS_MiB | Pi_peak_temperature_C | Pi_throughput_windows_per_s | deployment_status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| RUNTIME_V4_FULL | 22765 | 28737 | non-trainable frozen references/policy | 2971538 | 5.4337 | 13.9448 | 4.57137 | 4.62118 | 237.891 | 54 | 209.109 | FORMAL_BASELINE |
| RUNTIME_V5_REGRESSION_CORE | 22765 | 844 | none | 289916 | 4.0716 | 14.7863 | 3.72454 | 3.76499 | 234.234 | 52.35 | 254.242 | FINAL_SIMPLIFIED_REGRESSION |
| RUNTIME_V5_QC2_CANDIDATE | 22765 | 844 | non-trainable frozen references/policy | 1065632 | 5.47655 | 19.3115 | 4.52254 | 4.57112 | 235.141 | 54.55 | 211.412 | VALID_CANDIDATE_NOT_PROMOTED |

Runtime v4 is the formal selective-output baseline. Runtime v5 regression is the selected simplified regression implementation. Runtime v5 QC2 is a valid candidate that was not promoted. Benchmarking does not alter method selection.
