# Metric Record Contract

Record each metric with: `metric_id`, `experiment_id`, `metric_name`, `value`, `unit`, `direction`, `sample_scope`, `client_scope`, `gas_scope`, `aggregation`, `seed_set`, `uncertainty`, `source_path`, `calculation_status`, and `notes`.

Do not collapse distinct metrics. In particular, keep Accuracy, ECE, RMSE, NRMSE, Coverage, Coverage+Review, Accepted RMSE, Route-correct RMSE, Latency, calibration latency, memory, parameter count, and communication Payload separate. Label a value `reported` when copied from an existing report and `recomputed` only when this run calculated it from confirmed raw records.
