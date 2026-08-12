# Canonical-v1 Q0 QC necessity protocol

Status: `PRE_RUN_FROZEN_NO_EXECUTION`

- Final regression backend: frozen `R84_CONCAT`, following R2-v2 decision
  `RETAIN_R84_DEVICE_DEPENDENT`.
- Targets: C3/C4/C5; C5 is primary, C3/C4 and target-stratified pooled results
  are secondary consistency evidence.
- Policies: full output, 5,000 seed42 matched-count random references,
  classification confidence, and five-fold raw-file-grouped regression
  dispersion.
- Coverage grid: 0.50 through 1.00 in 0.01 increments.
- Existing equal-mean Q4: recorded as
  `Q4_CANONICAL_INPUTS_UNAVAILABLE` unless all three exact canonical inputs can
  be proven. No legacy input or substitute score is allowed.
- No model, threshold, split, feature, regression backend, or QC formula search.
- Target test is opened only after `qc_policy_lock.json` is written.
- This pre-run commit does not authorize Q1 or formal execution.
