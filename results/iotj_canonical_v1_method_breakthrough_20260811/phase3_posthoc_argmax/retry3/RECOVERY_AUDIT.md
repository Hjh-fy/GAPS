# Phase 3 report-only recovery audit

- Final status: `PASS`
- Scientific endpoint: `CAN-V1-MB-P3-POSTHOC-R84-S42`
- Selected classifier: immutable `I0+B20`, step 100
- Classifier SHA256: `857f3954003bffad1af716002a1bd2915923389faec31b69f5c72e563aaa212c`

The original `retry3` execution completed calibration fitting, wrote and
verified the calibration lock, opened the sealed C5 test, evaluated all four
R84 scopes, and wrote the prediction-derived CSV files and endpoint manifest.
It then failed in Markdown rendering because the report code requested the
nonexistent column `evaluation_scope`; the canonical result schema uses
`scope`.

The recovery was deliberately report-only. The finalizer validated the frozen
calibration-model hash, sealed-test-open record, completed endpoint manifest,
classifier SHA, and no-test-selection flag, then consumed the already-written
CSV files. It did not call classifier inference, R84 fitting, or target-test
evaluation. Therefore C5 test was opened exactly once for this endpoint and no
completed endpoint was rerun.

The field correction and report-only recovery path are covered by
`test_phase3_finalizer_uses_scope_column_without_reopening_test`.
