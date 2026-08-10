# A0T versus A4 regression report

## Input contract and provenance

- Six immutable round25 adapted classifier checkpoints: A0T/A4 × C3/C4/C5.
- Canonical-v1 target calibration/test roles; fixed R84_FED_H1 alpha table; seed42.
- Metrics are recomputed from raw endpoint prediction records. No confidence interval or significance test is reported because there is one seed and windows are not independent clients.

## Primary result

- C5 S_ALL RMSE: A0T 22.1560 ppm; A4 18.4765 ppm; delta -3.6794 ppm.
- Pooled S_ALL RMSE: A0T 15.1925 ppm; A4 13.3144 ppm; delta -1.8781 ppm.
- Decision: `REGRESSION_ADVANTAGE_SUPPORTED`.

## Anomalies and sensitivity

- C5 CO and Methane worsen under A4 despite total C5 improvement.
- Frozen A4 QC thresholds transfer to A0T with a substantially different achieved coverage; this is a fixed-policy transfer comparison, not an equal-coverage refit.
- Correct-route mappings do not improve; the quantitative advantage is driven by the identity/severity of routing mistakes.

## Evidence files

- `regression_comparison.csv`, `per_gas_regression_comparison.csv`, `routing_scope_summary.csv`
- `qc_comparison.csv`, `C5_A0T_VS_A4_REGRESSION.md`
- `A0T_VS_GAPS_FINAL_CONCLUSION.md`
