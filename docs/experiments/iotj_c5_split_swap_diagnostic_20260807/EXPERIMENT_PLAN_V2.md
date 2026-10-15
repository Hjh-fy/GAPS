# C5 legacy-dataset diagnostic amendment

The fixed-new-assets cross-prediction is retained as diagnostic arm D1. It
does not answer whether the legacy end-to-end result itself is reproducible,
so a second frozen arm is added.

## D1: fixed role-aware assets on legacy test

- Role-aware FCL-RW-GAPS-C5 round-25 checkpoint.
- Role-aware frozen R84 models; no refit.
- Legacy C5 test prediction.
- Leakage-risk diagnostic only because split membership crosses protocols.

## D2: exact legacy regression replay

- Legacy FCL-E3-GAPS-C5 round-25 checkpoint.
- Legacy C5 calibration: 320 windows.
- Frozen alpha grid and deterministic 60-fit/20-validation-per-gas rule.
- Persist model and selection lock before opening the legacy 1,360-window test.
- No classifier training, checkpoint selection, test selection, or tuning.
- Acceptance: reproduce the existing legacy C5 R84 summary within numerical
  precision. Any mismatch is a blocking implementation/provenance finding.
