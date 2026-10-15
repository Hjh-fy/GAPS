# Phase-2 post-run experiment audit

## Verdict: approved

- Five new endpoints and one exact G1 reuse endpoint form the complete registered matrix.
- Every new endpoint independently reloaded its registered original round25 source-state fingerprint and stopped at fixed step100.
- B20 uses the canonical 320-window order required for exact G1 reuse; B05 is the frozen nested 80-window calibration view with 40/40 strata.
- C5 test opened only after all six endpoint gates passed. It was not used for training, stopping, hyperparameter selection, checkpoint selection, or method selection.
- All 33 immutable Phase-2 evidence files match `sha256_index.json`.
- Favorable I2+B20 performance does not override the registered cross-budget decision or the Phase-3 simplest-effective selection rule.

