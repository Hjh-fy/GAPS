# Phase 4 pre-execution audit

## Verdict: PASS

- Predecessor Phase 3 is approved at commit `e03dd02` and fixes the exact
  `I0+B20` step100 classifier and frozen C5 R84 models.
- Classifier SHA256 is
  `857f3954003bffad1af716002a1bd2915923389faec31b69f5c72e563aaa212c`.
- H1 SHA256 remains
  `d32217a30f491ba46be436f3baf469b764b54a08d4d542b4eb71dbc007338ecc`.
- The 4x4 matrix is built only from the 320-window C5 B20 calibration set using
  `max(0, mean(SE_forced_j-SE_correct_c))`, with diagonal zero.
- The router is exactly `argmin_j sum_c p(c|x) C(c,j)`; lambda and threshold
  are unavailable and no search path exists.
- The matrix and its SHA lock must exist before semantic C5 test access.
- Test evaluation uses the fixed endpoint once; it cannot select a checkpoint,
  matrix, formula, threshold, lambda, or stopping point.
- Grouped bootstrap is fixed to raw filename, 2000 replicates, seed42.
- Implementation commit is `a669dac`; four Phase 4 protocol tests pass.

The run may proceed without further authorization. Any provenance, lock, or
shape mismatch must fail closed.
