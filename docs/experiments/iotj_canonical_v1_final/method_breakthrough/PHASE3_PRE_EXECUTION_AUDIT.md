# Phase-3 pre-execution audit

## Verdict: PASS

- Phase-2 predecessor decision: `DG_TO_COMMISSIONING_NOT_SUPPORTED`.
- Registered selection rule chooses the first I0/I1 identity within 0.01 Macro-F1 of the best B20 row. I0 is 0.007276 below I2, so I0+B20 is selected.
- This is a method-identity rule fixed before Phase 3; it is not round, step, checkpoint, or alpha selection.
- Classifier input is exact immutable G1 I0+B20 step100 checkpoint SHA `857f3954003bffad1af716002a1bd2915923389faec31b69f5c72e563aaa212c`.
- H1 manifest SHA remains `d32217a30f491ba46be436f3baf469b764b54a08d4d542b4eb71dbc007338ecc`.
- C5 alphas remain `{0:1.0, 1:0.01, 2:10.0, 3:0.1}`; no alpha search is authorized.
- Phase 3 fits R84 on the canonical 320-window C5 calibration set, locks model/hash, and only then opens C5 test.
- Classifier training, QC, cost-aware routing, and any target-test selection are disabled.

