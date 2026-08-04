# Experiment audit

Verdict: **PASS for descriptive no-fit cross-target capability evidence**.

- All three rows use seed 42 target-matched GAPS endpoints from the same frozen classification study.
- The identical Federated-H1 source model and 104-D feature construction are used for C3/C4/C5.
- Target calibration files are not accessed; no target Ridge, alpha selection, checkpoint selection, QC, or threshold search occurs.
- Target test labels are used only to calculate metrics, route-correct slices, and the explicitly diagnostic oracle-route result.
- C3/C4/C5 have different test sizes and concentration/window distributions. Results must be reported per target and must not be interpreted as a controlled device-only causal effect.
- Single seed 42 supports descriptive endpoint evidence, not stability or uncertainty claims.
- The C5 A4+R84 personalized reference follows a different router and target-calibration protocol and is kept separate.
