# Experiment audit

Verdict: **PASS for target-specific full-pipeline capability; restricted for causal cross-target comparison**.

- Three unique experiment IDs, whole-file provenance hashes, and ordered state-content fingerprints are recorded; state-content fingerprints are the checkpoint equality basis.
- Every target uses the same 83-D sensor profile, Federated-H1 asset, Ridge family, alpha grid, 75/25 calibration split rule, refit rule, seed, and metric definitions.
- Each calibration lock is persisted and validated before that target's test loader is called.
- Target test is not used for alpha, checkpoint, threshold, or model selection.
- C4 has half the calibration windows of C3/C5. Report per-target capability, but do not claim a device-only causal ranking.
- Single seed 42 does not support stability or uncertainty claims.
- The formal A4-C5+R84 result is a different-router reference and is not pooled into the GAPS-router matrix.
