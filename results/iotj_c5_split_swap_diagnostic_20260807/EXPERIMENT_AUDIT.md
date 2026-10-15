# Experiment audit

Verdict: **diagnostic PASS; formal comparison restricted**.

- No classifier training, regression refit, alpha selection, checkpoint
  selection, or QC selection was performed in D1.
- D1 uses frozen role-aware checkpoint and R84 assets on the legacy test.
- D1 is marked LEAKAGE_RISK_DIAGNOSTIC_ONLY because cross-split disjointness
  is not established.
- D2 replay refits only on the legacy calibration split and opens legacy test
  after the persisted calibration lock.
- D2 reproduces six legacy result artifacts byte-for-byte.
- The two data roots have equal sample counts but conflicting numeric and
  provenance identity; they must not be described as the same split.
- The two requested normalization arms in D1 are numerically identical
  because the frozen test-loader contract sets normalize=False. They are one
  effective prediction condition, not independent evidence.
