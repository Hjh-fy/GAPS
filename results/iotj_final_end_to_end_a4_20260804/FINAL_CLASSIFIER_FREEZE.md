# Final classifier freeze

- Formal router: server-centric A4, round 25, seed 42, LE1, batch size 32.
- C5 is complete: accuracy=0.993382, macro-F1=0.993390.
- C3/C4 are blocked because immutable same-protocol A4 endpoints are unavailable.
- No classifier was retrained and no full-GAPS checkpoint was substituted.
- Checkpoint equality uses ordered state-content fingerprint; whole-file SHA-256 is provenance only.
