# Phase 4 experiment audit

## Verdict: PASS

- canonical-v1, classifier, H1, and R84 hashes match the frozen Phase 3 baseline.
- The expected-cost matrix uses calibration true class/concentration only and was locked before semantic target-test access.
- Test data was used once for fixed-policy evaluation only.
- No model fitting, alpha search, threshold, lambda, checkpoint selection, QC, or algorithm search occurred.
- Grouped bootstrap used complete raw filenames, 2000 replicates, and seed42.
