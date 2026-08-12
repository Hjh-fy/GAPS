# Canonical-v1 R1 frozen protocol

Status: `DESIGN_FREEZE_READY_FORMAL_NOT_STARTED`. R0-v2 PASS model lock is the sole source prior. Target 83D and R84 caches must be freshly reconstructed from each target's canonical 50x8 arrays under this study ID. R84 adds exactly one R0-v2 FedRidge prediction column. The frozen C0 interleaved classifiers are reused byte-for-byte; no classifier training occurs.

Calibration alone performs per-target/method/gas deterministic five-fold raw-filename-grouped CV. All alpha/model/classifier/cache/bootstrap-design locks precede any target-test tensor or label opening. Evaluation uses S_ALL, S_CC, Oracle_ALL, Oracle_CC, the registered metrics/slices, and 5,000 paired raw-file group bootstrap replicates. No legacy cache, coefficient, scaler, alpha, QC asset, target-test selection, or hyperparameter search is accepted.

The frozen split's raw-file/raw-time neighborhood overlap remains a stated limitation. `preflight`, `run`, and `audit` remain fail-closed until the implementation receives independent review and a separately authorized freeze HEAD.

The audit is a functional and provenance baseline. It checks indexed evidence,
locks and access ordering, recomputes saved prediction metrics and slices, and
checks the frozen bootstrap summary contract. It does not claim adversarial or
cryptographic threat-model completeness.
