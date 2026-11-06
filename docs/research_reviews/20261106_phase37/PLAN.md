# Phase3.7 Numerical Bridge Implementation Plan

**Goal:** One unchanged window supplies the frozen classifier and independent canonical71 extractor; predicted class alone selects the frozen F-SVR71 head. Validate only C5Day18A4.

**Architecture:** bridge.py is a model-independent, fail-closed inference interface; assets.py supplies hash-checked frozen adapters. prepare.py creates source/tolerance locks before numeric comparison; validate.py uses only32Calibration windows and16approvedrawexposures. Tests inject faults without training. Future offline six-view evaluation is a document-only contract.

**Tech Stack:** installed Python/PyTorch/numpy/sklearn/joblib; unchanged Phase33R classification and v5 extractor/SVR functions.

**Spec:** user Phase3.7 A–E and Phase3.6 fixed4bf929fbbaa456e0a21599ad39fd01e8145a0406.

Global constraints: no fit/scaler refit/optimizer, no Day22 inference/evaluation, no historicalTest curves, no Pi, no weight/config modifications. Current numeric access is whitelist-only Calibration and exact model assets. Fixed all6 class endpoints and4F-SVR heads. No model selection or tolerance relaxation after results.

Review focus: class failure has no Oracle fallback; rawNaN/Inf rejected BEFORE clip; missing/wrong-version head unavailable; wrong-class routes actually call predictedhead;64encoder is never71; feature names/order/type/finite validated; all existing frozen files preserved.

- [ ] Lock source identities, fixed tolerances and available reference coverage (metadata-only preparation).
- [ ] Write failing interface/fault tests; confirm expected missing-feature failure.
- [ ] Implement minimal bridge and unchanged numeric adapters; run synthetic fault tests.
- [ ] Rebuild only16Day18raw/32windows and canonical71; verify saved signals/features exactly, all4head loading/scalers/support, saved Calibration references where available and direct frozen implementation parity.
- [ ] Validate all6 classifiers on same32windows, checkpoint reload/single-batch parity; preserve absence of historical32logits as a limitation.
- [ ] Freeze future6×840 outputs/identity/metric contract without executing it; no mixed-gas overallNRMSE or joint-success threshold.
- [ ] Independent review, fresh sourceSHA checks, seal and publish sanitized GitHub review.

Only new files under research/fedongas_phase37_20261106 and review publication copies. Existing isolated managed worktree reused. No automatic localgit/main merge; publish only authorized research-review branch.
