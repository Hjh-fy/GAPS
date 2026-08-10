# Gate 3 Experiment Audit

Status: `PASS`.

- All three final methods independently deep-copied the same verified source-only round25 model state.
- The 80L and 240U pools are disjoint and exhaust the frozen 320 calibration identities; calibration and test identities are disjoint.
- The unlabeled dataset exposes only X and physical identity. Target phase, concentration, and hidden class truth are absent from training batches.
- GAPS selection evaluated exactly six pre-registered configurations on two deterministic labeled folds; test data did not enter selection.
- Each final endpoint used exactly 100 Adam updates at 5e-4, batch size 32, seed42.
- MME is explicitly reported as compatible rather than exact because the frozen biased linear classifier is retained.
- Hidden unlabeled truth opened only after endpoint locks and is used solely for the post-hoc pseudo-label diagnostic.
- C5 test opened once after all endpoint and selection locks; no checkpoint or hyperparameter was selected from it.
## Verification

- Focused Gate-1/Gate-2/Gate-3 safety suite: `25 passed`.
- `python -m compileall -q gaps_flower scripts tests/test_iotj_c5_ssda_g3.py`: PASS.
- Evidence SHA index: all 35 indexed files verified.
- Prediction-only metric recomputation: all 3×1360 rows reproduce the published Accuracy, Macro-F1, NLL, and ECE.
- Full historical repository suite: `1179 passed, 5 skipped, 33 failed, 9 setup errors`. The non-passing items are outside the G3 dependency/touch set and arise from legacy missing result/runtime assets, tests that require absent historical output directories, and Windows path-length failure in a tracked-only checkout fixture. The audit therefore does not claim a globally green repository.
