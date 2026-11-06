# Phase3.8 one-shot fixed-view offline evaluation

Authority: user independently approved OFFLINE_JOINT_EVAL_C5_DAY22_SIX_FIXED_VIEWS once; anchor ba7a143677404155e93083d0c43e15e9e72b4658. Preserve Phase3.7 and all earlier assets.

1. Implement a new permission-only Day22 loader around unchanged P37 FrozenAssets/JointBridge. Temporarily bind its install_guard callback to the new purpose-scoped whitelist during construction; restore callback afterward. No numeric adapter, kernel, scaler, loss or checkpoint edits.
2. Test routing audit and scoring with synthetic records only; review new entry before one-shot registration/execution.
3. Lock current code, old contract/model/scaler identities,40exposures/840window identities, original cacheSHA, full-flow/conditional metric semantics. Fresh explicit authorization is not the old P37 template or SOURCE/pilot authorization.
4. Consume authorization at run1 start, verify models/state/scalers and rebuild only40Day22 exposures using original strict preprocessor. Require all840 signal/71D values bitwise equal to frozen caches and exact identity/QC agreement before first joint prediction.
5. For B0–B5, run single-window P37 JointBridge once perwindow; count unchanged classifier/head adapter calls with transparent proxies. Compare actual captured single-window logits/argmax to saved Phase35 final batch predictions under unchanged P37 tolerance. Drift stops corresponding view; no changedbatch/threshold/algorithm/retry.
6. Save each complete/partial view and stateSHA. No score/ranking until all6 views locked. UNAVAILABLE is explicit; input/model/schema inconsistency is a blocking error. Any missing/drift view means incomplete matrix, no partial winner or automatic repeat.
7. Only if all6 complete, derive full-flow classification, perpredictedgas availability/calls, conditional correct-class regression raw/clip bygas/concentration/exposure. Empty subsets get undefined metrics and0coverage, never zero error. Exposure equality is within each retained subset and coverage is reported.
8. Preserve all statuses, wrong routes, original identities and tolerances; no mixed-gas NRMSЕ/jointsuccess score. Independent CSV audit, sourceSHA seal, PC/Pi readiness contract, compact GitHub review/index/fixedcommit.

Ruling: old P37 Python guard permits Calibration only; user explicitly authorized a new Day22 loader. The only runtime substitution is its guard-registration callback, registered with this plan and codeSHA, with a new equally restrictive whitelist. Old file contents and all numerical callables remain unchanged.

Frozen evaluation semantics: group NRMSE normalized by each truegas Calibration support width (Ethanol/Ethylene112.5ppm, CO/Methane225ppm), exposed as pergas percent only. No acrossgas scalar. No new confidence intervals or seed statistics. Retrospective one seed; six views share40 exposures.
