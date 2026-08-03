# Frozen P0-U Protocol

U1 and U2 independently deep-copy the same hash-pinned P0A round25 model. `FeatureOnlyCalibrationDataset` loads only `calibration_features.npy` and returns one tensor; adaptation functions reject tuples, lists, and mappings at runtime.

U1 performs 100 Adam steps at 5e-4 with source CE and unconditional CORAL/global MMD²/Wasserstein-min adversarial alignment. All class-conditioned target operations, target CE, target prototype anchor, same-class-phase MMD and pseudo labels are unavailable or disabled by construction.

U2 uses a frozen source teacher for all 100 steps. Pseudo labels are `argmax(teacher(x))`; only predictions with confidence at least 0.90 enter CE. The threshold is fixed, and neither calibration truth nor C5 test can modify it.

After both branches finish, calibration truth is opened once to audit the precision of the already-consumed teacher pseudo labels. Then and only then is the C5 sealed test opened for one final evaluation per branch. Existing P0 Source-only and Simple Target-CE round25 rows are copied read-only into the unified table.
