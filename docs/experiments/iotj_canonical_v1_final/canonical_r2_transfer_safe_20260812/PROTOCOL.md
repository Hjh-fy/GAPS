# Canonical-v1 R2 transfer-safe regression

R2 is triggered only by the completed R1 decision `CANONICAL_R84_DEVICE_DEPENDENT`. It reuses R1's frozen target calibration/test identities, canonical 83D features, source-prior predictions, R84 baseline, classifier routes, and seed42 bootstrap design.

Exactly two candidates are allowed: residual transfer and shrinkage transfer. Residual-transfer alpha and shrinkage beta are selected by five-fold raw-filename-grouped target-calibration RMSE; target test is unavailable until `selection_lock.json` exists. A candidate is retained only with at least 3% pooled S_ALL RMSE improvement over R84, no gas degradation above 5%, and a paired grouped-bootstrap RMSE-delta 95% interval entirely below zero. No difficult-case exception is enabled.

Formal output is immutable at `results/iotj_canonical_v1_final/canonical_r2_transfer_safe_20260812/`. R2 stops after its decision and does not start Q0.
