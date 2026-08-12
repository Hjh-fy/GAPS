# Canonical-v1 Q1 conformal-style QC pre-run freeze

Q1 is released because Q0 returned `MULTISIGNAL_QC_NOT_ESTABLISHED` and the
registered regression-only uncertainty has lower NRMSE-AURC than confidence on
both C5 and the target-stratified pooled scope. The regression backend remains
the frozen `R84_CONCAT` model.

For each target and gas, calibration raw-file groups are deterministically
divided into balanced fit and conformal-calibration subsets. The frozen R84
alpha is reused. Absolute residuals on the conformal subset define a fixed 90%
nominal group-aware conformal-style interval. This is an empirical prediction
interval; no exact iid coverage guarantee is claimed.

The one registered comparison contains confidence-only, interval-width-only,
and the equal mean of calibration-ECDF-normalized confidence risk and interval
width. Both weights are fixed at 0.5 and no weight search is allowed. Support
requires at least 5% relative NRMSE-AURC improvement over confidence on both C5
and pooled; otherwise `CONFIDENCE_QC_FINAL` is retained.

Target test data are opened only after the split models, residual radii, CDF
references, and policy weights are locked.
