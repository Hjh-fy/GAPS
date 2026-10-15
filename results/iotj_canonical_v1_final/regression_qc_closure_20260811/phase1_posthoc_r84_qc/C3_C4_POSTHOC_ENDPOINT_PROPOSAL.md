# C3/C4 final post-hoc endpoint proposal

Status: `PROPOSAL_ONLY_NOT_EXECUTED`.

To extend the closure, create one pre-run freeze for C3 and C4 using the same
source-only round-25 checkpoint, fixed post-hoc commissioning identity, 100
steps, seed 42, canonical target-specific calibration manifests, and sealed
tests. Lock each adapted checkpoint before test evaluation, then apply the
already frozen target R84 alphas and calibration-locked QC formula. Do not use
C3/C4 target-test performance to select the post-hoc identity or endpoint.
