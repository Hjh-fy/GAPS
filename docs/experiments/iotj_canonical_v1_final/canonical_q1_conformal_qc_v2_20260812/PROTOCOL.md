# Canonical-v1 Q1-v2 conformal-style QC pre-run freeze

Q1-v2 is an independently versioned repair of the failed Q1-v1 execution. The
only implementation change is that `empirical_interval` accepts a scalar or a
shape-compatible per-sample radius vector. Q1-v1 is invalid and superseded; its
partial result directory is preserved and is not resumed.

All scientific settings remain frozen: canonical-v1, targets C3/C4/C5, final
`R84_CONCAT`, deterministic raw-file-group-aware calibration split, 90% nominal
absolute-residual empirical interval, coverage grid 0.50--1.00, calibration-ECDF
normalization, fixed equal weights 0.5/0.5, no weight search, and the 5% C5 plus
pooled NRMSE-AURC support gate.
