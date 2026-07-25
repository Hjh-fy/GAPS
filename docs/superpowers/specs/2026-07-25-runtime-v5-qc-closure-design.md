# Runtime v5 QC closure design

## Goal and evidence boundary

Close the independent QC path for the frozen B5 seed-42 classifier, real-topology
Federated H1 source prior, and 105D C5 per-gas Ridge runtime candidate.  The only
supported claim is that Runtime v5 uses calibration-derived confidence,
representation-distance, and source-to-target regression-consistency signals for
selective output.  Runtime v4 and its HC95/HC90 assets remain read-only.

## Calibration protocol

The 320 C5 calibration windows are grouped by `filename`.  The audited calibration
subset contains 80 source files with variable group sizes of 1--7 rows; every row
sharing a filename appears in exactly one of five folds.  A deterministic greedy
assignment balances total rows, gas, concentration strata, and group count.
For each held-out fold, the other four folds alone fit the target Ridge heads,
B5 feature prototype/support reference, component ECDFs, and per-predicted-gas
regression-consistency MAD scales.  No 1360-row test artifact is opened during
candidate development or locking.

This is protocol amendment v2.  The previous fixed-four-row assumption was
rejected before test opening: across calibration plus test each filename has 21
windows, while the historical window-level resplit leaves 1--7 of those windows
in calibration.  Filename grouping applies only to calibration OOF folds.  The
historical calibration/test split remains window-level, so no original-file-level
independence claim is permitted.

## Frozen risk candidates

- QC1 confidence: mean of normalized entropy and inverse top1-top2 margin.
- QC2 confidence_distance: mean of the QC1 confidence group and the mean of
  prototype-distance and nearest-support-distance percentiles.
- QC3 confidence_distance_regconsistency: mean of the confidence group, distance
  group, and the percentile of per-gas MAD-normalized absolute disagreement
  between the target Ridge and Federated H1 predictions.

Every raw component is monotonically mapped through an empirical CDF fitted only
on the corresponding fold's training rows.  Missing, unknown, duplicate, or
non-finite inputs fail closed.  The candidate is selected only from OOF records;
the simplest passing candidate wins unless a more complex candidate improves
accepted RMSE by at least 0.25 ppm at HC95 or HC90 with yield loss no greater
than 0.01.

## Lock and test boundary

After OOF selection, the selected candidate is refit on all 320 calibration rows.
The immutable `qc_selection_lock.json` binds the selected formula, full-calibration
references, ECDFs, MAD scales, HC95/HC90 thresholds, decision rules, assets, and
hashes while recording `test_opened_after_lock=false`.  A separate test command
validates that lock and then opens the 1360 rows exactly once.  Test metrics cannot
change the candidate or any policy value.

HC95 accepts risk at or below the calibration 0.95 quantile and rejects above the
0.9875 quantile.  HC90 uses 0.90 and 0.975.  Intermediate rows are review; only
accepted rows emit `auto_output_ppm`.

## Runtime integration and verification

An independent v5 QC policy/bundle layer extends the Runtime v5 candidate without
importing H2, H3, H2.3, all-prior, legacy rescue, or v4 risk assets.  Offline and
runtime paths must agree on row key, route, prediction, every raw risk component,
normalized/deployment risk, decision, and automatic output at the registered
tolerances.  Promotion remains limited to the four predeclared Q10 outcomes.
