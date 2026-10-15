# Formal Regression QC and Oracle-Route Metric Extension

## Goal

Extend the frozen A6/B5/B2 C5 formal-regression analysis so every QC workpoint reports comparable actual-route, nonreject, and forced-oracle-route regression metrics. The extension is analysis-only: it must not change classifier checkpoints, H8 model selection, calibration splits, risk scores, or frozen HC95/HC90 thresholds.

## Metric Definitions

For a workpoint, let `A` be rows assigned `accept`, `V` rows assigned `review`, and `J` rows assigned `reject`.

- `Accepted = A`.
- `Nonreject = A union V`; only `reject` rows are excluded.
- `automatic_yield = |A| / N`.
- `nonreject_coverage = |A union V| / N`.
- Actual-route metrics use the deployed H8 prediction routed by the classifier's `pred_class`.
- Oracle-route metrics retain the same workpoint membership but recompute the complete H8 prediction with `route_class=true_class`.

For each subset, RMSE is computed in ppm. NRMSE uses the existing row-wise class-range normalization and then takes the root mean square, preserving the formal-regression definition.

## Important Boundary

Oracle-route is not the existing `S_CC` slice.

- `S_CC` drops every originally misclassified row and evaluates only naturally correct routes.
- Forced oracle-route keeps all rows in the selected workpoint subset and routes every source head and the final C5 Ridge head using `true_class`.

Therefore the FULL oracle-route result contains all 1,360 C5 test windows and isolates routing error from the regressor's intrinsic error.

## Data Flow

1. Reuse the existing C1/C2 source data, C5 calibration/test data, classifier prediction stream, seed 42, 25% concentration-stratified calibration validation split, and H8 hyperparameter grids.
2. Fit the three H8 source predictors exactly as before: per-gas Ridge, per-gas shallow MLP, and shared shallow MLP.
3. Fit the final per-gas C5 Ridge heads on 104 rich features plus the three source predictions, exactly as before.
4. Produce the existing actual-route test stream using `route_class=pred_class`.
5. Produce a second test stream by copying the same 1,360 rows, setting `route_class=true_class`, and recomputing all three source predictions and the final C5 Ridge prediction.
6. Join the oracle prediction back to QC records by stable `(client, split, sample_index)` identity.
7. Apply the already-frozen FULL/HC95/HC90 decisions. Do not refit QC scores or thresholds from test truth.
8. Calculate actual-route and oracle-route metrics on Accepted and Nonreject subsets.

## Output Schema

`qc_operational_comparison.csv` and the Markdown report gain these fields:

- `nonreject_N`
- `nonreject_RMSE`
- `nonreject_NRMSE`
- `oracle_accept_RMSE`
- `oracle_accept_NRMSE`
- `oracle_nonreject_RMSE`
- `oracle_nonreject_NRMSE`

Existing fields remain unchanged. At FULL, `accept_N=nonreject_N=1360`, so actual Accepted and Nonreject metrics must match; oracle Accepted and Nonreject metrics must also match.

The H8 output directory additionally contains a row-level oracle-route prediction CSV. This is a derived diagnostic artifact and does not replace the deployable actual-route output.

## HC95 and HC90

The labels describe calibration-validation targets, not guaranteed test-set coverage:

- HC95: accept threshold is the 0.95 quantile of calibration-validation risk; reject threshold is the 0.9875 quantile. Values between them are `review`.
- HC90: accept threshold is the 0.90 quantile; reject threshold is the 0.975 quantile.
- FULL: every row is accepted.

Thresholds use deployment-visible risk only. Test labels are used solely after decisions are frozen to calculate evaluation metrics. Reports must show realized test counts and coverage rather than relabeling them as exact 95% or 90% test acceptance.

## Error Handling

- Fail if an oracle prediction is missing, duplicated, non-finite, or cannot be joined one-to-one to a QC row.
- Fail if workpoint decisions do not partition all test rows into exactly one of accept/review/reject.
- Fail if FULL does not contain all 1,360 rows or if FULL Accepted and Nonreject metrics differ.
- Preserve the existing class-range NRMSE mapping and reject unknown true classes.

## Verification

1. Unit-test H8 oracle routing with deliberately wrong predicted classes and confirm every source/final head follows `true_class`.
2. Unit-test Accepted and Nonreject subset counts and RMSE/NRMSE calculations.
3. Unit-test one-to-one oracle joins and failure cases.
4. Re-run existing H8 and high-coverage QC tests to prove actual-route outputs and thresholds are unchanged.
5. Rebuild A6/B5/B2 summaries and verify FULL invariants plus the expected 1,360-row oracle count.

## Non-Goals

- No retraining of A6/B5/B2 classification checkpoints.
- No new QC risk component, threshold tuning, or test-label-driven workpoint selection.
- No replacement of H8 by H2.3+ or neural regression.
- No mutation of previously reported actual-route metrics.
