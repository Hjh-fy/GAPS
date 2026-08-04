# GAPS cross-target regression router analysis

## Scope

This is a frozen-checkpoint, no-fit evaluation. It does not train or tune a classifier, regression head, calibration rule, or QC threshold. Existing assets remain read-only and outputs are written to `results/iotj_gaps_cross_target_regression_router_analysis_20260804`.

## Hypotheses

- `XTR-H1`: The frozen target-matched GAPS routers for C3, C4, and C5 can be coupled to the same frozen Federated-H1 source Ridge to quantify end-to-end source-regression transfer on each target.
- `XTR-H2`: The difference between predicted-route and oracle-route H1 errors quantifies route sensitivity, while route-correct metrics quantify regression capability conditional on correct classification.

## Frozen matrix

- Router checkpoints: `FCL-E3-GAPS-C3`, `FCL-E3-GAPS-C4`, and `FCL-E3-GAPS-C5`, seed 42, fixed round-25 adapted endpoint.
- Regression head: the same four per-gas models from `federated_h1_manifest.json` for all targets.
- Test routes: existing `per_window_predictions.csv`; no checkpoint inference is rerun.
- Target calibration: unavailable to this analysis; no target regression fitting or alpha selection.
- Target test: used once for descriptive fixed-endpoint evaluation only.
- QC: none; all outputs are evaluated.

## Metrics

- Classification: accuracy and macro-F1 copied from the frozen classification result.
- Regression: RMSE, MAE, R2, and class-range-normalized RMSE.
- Scopes: routed `S_ALL`, routed `S_CC`, oracle true-class route, wrong-route subset, and per-gas routed `S_ALL`/`S_CC`.
- Route decomposition: misroute rate and routed-minus-oracle RMSE/NRMSE.

## Claim boundary

The study supports target-specific descriptive capability under seed 42. It is not a comparison of target-personalized regression heads and does not estimate seed uncertainty. The existing C5 A4+R84 result may be shown only as a separately labeled personalized reference.
