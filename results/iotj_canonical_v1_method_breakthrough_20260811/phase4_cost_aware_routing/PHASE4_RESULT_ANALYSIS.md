# Phase 4 Expected Downstream-Cost Routing

## Protocol

The 4x4 cost matrix was estimated only from the frozen 320-window C5 B20 calibration set. The policy is the parameter-free rule `argmin_j sum_c p(c|x) C(c,j)`. The matrix was hashed before the sealed C5 test opened. No lambda, threshold, checkpoint, or hyperparameter search was performed.

## Result

- Argmax RMSE / Macro-F1: 28.057496 ppm / 0.976544
- Expected-cost RMSE / Macro-F1: 28.370587 ppm / 0.973525
- Relative RMSE improvement: -1.115890%
- Macro-F1 drop: 0.003019
- Route changes: 17
- Grouped raw-filename bootstrap P(delta RMSE < 0): 0.424500
- Bootstrap 95% interval: [-2.644208, 3.751293] ppm

## Decision

`COST_AWARE_ROUTING_NOT_SUPPORTED`

The interval crossing zero is reported descriptively and is not an additional hard gate beyond the preregistered decision rule.
