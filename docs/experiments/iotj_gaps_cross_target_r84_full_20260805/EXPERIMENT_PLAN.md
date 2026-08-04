# Target-matched GAPS + full R84 regression evaluation

## Objective

Replicate the frozen C5 `R84_FED_H1` target-personalization workflow for C3, C4, and C5, using each target's frozen GAPS round-25 adapted classifier as the deployment router. Existing inputs are read-only and all outputs go to `results/iotj_gaps_cross_target_r84_full_20260805`.

## Frozen protocol

- Seed: 42 only.
- Routers: `FCL-E3-GAPS-C3`, `FCL-E3-GAPS-C4`, and `FCL-E3-GAPS-C5` fixed round-25 adapted checkpoints.
- Regression input: 83-D sensor statistics plus the routed frozen Federated-H1 prediction (`R84_FED_H1`).
- Regression model: one target Ridge per gas.
- Alpha candidates: `[0, 0.01, 0.1, 1, 10, 100, 1000]`, exactly matching the frozen C5 formal protocol.
- Calibration split: deterministic concentration-stratified 75/25 within each gas; C3/C5 use 60 fit + 20 validation from 80 windows per gas, while C4 uses 30 + 10 from 40 windows per gas.
- After alpha selection, each target/gas model is refit on all available calibration windows.
- Calibration lock must be persisted and validated before target test is opened.
- Target test is used once for fixed-endpoint evaluation and never for selection.
- No classifier training, checkpoint selection, QC, threshold search, or new ablation.

## Metrics

Report RMSE, MAE, R2, and class-range-normalized RMSE for `S_ALL`, `S_CC`, each gas, each concentration, route-correct windows, and misrouted windows. Preserve calibration alpha grids and per-window predictions.

## Comparison boundary

The three target rows measure full available-calibration capability, not a device-only causal effect: C4 has 160 calibration windows, whereas C3 and C5 have 320. The existing formal `A4-C5 + R84_FED_H1` result is a separate reported reference because its router differs.
