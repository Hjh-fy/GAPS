# Source-Augmented Target Ridge Evaluation

Question:

- Do source-trained lightweight heads add useful information to target-calibrated direct heads?

Protocol:

- Source heads: trained on C1/C2 train, selected on C1/C2 calibration.
- Target heads: per-client/per-gas Ridge, selected on target calibration holdout.
- Target test: fixed-DA predicted gas route.
- QC: not used.

Feature sets:

- `target_ridge_rich_only`: rich target window statistics only.
- `target_ridge_plus_source_preds`: rich target stats plus source Ridge/MLP/shared-MLP ppm predictions.
- `*_plus_c4_rescue`: same prediction plus the existing calibration-selected C4 route-rescue gate.

## Target Test RMSE

| mode | ALL | C3-CO | C4-CO | C5-CO | C3-CO_high_200_250 | C4-CO_high_200_250 | C5-CO_high_200_250 | nonCO_ALL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_final_ppm | 22.94 | 32.31 |  |  | 38.16 |  |  | 19.34 |
| target_ridge_rich_only | 15.91 | 12.84 |  |  | 13.56 |  |  | 15.04 |
| target_ridge_rich_only_plus_c4_rescue | 15.91 | 12.84 |  |  | 13.56 |  |  | 15.04 |
| target_ridge_plus_source_preds | 16.01 | 10.73 |  |  | 11.04 |  |  | 15.24 |
| target_ridge_plus_source_preds_plus_c4_rescue | 16.01 | 10.73 |  |  | 11.04 |  |  | 15.24 |

## Fit Audit

| feature set | client | gas | train N | val N | alpha | val RMSE |
|---|---|---|---:|---:|---:|---:|
| rich_only | C1 | Ethanol | 130 | 40 | 0.01 | 6.53 |
| rich_plus_source_preds | C1 | Ethanol | 130 | 40 | 0.01 | 5.43 |
| rich_only | C1 | CO | 130 | 40 | 1 | 17.82 |
| rich_plus_source_preds | C1 | CO | 130 | 40 | 10 | 17.52 |
| rich_only | C1 | Ethylene | 130 | 40 | 10 | 3.25 |
| rich_plus_source_preds | C1 | Ethylene | 130 | 40 | 10 | 2.59 |
| rich_only | C1 | Methane | 130 | 40 | 10 | 9.37 |
| rich_plus_source_preds | C1 | Methane | 130 | 40 | 0.01 | 8.90 |
| rich_only | C2 | Ethanol | 130 | 40 | 100 | 7.40 |
| rich_plus_source_preds | C2 | Ethanol | 130 | 40 | 100 | 6.80 |
| rich_only | C2 | CO | 130 | 40 | 0.01 | 12.54 |
| rich_plus_source_preds | C2 | CO | 130 | 40 | 0.01 | 11.33 |
| rich_only | C2 | Ethylene | 130 | 40 | 10 | 4.53 |
| rich_plus_source_preds | C2 | Ethylene | 130 | 40 | 1 | 2.69 |
| rich_only | C2 | Methane | 130 | 40 | 0.01 | 9.62 |
| rich_plus_source_preds | C2 | Methane | 130 | 40 | 10 | 11.00 |
| rich_only | C3 | Ethanol | 130 | 40 | 0.01 | 4.24 |
| rich_plus_source_preds | C3 | Ethanol | 130 | 40 | 0.01 | 3.89 |
| rich_only | C3 | CO | 130 | 40 | 1 | 11.90 |
| rich_plus_source_preds | C3 | CO | 130 | 40 | 1 | 9.42 |
| rich_only | C3 | Ethylene | 130 | 40 | 0.01 | 5.46 |
| rich_plus_source_preds | C3 | Ethylene | 130 | 40 | 0.01 | 5.29 |
| rich_only | C3 | Methane | 130 | 40 | 0.01 | 5.31 |
| rich_plus_source_preds | C3 | Methane | 130 | 40 | 0.01 | 4.82 |

## Reading

- If source predictions add useful transferable information, `target_ridge_plus_source_preds` should beat `target_ridge_rich_only`.
- If not, the target-only direct-head path is simpler and should remain the mainline.
