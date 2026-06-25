# H8 Calibration-Only Selector

Goal: decide whether the H8 CO-specialist switch should enter the next formal/runtime stage without using target test metrics.

- Output CSV: `results/h8_calibration_selector_20260625/h8_calibration_selector_clients.csv`
- Output profile: `results/h8_calibration_selector_20260625/h8_pred_co_source_aug_selector_profile.json`

## Selection Criteria

- `pred_class == CO` switch precision on calibration must be at least 0.95.
- switch false-positive rate must be at most 0.05.
- CO recall must be at least 0.90.
- CO-high recall must be at least 0.80.
- source-augmented target Ridge must improve CO calibration-val RMSE vs rich-only target Ridge.

## Client Decisions

| client | enable | precision | FP | CO recall | high recall | rich CO val | src-aug CO val | delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C3 | 1 | 0.988 | 0.012 | 1.000 | 1.000 | 11.59 | 8.06 | -3.52 |
| C4 | 1 | 1.000 | 0.000 | 0.963 | 0.875 | 18.17 | 16.17 | -2.00 |
| C5 | 1 | 0.987 | 0.013 | 0.938 | 0.958 | 11.79 | 10.63 | -1.16 |

## Decision

- Enabled clients: C3, C4, C5.
- This is not a deployment artifact yet. It is a calibration-only analysis profile.
- Next required step: implement/export source-aug target Ridge runtime support, then run parity against the H8 analysis CSV.
