# Source Lightweight Heads + Full Residual auto_v2

Criterion: no-QC full-set target test final ppm. Test rows are not used for selection.

## Protocol

- Source heads are fitted on C1/C2 train.
- Source hyperparameters are selected on C1/C2 calibration.
- Target calibration is split deterministically into calibration-fit and calibration-validation.
- Residual candidates are selected per target client and gas using calibration-validation only.
- Selected residual candidates are refit on full target calibration.
- Target test is used only once for final reporting.

## Target Test RMSE

| mode | ALL | C3-CO | C4-CO | C5-CO | C3-CO_high_200_250 | C4-CO_high_200_250 | C5-CO_high_200_250 | nonCO_ALL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_final_ppm | 27.34 | 33.70 | 56.59 | 46.12 | 41.70 | 95.32 | 60.00 | 19.00 |
| source_ridge_forced_identity | 65.42 | 89.18 | 130.12 | 46.91 | 33.00 | 193.91 | 62.91 | 52.77 |
| source_ridge_forced_affine | 40.17 | 46.23 | 82.34 | 44.23 | 59.10 | 119.14 | 61.75 | 32.63 |
| source_ridge_forced_ridge_phase | 22.91 | 15.53 | 53.94 | 26.84 | 21.50 | 96.35 | 29.53 | 18.85 |
| source_ridge_forced_piecewise_ridge | 23.57 | 17.04 | 48.35 | 31.14 | 21.41 | 85.87 | 38.56 | 20.36 |
| source_ridge_val_selected | 22.62 | 17.04 | 48.35 | 25.83 | 21.41 | 85.87 | 29.53 | 19.54 |
| source_per_gas_mlp_forced_identity | 71.28 | 97.83 | 118.55 | 107.91 | 32.27 | 90.61 | 37.94 | 55.07 |
| source_per_gas_mlp_forced_affine | 39.41 | 50.59 | 73.29 | 59.68 | 65.03 | 113.31 | 78.35 | 29.96 |
| source_per_gas_mlp_forced_ridge_phase | 22.83 | 15.23 | 53.69 | 24.87 | 19.99 | 96.33 | 27.62 | 19.04 |
| source_per_gas_mlp_forced_piecewise_ridge | 23.32 | 15.63 | 53.42 | 25.20 | 20.59 | 95.83 | 28.54 | 19.79 |
| source_per_gas_mlp_val_selected | 22.76 | 15.73 | 53.71 | 24.87 | 20.59 | 96.37 | 27.62 | 18.85 |
| source_shared_mlp_forced_identity | 63.89 | 79.10 | 73.68 | 64.12 | 32.22 | 119.75 | 49.15 | 60.05 |
| source_shared_mlp_forced_affine | 36.87 | 40.83 | 60.33 | 52.08 | 54.65 | 102.46 | 59.87 | 31.69 |
| source_shared_mlp_forced_ridge_phase | 22.92 | 15.26 | 53.60 | 27.25 | 20.88 | 96.22 | 29.73 | 18.93 |
| source_shared_mlp_forced_piecewise_ridge | 21.96 | 14.27 | 53.02 | 27.79 | 19.42 | 95.73 | 34.88 | 17.56 |
| source_shared_mlp_val_selected | 22.63 | 14.27 | 53.14 | 27.25 | 19.42 | 95.38 | 29.73 | 18.70 |

## Calibration-val Selection

| base | client | gas | selected mode | val RMSE | n cal |
|---|---|---|---|---:|---:|
| source_ridge | C3 | Ethanol | ridge_phase | 4.77 | 170 |
| source_ridge | C3 | CO | piecewise_ridge | 8.34 | 170 |
| source_ridge | C3 | Ethylene | ridge_phase | 5.06 | 170 |
| source_ridge | C3 | Methane | ridge_phase | 3.89 | 170 |
| source_ridge | C4 | Ethanol | piecewise_ridge | 4.83 | 80 |
| source_ridge | C4 | CO | ridge_phase | 18.62 | 80 |
| source_ridge | C4 | Ethylene | piecewise_ridge | 3.69 | 80 |
| source_ridge | C4 | Methane | piecewise_ridge | 9.78 | 80 |
| source_ridge | C5 | Ethanol | ridge_phase | 4.73 | 80 |
| source_ridge | C5 | CO | ridge_phase | 13.42 | 80 |
| source_ridge | C5 | Ethylene | ridge_phase | 3.28 | 80 |
| source_ridge | C5 | Methane | piecewise_ridge | 8.67 | 80 |
| source_per_gas_mlp | C3 | Ethanol | piecewise_ridge | 3.39 | 170 |
| source_per_gas_mlp | C3 | CO | piecewise_ridge | 7.85 | 170 |
| source_per_gas_mlp | C3 | Ethylene | ridge_phase | 5.09 | 170 |
| source_per_gas_mlp | C3 | Methane | ridge_phase | 3.10 | 170 |
| source_per_gas_mlp | C4 | Ethanol | ridge_phase | 4.00 | 80 |
| source_per_gas_mlp | C4 | CO | piecewise_ridge | 16.54 | 80 |
| source_per_gas_mlp | C4 | Ethylene | ridge_phase | 4.71 | 80 |
| source_per_gas_mlp | C4 | Methane | ridge_phase | 9.02 | 80 |
| source_per_gas_mlp | C5 | Ethanol | ridge_phase | 3.98 | 80 |
| source_per_gas_mlp | C5 | CO | ridge_phase | 9.81 | 80 |
| source_per_gas_mlp | C5 | Ethylene | ridge_phase | 2.51 | 80 |
| source_per_gas_mlp | C5 | Methane | ridge_phase | 9.18 | 80 |
| source_shared_mlp | C3 | Ethanol | piecewise_ridge | 3.48 | 170 |
| source_shared_mlp | C3 | CO | piecewise_ridge | 7.59 | 170 |
| source_shared_mlp | C3 | Ethylene | piecewise_ridge | 3.89 | 170 |
| source_shared_mlp | C3 | Methane | piecewise_ridge | 3.17 | 170 |
| source_shared_mlp | C4 | Ethanol | piecewise_ridge | 4.41 | 80 |
| source_shared_mlp | C4 | CO | ridge_phase | 16.88 | 80 |
| source_shared_mlp | C4 | Ethylene | ridge_phase | 4.30 | 80 |
| source_shared_mlp | C4 | Methane | ridge_phase | 9.29 | 80 |
| source_shared_mlp | C5 | Ethanol | ridge_phase | 4.10 | 80 |
| source_shared_mlp | C5 | CO | ridge_phase | 11.41 | 80 |
| source_shared_mlp | C5 | Ethylene | ridge_phase | 3.59 | 80 |
| source_shared_mlp | C5 | Methane | ridge_phase | 9.13 | 80 |

## Source Oracle Test RMSE

| mode | ALL | C1-CO | C2-CO | nonCO_ALL |
|---|---:|---:|---:|---:|
| source_ridge_source_oracle | 8.43 | 14.22 | 12.77 | 5.82 |
| source_per_gas_mlp_source_oracle | 5.25 | 9.81 | 9.26 | 2.53 |
| source_shared_mlp_source_oracle | 6.79 | 10.23 | 12.55 | 4.23 |

## Interpretation

- `forced_identity` is the source-lightweight direct-transfer baseline.
- `forced_affine` should broadly match the previous source-lightweight + target-affine diagnostic.
- `forced_ridge_phase` and `forced_piecewise_ridge` test whether rich residual calibration can rescue the source head.
- `val_selected` is the fair auto_v2-style result because the mode is chosen using calibration-validation only.
- A lightweight head should not replace R3aK16 unless `val_selected` approaches the original baseline and ideally the H2.3/H8 mainline.
