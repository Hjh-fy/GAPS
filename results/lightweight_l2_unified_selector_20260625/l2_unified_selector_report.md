# L2 Unified Lightweight Selector

Scope: C12 -> C345 target test, no-QC full-set.

This experiment selects one candidate per target client/gas using calibration-validation RMSE only.

## Test RMSE

| mode | ALL | C3-CO | C4-CO | C5-CO | C3-CO_high_200_250 | C4-CO_high_200_250 | C5-CO_high_200_250 | nonCO_ALL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_final_ppm | 27.34 | 33.70 | 56.59 | 46.12 | 41.70 | 95.32 | 60.00 | 19.00 |
| target_ridge_direct | 23.67 | 18.70 | 54.21 | 30.67 | 24.49 | 96.70 | 31.66 | 19.07 |
| target_mlp_direct | 21.82 | 16.15 | 48.85 | 31.03 | 20.02 | 85.05 | 39.41 | 17.61 |
| l1_source_ridge_full_auto_v2 | 22.62 | 17.04 | 48.35 | 25.83 | 21.41 | 85.87 | 29.53 | 19.54 |
| l1_source_per_gas_mlp_full_auto_v2 | 22.76 | 15.73 | 53.71 | 24.87 | 20.59 | 96.37 | 27.62 | 18.85 |
| l1_source_shared_mlp_full_auto_v2 | 22.63 | 14.27 | 53.14 | 27.25 | 19.42 | 95.38 | 29.73 | 18.70 |
| L2_unified_val_selector | 23.58 | 14.27 | 57.35 | 28.06 | 19.42 | 96.82 | 27.62 | 19.12 |
| L2_client_val_selector | 22.63 | 14.27 | 53.71 | 24.87 | 19.42 | 96.37 | 27.62 | 18.84 |

## Selection Counts

| selected candidate | count |
|---|---:|
| l1_source_per_gas_mlp_full_auto_v2 | 6 |
| l1_source_shared_mlp_full_auto_v2 | 2 |
| baseline_final_ppm | 2 |
| target_ridge_direct | 2 |

## Selected Profile

| client | gas | selected | val RMSE |
|---|---|---|---:|
| C3 | Ethanol | l1_source_per_gas_mlp_full_auto_v2 | 3.39 |
| C3 | CO | l1_source_shared_mlp_full_auto_v2 | 7.59 |
| C3 | Ethylene | l1_source_shared_mlp_full_auto_v2 | 3.89 |
| C3 | Methane | l1_source_per_gas_mlp_full_auto_v2 | 3.10 |
| C4 | Ethanol | l1_source_per_gas_mlp_full_auto_v2 | 4.00 |
| C4 | CO | baseline_final_ppm | 13.34 |
| C4 | Ethylene | target_ridge_direct | 3.13 |
| C4 | Methane | baseline_final_ppm | 8.31 |
| C5 | Ethanol | l1_source_per_gas_mlp_full_auto_v2 | 3.98 |
| C5 | CO | l1_source_per_gas_mlp_full_auto_v2 | 9.81 |
| C5 | Ethylene | l1_source_per_gas_mlp_full_auto_v2 | 2.51 |
| C5 | Methane | target_ridge_direct | 6.29 |

## Client-Level Conservative Profile

| client | selected | client val RMSE |
|---|---|---:|
| C3 | l1_source_shared_mlp_full_auto_v2 | 4.87 |
| C4 | l1_source_per_gas_mlp_full_auto_v2 | 9.91 |
| C5 | l1_source_per_gas_mlp_full_auto_v2 | 7.12 |

## Reading

- If L2 selects lightweight candidates frequently and improves test metrics, lightweight source heads provide useful target signal beyond direct target heads.
- If L2 mostly selects target direct heads, lightweight heads remain deployment-lite candidates rather than performance-mainline candidates.
- `L2_unified_val_selector` selects per client/gas and may overfit small calibration-validation cells.
- `L2_client_val_selector` is the conservative variant: one candidate per client based on aggregate client-level validation RMSE.
- Test metrics here do not include C4 route rescue or H8 CO-specialist switching; compare against H2.3/H8 mainline reports separately.
