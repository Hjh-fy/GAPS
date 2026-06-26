# Low-Calibration Stress Profiles

This stress test refits target-side calibration/profile heads using stratified subsets of the target calibration split.
QC is not used. Test metrics are reported after calibration-only fitting and selector decisions.

## Selector Decisions

| calib_ratio | subset_N | selected_profile | h8_enabled | gate_enabled | pred_CO_precision | h2_CO | h8_CO | calib_gate_false | calib_gate_hits | test_gate_false | test_gate_hits |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20 | 1320 | H2_3_lowcal | 0 | 1 | 0.991 | 24.54 | 24.81 | 0 | 3 | 0 | 14 |
| 10 | 640 | H2_3_lowcal | 0 | 1 | 0.994 | 17.87 | 21.11 | 0 | 1 | 0 | 14 |
| 5 | 320 | H2_3_lowcal | 0 | 1 | 0.987 | 22.58 | 23.15 | 0 | 1 | 0 | 14 |

## Test RMSE

### Calibration 20%

| mode | ALL | C3-CO | C4-CO | C5-CO | C4-CO_high_200_250 | C5-CO_high_200_250 | nonCO_ALL |
|---|---:|---:|---:|---:|---:|---:|---:|
| B0_baseline_final | 27.34 | 33.70 | 56.59 | 46.12 | 95.32 | 60.00 | 19.00 |
| H2_3_lowcal | 18.86 | 16.13 | 26.29 | 30.68 | 43.48 | 39.40 | 17.16 |
| H8_plus_C4_forced_lowcal | 18.24 | 14.97 | 25.35 | 23.57 | 43.34 | 27.55 | 17.49 |
| H8_C4_selector_lowcal | 18.86 | 16.13 | 26.29 | 30.68 | 43.48 | 39.40 | 17.16 |

### Calibration 10%

| mode | ALL | C3-CO | C4-CO | C5-CO | C4-CO_high_200_250 | C5-CO_high_200_250 | nonCO_ALL |
|---|---:|---:|---:|---:|---:|---:|---:|
| B0_baseline_final | 27.34 | 33.70 | 56.59 | 46.12 | 95.32 | 60.00 | 19.00 |
| H2_3_lowcal | 23.23 | 24.97 | 30.11 | 32.79 | 42.53 | 39.06 | 21.21 |
| H8_plus_C4_forced_lowcal | 22.60 | 14.83 | 35.58 | 32.45 | 44.34 | 44.75 | 21.21 |
| H8_C4_selector_lowcal | 23.23 | 24.97 | 30.11 | 32.79 | 42.53 | 39.06 | 21.21 |

### Calibration 5%

| mode | ALL | C3-CO | C4-CO | C5-CO | C4-CO_high_200_250 | C5-CO_high_200_250 | nonCO_ALL |
|---|---:|---:|---:|---:|---:|---:|---:|
| B0_baseline_final | 27.34 | 33.70 | 56.59 | 46.12 | 95.32 | 60.00 | 19.00 |
| H2_3_lowcal | 27.47 | 32.01 | 31.28 | 40.49 | 48.71 | 45.02 | 24.83 |
| H8_plus_C4_forced_lowcal | 26.87 | 24.19 | 31.22 | 43.65 | 48.73 | 43.76 | 24.99 |
| H8_C4_selector_lowcal | 27.47 | 32.01 | 31.28 | 40.49 | 48.71 | 45.02 | 24.83 |

## Reading

- `H8_C4_selector_lowcal` uses calibration-only stress rules and can fall back to H2.3.
- This is a profile-refit stress test, not a new exported runtime bundle.
- If low-ratio gates lose support or produce false hits, the co-priority specialist should be restricted to the full 20% calibration setting.
