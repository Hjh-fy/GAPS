# R3aK16 Structure Ablation Matrix Report

- Generated from existing PC evaluation artifacts.
- Evaluation target: C12 -> C345 target test, no-QC full-set.
- Prediction column: `calibrated_ppm` as final; `pred_ppm` retained in detail CSV. These artifacts are structure candidates after the existing target-side specialist/calibration flow, not raw source-only transfer.
- CO high definition: true CO rows with `true_ppm >= 200`.
- Summary CSV: `results/r3ak16_structure_ablation_20260625/structure_ablation_summary.csv`
- Detail CSV: `results/r3ak16_structure_ablation_20260625/structure_ablation_scope_metrics.csv`

## Current Matrix

| candidate | role | depth | response | dct_k | tcn | shared | ratio | full params | reg params |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M0_R3aK16_depth4_dct16 | current baseline | 4 | dct | 16 |  |  |  | 448023 | 411850 |
| M1_R3aK8_depth4_dct8 | lighter DCT branch | 4 | dct | 8 |  |  |  | 443799 | 407626 |
| M2_R3b_depth4_msconv16 | classic local conv response branch | 4 | msconv | 8 |  |  |  | 452135 | 415962 |
| M3_S2_tcnadapter_k3g005 | response adapter stress test | 4 | none | 8 | 1 |  |  | 441927 | 405754 |
| M4_T9fix_shared_trunk | shared-private head ablation | 4 | none | 8 | 0 | 1 | 0 | 143319 | 107146 |
| M5_T10afix_ratio_dct | ratio auxiliary branch | 4 | dct | 16 | 0 | 0 | 1 | 451456 | 415283 |

## No-QC Full-Set Metrics

| candidate | N | ALL RMSE | ALL NRMSE | CO RMSE | CO high | nonCO | C3 CO | C4 CO | C5 CO | C3 high | C4 high | C5 high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M0_R3aK16_depth4_dct16 | 1320 | 23.31 | 0.16 | 32.06 | 40.64 | 19.55 | 25.14 | 24.09 | 48.12 | 28.05 | 30.24 | 65.01 |
| M1_R3aK8_depth4_dct8 | 1320 | 24.39 | 0.15 | 35.89 | 44.83 | 19.08 | 31.05 | 24.13 | 51.79 | 39.90 | 29.42 | 63.57 |
| M5_T10afix_ratio_dct | 1320 | 26.56 | 0.16 | 36.39 | 44.45 | 22.34 | 29.30 | 30.51 | 52.04 | 36.76 | 23.15 | 68.86 |
| M3_S2_tcnadapter_k3g005 | 1320 | 26.90 | 0.16 | 38.72 | 52.64 | 21.57 | 35.12 | 24.12 | 54.60 | 49.02 | 29.03 | 74.05 |
| M2_R3b_depth4_msconv16 | 1320 | 27.86 | 0.16 | 42.10 | 60.02 | 21.07 | 28.26 | 24.73 | 70.74 | 36.73 | 30.05 | 105.31 |
| M4_T9fix_shared_trunk | 1320 | 30.35 | 0.19 | 42.24 | 48.17 | 25.16 | 36.87 | 22.19 | 63.08 | 47.73 | 26.48 | 63.48 |

## Delta vs Baseline

| candidate | reg params % | ALL RMSE delta | CO delta | CO high delta | nonCO delta |
| --- | --- | --- | --- | --- | --- |
| M0_R3aK16_depth4_dct16 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| M1_R3aK8_depth4_dct8 | -1.03 | 1.08 | 3.83 | 4.19 | -0.47 |
| M5_T10afix_ratio_dct | 0.83 | 3.24 | 4.33 | 3.81 | 2.79 |
| M3_S2_tcnadapter_k3g005 | -1.48 | 3.59 | 6.66 | 12.00 | 2.02 |
| M2_R3b_depth4_msconv16 | 1.00 | 4.55 | 10.04 | 19.38 | 1.52 |
| M4_T9fix_shared_trunk | -73.98 | 7.03 | 10.18 | 7.53 | 5.61 |

## Interpretation

- Best existing structure by ALL RMSE is `M0_R3aK16_depth4_dct16` (23.31 RMSE, 0.1556 NRMSE).
- Baseline remains the best among the existing structure candidates by ALL RMSE.
- `M4_T9fix_shared_trunk` is the only truly lightweight existing neural-structure candidate (about 74% fewer regression-branch parameters than M0), but its ALL RMSE and nonCO RMSE are clearly worse.
- `M1_R3aK8` trims only about 1% of regression-branch parameters; that is not enough to count as meaningful lightweight simplification, and it worsens ALL/CO metrics.
- `M5_T10afix_ratio_dct` helps C4 CO high in this artifact set, but the global ALL/nonCO trade-off is too large to promote as a mainline structure.
- This table only evaluates the base regression structure plus existing calibration output. It does not yet include the stronger target direct-head auto_v2 candidates such as H2.3/H8.
- If a lighter head cannot beat the baseline before target calibration, it should still be tested with the same target calibration before being rejected, because previous light-source-only tests showed direct transfer can collapse.

## Planned Missing Lightweight Experiments

| candidate | purpose | command skeleton |
| --- | --- | --- |
| M6_depth2_dct16 | Test whether the residual head can be made shallower while retaining DCT response statistics. | python -m gaps_flower.regression_server --reg-head-depth 2 --reg-response-branch dct --reg-dct-k 16 ... |
| M7_depth2_none | Classic compact MLP head without explicit response branch. | python -m gaps_flower.regression_server --reg-head-depth 2 --reg-response-branch none ... |
| M8_depth4_none | Keep deep head but remove response branch to isolate DCT contribution. | python -m gaps_flower.regression_server --reg-head-depth 4 --reg-response-branch none ... |

## Promotion Rule

- P0: lower no-QC full-set ALL RMSE / NRMSE after the same target calibration flow.
- P1: lower CO and CO-high RMSE, especially C4/C5, without obvious nonCO regression damage.
- P2: prefer smaller regression branch parameters only when P0/P1 are not worse.
- QC remains out of this selection loop.
