# H2.3 No-B0 Feature Ablation (c12_c345)

This ablation separates target direct-head training from B0/R3aK16/auto_v2 baseline and C4 route-rescue usage.

## Feature Check

- Direct-head feature count: target-window rich feature_dict; see run_regression_head_ablation.add_target_features
- B0/R3aK16/QC-risk feature keys found in direct-head feature_dict: []

## ALL Metrics

| mode | role | ALL RMSE | ALL NRMSE | macro-client RMSE | macro-client NRMSE | rescue/override hits |
|---|---|---:|---:|---:|---:|---:|
| A0_B0_R3aK16_auto_v2 | B0 baseline | 27.34 | 0.1578 | 28.63 | 0.1633 | 0 |
| A1_H2_3_current_with_B0_rescue | current H2.3 profile with B0-dependent C4 rescue | 18.62 | 0.1326 | 18.63 | 0.1279 | 15 |
| A2_H2_3_direct_only_no_B0_rescue | fixed H2.3 direct heads without B0-dependent C4 rescue | 22.39 | 0.1436 | 23.55 | 0.1478 | 0 |
| A3_no_auto_v2_ppm_feature | same direct-head features; no auto_v2 ppm feature present | 22.39 | 0.1436 | 23.55 | 0.1478 | 0 |
| A4_no_r3ak16_ppm_feature | same direct-head features; no R3aK16 ppm feature present | 22.39 | 0.1436 | 23.55 | 0.1478 | 0 |
| A5_no_risk_feature | same direct-head features; no risk feature present | 22.39 | 0.1436 | 23.55 | 0.1478 | 0 |
| A6_no_ppm_no_risk_direct_only | clean direct target-head output | 22.39 | 0.1436 | 23.55 | 0.1478 | 0 |

## Per-Client NRMSE

| mode | C3 NRMSE | C4 NRMSE | C5 NRMSE | macro-client NRMSE |
|---|---:|---:|---:|---:|
| A0_B0_R3aK16_auto_v2 | 0.1108 | 0.1520 | 0.2272 | 0.1633 |
| A1_H2_3_current_with_B0_rescue | 0.1023 | 0.0713 | 0.2100 | 0.1279 |
| A2_H2_3_direct_only_no_B0_rescue | 0.1023 | 0.1311 | 0.2100 | 0.1478 |
| A3_no_auto_v2_ppm_feature | 0.1023 | 0.1311 | 0.2100 | 0.1478 |
| A4_no_r3ak16_ppm_feature | 0.1023 | 0.1311 | 0.2100 | 0.1478 |
| A5_no_risk_feature | 0.1023 | 0.1311 | 0.2100 | 0.1478 |
| A6_no_ppm_no_risk_direct_only | 0.1023 | 0.1311 | 0.2100 | 0.1478 |

## Decision

- Current H2.3 macro-client NRMSE: 0.1279
- No-B0 direct-only macro-client NRMSE: 0.1478
- Relative macro-client NRMSE gap: 0.1561
- C4 NRMSE current -> direct-only: 0.0713 -> 0.1311
- C4 high-CO NRMSE current -> direct-only: 0.1522 -> 0.4298
- Run reverse under the forward gate: False
- Reading: Forward no-B0 is not close to current H2.3; keep B0/R3aK16/auto_v2 as runtime support layer.

## Reading

- If A2-A6 match each other, B0/R3aK16/auto_v2 ppm and QC-risk scalars are not direct-head training features.
- If A1 is better than A2-A6, the gain comes from the B0/risk-dependent route-rescue/profile layer, not from direct-head feature training.
- If A2-A6 remain close to A1 by macro-client NRMSE, the thesis mainline can be simplified toward encoder/classifier + target direct-head calibration.
