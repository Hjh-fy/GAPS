# L3 Lightweight-Base Hybrid Matrix

Scope: C12 -> C345 target test, no-QC full-set.

This matrix explicitly tests lightweight-source outputs with H2.3/H8-style combinations.

Important distinction:

- `strict lightweight base` uses lightweight source-head full-auto_v2 predictions directly.
- `lightweight CO switch + H2.3 fallback` uses a lightweight candidate only when `pred_class == CO`; otherwise it keeps H2.3.
- `H8` uses source-augmented target Ridge as the CO specialist, not a pure lightweight base replacement.

Formal C4 rescue gate reused here:

```json
{
  "class_label": "ethanol",
  "pred_classes": "0",
  "phase": "any",
  "max_final": 20.0,
  "min_risk": 6.0,
  "max_conf_margin": 1.0,
  "rescue_ppm": 250.0,
  "hit_N": 3,
  "true_c4_high_hits": 3,
  "false_hits": 0,
  "calib_c4_high_N": 24,
  "calib_c4_high_recall": 0.125
}
```

## Test Metrics

| mode | family | ALL | NRMSE | C3 CO | C4 CO | C5 CO | C3 high | C4 high | C5 high | nonCO |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A0_baseline_final | reference | 27.34 | 0.16 | 33.70 | 56.59 | 46.12 | 41.70 | 95.32 | 60.00 | 19.00 |
| H2_3_R3aK16_current_mainline | reference | 18.62 | 0.13 | 16.15 | 22.02 | 26.85 | 20.02 | 34.24 | 34.82 | 17.83 |
| H8_R3aK16_source_aug_CO_else_H2_3 | reference | 18.47 | 0.14 | 14.97 | 19.76 | 23.69 | 19.93 | 32.22 | 27.54 | 18.38 |
| source_aug_target_Ridge_plus_C4_rescue | source-augmented target Ridge reference | 19.19 | 0.14 | 14.81 | 20.17 | 26.79 | 19.93 | 33.05 | 30.72 | 18.98 |
| H8_plus_formal_C4_rescue | reference | 18.30 | 0.14 | 14.97 | 17.16 | 23.69 | 19.93 | 26.79 | 27.54 | 18.38 |
| L3_light_source_ridge_full_auto_v2 | strict lightweight base | 22.62 | 0.15 | 17.04 | 48.35 | 25.83 | 21.41 | 85.87 | 29.53 | 19.54 |
| L3_light_source_per_gas_mlp_full_auto_v2 | strict lightweight base | 22.76 | 0.15 | 15.73 | 53.71 | 24.87 | 20.59 | 96.37 | 27.62 | 18.85 |
| L3_light_source_shared_mlp_full_auto_v2 | strict lightweight base | 22.63 | 0.15 | 14.27 | 53.14 | 27.25 | 19.42 | 95.38 | 29.73 | 18.70 |
| L3_light_H2_3_analog | lightweight H2.3 analogy | 22.25 | 0.15 | 15.73 | 48.35 | 24.87 | 20.59 | 85.87 | 27.62 | 19.26 |
| L3_light_H2_3_analog_plus_formal_C4_rescue | lightweight H2.3 analogy + formal rescue | 19.69 | 0.14 | 15.73 | 25.19 | 24.87 | 20.59 | 41.16 | 27.62 | 19.26 |
| L3_light_client_val_profile | lightweight client-val profile | 22.63 | 0.15 | 14.27 | 53.71 | 24.87 | 19.42 | 96.37 | 27.62 | 18.84 |
| L3_light_client_val_profile_plus_formal_C4_rescue | lightweight client-val profile + formal rescue | 19.30 | 0.14 | 14.27 | 25.82 | 24.87 | 19.42 | 43.52 | 27.62 | 18.84 |
| L3_source_ridge_CO_else_H2_3 | lightweight CO switch + H2.3 fallback | 18.95 | 0.14 | 17.14 | 20.96 | 25.68 | 21.41 | 32.40 | 29.35 | 18.38 |
| L3_source_ridge_CO_else_H2_3_plus_formal_C4_rescue | lightweight CO switch + H2.3 fallback + formal rescue | 18.79 | 0.14 | 17.14 | 18.53 | 25.68 | 21.41 | 27.02 | 29.35 | 18.38 |
| L3_source_per_gas_mlp_CO_else_H2_3 | lightweight CO switch + H2.3 fallback | 18.47 | 0.13 | 15.85 | 20.48 | 23.59 | 20.59 | 32.70 | 27.42 | 18.20 |
| L3_source_per_gas_mlp_CO_else_H2_3_plus_formal_C4_rescue | lightweight CO switch + H2.3 fallback + formal rescue | 18.30 | 0.13 | 15.85 | 17.98 | 23.59 | 20.59 | 27.38 | 27.42 | 18.20 |
| L3_source_shared_mlp_CO_else_H2_3 | lightweight CO switch + H2.3 fallback | 18.60 | 0.14 | 14.60 | 20.06 | 25.80 | 19.42 | 32.01 | 29.54 | 18.34 |
| L3_source_shared_mlp_CO_else_H2_3_plus_formal_C4_rescue | lightweight CO switch + H2.3 fallback + formal rescue | 18.44 | 0.14 | 14.60 | 17.50 | 25.80 | 19.42 | 26.54 | 29.54 | 18.34 |

## Reading

- The strict lightweight bases remain above 21.9-22.8 ALL RMSE, so they do not replace H2.3/H8 as performance mainline.
- Lightweight H2.3-style/client profiles mainly inherit the C4 high-CO weakness; formal C4 rescue helps but does not close the gap.
- Lightweight CO-switch variants are the closest pure-lightweight analogue to H8: use a lightweight candidate only when `pred_class == CO`, otherwise keep H2.3.
- The per-gas-MLP CO-switch plus formal C4 rescue nearly ties H8 + formal C4 rescue on ALL RMSE and has slightly better NRMSE/nonCO, so it is a credible deployment-lite CO-specialist candidate.
- The shared-MLP CO-switch plus formal C4 rescue has the best C4 high-CO RMSE in this matrix, but gives up more ALL/C5 performance.
- The practical conclusion is not full lightweight replacement. The useful lightweight form is selective CO-specialist switching with H2.3 fallback.
