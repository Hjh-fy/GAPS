# Formal C4 Route-Rescue Selector

Selection uses target calibration only. Test is used only after selecting the fixed gate.

## Selected Gate

```json
{"class_label": "ethanol", "pred_classes": "0", "phase": "any", "max_final": 20.0, "min_risk": 6.0, "max_conf_margin": 1.0, "rescue_ppm": 250.0, "hit_N": 3, "true_c4_high_hits": 3, "false_hits": 0, "calib_c4_high_N": 24, "calib_c4_high_recall": 0.125}
```

## Calibration Candidate Ranking

| rank | classes | phase | max_final | min_risk | max_margin | rescue | hits | true high | false | recall |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0 | any | 20 | 2.0 | 1.0 | 250 | 3 | 3 | 0 | 0.125 |
| 2 | 0 | any | 20 | 4.0 | 1.0 | 250 | 3 | 3 | 0 | 0.125 |
| 3 | 0 | any | 20 | 6.0 | 1.0 | 250 | 3 | 3 | 0 | 0.125 |
| 4 | 0 | any | 30 | 2.0 | 1.0 | 250 | 3 | 3 | 0 | 0.125 |
| 5 | 0 | any | 30 | 4.0 | 1.0 | 250 | 3 | 3 | 0 | 0.125 |
| 6 | 0 | any | 30 | 6.0 | 1.0 | 250 | 3 | 3 | 0 | 0.125 |
| 7 | 0 | any | 50 | 2.0 | 1.0 | 250 | 3 | 3 | 0 | 0.125 |
| 8 | 0 | any | 50 | 4.0 | 1.0 | 250 | 3 | 3 | 0 | 0.125 |
| 9 | 0 | any | 50 | 6.0 | 1.0 | 250 | 3 | 3 | 0 | 0.125 |
| 10 | 0 | recovery | 20 | 2.0 | 1.0 | 250 | 3 | 3 | 0 | 0.125 |
| 11 | 0 | recovery | 20 | 4.0 | 1.0 | 250 | 3 | 3 | 0 | 0.125 |
| 12 | 0 | recovery | 20 | 6.0 | 1.0 | 250 | 3 | 3 | 0 | 0.125 |
| 13 | 0 | recovery | 30 | 2.0 | 1.0 | 250 | 3 | 3 | 0 | 0.125 |
| 14 | 0 | recovery | 30 | 4.0 | 1.0 | 250 | 3 | 3 | 0 | 0.125 |
| 15 | 0 | recovery | 30 | 6.0 | 1.0 | 250 | 3 | 3 | 0 | 0.125 |

## Test Summary

| mode | scope | RMSE | Bias | P90AE | N |
|---|---|---:|---:|---:|---:|
| H8_pred_CO_source_aug | ALL | 18.47 | 0.51 | 13.47 | 5400 |
| H8_pred_CO_source_aug | C4-CO_high_200_250 | 32.22 | -4.58 | 24.69 | 102 |
| H8_pred_CO_source_aug | C4-nonCO | 8.86 | 0.46 | 11.43 | 1020 |
| H8_pred_CO_source_aug | nonCO_ALL | 18.38 | 0.97 | 10.59 | 4050 |
| H8_plus_formal_c4_route_rescue | ALL | 18.30 | 0.55 | 13.47 | 5400 |
| H8_plus_formal_c4_route_rescue | C4-CO_high_200_250 | 26.79 | -2.26 | 24.69 | 102 |
| H8_plus_formal_c4_route_rescue | C4-nonCO | 8.86 | 0.46 | 11.43 | 1020 |
| H8_plus_formal_c4_route_rescue | nonCO_ALL | 18.38 | 0.97 | 10.59 | 4050 |

## Test Gate Audit

| label | hit | true high | false | ALL | C4 high | C4 nonCO | nonCO |
|---|---:|---:|---:|---:|---:|---:|---:|
| H8_pred_CO_source_aug | 0 | 0 | 0 | 18.47 | 32.22 | 8.86 | 18.38 |
| H8_plus_formal_c4_route_rescue | 14 | 14 | 0 | 18.30 | 26.79 | 8.86 | 18.38 |

## Reading

- This is stricter than the previous upper-bound sweep because the gate is selected on calibration.
- A useful gate should reduce C4 high-CO without adding false hits on C4/nonCO or nonCO overall.
- If the selected calibration gate does not reproduce the test upper bound, the C4 rescue pattern is likely file/repeat-specific and needs stronger validation.
