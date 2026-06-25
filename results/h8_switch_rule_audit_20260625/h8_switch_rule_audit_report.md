# H8 Switch Rule Visibility Audit

Question: is the H8 `pred_class == CO` switch a deployment-visible rule with reasonable calibration/test support?

- Input target predictions: `results/timeaware_2080_c12src_c345tgt_fixed_da_r25_r3ak16_auto_v2_eval/ppm_layer_co_audit/target_layer_predictions.csv`
- Output CSV: `results/h8_switch_rule_audit_20260625/h8_pred_co_switch_split_audit.csv`
- Rule audited here: switch to CO specialist when the deployed classifier predicts CO.
- This audit does not evaluate ppm improvement; it checks support, false positives, and leakage risk.

## Split Audit

| split | client | N | switch | true CO | nonCO | CO high | precision | FP rate | CO recall | high recall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| calibration | ALL | 1320 | 325 | 322 | 3 | 95 | 0.991 | 0.009 | 0.976 | 0.960 |
| calibration | C3 | 680 | 172 | 170 | 2 | 51 | 0.988 | 0.012 | 1.000 | 1.000 |
| calibration | C4 | 320 | 77 | 77 | 0 | 21 | 1.000 | 0.000 | 0.963 | 0.875 |
| calibration | C5 | 320 | 76 | 75 | 1 | 23 | 0.987 | 0.013 | 0.938 | 0.958 |
| test | ALL | 5400 | 1341 | 1306 | 35 | 387 | 0.974 | 0.026 | 0.967 | 0.956 |
| test | C3 | 2680 | 685 | 668 | 17 | 201 | 0.975 | 0.025 | 0.997 | 1.000 |
| test | C4 | 1360 | 322 | 321 | 1 | 85 | 0.997 | 0.003 | 0.944 | 0.833 |
| test | C5 | 1360 | 334 | 317 | 17 | 101 | 0.949 | 0.051 | 0.932 | 0.990 |

## Existing H8 Test Switch Audit

```csv
switch_rule,switched_N,true_CO_N,nonCO_N,by_client,by_pred_gas,key_field_mismatches
h8_pred_co_switch,1341,1306,35,"{""C3"": 685, ""C4"": 322, ""C5"": 334}","{""CO"": 1341}",0
h8_pred_co_or_c4_gate_switch,1356,1321,35,"{""C3"": 685, ""C4"": 337, ""C5"": 334}","{""CO"": 1341, ""Ethanol"": 15}",0
```

## Reading

- The rule uses only deployment-visible `pred_class`, so it is eligible for runtime use.
- Calibration support and false-positive rate should be compared with test before promoting H8.
- If calibration and test switch behavior are aligned, the next step is a formal selector that chooses H2.3 vs H8 using calibration-validation only.
- H8 still needs a richer runtime artifact because its CO specialist depends on source-head predictions, not just the existing target Ridge policy.
