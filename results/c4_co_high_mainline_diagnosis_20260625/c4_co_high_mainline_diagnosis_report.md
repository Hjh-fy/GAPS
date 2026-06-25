# C4 High-CO Mainline Diagnosis

Scope: C4, true CO, true ppm >= 200, target test, no QC filtering.

## Overall

| candidate | N | RMSE | MAE | Bias | P90AE | Under50Rate |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 102 | 95.32 | 52.02 | -33.67 | 233.80 | 0.17 |
| h2_3 | 102 | 34.24 | 15.63 | -9.67 | 39.42 | 0.02 |
| source_aug | 102 | 33.05 | 14.16 | -4.70 | 24.69 | 0.02 |
| h8 | 102 | 32.22 | 14.04 | -4.58 | 24.69 | 0.02 |

## By Response Phase

| response_phase | candidate | N | RMSE | Bias | P90AE | Under50Rate |
|---|---|---:|---:|---:|---:|---:|
| main_response | baseline | 49 | 33.60 | 10.69 | 35.71 | 0.02 |
| main_response | h2_3 | 49 | 28.32 | -2.47 | 12.63 | 0.02 |
| main_response | source_aug | 49 | 28.59 | 0.94 | 17.01 | 0.02 |
| main_response | h8 | 49 | 28.59 | 0.94 | 17.01 | 0.02 |
| recovery | baseline | 53 | 128.22 | -74.68 | 234.07 | 0.30 |
| recovery | h2_3 | 53 | 38.92 | -16.33 | 44.64 | 0.02 |
| recovery | source_aug | 53 | 36.70 | -9.92 | 26.63 | 0.02 |
| recovery | h8 | 53 | 35.24 | -9.69 | 26.63 | 0.02 |

## By Predicted Route

| route_group | candidate | N | RMSE | Bias | P90AE | Under50Rate |
|---|---|---:|---:|---:|---:|---:|
| pred_CO | baseline | 85 | 20.84 | 5.28 | 35.38 | 0.00 |
| pred_CO | h2_3 | 85 | 19.70 | -7.04 | 38.41 | 0.00 |
| pred_CO | source_aug | 85 | 15.05 | -0.94 | 23.00 | 0.00 |
| pred_CO | h8 | 85 | 15.05 | -0.94 | 23.00 | 0.00 |
| pred_nonCO | baseline | 17 | 228.78 | -228.40 | 234.14 | 1.00 |
| pred_nonCO | h2_3 | 17 | 71.38 | -22.82 | 90.00 | 0.12 |
| pred_nonCO | source_aug | 17 | 73.64 | -23.53 | 90.00 | 0.12 |
| pred_nonCO | h8 | 17 | 71.38 | -22.82 | 90.00 | 0.12 |

## Top H8 Absolute Errors

| sample | phase | pred | true | h2.3 | h8 | h8 abs err | filename | repeat |
|---:|---|---|---:|---:|---:|---:|---|---:|
| 385 | recovery | Ethylene | 250.0 | 24.5 | 24.5 | 225.5 | B4_GCO_F100_R1.txt | 1.0 |
| 982 | main_response | Ethanol | 200.0 | 12.5 | 12.5 | 187.5 | B4_GCO_F080_R2.txt | 2.0 |
| 273 | recovery | CO | 250.0 | 196.7 | 205.1 | 44.9 | B4_GCO_F100_R1.txt | 1.0 |
| 8 | recovery | CO | 250.0 | 194.0 | 206.6 | 43.4 | B4_GCO_F100_R1.txt | 1.0 |
| 974 | recovery | CO | 250.0 | 194.1 | 209.0 | 41.0 | B4_GCO_F100_R1.txt | 1.0 |
| 927 | main_response | CO | 200.0 | 208.4 | 229.9 | 29.9 | B4_GCO_F080_R1.txt | 1.0 |
| 485 | recovery | CO | 250.0 | 201.1 | 221.2 | 28.8 | B4_GCO_F100_R1.txt | 1.0 |
| 453 | recovery | CO | 250.0 | 206.8 | 223.0 | 27.0 | B4_GCO_F100_R1.txt | 1.0 |
| 528 | recovery | CO | 225.0 | 188.6 | 199.8 | 25.2 | B4_GCO_F090_R2.txt | 2.0 |
| 783 | recovery | Ethanol | 225.0 | 250.0 | 250.0 | 25.0 | B4_GCO_F090_R2.txt | 2.0 |
| 599 | recovery | CO | 250.0 | 209.5 | 225.2 | 24.8 | B4_GCO_F100_R1.txt | 1.0 |
| 1355 | recovery | CO | 250.0 | 205.0 | 226.3 | 23.7 | B4_GCO_F100_R1.txt | 1.0 |

## Reading

- If `route_group=pred_nonCO` has high Under50Rate, the failure is route-driven and residual correction cannot fully fix it.
- If `route_group=pred_CO` still has high RMSE/Bias, the concentration mapping itself remains weak for C4 high CO.
- Phase-specific concentration bias points to recovery/main-response mismatch and should guide the next C4-specific candidate.
