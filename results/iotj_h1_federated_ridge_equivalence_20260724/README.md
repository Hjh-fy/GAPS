# H1 pooled-to-federated sufficient-statistics equivalence

- Decision: `PRACTICAL_EQUIVALENCE`
- Formal commit: `e12a4eb61180c8819b9f6e87dee71b103ac040a8`
- Source: C1/C2; target: C5; B5 seed 42; C5 calibration/test: 320/1360.
- H1 is a 104D per-gas custom closed-form Ridge. The federated path exchanges
  feature moments, normal equations, and calibration SSE/count only.
- Maximum scaler difference: `1.9539925233402755e-14`
- Maximum coefficient/intercept difference: `1.7438189274798788e-06`
- Maximum C5 H1 prediction difference: `2.0809125089726876e-08` ppm
- Ridge+H1 pooled S_ALL/S_CC RMSE: `25.648978143131` /
  `11.341598573030` ppm
- Ridge+H1 federated S_ALL/S_CC RMSE: `25.648978143013` /
  `11.341598573025` ppm

Evidence boundary: source raw samples remain local and only aggregated
sufficient statistics are used to reconstruct the global Ridge solution.
This audit does not claim secure aggregation, differential privacy,
cryptographic privacy, or that sufficient statistics are non-leaking.
Runtime v4 and QC remain unchanged.
