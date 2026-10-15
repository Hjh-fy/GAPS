# Full target-personalized R84 regression analysis

Each target uses its frozen target-matched GAPS router and a separately calibrated 84-D target Ridge (83-D sensor statistics + frozen Federated-H1). Target test was opened only after the corresponding calibration lock was persisted and validated.

| Target | Calibration N | Router accuracy | S_ALL RMSE | S_ALL MAE | S_ALL R2 | S_ALL NRMSE | S_CC N | S_CC RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C3 | 320 | 98.97% | 20.481 | 7.579 | 0.9048 | 0.1284 | 673 | 9.996 |
| C4 | 160 | 99.06% | 21.400 | 12.744 | 0.8960 | 0.1675 | 317 | 19.455 |
| C5 | 320 | 98.46% | 16.093 | 7.700 | 0.9412 | 0.1184 | 1339 | 11.797 |

## Interpretation

- C3 has the best correct-route regression (S_CC RMSE 9.996 ppm), but seven misroutes raise end-to-end S_ALL RMSE to 20.481 ppm; the misrouted subset RMSE is 176.472 ppm.
- C4 has only three misroutes, yet its S_CC RMSE remains 19.455 ppm. The principal observed limitation is therefore not routing frequency; Ethanol has the largest C4 correct-route RMSE (30.744 ppm). C4 also has only half the target calibration budget, so this cannot be attributed to device shift alone.
- C5 has 21 misroutes: its S_CC RMSE is 11.797 ppm and S_ALL RMSE is 16.093 ppm. The separately reported formal A4-C5+R84 reference is 11.462/12.855 ppm for S_CC/S_ALL with 1351 correct routes, showing that the different router mostly changes end-to-end performance and only modestly changes correct-route regression.

C3/C5 use 320 calibration windows and C4 uses 160; therefore cross-target differences combine device/domain effects and calibration-budget differences. These are seed-42 fixed-endpoint capability results, not uncertainty estimates or a device-only causal ranking. The formal A4-C5+R84 row is retained separately because its router differs.
