# Final post-hoc R84/QC closure

Decision: `PIPELINE_CLOSURE_PASS_C5_ONLY`.

The formal post-hoc lifecycle endpoint is available only for C5. It uses the
fixed step-100 post-hoc classifier (SHA256 `857f3954003bffad1af716002a1bd2915923389faec31b69f5c72e563aaa212c`), the
frozen C5 R84/H1 alphas, 320 canonical calibration windows, and 1360 sealed-test
windows. No classifier training, alpha search, test-based refit, or QC formula
change occurred in this closure.

## C5 result

- Classification: Accuracy 0.9764705882; Macro-F1 0.9765440505.
- R84 S_ALL: RMSE 28.057496 ppm, MAE 10.691171,
  NRMSE 0.199727, R2 0.821260, Bias 1.167081.
- R84 S_CC: RMSE 13.110113 ppm.
- R84 Oracle_ALL: RMSE 14.448829 ppm.
- Routing gap S_ALL-S_CC: 14.947383 ppm.
- Oracle gap S_ALL-Oracle_ALL: 13.608668 ppm.

S_CC and Oracle_CC are identical by construction on correctly routed samples;
this is not independent evidence of regression-map improvement.

## QC workpoints

HC90 transfers to test coverage 0.834559
(error -0.065441) with accepted RMSE
24.642191 ppm. HC95 transfers to
0.893382 (error
-0.056618) with accepted RMSE
25.178852 ppm.

C3/C4 remain blocked because no formal final post-hoc endpoint exists. Historical
interleaved-A4 endpoints were not substituted.
