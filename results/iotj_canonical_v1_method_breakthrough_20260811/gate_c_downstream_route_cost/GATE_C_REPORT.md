# Gate C Downstream Routing Cost Audit

## [Scientific Question]

Can equal classification-error counts produce materially different quantitative risk because true-gas to routed-gas errors have heterogeneous downstream concentration costs?

## [Protocol]

The 4x4 matrix was constructed only from the 320-window canonical-v1 C5 calibration set by forcing every sample through each frozen R84_FED_H1 gas route. The primary off-diagonal cost is max(0, mean incremental squared ppm error versus the correct route). The matrix was hashed and locked before existing C5 test predictions were opened for post-hoc explanation. Grouped bootstrap resampled complete raw filenames for 2000 seed42 replicates.

## [Primary Result]

- A0T C5 S_ALL RMSE: 22.155951 ppm
- A4 C5 S_ALL RMSE: 18.476540 ppm
- A4 - A0T RMSE: -3.679410 ppm
- Positive off-diagonal cost CV: 0.638050
- A0T/A4 misroutes: 8/8
- Misroute union windows: 14
- Positive-contribution raw files: 3
- Top-one/top-two positive file shares: 0.5194/0.8657
- Grouped-bootstrap A4-A0T RMSE 95% CI: [-11.202361, 3.312817] ppm; P(delta<0)=0.8300

## [Negative Result / Limitation]

The cost matrix is calibration-estimated on one target device and seed42. It motivates but does not validate a cost-aware decision rule. Test misroute decomposition is strictly post-hoc and cannot alter the matrix or routing policy.

## [Leakage Audit]

`CALIBRATION_COST_MATRIX_LOCK.json` was written before reading test prediction CSVs. No test row, label, error, probability, or filename entered cost construction, thresholding, model fitting, or hyperparameter selection.

## [Decision]

`COST_AWARE_ROUTING_MOTIVATED`.

## [Paper Implication]

Classification accuracy alone is insufficient to characterize quantitative routing risk when off-diagonal downstream costs are heterogeneous. A cost-aware router remains a separately gated future method, not a supported component yet.

## [Next Action]

`GO_GATE_D` is the registered scientific recommendation. This task stops here and does not execute Gate D/E/F.
