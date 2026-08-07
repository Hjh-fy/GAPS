# P1 window identity

- Experiment groups in both datasets: 80/80; every group has 21/21 windows.
- Hungarian one-to-one physical matches: 1680/1680.
- Numerically bit-identical: 0/1680.
- Median/P95/max matched-window RMSE: 0.00134627463 / 0.049095004 / 2.38930632.
- Median matched max-absolute difference: 0.00639920789.
- Tolerance matches (atol 1e-6/1e-5/1e-4/1e-3): 0/0/0/0.
- Same calibration/test membership after physical matching: 1180/1680.

Conclusion: the datasets represent the same named physical experiments and the same 21 nominal time positions per experiment, but are not numerically identical windows and have different split membership.
