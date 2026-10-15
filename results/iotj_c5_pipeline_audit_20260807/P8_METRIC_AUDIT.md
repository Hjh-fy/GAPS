# P8 independent metric audit

PASS. This script imports no project evaluator or RMSE helper. It reads persisted row records and directly computes `sqrt(mean((pred-true)**2))` in NumPy. Maximum absolute RMSE difference versus the P6 summary is `0` (required < 1e-8). S_CC is independently filtered by `route_correct == 1`. Per-arm, per-gas and per-concentration N/RMSE/MAE/Bias/R2 are in `p8_independent_metric_check.csv`.
