# Strict non-overlap robustness analysis

This supplementary sensitivity changes target membership only; canonical-v1 remains the primary protocol. The strict split has zero exact-window, raw-file, and raw-time overlap and retains the frozen calibration N.

The preregistered descriptive collapse flags are an absolute Macro-F1 loss of at least 0.20 or an S_ALL RMSE ratio of at least 2.0. They are reporting flags, not tuning or acceptance criteria.

| Target | ΔMacro-F1 | ΔS_ALL RMSE (ppm) | RMSE ratio | Classification flag | Regression flag |
|---|---:|---:|---:|---|---|
| C3 | -0.000499 | 1.494 | 1.160 | False | False |
| C4 | -0.004938 | 0.471 | 1.034 | False | False |
| C5 | -0.300519 | 54.461 | 3.948 | True | True |

Overall collapse flag: **True**.

All drops and improvements are retained. No retraining, hyperparameter change, outlier removal, or replacement of the canonical main results is authorized by this analysis.
