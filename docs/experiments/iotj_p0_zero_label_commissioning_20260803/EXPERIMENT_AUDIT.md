# P0-U Experiment Audit

## Verdict: approved for seed42 descriptive evidence

Static and runtime label-access audits passed. U1/U2 target training APIs received x-only tensors; U1 label-conditioned losses were unavailable/disabled; U2 pseudo labels came only from a frozen source teacher at the fixed 0.90 threshold. Calibration truth opened once after both training branches solely for pseudo-label precision, and C5 test opened afterward for final evaluation. Both branches identify the same hash-pinned source checkpoint. No hyperparameter, threshold, early-stopping, or checkpoint search occurred.

## Limitations

Seed42 only. Existing Source-only and supervised Target-CE rows are read-only references. Results do not support uncertainty, significance, or automatic follow-up optimization.
