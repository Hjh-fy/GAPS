# Result Analysis

All results are fixed-endpoint, seed42-only descriptive evidence. No target test was used for learning-rate selection, threshold selection, stopping, hyperparameter search or checkpoint selection.

## Baseline interpretation

SCAFFOLD is implemented with its canonical SGD-style control-variate update, whereas FedAvg, FedProx, and GAPS use the frozen Adam optimizer adopted by the experimental system. Therefore, the comparison represents standard algorithm-level baselines rather than an optimizer-controlled single-factor ablation.

## Per-target fixed-endpoint summary

- C3: highest fixed-endpoint macro-F1 is GAPS (0.989693); this is descriptive seed42 evidence.
- C4: highest fixed-endpoint macro-F1 is GAPS (0.990624); this is descriptive seed42 evidence.
- C5: highest fixed-endpoint macro-F1 is GAPS (0.984543); this is descriptive seed42 evidence.

## Interpretation limits

The comparison uses one registered seed and one C1/C2 source split. Differences must not be described as multi-seed stability, universal cross-device superiority or optimizer-controlled causal effects. E2 isolates canonical x-only post-hoc adaptation from GAPS's registered class/phase calibration use. Communication totals are deterministic model/control payload estimates; GAPS statistic JSON is identified separately rather than misreported as exact wire bytes.
