# P7 R84 feature audit

`p7_r84_feature_comparison.csv` contains pairwise physical-window feature differences for legacy vs current interpolation and legacy vs time-bin. `p7_r84_feature_family_summary.csv` aggregates the observed differences into global and per-channel statistic families. It intentionally does **not** infer concentration-feature correlation or coefficient-of-variation from the absolute-difference table; those quantities were not persisted per window and are therefore marked unavailable rather than reconstructed from test outcomes.
