# P4 R84 feature audit

- Actual R84 = 83 deterministic sensor statistics plus one frozen Federated-H1 source Ridge prediction.
- DCT dimension: 0. Encoder/reg_feat dimension: 0. Target phase/class metadata dimension: 0.
- Per-head Ridge standardizes each feature using its calibration-fit mean/std; no dataset `norm_stats.npz` is applied inside this feature builder.
- Feature ordering is lexicographically sorted and SHA-bound per row in `p4_matched_r84_features.csv`.
- Across 1,680 Hungarian-matched physical windows, median/P95/max 84-D RMSE = 0.338645/3.35807/24.6346.

The R84 implementation is common to OLD and NEW. Its values differ because the input windows differ numerically, not because two R84 builders were selected.
