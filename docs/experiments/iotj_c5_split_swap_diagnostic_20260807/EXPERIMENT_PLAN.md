# C5 legacy-dataset split-swap prediction diagnostic

## Question

Does the higher C5 role-aware result (S_CC RMSE=20.286) primarily reflect
the changed C5 calibration/test window membership rather than an RMSE
calculation defect?

## Frozen diagnostic

- Experiment ID: DIAG-C5-SPLIT-SWAP-20260807.
- Target: C5; seed 42.
- Checkpoint: the completed role-aware FCL-RW-GAPS-C5 round-25 adapted
  checkpoint.
- Regression models: the already fitted role-aware C5 R84_FED_H1 models.
- Prediction data: the legacy
  client_data_c1234src_c5tgt_2080_timeaware_60_170_window_fullgrid C5 test
  split.
- No classifier training, regression refit, alpha selection, checkpoint
  selection, QC, or hyperparameter search.
- Report routed S_ALL, routed S_CC, and oracle-route regression metrics.
- Run classifier routing once with the legacy dataset's native normalization
  and once with the role-aware checkpoint's normalization. This exposes the
  normalization-root confound without selecting between the two results.

## Interpretation boundary

This is a diagnostic, not formal evidence. The role-aware checkpoint consumed
the role-aware C5 calibration split, and the legacy test split is not proven
disjoint from that calibration split. Any result is therefore marked
LEAKAGE_RISK_DIAGNOSTIC_ONLY and cannot replace either formal result.

## Acceptance checks

- All model/checkpoint inputs are fingerprinted before prediction.
- The legacy test contains exactly 1,360 rows.
- The fixed role-aware R84 model JSON is not modified.
- Metrics recomputed from saved prediction records match the summary.
