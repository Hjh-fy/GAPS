# Role-aware target-specific GAPS + R84 correction study

## Objective

Correct the legacy-split cross-target diagnostic by using the registered
`C1/C2 source -> C3/C4/C5 target` role-aware dataset. Existing result roots
remain read-only. The new output root is
`results/iotj_gaps_roleaware_r84_full_20260805`.

## Frozen classification protocol

- Source clients: C1 and C2 only.
- Target branches: C3, C4, and C5, each trained independently.
- Dataset: `client_data_c12src_c345tgt_2080_timeaware_60_170_window_fullgrid`.
- Target split: role-aware 20% calibration / 80% sealed test; actual counts are
  C3 680/2680, C4 320/1360, and C5 320/1360.
- GAPS: exact FCL-E3 configuration: 25 rounds, local epoch 1, batch size 32,
  Adam lr 5e-4, selective warm-up rounds 1-5, target-specific server DA,
  100 DA steps per round, seed 42.
- No target test use during training, adaptation, checkpoint selection, or
  hyperparameter selection. Fixed round 25 is the endpoint.
- No source-only Flower checkpoint is retrained or replaced; these are new
  target-specific full GAPS correction runs because GAPS interleaves source
  updates and target adaptation.

## Frozen regression protocol

- Router: the matching corrected round-25 target-specific GAPS checkpoint.
- Input: R84_FED_H1 = 83-D sensor statistics + routed immutable Federated-H1.
- One target Ridge per gas; alpha grid `[0, .01, .1, 1, 10, 100, 1000]`.
- Calibration-only deterministic concentration-stratified selection; refit on
  all calibration rows before the target test is opened.
- Metrics: Accuracy/Macro-F1/NLL/ECE for routing; RMSE/MAE/R2/NRMSE for S_ALL,
  S_CC, gas, concentration, correct-route, and misrouted scopes.
- Seed 42 only; no uncertainty or stability claim.

## Leakage gate and stopping rule

The existing FCL-E3 checkpoints used a different legacy calibration split.
They are unavailable for this formal test because checkpoint-to-test
non-overlap cannot be established and nearest-window provenance maps about
80% of their old calibration rows into the role-aware test. Any new run that
opens target test before its calibration/checkpoint lock is persisted fails
closed. No tuning or matrix changes are allowed after the pre-run freeze.
