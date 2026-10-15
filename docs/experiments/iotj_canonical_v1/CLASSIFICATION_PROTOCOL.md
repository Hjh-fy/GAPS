# Canonical v1 Formal Classification Protocol

## Frozen endpoint

- Study: `iotj_canonical_v1_final_20260808`
- Source clients: C1 and C2 only.
- Target commissioning devices: C3, C4, and C5, each run independently.
- Dataset: `dataset/iotj_canonical_v1/`, aggregate SHA256 `2f810d7e93cae5f361923184e9dc87d5ae59e0f59be9f52aff7e14f9f33e94f6`.
- Model and algorithm: the final frozen A4 GAPS router with the TCN-attention architecture, `ce_stats` client profile, client semantic/replay disabled, selective aggregation disabled, and the frozen A4 server adaptation configuration.
- Optimizer: Adam, learning rate `5e-4`.
- Federated endpoint: 25 rounds, 1 local epoch per round, batch size 32, seed 42.
- Aggregation: A4 uses ordinary FedAvg aggregation throughout; selective aggregation is unavailable/disabled.
- Server adaptation: fixed 100 steps, target-specific calibration split only.
- DA input contract: explicit temporal window length 50 and sensor dimension 8. The historical default remains 100; this declaration changes only input-shape validation and does not alter the model or losses.

Each target run starts from a fresh seed-42 random initialization. Historical source or adapted checkpoints are unavailable to the execution API, and no `checkpoint` or `resume` argument appears in the formal commands.

The local-epoch setting remains 1 to preserve the preceding frozen GAPS training configuration. The canonical rerun is intended to isolate the preprocessing change; no training-budget expansion is part of the formal comparison.

## Target information gate

The frozen GAPS server adaptation consumes source C1/C2 calibration fields and the selected target device's calibration `x`, class, and phase fields. Target concentration is not an adaptation input. The target test arrays and labels remain sealed throughout all three 25-round executions and cannot be used for loss construction, checkpoint selection, early stopping, hyperparameter selection, or method selection.

The target tests may be opened only after all three fixed round-25 completion markers exist. Evaluation is then one-time and includes ALL, C3, C4, and C5 metrics. No numerical outcome can reopen preprocessing selection or trigger tuning.

## Fail-closed gates

Execution stops before Flower starts if the local dataset preflight is not `PASS`, the aggregate hash differs, split counts differ, a remote canonical data root is absent or inconsistent, a previous formal output exists, a residual Flower process exists, a fixed command contains checkpoint reuse, or the pre-run protocol fingerprint changes.
