# P0-U Zero-Label Target Commissioning Study

## Scope

- Reuse the immutable P0A round25 source checkpoint; no Flower/source retraining.
- C1/C2 source reference data and C5 calibration inputs; C5 target class labels are unavailable to U1/U2 training APIs.
- Seed42, 100 steps, Adam, model LR 5e-4, no hyperparameter search.

## Hypotheses

| ID | Falsifiable hypothesis | Baseline | Intervention | Primary metric | Expected evidence | Acceptance criterion |
|---|---|---|---|---|---|---|
| P0U-H1 | Label-free global alignment improves over frozen source-only at round25. | Source-only round25 | U1 unconditional alignment | C5 test Macro-F1 | fixed post-commissioning result | signed descriptive difference |
| P0U-H2 | Fixed-threshold pseudo-label self-training improves over frozen source-only without target labels. | Source-only round25 | U2 frozen-teacher pseudo CE | C5 test Macro-F1 | fixed post-commissioning result plus coverage | signed descriptive difference |
| P0U-H3 | Zero-label commissioning approaches supervised simple Target-CE. | Simple Target-CE round25 | U1/U2 | C5 test Macro-F1 | unified four-row table | descriptive gap; no significance claim |

## Fixed protocol

- Checkpoint: P0A `server_round_025.pth`, SHA-256 `4313c375a8fa2e929de9d65637a2196f6c0f0752c2dc78112020b8727351751c`.
- U1: source CE weight 1; unconditional CORAL 0.5; unconditional global MMD² 0.5; unconditional Wasserstein-min adversarial feature loss 0.5. Three critic updates per step, critic LR 1e-3, GP 10. Target API receives tensors only.
- U2: frozen source teacher, confidence threshold 0.90 fixed before execution, high-confidence pseudo-label CE only. No pseudo-label refresh policy search and no true target label access.
- C5 calibration labels may be opened once only after both U1/U2 training completes, solely for post-hoc pseudo-label precision audit.
- C5 sealed test loader may be constructed only after both 100-step branches and checkpoints complete.

## Stopping rules and limitations

- Fail closed on any target-label-bearing object at either training API, checkpoint mismatch, threshold mismatch, target-test pre-access, or missing diagnostic row.
- No early stopping, threshold/configuration/checkpoint selection, pseudo-label search, or automatic follow-up optimization.
- Seed42 only; no uncertainty, significance, or stability claim.
