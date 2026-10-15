# C5 Low-Label Commissioning Design

## Objective

Measure whether canonical-v1 GAPS/A4 retains C5 classification performance, calibration quality, and source discrimination better than equal-label A0T when the nominal commissioning budget is reduced from the existing 20% pool to 15%, 10%, and 5%.

This is a single-seed, canonical window-level budget sensitivity study. It is not strict unseen-experiment generalization and cannot override the existing strict C5 collapse result.

## Frozen experiment matrix

Six new experiments are authorized:

| Method | Budget | Experiment ID |
|---|---:|---|
| A0T | 15% | `CAN-V1-C5-LB-A0T-B15-S42` |
| A0T | 10% | `CAN-V1-C5-LB-A0T-B10-S42` |
| A0T | 5% | `CAN-V1-C5-LB-A0T-B05-S42` |
| GAPS/A4 | 15% | `CAN-V1-C5-LB-A4-B15-S42` |
| GAPS/A4 | 10% | `CAN-V1-C5-LB-A4-B10-S42` |
| GAPS/A4 | 5% | `CAN-V1-C5-LB-A4-B05-S42` |

The existing 20% A0T and GAPS/A4 results are reused only as reference rows. Their target-adapted checkpoints are never used for initialization.

## Data and nested allocation

The immutable source is `dataset/iotj_canonical_v1/client_5`, preprocessing `HZ5_MEAN_W10S`, dataset aggregate SHA-256 `2f810d7e93cae5f361923184e9dc87d5ae59e0f59be9f52aff7e14f9f33e94f6`.

The existing C5 calibration pool has 320 windows and exactly 40 `class_id × concentration` strata with 8 windows in every stratum. The deterministic family retains, per stratum:

- 20%: 8 windows, total 320;
- 15%: 6 windows, total 240;
- 10%: 4 windows, total 160;
- 5%: 2 windows, total 80.

Within every stratum, identities are ordered deterministically to maximize raw-file and repeat diversity before adding additional windows. The same ordering is used for all budgets, guaranteeing `5% ⊂ 10% ⊂ 15% ⊂ 20%`. No sample is duplicated. The canonical C5 test arrays and identities remain unchanged and unavailable to subset construction.

Budget directories contain indexed copies of existing canonical calibration arrays and metadata only. They do not regenerate or transform features.

## Training protocol

Every new run starts from fresh seed42 initialization and executes the complete frozen protocol:

- C1/C2 source clients;
- 25 Flower rounds;
- `local_epochs=1`, batch size 32;
- Adam, learning rate `5e-4`;
- 100 target-adaptation steps after each round;
- fixed round25 endpoint;
- no checkpoint reuse, early stopping, hyperparameter search, or target-test selection.

A0T retains target-CE-only commissioning and disables every non-CE target loss. GAPS/A4 retains the exact `ce_stats`, A4, class/phase-conditioned alignment, prototype, consistency, residual, and stage settings of the canonical 20% run. Only the target calibration directory changes.

## Test discipline and evaluation

All six fixed endpoints and checkpoint SHA-256 values must exist before the C5 test is opened once. A common evaluator then reports C5 Accuracy, Macro-F1, NLL, ECE, four-class precision/recall/F1, and confusion matrices. It also evaluates merged C1+C2 source test Accuracy/Macro-F1 and calculates retention relative to the frozen canonical FedAvg round25 source endpoint.

No test metric may influence subset membership, stopping, checkpoint selection, or configuration.

## Decision rules

The primary comparison is `GAPS/A4 Macro-F1 − A0T Macro-F1` at 20%, 15%, 10%, and 5%. Degradation from the reused 20% endpoint is reported separately for each method.

- `LABEL_EFFICIENCY_SUPPORTED`: at 10% or 5%, A4 exceeds A0T by at least 1 percentage point or A0T collapses while A4 remains stable.
- `LABEL_EFFICIENCY_NOT_SUPPORTED`: A0T matches or exceeds A4, or all differences remain practically negligible.
- `DEVICE_DEPENDENT/INCONCLUSIVE`: non-monotonic or anomalous outcomes cannot be attributed to label quantity after composition and implementation audits.

Multi-seed work is proposal-only and is never launched automatically.

## Fail-closed conditions

Execution stops before training if any dataset hash differs; any budget count or stratum count differs; nesting fails; calibration/test exact identity overlap is nonzero; commands reference target test; a run reuses a checkpoint; A0T/A4 settings differ from their frozen canonical settings beyond calibration path and experiment identity; or remote/local manifest hashes disagree.

After the six-run evaluation and audit, the workflow stops. C3/C4, lower budgets, regression, QC, additional methods, retuning, and automatic multi-seed execution are out of scope.
