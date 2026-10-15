# R1-M2 Baseline Fairness Experiment Plan

## Research brief and scope

- Brief source: reviewer concern R1-M2 (baseline fairness) for the GAPS IoTJ manuscript.
- Target venue/audience: IEEE Internet of Things Journal reviewers.
- Resource budget: five new configurations, one locked formal run per configuration, all with seed 42. No multi-seed stability or significance claim is authorized by this plan.
- Existing references: canonical GAPS B5 seed-42 result in `results/iotj_b5_multiseed_20260724`; canonical historical FedAvg/A0 seed-42 checkpoint in `results/iotj_classification_ablation_20260711_v2r1` (must retain its older-code provenance).

## Hypotheses

| ID | Falsifiable hypothesis | Baseline | Intervention | Primary metric | Expected Evidence | Acceptance criterion |
|---|---|---|---|---|---|---|
| H-R1M2-01 | Source knowledge improves target-device classification relative to training the same TCN only on the allowed target calibration labels. | Target-only TCN | GAPS B5 | C5 test macro-F1 | GAPS B5 exceeds target-only under equal target-label access and declared update budgets. | Report the observed difference without a superiority claim if GAPS does not exceed target-only. |
| H-R1M2-02 | Federated optimization does not receive an undeclared advantage over centralized source training. | Centralized source-only TCN | FedAvg source-only/A0 | C5 test macro-F1 | Comparable or explicitly explained performance under equal source samples, epochs, backbone, optimizer and seed. | Any material difference must be attributed to optimization topology, not described as domain adaptation. |
| H-R1M2-03 | The GAPS gain is not explained solely by a standard non-IID federated optimizer. | FedProx source-only | GAPS B5 | C5 test macro-F1 | GAPS B5 exceeds FedProx with no target labels exposed to FedProx. | If FedProx matches/exceeds GAPS, narrow the mechanism claim. |
| H-R1M2-04 | GAPS client mechanisms and selective aggregation add value beyond target-label-assisted adaptation alone. | FedAvg + same target adapter | GAPS B5 | C5 test macro-F1 | GAPS B5 exceeds the adapter-matched FedAvg comparator. | If not, attribute the gain primarily to target-assisted server adaptation and revise the claim. |
| H-R1M2-05 | GAPS provides benefit beyond an electronic-nose calibration-transfer baseline. | Regularized DS + FedAvg checkpoint | GAPS B5 | C5 test macro-F1 | GAPS B5 exceeds DS under the same C5 calibration/test boundary. | If not, avoid claiming superiority over calibration transfer. |

## Fixed protocol

- Source clients: C1 and C2; each source train split has 2,360 windows.
- Target client: C5; calibration has 320 windows and the sealed test split has 1,360 windows.
- Split protocol: `dataset/client_data_c1234src_c5tgt_2080_timeaware_60_170_window_fullgrid`, existing time-aware split manifest, split seed 42. The C5 test split is evaluation-only.
- Model/checkpoint policy: same repository TCN architecture and classification head. Final-round checkpoints only; no C5-test model selection. Checkpoint and source-tree SHA-256 hashes are mandatory.
- Seeds: exactly seed 42 for initialization, loader order and stochastic transforms. These runs cannot support variance or statistical-significance claims.
- Common optimizer budget: Adam, learning rate 5e-4, batch size 32. Source training uses 25 rounds x 5 local epochs. Target-only uses exactly 2,500 calibration optimizer steps, matching B5's 25 x 100 server-adaptation steps.
- FedProx: proximal coefficient mu=0.01, fixed before execution; no target data and no post-hoc tuning.
- FedAvg + same target adapter: CE-only C1/C2 updates, sample-weighted FedAvg, client feature-statistic upload enabled only for the server adapter, and the exact B5 seed-42 target calibration adapter settings. Selective aggregation, client alignment, replay distillation and prototype decoupling are disabled.
- DS: regularized affine target-to-source direct standardization fitted from matched source/C5 calibration strata; regularization is selected using calibration-only folds, then locked before a single test evaluation. The classifier is the frozen FedAvg/A0 checkpoint.
- DA / calibration / QC controls: target labels are prohibited for centralized source-only, FedProx and source-only FedAvg. Target-only, FedAvg+same adapter and DS may use only the fixed 320-window C5 calibration split. No additional QC exclusion is introduced.
- Metrics: C5 test accuracy, macro-F1, per-class recall, NLL, ECE, confusion matrix; wall-clock time, peak memory where observable, transmitted parameter/statistic bytes and total communication rounds. Downstream regression replay is secondary and uses the same locked representation/result provenance.

## Risks, unknowns, conflicts, and stopping rules

- The historical A0 checkpoint was produced by an older code revision. It may be reused only as a read-only, hash-pinned comparator with the incompatibility disclosed; otherwise A0 must be rerun from the new runtime commit.
- DS requires matched gas/concentration/phase strata. If the pairing audit fails, the run is marked blocked rather than relaxing the split or inspecting C5 test labels.
- A run is invalid if it changes the split, seed, backbone, target-label budget, optimizer budget or opens C5 test for tuning.
- Runtime failures may be retried from scratch with the identical locked manifest. Hyperparameter changes create a new experiment ID and are outside this five-run plan.
- All artifacts must be registered before execution, then committed and pushed to the active GitHub branch after validation.
