# B2/B5 Cross-Direction Classification Experiment Specification

**Status:** Approved on 2026-07-13

## Research Question

Test whether the compact B2 classifier remains competitive with the corrected full B5 classifier when the source and target data domains change. The experiment is an appendix/generalization study and does not replace the frozen primary C1/C2-to-C5 protocol.

The supported conclusion is limited to the incremental B5 factors. If B2 is non-inferior, the evidence argues that adding corrected CORAL, cross-domain class-phase MMD, and adversarial alignment on top of the shared semantic core and global/class MMD does not provide a stable additive benefit. It does not show that prototype semantics, replay, selective aggregation, or global/class MMD are unnecessary because B2 and B5 share those mechanisms.

## Frozen Directions And Data

| Direction ID | Sources | Target | Data root | Source train N | Target calibration/test N |
|---|---|---|---|---:|---:|
| F1_C1_TO_C5 | C1 | C5 | `dataset/client_data_c1234src_c5tgt_2080_timeaware_60_170_window_fullgrid` | C1=2360 | 320/1360 |
| R1_C5_TO_C1 | C5 | C1 | `dataset/client_data_c2345src_c1tgt_2080_timeaware_60_170_window_fullgrid` | C5=1200 | 680/2680 |
| R2_C45_TO_C1 | C4,C5 | C1 | `dataset/client_data_c2345src_c1tgt_2080_timeaware_60_170_window_fullgrid` | C4=1200, C5=1200 | 680/2680 |

All active train, calibration, and test splits contain four balanced classes. The existing advisor-approved window-level class/concentration-stratified split is reused. No dataset is regenerated unless an integrity check detects a missing active array, count mismatch, class imbalance, or provenance mismatch.

## Frozen Methods

Both methods use 25 Flower rounds, 5 local epochs, batch size 32, client Adam learning rate `5e-4`, gradient clipping 5, 100 server adaptation steps per round, server learning rate `5e-4`, `proto_replay`, GAPS selective aggregation, prototype EMA 0.8, warmup 3, and minimum aggregation scale 0.3.

Both methods use prototype anchor 0.3, prototype fit 0.05, consistency 2.0, device residual 0.1, target CE 0, and detached prototype pair-L2 weight 0. Target calibration labels are available to class-conditional adaptation terms, so the method is calibration-assisted rather than unsupervised.

- B2 adds conventional global MMD-squared 0.5 and class-conditional MMD-squared 0.5.
- B5 adds B2 plus class-conditional CORAL 0.5, cross-domain same-class/same-phase MMD-squared 0.2, and corrected Wasserstein feature alignment 0.5.

No weights or training budgets may be tuned after a target test result is opened.

## Real Device Topology

| Direction | Raspberry Pi | Windows PC | Alibaba Cloud ECS |
|---|---|---|---|
| C1 to C5 | C1 source client | tunnel host only | Flower server, source validation, C5 calibration adaptation |
| C5 to C1 | C5 source client | tunnel host only | Flower server, source validation, C1 calibration adaptation |
| C4,C5 to C1 | C4 source client | C5 source client | Flower server, source validation, C1 calibration adaptation |

The Raspberry Pi endpoint is `gaps@192.168.31.184`. All reportable training uses these physical edge clients and ECS; local simulated training is prohibited. The controller monitors Pi temperature and throttling and refuses to overwrite partial remote runs.

## Execution Order

Seed-42 screening runs execute sequentially in this exact order:

1. F1_C1_TO_C5 B2
2. F1_C1_TO_C5 B5
3. R1_C5_TO_C1 B2
4. R1_C5_TO_C1 B5
5. R2_C45_TO_C1 B2
6. R2_C45_TO_C1 B5

Each run must recover round-25 history, final aggregated and adapted checkpoints, 25 client-stat files, DA diagnostics, run configuration, logs, and provenance hashes before the next run begins. Seeds 43-46 are a second confirmation stage and are not mixed with the seed-42 screening interpretation.

## Metrics And Paired Analysis

Primary metrics are test accuracy and macro-F1. Secondary metrics are per-class recall, worst-class recall, confusion matrix, NLL, ECE, mean confidence, wall time, server adaptation time, client time, communication payload statistics, Pi temperature, and throttling state.

For every direction, B2 and B5 use the same target rows. Persist per-window probabilities and predictions, then compute:

- B2 minus B5 accuracy and macro-F1 in percentage points;
- paired correctness counts and an exact McNemar test;
- bootstrap confidence intervals for metric differences using fixed resampling seeds;
- mean and sample standard deviation across training seeds after confirmation.

## Decision Rule

B2 is treated as practically non-inferior on a direction when both accuracy and macro-F1 are no more than 0.5 percentage points below B5 and worst-class recall does not show a material collapse. The simplified-method paper claim requires that B5 show no consistent improvement greater than 0.5 percentage points across directions and confirmation seeds.

Seed 42 is screening evidence only because B2 was selected after opening the original C1/C2-to-C5 ranking. A final method claim requires paired seeds 43-46. If a direction contradicts the screening hypothesis, report it and use confirmation seeds to distinguish instability from a real direction-specific advantage.

## Artifact Contract

- Commands: `results/iotj_b2_b5_cross_direction_20260713_commands`
- Training: `results/iotj_b2_b5_cross_direction_20260713`
- Local client logs: `results/iotj_b2_b5_cross_direction_20260713_local_logs`
- Evaluation: `results/iotj_b2_b5_cross_direction_20260713_summary`
- Runtime configuration: `configs/iotj_b2_b5_cross_direction_20260713.json`

Every manifest records direction ID, source/target clients, data-root hashes, training seed, B2/B5 loss weights, physical executor for each source client, exact commands, code revision, and output directory. Test labels are used only after training for evaluation.
