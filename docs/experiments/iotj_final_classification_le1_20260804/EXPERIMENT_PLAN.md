# Experiment Plan

## Research brief and scope

- Brief source: user-approved GAPS IoT-J final classification closure, including the canonical SCAFFOLD and canonical unsupervised E2 amendments.
- Target venue/audience: IEEE Internet of Things Journal reviewers.
- Resource budget: seed 42 only; 10 new 25-round federated runs; 9 fixed 100-step adaptation branches; no search or multi-seed extension.

## Hypotheses

| ID | Falsifiable hypothesis | Baseline | Intervention | Primary metric | Expected Evidence | Acceptance criterion |
|---|---|---|---|---|---|---|
| FCL-H1 | C1+C2 and each target exhibit nonzero input/embedding distribution discrepancy under the frozen preprocessing and FedAvg encoder. | within-source C1 versus C2 | C1+C2 versus C3/C4/C5 | global MMD²; CORAL distance | E0 discrepancy tables and Fig. 1 | supported only if target discrepancies are reproducibly nonzero under the preregistered estimators; otherwise falsified |
| FCL-H2 | At least one canonical FL baseline changes mean C3/C4/C5 macro-F1 relative to FedAvg under the frozen LE1 protocol. | FedAvg | FedProx; SCAFFOLD | mean target macro-F1 | E1 main comparison with source retention and cost | direction and magnitude reported; equality/no material change falsifies the performance-difference hypothesis |
| FCL-H3 | Canonical x-only CORAL, MMD, or DANN improves mean target macro-F1 over its shared FedAvg source checkpoint. | FedAvg round25 | E2 CORAL/MMD/DANN | mean target macro-F1 | paired checkpoint-controlled E2 comparison | supported only for methods whose fixed-endpoint mean is higher; otherwise falsified; no causal component claim |
| FCL-H4 | Full GAPS improves mean C3/C4/C5 macro-F1 over the best standard baseline under the registered protocol. | best of E1/E2 | Full GAPS | mean target macro-F1 | main table plus target slices, source retention and costs | supported only if fixed round25 GAPS mean is higher; otherwise falsified |
| FCL-H5 | The cumulative C5 hierarchy identifies measurable marginal changes from semantic, replay, selective aggregation and server DA. | preceding registered ablation row | A1-A6 cumulative additions | C5 macro-F1 | E4 table and Fig. 6 | each marginal delta is reported regardless of sign; zero/opposite deltas falsify monotonic-contribution expectations |
| FCL-H6 | GAPS performance gains co-occur with lower registered source-target embedding discrepancy. | FedAvg | Full GAPS | macro-F1 gain and discrepancy delta | discrepancy-performance table and Figs. 3/9 | descriptive association only; inconsistent direction falsifies the co-occurrence hypothesis |

## Fixed protocol

- Source clients: C1;C2.
- Target clients: C3;C4;C5.
- Split protocol: time-aware `60_170_window_fullgrid`; source train for FL; source calibration for E2 source batches; target calibration for registered adaptation; target test sealed until fixed endpoint; target train excluded.
- Model/checkpoint policy: common classification TCN; FedAvg/E2 reuse P0A round25; ordered state-content fingerprint defines equality and whole-file SHA-256 `4313c375a8fa2e929de9d65637a2196f6c0f0752c2dc78112020b8727351751c` is provenance only; all new FL results use round25.
- Seeds: 42 only.
- DA / calibration / QC controls: E1 has no DA; E2 is canonical x-only unconditional DA; E3/E4 server DA follows the locked GAPS configuration; no QC; no post-hoc calibration or threshold tuning.
- Optimizers: FedAvg/FedProx/GAPS/E2 Adam lr 5e-4; SCAFFOLD canonical SGD lr 5e-4, no momentum/scheduler.
- SCAFFOLD SGD5e-4 is not asserted equivalent to Adam5e-4. A discarded C1/C2-only numerical validity run must pass fixed CE/discrimination/finite/norm gates or SCAFFOLD fails closed without lr search.
- Common training: 25 rounds, LE1, batch32; formal endpoint round25.
- E2: 100 steps, batch32, coefficient 0.5, fixed endpoint.
- Target information: E2 calibration x only; Full GAPS/A4-A6 calibration x/class/phase; concentration unused; A0-A3 use no target calibration during training. Target test leakage outside final evaluation is a hard failure.
- Selective boundary: rounds1-5 FedAvg; round6 onward selective when registered semantic inputs exist.
- E0 includes per-channel sensor-space shift statistics. E1 includes per-target source-target macro-F1 gaps.
- A0-A6 must emit `ablation_loss_activity.csv` with configured/available/observed activity fields.

## Risks, unknowns, conflicts, and stopping rules

- Limitation: seed42 only; no stability claim.
- P0A checkpoint is an external read-only input to the new worktree. It must be copied and hash-verified before use.
- Historical FedProx LE5 and historical GAPS warmup3 conflict with the current protocol and are excluded.
- E2 label-access, source/target split, checkpoint equality or endpoint audit failure is blocking.
- No hyperparameter change is permitted after target-test access.
- Stop after the registered matrix is complete and audited; do not launch optimization extensions.
