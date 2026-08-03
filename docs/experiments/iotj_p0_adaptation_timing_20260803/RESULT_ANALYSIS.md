# P0-I Result Analysis

## Input contract and provenance

- Experiments: P0-I2 post-hoc UDA2500 and P0-I3 interleaved UDA25×100, with audited read-only I0/I1/S1 references.
- Seed set: `{42}` only. C5 test scope is 1,360 sealed windows; C1/C2 source retention is computed separately.
- All values below are recomputed post-hoc from fixed checkpoints after both training runs completed. No checkpoint selection was performed.
- Because there is only one seed and one target client, no standard deviation, confidence interval, significance test, or population-level effect size is claimed. Window-level predictions are not treated as independent client replicates.

## Unified endpoint results

| Method | Target labels | UDA steps | Accuracy | Macro-F1 | NLL | ECE | Source mean F1 | UDA seconds |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| I0 Source-only | none | 0 | 0.3250 | 0.2352 | 4.8785 | 0.6634 | 0.9926 | 0.00 |
| I1 Post-hoc UDA100 | none, x-only | 100 | 0.6074 | 0.5921 | 2.3375 | 0.3174 | 0.9927 | 8.89 |
| I2 Post-hoc UDA2500 | none, x-only | 2500 | 0.4809 | 0.4901 | 3.1907 | 0.4660 | 0.9904 | 847.89 |
| I3 Interleaved UDA25×100 | none, x-only | 2500 | 0.5191 | 0.4927 | 3.0768 | 0.4042 | 0.9934 | 702.39 |
| S1 Target-CE100 | C5 calibration labels | 100 | 0.9765 | 0.9765 | 0.1046 | 0.0076 | 0.6565 | 3.41 |

## Required questions

### Q1 — Do more post-hoc UDA steps help?

No at the fixed formal endpoint. I2 step2500 is 0.1020 lower in Macro-F1 than I1 step100 (0.4901 versus 0.5921), with worse NLL and ECE. The diagnostic trajectory peaks at step500 (Macro-F1 0.7852) and then degrades, but step500 is explicitly non-selective post-hoc evidence and must not replace the registered step2500 endpoint. This is evidence of long-horizon instability, not a basis for retrospective model selection.

### Q2 — Does adaptation timing matter at the same 2500-step budget?

Only marginally at the registered terminal checkpoint. I3 round25 POST exceeds I2 step2500 by 0.00263 Macro-F1 (0.263 percentage points) and 0.0382 Accuracy, while improving NLL by 0.1139 and ECE by 0.0618. Under seed42-only evidence, this is not a meaningful terminal Macro-F1 separation. The primary conclusion is that interleaving did not rescue the 2500-step endpoint.

### Q3 — Is interleaved adaptation persistently carried into later FL rounds?

Yes in the limited mechanistic sense that the next-round PRE model differs favorably from same-round pure FedAvg: PRE Macro-F1 is higher in 24 of 25 rounds, with an average descriptive difference of +0.2263. This is backed by exact lineage: every round 2–25 client initialization fingerprint equals the preceding POST state fingerprint.

However, the immediate 100-step UDA update is unstable: POST exceeds PRE in only 12 of 25 rounds, and the mean POST-minus-PRE difference is −0.00781. At rounds 5, 20, and 25, UDA reduces Macro-F1. Therefore the evidence supports trajectory influence/persistence of earlier updates, but not reliable repeated repair or persistent representation preservation by each current-round UDA operation.

## Selected round trajectory

| Round | Pure target F1 | I3 PRE F1 | I3 POST F1 |
|---:|---:|---:|---:|
| 1 | 0.5720 | 0.5720 | 0.8016 |
| 5 | 0.2951 | 0.4939 | 0.4108 |
| 10 | 0.2760 | 0.4582 | 0.6180 |
| 15 | 0.2836 | 0.4315 | 0.5187 |
| 20 | 0.2916 | 0.5714 | 0.3670 |
| 25 | 0.2352 | 0.5104 | 0.4927 |

## Domain discrepancy and source retention

At round25, global MMD2 is 0.16675 for pure FedAvg, 0.01639 for I3 PRE, and 0.03250 for I3 POST; CORAL discrepancy is respectively 1.1063e-5, 1.2556e-6, and 2.1296e-6. Interleaving therefore strongly suppresses feature-distribution discrepancy, including before the current round's UDA. Yet lower discrepancy does not translate monotonically into better target F1: round25 POST has higher MMD2/CORAL and lower F1 than PRE.

Round25 source mean Macro-F1 remains high: 0.9926 pure, 0.9941 I3 PRE, and 0.9934 I3 POST. The poor target endpoint is therefore not explained by catastrophic loss of source discrimination. The round25 source-target gaps are 0.7574 pure, 0.4837 PRE, and 0.5007 POST.

## Anomalies and sensitivity notes

- I2 has a non-monotonic target trajectory and a diagnostic maximum at step500. This is retained as an anomaly; no observation or checkpoint is deleted.
- I3 UDA can increase discrepancy and reduce target F1 in the same round (notably rounds 5, 20, and 25), so distribution alignment alone is insufficient as a success proxy.
- Source retention is near ceiling across unsupervised methods, while S1 trades source retention for very high target performance. S1 is not label-fair and remains a supervised upper-reference path.
- Seed42-only results cannot establish statistical generality.

## Proposed paper use

Use the unified endpoint table as a mechanism-study table, not a performance claim. Use the target-F1, domain-discrepancy, source-target-gap, and post-hoc trajectory figures to show: (i) excessive UDA duration is unstable; (ii) interleaving changes the federated trajectory; and (iii) timing alone does not solve the terminal transfer gap.

## Audit handoff

Label-access audit: PASS. Interleaved lineage audit: PASS. Strict experiment audit: PASS. Comparability is restricted to the frozen seed42 protocol; no multi-seed inference is approved.
