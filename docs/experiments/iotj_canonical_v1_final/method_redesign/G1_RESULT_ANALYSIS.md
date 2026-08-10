# Gate 1 Result Analysis

## Input contract and provenance

- Experiment IDs: `CAN-V1-MR-G1-SOURCE`, `CAN-V1-MR-G1-A0T-FULL`, `CAN-V1-MR-G1-A4`, `CAN-V1-MR-G1-HEAD`.
- Result path: `results/iotj_canonical_v1_method_redesign_20260811/gate1_posthoc/`.
- Source state fingerprint: `cad6726ec29fb574314a5f2a45ed9800d1d90906b81cbd3ba8f8efb48a0df5d7` for every method.
- Metric direction: higher is better for Accuracy/Macro-F1; lower is better for NLL/ECE, time, trainable parameters, and source-retention loss magnitude.
- Sample scope: C1 test N=680, C2 test N=680, merged source N=1360, C5 test N=1360. One frozen seed (42) is reported; windows are not treated as independent clients or seeds.
- Values below are recomputed by the Gate-1 evaluator from per-window probabilities after all three step-100 endpoints were locked. No confidence interval or significance test is valid across seeds because the frozen scope has one seed.

## Descriptive results

| Method | C1 F1 | C2 F1 | C1+C2 F1 | C5 F1 | C5 Accuracy | C5 NLL | C5 ECE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Source-only | 0.998529 | 1.000000 | 0.999265 | 0.368610 | 0.478676 | 4.082857 | 0.505797 |
| Posthoc A0T-full | 0.613559 | 0.708012 | 0.662252 | 0.976544 | 0.976471 | 0.071292 | 0.010052 |
| Posthoc A4 | 0.998529 | 0.998529 | 0.998529 | 0.894334 | 0.893382 | 0.601079 | 0.089622 |
| Posthoc Target-head | 0.998529 | 1.000000 | 0.999265 | 0.368610 | 0.478676 | 3.742837 | 0.503759 |

## Effect sizes and system trade-offs

- A0T-full improves C5 Macro-F1 by `+0.607935` over zero-shot and is `-0.017595` below historical interleaved A0T.
- A4 improves C5 Macro-F1 by `+0.525725` over zero-shot, but is `-0.082210` below post-hoc A0T-full and `-0.099792` below historical interleaved A4.
- Target-head changes no C5 class decision and therefore has `0.000000` Macro-F1 gain. It does reduce C5 NLL by `0.340021`, which is insufficient for the commissioned classification endpoint.
- A0T-full uses 22,765 trainable parameters, takes 13.081 s, and changes parameters by relative L2 displacement 0.030134. Its hypothetical sharing retention delta on C1/C2 averages `-0.338479`.
- A4 uses 22,765 trainable parameters, takes 136.460 s, and changes parameters by 0.040368. Its source retention delta averages only `-0.000735`.
- Target-head uses 3,396 trainable parameters (14.92%), takes 0.774 s, and has zero source Macro-F1 loss, but fails the C5 effectiveness condition.

## A4 activity interpretation

Source CE, CORAL, global MMD, class MMD, domain adversarial loss, and cross-domain same-class-phase MMD were active for all 100 steps. Prototype anchor, client-prototype loss, prototype MMD, consistency, and device residual were inactive because a CE-only source endpoint contains no interleaved semantic/client state. Target CE was available but fixed to weight zero by A4. This is a faithful test of the registered A4 post-hoc behavior with unavailable inputs preserved as unavailable, not reconstructed.

## Anomalies and sensitivity

- Target-head changes NLL and parameter state but not predicted classes. The prediction file confirms this is a decision-boundary outcome, not a missing optimizer update.
- A0T-full has the strongest C5 result but severe diagnostic source forgetting. This does not damage deployed C1/C2 because the global source checkpoint is immutable; it does rule out sharing the personalized full checkpoint as a global replacement.
- A4's retention advantage is substantial, but its C5 deficit exceeds the pre-registered 0.5-point equivalence window, so retention cannot rescue it as the core method.
- Historical interleaved comparisons differ in exposure timing and source-update count and are descriptive, not optimizer-controlled single-factor comparisons.

## Decision

- `POSTHOC_LIFECYCLE_SUPPORTED`
- A4: `RETIRE_AS_CORE`
- Target-head: `NOT_COMPETITIVE`
- `INTERLEAVED_DEPENDENCY_RISK=false`, because post-hoc A0T remains within five percentage points of its interleaved reference even though A4 does not.

The supported claim is narrow: one-time supervised commissioning can recover canonical window-level C5 after immutable C1/C2 source FL. The evidence does not support strict experiment-independent generalization or post-hoc A4 superiority.

