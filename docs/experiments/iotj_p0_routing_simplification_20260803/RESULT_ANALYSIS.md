# P0 Routing Simplification Result Analysis

## Input contract and provenance

- Experiments: `P0A-PURE-FEDAVG-LE1-S42`, `P0B-SIMPLE-TARGET-CE-S42`, and `P0B-FULL-TARGET-ADAPTER-S42`.
- Scope: seed42; C5 sealed test, 1,360 windows; fixed round25 formal comparison.
- Values below are reported from the generated CSV. Percentage-point differences and time ratios are recomputed here.
- The 25 round-wise values are correlated retrospective checkpoints from one training trajectory, not independent seeds. No SD, confidence interval, p-value, or statistical effect size is reported.

## Fixed round25 formal comparison

| Method | Accuracy | Macro-F1 | NLL | ECE | Commissioning time |
|---|---:|---:|---:|---:|---:|
| Source-only | 32.500% | 23.521% | 4.8785 | 0.6634 | 0.33 s evaluation |
| Simple Target-CE | **97.647%** | **97.653%** | **0.1046** | **0.0076** | **3.41 s** |
| Full target adapter | 89.779% | 89.959% | 0.4834 | 0.0733 | 79.16 s |

At the predeclared round25 checkpoint, simple Target-CE exceeds the full adapter by 7.694 Macro-F1 percentage points and 7.868 Accuracy points. Its measured commissioning time is about 23.22 times lower. It exceeds source-only by 74.132 Macro-F1 points.

## Round-wise diagnostic findings

- Source-only Macro-F1 is 57.197% at round1, reaches a descriptive maximum of 58.946% at round2, and falls to 23.521% at round25. This is evidence of source-domain optimization failing to preserve direct C5 transfer, but round2 cannot be selected because the curve is target-test post hoc.
- Simple Target-CE is robust across the frozen source trajectory: its observed Macro-F1 range is 95.568%–98.678%, with 97.653% at round25. The descriptive maximum occurs at round17 but is not model-selection evidence.
- Full target adaptation ranges from 86.545% to 93.202%, with 89.959% at round25. It is consistently less effective than simple Target-CE for the fixed protocol.
- The P0-A clients nevertheless fit their own source training data well: C1 CE changes from 1.0189 to 0.0309 and C2 from 0.9961 to 0.00944 between rounds 1 and 25. Thus the poor late-round direct C5 result is not explained by failure to optimize source CE.

## Server-loss activity

The active full-adapter terms are CORAL, global MMD, class MMD, stage MMD, and adversarial alignment. Prototype anchor is zero in all steps. Prototype loss, consistency, and residual loss are `ZERO_NO_INPUT_STATISTICS`; prototype MMD and target CE are `ZERO_BY_CONFIGURATION`. Therefore the observed full-adapter performance must not be attributed to inactive prototype or residual mechanisms.

## Interpretation and paper recommendation

For this seed42 P0 protocol, the simplest defensible router is: finish CE-only LE1 FedAvg source training, freeze round25 by protocol, and perform one 100-step full-model Target-CE commissioning pass using C5 calibration. Repeating target adaptation during every FL round is unnecessary for deployment; the round-wise branches are diagnostic counterfactuals, while only round25 is formal.

The paper can state that simple commissioning is the best of the three locked P0 paths at seed42 and is substantially cheaper than full DA. It cannot state multi-seed stability, statistical superiority, or that round17/round2 should be deployed. Historical B5 is not part of this P0 comparison.

`checkpoint_bytes` records actual serialized artifact size, but source and adapted checkpoints contain different metadata payloads. It must not be interpreted as a model-parameter or storage-efficiency comparison without normalizing the serialization schema.
