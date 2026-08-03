# P0-U Zero-Label Commissioning Result Analysis

## Input contract and provenance

- Seed42 only; all methods originate from the same P0A round25 checkpoint (`4313c375…1751c`).
- Formal scope is the C5 sealed test with 1,360 windows, opened only after both U1/U2 100-step branches completed.
- Source-only and Simple Target-CE values are copied/reported from the audited P0 round25 table. U1/U2 values are reported from the new audited comparison CSV. Percentage-point differences and time ratios below are recomputed.
- One seed does not support SD, confidence intervals, p-values, or statistical effect sizes. C5 windows are not treated as independent experimental replicates.

## Unified formal comparison

| Method | Target-label access | Accuracy | Macro-F1 | NLL | ECE | Commissioning |
|---|---|---:|---:|---:|---:|---:|
| Source-only | none | 32.500% | 23.521% | 4.8785 | 0.6634 | — |
| U1 Unsupervised global alignment | x only | 60.735% | **59.213%** | 2.3375 | 0.3174 | 8.89 s |
| U2 Pseudo-label self-training | x only | 33.235% | 24.156% | 4.9467 | 0.6596 | 3.52 s |
| Simple Target-CE reference | 320 true calibration labels | **97.647%** | **97.653%** | **0.1046** | **0.0076** | 3.41 s |

U1 improves over Source-only by 35.692 Macro-F1 percentage points and 28.235 Accuracy points, while remaining 38.440 Macro-F1 points below supervised Target-CE. U1 takes about 2.61 times the commissioning time of Target-CE in this local CPU measurement.

U2 improves Macro-F1 over Source-only by only 0.635 points. Its NLL is 0.0682 worse than Source-only, so the small accuracy/F1 movement is not accompanied by better probabilistic fit.

## Pseudo-label failure diagnostic

- At the fixed 0.90 threshold, post-hoc coverage is 96.5625% (309/320), so U2 did not fail because too few samples were selected.
- Post-hoc pseudo-label precision is only 35.599%, despite a mean selected confidence of 99.600%.
- Selected class counts are `[13, 2, 110, 184]`, showing severe collapse toward classes 2 and 3 and near-elimination of class 1.
- The teacher is therefore highly confident but poorly calibrated under the C5 shift. Training on these labels mostly reinforces the frozen source router's errors.

Calibration truth was used only for this one post-hoc diagnosis after training. The precision result did not affect the threshold, training, checkpoint, or final test procedure.

## U1 activity

All 100 U1 steps use source CE plus unconditional CORAL, global MMD² and Wasserstein-min adversarial alignment. Mean raw source CE is 0.03264, mean global MMD² is 0.08018, and mean adversarial feature loss is 0.11226. Target CE, conditional CORAL, class MMD, same-class-phase MMD, target prototype anchor, semantic label matching, and pseudo labels are unavailable or disabled by construction.

## Interpretation

U1 demonstrates that zero-label C5 features contain useful global shift information and materially recover direct-transfer performance. However, it does not approach the supervised commissioning result. U2 demonstrates that naive high-confidence self-training is unsafe in this setting: the source router is confidently wrong on C5, and threshold 0.90 does not filter those errors.

The defensible seed42 conclusion is: unconditional x-only alignment is the stronger zero-label option, but supervised Target-CE remains decisively better when calibration labels are available. Per the frozen task boundary, no new threshold, teacher, entropy, clustering, or consistency optimization should be launched automatically.
