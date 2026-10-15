# P0-U Comparison Plan

| Hypothesis | Factor | Levels | Held constants | Primary metric | Confound check | Stopping rule |
|---|---|---|---|---|---|---|
| P0U-H1 | target global alignment | off; U1 | checkpoint, steps, LR, seed | C5 Macro-F1 | target loader must be x-only | fail on any target label object |
| P0U-H2 | pseudo self-training | off; U2 frozen teacher threshold0.90 | checkpoint, steps, LR, seed | C5 Macro-F1 | pseudo labels only from logits | fail on threshold/config search |
| P0U-H3 | target supervision | zero-label U1/U2; supervised Target-CE reference | same P0A round25 origin | C5 Macro-F1 | declare label access | no causal/statistical claim from one seed |

No additional method, threshold, teacher schedule, seed, calibration budget, or hyperparameter branch is authorized.
