# Laboratory Three-Gas All-Concentration Time-Purged Plan

## Research brief and scope

- Brief source: evaluate why laboratory P2-to-P3 fold-1 window Accuracy
  (91.30%) is below the public-dataset 98%--99% range by adding an auxiliary
  in-distribution split whose calibration and evaluation sets both cover every
  retained concentration.
- Target audience: internal laboratory screening; this protocol does not
  replace concentration-group-aware five-fold evaluation.
- Resource budget: one P2-to-P3 run, seed 42, 25 Flower rounds, 3 local epochs,
  fixed strong DA with 100 server steps per round. Run only after the active
  concentration-held-out queue finishes.

## Hypotheses

| ID | Falsifiable hypothesis | Baseline | Intervention | Primary metric | Expected Evidence | Acceptance criterion |
|---|---|---|---|---|---|---|
| `H-LAB-ID-01` | Under the same P2-only normalization and training configuration, a time-purged split covering every concentration in calibration and test has higher P3 adapted window Accuracy than concentration-held-out fold 1 | P2-to-P3 fold-1 adapted Accuracy=0.913043 | all-concentration time-purged split | P3 adapted test window Accuracy | one audited descriptive delta with unchanged model/DA controls | all integrity checks pass and delta is reported regardless of sign |
| `H-LAB-ID-02` | The all-concentration result can be produced without sharing raw time samples across calibration and train/test | invalid random split of overlapping windows (not executed) | fixed calibration indices plus one-window purge on both sides | raw-sample overlap count | machine-readable split audit | overlap count=0 for every client/exposure/split pair |

## Fixed protocol

- Source clients: P2 / logical C2 / cloud B.
- Target client: P3 / cloud A calibration and held-out test.
- Retained exposures: positions 2--6 for all three gases, both v1/v2 sessions;
  30 exposures per platform.
- Base windows: 100 s length, 50 s stride, 23 windows per exposure.
- Within each exposure:
  - calibration window indices: `3,11,19` (zero-based), covering early,
    middle, and late gas response;
  - purge indices: `2,4,10,12,18,20`, because they overlap a calibration
    window in raw time;
  - train/test indices: the remaining 14 windows.
- P2: 14 windows/exposure for train, 3 for source calibration.
- P3: 3 windows/exposure for target calibration, 14 for target test.
- Normalization: mean/std fit only on P2 train windows.
- Model/checkpoint policy: `strong_cls`; select round using P2 calibration
  exposure Macro-F1, then window Macro-F1, then earliest round.
- Seed: 42.
- DA/calibration/QC: `fixed_da_strong`, 100 steps/round, P3
  calibration-assisted, no QC.
- Held constants: channels 1/2/4/6/8/9, 100-point input, three classes,
  whole-exposure phase label, nominal gas boundaries, batch size 32.

## Risks, unknowns, conflicts, and stopping rules

- Calibration and test share the same physical exposure but not the same raw
  time samples. This measures within-exposure/in-distribution performance and
  is not independent-exposure generalization.
- Exact gas boundaries remain unknown; evidence stays
  `preliminary_nominal_boundary_screening`.
- One seed cannot establish stability.
- Any raw-sample overlap, missing concentration, non-finite value, class
  imbalance, wrong client identity, incomplete checkpoint set, or target-test
  use before round selection fails closed.
- Performance does not decide whether to rerun or tune parameters.
