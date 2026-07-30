# Laboratory Three-Gas All-Concentration Time-Purged Ablation Plan

| Hypothesis ID | Factor | Levels | Held constants | Primary metric | Expected Evidence | Confound check | Stopping rule |
|---|---|---|---|---|---|---|---|
| `H-LAB-ID-01` | concentration coverage | held-out lowest concentration fold 1; all-concentration time-purged | P2 source, P3 target, model, channels, normalization clients, seed, rounds, local epochs, DA | P3 adapted window Accuracy | descriptive performance gap | label the split and independence change; no causal model-regression wording | stop after one audited all-concentration run |
| `H-LAB-ID-02` | window split safety | random overlapping split (prohibited); fixed time-purged split | exposure inventory and preprocessing | raw-sample overlap count | validator evidence of zero overlap | verify using window start/end intervals, not window IDs only | fail closed on any overlap |

## Required baselines

- Audited P2-to-P3 source-only-normalized fold-1 result:
  adapted window Accuracy 0.913043 and Macro-F1 0.911300.
- Unadapted checkpoint from the same new all-concentration run.

## Resource budget and execution order

1. Build and validate the new dataset locally.
2. Freeze a new immutable dataset/source/protocol identity.
3. Wait for the existing folds 2--5 queue to finish and verify clean processes.
4. Deploy to cloud A and cloud B.
5. Run one 25-round P2-to-P3 experiment and postflight audit.
6. Report both all-concentration and concentration-held-out results with
   different task labels.

## Unknown or conflicting protocol fields

- Exact reviewed gas boundaries: unknown.
- Multi-seed variance: unknown.
- Fully independent calibration and test exposures at every exact
  concentration: unavailable in the present acquisition because there is only
  one physical exposure for most exact platform/gas/concentration tuples.
