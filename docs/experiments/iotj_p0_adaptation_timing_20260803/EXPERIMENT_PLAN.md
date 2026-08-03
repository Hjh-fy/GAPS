# P0-I Unsupervised Adaptation Timing Study

Status: protocol frozen before formal execution.

## Research question

With seed 42, the same C1/C2 CE-only Flower setting, the same C5 calibration features, the same audited U1 objective, and 2,500 UDA steps, does interleaving 100 UDA steps after every FedAvg round change transfer behavior relative to applying all 2,500 steps after round 25?

## Fixed comparisons

| ID | Schedule | UDA steps | Formal endpoint | Target labels |
|---|---|---:|---|---|
| I0 | Source only | 0 | round 25 | none |
| I1 | Post-hoc UDA | 100 | step 100 | none |
| I2 | Post-hoc UDA | 2,500 | step 2,500 | none |
| I3 | 100 steps after each of 25 rounds | 2,500 | round 25 POST-UDA | none |
| S1 | Post-hoc target CE | 100 | step 100 | C5 calibration labels |

I0, I1, and S1 are read-only audited references. I2 and I3 are the only new training runs. No promotion threshold, checkpoint selection, early stopping, hyperparameter search, component ablation, or additional seed is permitted.

## Decision analysis

The primary contrast is I2 versus I3. I1 versus I2 diagnoses step-budget effects. I3 PRE versus same-round pure FedAvg tests whether prior UDA updates persist into the subsequent federated trajectory; POST-only gains followed by PRE regression are interpreted as repeated repair, not persistent representation preservation.

## Required evidence

Target Accuracy/Macro-F1/NLL/ECE, C1/C2 source retention, label-free feature MMD2 and CORAL discrepancy, source-target Macro-F1 gap, wall time, complete loss trajectories, exact parameter lineage, label-access audit, strict experiment audit, and fixed endpoint declarations.
