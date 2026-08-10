# Ablation plan

| Hypothesis ID | Factor | Levels | Held constants | Primary metric | Expected evidence | Confound check | Stopping rule |
|---|---|---|---|---|---|---|---|
| H-C5-LB-01 | C5 calibration quantity | 20%, 15%, 10%, 5% | canonical preprocessing/test; seed42; 25 rounds; LE1; optimizer/LR; adaptation steps; method-specific loss surface | C5 Macro-F1 | per-method degradation and same-budget A4−A0T | all budgets retain 40/40 strata; verify nesting and source retention | stop after six new endpoints and unified evaluation |

## Required baselines

Existing formal C5 20% A0T and GAPS/A4 rows are reused without training. Canonical FedAvg round25 is used only as the source-retention reference.
