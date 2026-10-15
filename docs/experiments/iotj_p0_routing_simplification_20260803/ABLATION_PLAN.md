# P0 Routing Simplification Comparison Plan

| Hypothesis | Factor | Levels | Held constants | Primary metric | Expected evidence | Confound check | Stopping rule |
|---|---|---|---|---|---|---|---|
| P0-H1 | source FL round | 1–25 | seed42, CE-only, LE1, TCN, C1/C2 | C5 Macro-F1 | retrospective convergence curve | test never selects round | stop on missing checkpoint/hash |
| P0-H2 | commissioning method | none; simple CE; full adapter | identical matching P0-A checkpoint, steps100, LR5e-4 | round25 C5 Macro-F1 | fixed formal comparison | target CE label access declared; inactive DA terms exposed | stop on config/split mismatch |
| P0-H3 | commissioning timing | each frozen round post hoc; fixed round25 formal | no adapted inheritance | C5 Macro-F1 | diagnostic curve plus round25 row | curves excluded from selection | stop if any branch does not reload source checkpoint |

## Required baselines

The comparison includes only `source_only`, `simple_target_ce`, and `full_target_adapter`. Historical B5 is deliberately excluded from the P0 round-wise table.

## Resource budget and order

One seed, one 25-round three-host source run, followed by 25 independent evaluations for each method. No additional algorithm, calibration-budget, regression, QC, or cross-direction branch is authorized.

## Unknowns or conflicts

No protocol field is unknown. Runtime and measured performance remain unknown until execution. Single-seed uncertainty remains an explicit limitation.
