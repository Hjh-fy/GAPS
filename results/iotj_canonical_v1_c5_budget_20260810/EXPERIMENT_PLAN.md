# Experiment plan: canonical-v1 C5 low-label commissioning

## Research brief and scope

- Question: whether GAPS/A4 is more label-efficient than equal-label A0T on C5 as the nominal calibration budget falls from the reused 20% row to 15%, 10%, and 5%.
- Source clients: C1, C2.
- Target client: C5.
- Resource budget: six fresh seed42 25-round Flower runs; no other runs.

## Hypothesis

| ID | Falsifiable hypothesis | Baseline | Intervention | Primary metric | Expected evidence | Acceptance criterion |
|---|---|---|---|---|---|---|
| H-C5-LB-01 | A4 degrades more slowly than A0T at constrained C5 label budgets | A0T at the same nested budget | Frozen A4 at the same nested budget | C5 Macro-F1 | A4−A0T across 20/15/10/5% | At 10% or 5%, gap ≥1 percentage point; otherwise the label-efficiency claim is not supported |

## Fixed protocol

- Dataset/preprocessing: canonical-v1 / HZ5_MEAN_W10S / 50×8.
- Split: frozen C5 calibration pool and unchanged C5 test.
- Budgets: 240/160/80 windows, 6/4/2 per each of 40 strata.
- Training: fresh seed42, 25 rounds, LE1, batch32, Adam 5e-4, 100 target-adaptation steps per round.
- Endpoint: fixed round25 only; no target-test selection or search.

## Risks and stopping rules

The single nested family is not multi-seed stability evidence. Results are canonical window-level sensitivity only and cannot override strict C5 collapse. Stop after six endpoints and one unified evaluation.
