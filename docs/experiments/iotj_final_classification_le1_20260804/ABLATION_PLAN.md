# Ablation Plan

| Hypothesis ID | Factor | Levels | Held constants | Primary metric | Expected Evidence | Confound check | Stopping rule |
|---|---|---|---|---|---|---|---|
| FCL-H2 | FL algorithm | FedAvg; FedProx(mu=0.01); canonical SCAFFOLD | C1+C2, model, rounds25, LE1, batch32, seed42, endpoint | mean C3/C4/C5 macro-F1 | algorithm-level baseline comparison | optimizer difference explicitly disclosed; SCAFFOLD is not optimizer-controlled | one fixed run per new method |
| FCL-H3 | canonical UDA objective | none; global CORAL; global MMD²; unconditional DANN | exact FedAvg checkpoint, target x-only, calibration feature set, source batches, Adam lr5e-4, 100 steps, coefficient0.5 | target macro-F1 | checkpoint-controlled UDA reference | runtime label audit; no conditional GAPS losses | 9 fixed branches, no tuning |
| FCL-H5 | cumulative GAPS components | A0-A6 | C5, rounds25, LE1, batch32, Adam lr5e-4, seed42, round25 | C5 macro-F1 | hierarchical marginal deltas | reuse A0/A6 only after exact config audit | finish A1-A5 once; no factorial expansion |

## Required baselines

- E1 FedAvg round25 from P0A, FedProx mu0.01, canonical SCAFFOLD SGD.
- E2 FedAvg checkpoint without adaptation plus canonical CORAL/MMD/DANN.
- E3 Full GAPS for C3/C4/C5.
- E4 A0-A6, with A0 reused from E1 and A6 reused from E3 C5.

## Resource budget and execution order

1. Preflight and immutable input import.
2. E0 diagnostics.
3. E1 FedProx and SCAFFOLD; FedAvg evaluation from reuse.
4. E2 nine lightweight branches.
5. E3 Full GAPS C3/C4/C5.
6. E4 A1-A5 C5.
7. Unified evaluation, figures, analysis and audit.

Budget: 10 new full 25-round FL runs and 9 new 100-step adaptations, seed42 only.

## Unknown or conflicting protocol fields

- No unresolved hyperparameter field: SCAFFOLD lr5e-4 and E2 coefficient0.5 are preregistered before formal execution.
- Excluded conflicts: historical FedProx LE5; historical GAPS selective warmup3.
- Source checkpoint physical location differs across worktrees; semantic identity is fixed by SHA-256 and tensor audit.
