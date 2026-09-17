# Ablation Plan

| Hypothesis ID | Factor | Levels | Held constants | Primary metric | Expected Evidence | Confound check | Stopping rule |
|---|---|---|---|---|---|---|---|
| H-CLS-01 | Classification method within information regime | FedAvg; FedProx; SCAFFOLD; CORAL; MMD; DANN; A0T; GAPS/A4 | Canonical data; seed42; fixed endpoint; per-method registered target access | Accuracy; Macro-F1 | Grouped C3/C4/C5/pooled table | Hash/schema match and explicit target-information column | Exclude any row that cannot be reconciled; never substitute legacy evidence |
| H-R84-01 | Frozen source-prior feature | R83 absent; R84 present | Same budget rows; C5 test; A4 routing; 83D order; folds; alpha grid/rule; metrics | delta NRMSE_range; delta RMSE | Paired effect and 95% grouped-bootstrap CI at each budget | R84 differs by exactly one immutable R0-v2 prediction column | Stop budget if identities, source lock, fold rule, or release order differ |
| H-R84-01 | Calibration budget | 5%; 10%; 15%; 20% | Nested frozen manifest; paired R83/R84; same test and source prior | R83/R84 NRMSE curves; delta NRMSE | Four complete budget pairs without a monotonicity assumption | Verify 5 subset 10 subset 15 subset 20 and zero test overlap | Stop all reduced-budget fits if nesting or manifest hash fails |
| H-R84-02 | Gas | Ethanol; CO; Ethylene; Methane | Same predictions and budget pair | Per-gas delta NRMSE; delta RMSE | Heterogeneity table | Report every gas; no post-hoc gas omission | Stop only if gas identity/range mapping conflicts with canonical R1 |
| H-QC-01 | Retained coverage | 70%; 75%; 80%; 85%; 90%; 95%; 100% | Frozen Q1 predictions; confidence score; metric code | Actual coverage; accepted NRMSE_range | Risk--coverage curve and AURC context | Curve points are diagnostic; HC90/HC95 remain separately locked | Stop if values do not reproduce authoritative rows |
| H-QC-02 | Retention policy | Confidence ranking; optional same-count random | Same canonical population and retained counts | NRMSE difference; random interval | Canonical random-reference band | No legacy values; deterministic seeds; no threshold refit | Mark skipped if identity-equivalent canonical random sampling cannot be established |

## Required baselines

- Classification: canonical FedAvg is the source-only FL anchor; CORAL/MMD/DANN are x-only DA references; A0T is the label-aware target-CE reference; GAPS/A4 is the proposed method. M0 is supplementary only.
- Regression: R83 is the target-only control at every budget. The 20% R83/R84 pair is an immutable reference and is not rerun.
- QC: 100% retained is the no-QC endpoint. HC90/HC95 are calibration-locked operating points. Same-count random retention is optional canonical post-processing, not a replacement for the no-QC baseline.

## Resource budget and execution order

1. Consolidate and audit existing canonical classification rows; no training.
2. Register source locks, manifest hashes, six reduced-budget configurations, commands, and output paths.
3. Run B15 R83 then R84, B10 R83 then R84, and B05 R83 then R84, with pair-level test release only after both locks.
4. Compute paired overall/per-gas effects and conditional grouped-bootstrap intervals.
5. Extract QC curve points and, only if identity audit passes, compute the optional canonical random band.
6. Produce figures, final audit, claim mapping, and the experiment-freeze report.

No automatic rerun, alpha adjustment, subset replacement, prior replacement, algorithm addition, target expansion, or multi-seed matrix is allowed in response to observed results.

## Unknown or conflicting protocol fields

- The exact implementation runner, resolved configs, command lines, output hashes, and execution commit are `unknown` until the registry/implementation phase; formal execution is blocked until these are fixed.
- Earlier 100%/50%/25% planning text conflicts with the later frozen 20%/15%/10%/5% manifest. For Experiment B, this design records the user-approved later grid as a scoped amendment; the earlier document remains historical and must not be silently edited.
- CORAL and DANN exist as canonical fixed endpoints but are not both present in the current consolidated comparison CSV. Their inclusion is conditional on exact provenance/schema audit, not on their metric values.
- A paired test bootstrap cannot estimate variation from choosing a different calibration subset. Any CI must be labeled conditional on the single frozen nested subset and seed42.
