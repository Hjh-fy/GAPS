# Experiment Audit

## Audit scope and intended claim

Pre-execution audit of the planned seed42 classification comparison. The intended future claim is an algorithm-level comparison of standard FL baselines, canonical x-only UDA references, and Full GAPS under the registered source/target split and fixed endpoints. This document approves only execution readiness, not result Evidence.

## Compared experiments

| Experiment ID | Split | Model | Checkpoint | DA | Calibration | QC | Seeds | Provenance |
|---|---|---|---|---|---|---|---|---|
| FCL-E1-* | timeaware fullgrid | classification TCN | round25 | none | none | none | 42 | approved design; P0A manifest or new run manifest |
| FCL-E2-* | same | same | exact P0A SHA-256 | canonical global CORAL/MMD/DANN | target x only | none | 42 | approved E2 amendment |
| FCL-E3-* | same | same | round25 | locked Full GAPS | registered target calibration | none | 42 | approved design |
| FCL-E4-* | same | same | round25 | registered per A0-A6 | registered per A0-A6 | none | 42 | approved hierarchical plan |

## Findings

| Finding ID | Severity | Check | Evidence | Impact | Required action | Status |
|---|---|---|---|---|---|---|
| PRE-01 | blocking | P0A physical input absent from new worktree | source file exists in old audited worktree; SHA-256 `4313c...751c` | E1 reuse/E2 cannot run until imported | copy read-only input and verify byte hash plus tensor schema | open until implementation preflight |
| PRE-02 | informational | historical FedProx comparability | historical formal run used LE5 | cannot reuse | run new LE1 with frozen mu0.01 | resolved by plan |
| PRE-03 | informational | historical GAPS comparability | historical run used selective warmup3 | cannot reuse | run new warmup5 | resolved by plan |
| PRE-04 | major | optimizer parity | canonical SCAFFOLD uses SGD; other main algorithms use Adam | comparison is not optimizer-controlled | add optimizer fields and mandatory analysis limitation | resolved by plan; verify output |
| PRE-05 | blocking | E2 target label isolation | code not yet implemented/tested | leakage would invalidate E2 | API-level x-only contract and runtime audit | open until tests pass |
| PRE-06 | blocking | sealed target test | execution runner not yet implemented | selection leakage risk | staged unlock only after fixed endpoints and audit log | open until tests and run audit pass |
| PRE-07 | major | seed coverage | seed42 only by explicit user decision | no stability inference | label all Evidence descriptive seed42 | resolved by plan; verify prose |

## Leakage assessment

The planned protocol prohibits target-test use in training, stopping, model selection, threshold selection, and hyperparameter selection. E2 additionally prohibits passing target class or phase labels into adaptation APIs. Full GAPS may use registered target calibration labels/phase only for its declared conditional server-DA mechanisms. Runtime enforcement remains blocking until implemented and tested.

## Baseline, completeness, and reproducibility assessment

Baseline coverage is complete at plan level: FedAvg, FedProx, canonical SCAFFOLD, canonical CORAL/MMD/DANN, and Full GAPS. Fixed endpoints, optimizer differences, reuse rules, dataset, seed and target budgets are explicit. Reproducibility remains pending implementation manifests, checkpoint hashes, commands, logs and completed artifacts.

## Verdict: blocked

Execution Evidence remains blocked until PRE-01, PRE-05 and PRE-06 pass. The design itself is ready for user review and implementation planning.

## Unknowns and handoff

- No unknown hyperparameter remains.
- Handoff from `experiment-planner` to `experiment-registry`: the plan, matrix and ablation plan in this directory are the inputs; existing P0 assets remain read-only.
- Handoff requested next action: user reviews this written specification; after approval, create the implementation plan and implement preflight/tests before any formal training.
