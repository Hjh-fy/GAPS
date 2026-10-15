# Experiment Audit

## Audit scope and intended claim

检查四组已训练 B5 checkpoint 的逐轮 C5 指标能否支持“不同 Server-DA
预算具有不同收敛轨迹”的回顾性工程诊断。

## Compared experiments

| Experiment ID | Split | Model | Checkpoint scope | DA | Calibration | QC | Seeds | Provenance |
|---|---|---|---|---:|---|---|---:|---|
| B5-LE1-DA100-RC | historical C5 test | B5 | adapted R1–R25 | 100 | frozen B5 server adaptation | off | 42 | canonical reference |
| B5-LE1-DA80-RC | same | B5 | adapted R1–R25 | 80 | same | off | 42 | canonical postflight |
| B5-LE1-DA50-RC | same | B5 | adapted R1–R25 | 50 | same | off | 42 | canonical postflight |
| B5-LE1-DA30-RC | same | B5 | adapted R1–R25 | 30 | same | off | 42 | observability contract blocked |

## Findings

| Finding ID | Severity | Check | Evidence | Impact | Required action | Status |
|---|---|---|---|---|---|---|
| RC01 | informational | checkpoint completeness | each group has exactly R1–R25 adapted checkpoints | full trajectories available | none | closed |
| RC02 | informational | metric completeness | 100 rows, each N=1360, finite metrics | no missing round | none | closed |
| RC03 | informational | round-25 parity | all four round-25 Accuracy values match the prior budget summary within floating-point representation | links curve to frozen terminal evaluation | none | closed |
| RC04 | major | repeated test access | historical C5 test evaluated at every round | invalid for early stopping or checkpoint selection | retain retrospective-only boundary | open |
| RC05 | major | seed coverage | only seed 42 | cannot establish convergence stability across seeds | do not make statistical/stability claim | open |
| RC06 | blocking for DA30 formal evidence | observability validator | C2 coverage 0.948214 < 0.95 in the training attempt | DA30 trajectory remains technical only | do not promote | open |

## Leakage assessment

No per-round target metric was available to, or used by, the already completed
training process. The current calculation therefore did not alter training.
However, because the historical test is inspected across 25 checkpoints, these
metrics cannot be used retrospectively to select a round without creating
test-based checkpoint selection.

## Baseline, completeness, and reproducibility assessment

The comparison holds seed, source/target roles, model, data split, local epochs
and rounds fixed. Round-25 parity is exact to floating-point representation.
Checkpoint SHA256 values are recorded for every row.

## Verdict: blocked

Approved only as a transparently labeled retrospective trajectory diagnostic.
Blocked for checkpoint selection, early stopping, formal budget promotion and
DA30 formal evidence.

## Unknowns and handoff

Cross-seed convergence variability is unknown. No new training is implied by
this audit.
