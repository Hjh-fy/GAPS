# Pre-run Experiment Audit

## Audit scope and intended claim

Audit Gate A source-diversity sensitivity, Gate B lightweight post-hoc commissioning, and Gate C downstream routing-cost motivation. This audit approves protocol execution only; it does not approve a favorable scientific claim.

## Compared experiments

| Experiment ID | Split | Model | Checkpoint | DA | Calibration | QC | Seeds | Provenance |
|---|---|---|---|---|---|---|---|---|
| CAN-V1-MB-A-S2-FEDAVG | canonical-v1 | canonical classifier | SHA `2d114a8a...fa` | none | none | none | 42 | G2 manifest |
| CAN-V1-MB-A-S2-DGP | canonical-v1 | canonical classifier | SHA `3a19f14e...63` | exact G2 DGP | none | none | 42 | G2 manifest |
| CAN-V1-MB-A-S4-* | derived role view | same | fixed round25, pending | none/exact G2 DGP | none | none | 42 | pre-run lock required |
| CAN-V1-MB-B0/B1/B3 | canonical C5 20/80 | same | immutable G1 hashes | none/CE | N=320 | none | 42 | G1 manifests |
| CAN-V1-MB-B2/B4 | canonical C5 20/80 | same | source fingerprint `cad6726e...d5d7` | localized CE | N=320 | none | 42 | pre-run lock required |
| CAN-V1-MB-C-AUDIT | canonical C5 20/80 | frozen R84 | A0T/A4 endpoint hashes | none | calibration-only matrix | unchanged | 42 | endpoint manifests |

## Findings

| Finding ID | Severity | Check | Evidence | Impact | Required action | Status |
|---|---|---|---|---|---|---|
| MB-01 | blocking if unresolved | C3/C4 source-train role | canonical-v1 has no C3/C4 train arrays | S4 cannot run by path substitution | build new identity-audited role view; preserve C5 exactly | resolved in design; runtime verification pending |
| MB-02 | major | S2 reuse equality | G2 records 25 rounds, LE1, Adam 5e-4, seed42, fixed endpoint, no C5 training access | valid only if hashes remain equal | strict manifest/hash gate | pending execution audit |
| MB-03 | informational | B3 identity | G1 `target_head` trains `feat_proj` and `classifier` only | it is B3, not B2 | rename semantically in new table; do not rerun | resolved |
| MB-04 | major | B4 serialization | new adapter would otherwise change runtime architecture | could violate deployment constraint | require exact classifier-fold equivalence or mark not implemented | pending TDD |
| MB-05 | blocking if violated | Gate C matrix information | target test cannot construct costs | leakage would invalidate motivation | matrix builder accepts calibration rows only; seal before test diagnostics | pending TDD |
| MB-06 | minor | one seed | seed42 cannot establish stability | limits generality | label all results sensitivity/descriptive; proposal only if threshold met | accepted limitation |

## Leakage assessment

- Gate A commands and APIs must contain no C5 data path, X, Y, phase, concentration, statistics, or calibration manifest.
- Gate B receives C5 calibration only. The sealed test opens after every endpoint and hash is locked.
- Gate C constructs and hashes the cost matrix from calibration rows before any test stream is read. Test truth is post-hoc diagnostic only.

## Baseline, completeness, and reproducibility assessment

The planned matrix contains all user-required A/B/C baselines. No method, source count, rank, LR, step count, threshold, or checkpoint is selected from C5 test. Existing results remain read-only and are reused only through immutable provenance.

## Verdict: approved

Approved to implement and execute subject to runtime fail-closed gates MB-01, MB-02, MB-04, and MB-05. A favorable evidence claim remains unapproved until post-run audit.

## Unknowns and handoff

- Pending values: S4 checkpoint hashes, B2/B4 checkpoint hashes, all new metrics.
- Handoff: experiment-registry candidate rows above to implementation; post-run artifacts return to result analysis and experiment audit.

