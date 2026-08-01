# Experiment Audit

## Audit scope and intended claim
Audit the descriptive claim that fixed round-25 A1/A4 models have different classification performance in early, stable, and full response-time scopes.

## Compared experiments
| Experiment | Protocol | Direction | Early 0–150 s | Stable | Full | Exposure (full) |
|---|---|---|---:|---:|---:|---:|
| A1-FULL-E1-P2P3-S42 | A1 | P2→P3 | 58.33% | 99.44% | 93.57% | 100.00% |
| A4-CTRL-E2-P2P3-LE1-S42 | A4 | P2→P3 | 65.00% | 98.61% | 93.81% | 100.00% |
| A1-FULL-E3-P1P3-S42 | A1 | P1→P3 | 50.00% | 96.94% | 90.24% | 100.00% |
| A1-FULL-E4-P12P3-S42 | A1 | P12→P3 | 51.67% | 97.22% | 90.71% | 100.00% |
| A1-FULL-E5-P2P1-S42 | A1 | P2→P1 | 68.33% | 98.06% | 93.81% | 100.00% |
| A1-FULL-E6-P3P1-S42 | A1 | P3→P1 | 68.33% | 99.72% | 95.24% | 100.00% |

All runs use seed 42, 25 federated rounds, one local epoch, 100 server-DA steps per round, `proto_replay`, `corrected_b2`, target CE weight 0, source-train-only normalization, and the last-round checkpoint policy.

## Findings
| Finding ID | Severity | Check | Evidence | Impact | Required action | Status |
|---|---|---|---|---|---|---|
| F1 | informational | checkpoint identity | all summaries reference round-25 adapted checkpoints | fixes selection boundary | none | closed |
| F2 | informational | primary-scope consistency | A1 matches full420 audit; A4 matches stable360 audit | prevents scope mixing | none | closed |
| F3 | major | seed coverage | seed set is only 42 | blocks inferential/general claims | add seeds only if stronger claim is needed | open |
| F4 | major | target concentration scope | calibration and test cover all retained concentrations | not unseen-concentration evidence | label claim boundary | open |
| F5 | minor | boundary precision | nominal gas boundaries | may shift early/stable assignment | replace when exact valve timestamps exist | open |
| F6 | major | P1+P2 budget match | no matched source-update/data-budget control | not a pure source-diversity ablation | add matched control before causal wording | open |

## Leakage assessment
Target calibration, purged, early, stable, and full windows use explicit non-overlapping time indices. Target test was opened only after fixed round selection. The same exposures and concentrations appear across target calibration and test time positions, so the result measures time-purged within-exposure adaptation, not unseen-exposure or unseen-concentration generalization.

## Baseline, completeness, and reproducibility assessment
All six planned runs and their valid postflight audits are present. Early plus stable confusion matrices exactly reproduce the full-window confusion matrix. Checkpoint hashes and source archive identity are recorded. Single-seed uncertainty remains unavailable.

## Verdict: approved
Approved only for single-seed descriptive, within-protocol evidence. Blocked for seed-robust, unseen-concentration, or causal source-diversity claims.

## Unknowns and handoff
Multi-seed variance and performance under exact valve timestamps remain `unknown`.
