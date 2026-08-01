# Result Analysis

## Input contract and provenance
- Experiment IDs: A1-FULL-E1-P2P3-S42, A4-CTRL-E2-P2P3-LE1-S42, A1-FULL-E3-P1P3-S42, A1-FULL-E4-P12P3-S42, A1-FULL-E5-P2P1-S42, A1-FULL-E6-P3P1-S42
- Metric schema: window Accuracy/Macro-F1 and exposure Accuracy/Macro-F1.
- Sample scope: target-board early60, stable360, and full420 windows.
- Reported versus recomputed values: formal primary metrics copied from postflight audits are `reported`; all three post-hoc scope metrics are `recomputed` from frozen checkpoints.

## Descriptive statistics
| Experiment | Protocol | Direction | Early 0–150 s | Stable | Full | Exposure (full) |
|---|---|---|---:|---:|---:|---:|
| A1-FULL-E1-P2P3-S42 | A1 | P2→P3 | 58.33% | 99.44% | 93.57% | 100.00% |
| A4-CTRL-E2-P2P3-LE1-S42 | A4 | P2→P3 | 65.00% | 98.61% | 93.81% | 100.00% |
| A1-FULL-E3-P1P3-S42 | A1 | P1→P3 | 50.00% | 96.94% | 90.24% | 100.00% |
| A1-FULL-E4-P12P3-S42 | A1 | P12→P3 | 51.67% | 97.22% | 90.71% | 100.00% |
| A1-FULL-E5-P2P1-S42 | A1 | P2→P1 | 68.33% | 98.06% | 93.81% | 100.00% |
| A1-FULL-E6-P3P1-S42 | A1 | P3→P1 | 68.33% | 99.72% | 95.24% | 100.00% |

Only seed 42 is available; mean, SD, 95% CI, significance tests, and seed-level effect sizes are therefore `unknown` and are not reported.

## Assumptions, comparisons, effect sizes, and corrections
- Each row is descriptive for one fixed source/target configuration and one fixed round-25 adapted checkpoint.
- Overlapping windows are not treated as independent clients or independent exposures.
- No inferential tests or multiple-comparison corrections are applicable with one seed.

## Anomalies and sensitivity analysis
- Early-window performance is substantially below stable-window performance in every run.
- Exposure-level accuracy can remain 100% even when many early windows are wrong because exposure prediction aggregates all windows.
- P1+P2→P3 has no matched source-update/data-budget control; its difference from single-source runs is not a pure diversity effect.

## Proposed paper tables and figures
- Main table: early/stable/full window Accuracy and Macro-F1 by transfer direction.
- Diagnostic figure: paired early versus stable accuracy for each direction.

## Unknowns, conflicts, and audit handoff
- Uncertainty across seeds: `unknown`.
- Generalization to unseen concentrations and future sessions: `unknown`.
- No metric/provenance conflict was detected for the declared primary scopes.
