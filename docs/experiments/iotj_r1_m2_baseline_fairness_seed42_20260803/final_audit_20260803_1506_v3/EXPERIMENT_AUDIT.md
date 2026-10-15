# R1-M2 Baseline Fairness Experiment Audit

## Audit scope and intended claim

Assess whether the five registered seed42 baselines close reviewer concern R1-M2 without target-test leakage or hidden training-budget changes.

## Findings

| ID | Severity | Finding | Impact | Status |
|---|---|---|---|---|
| R1M2-A01 | informational | All five runs use seed42 and C5 sealed test n=1,360; artifact hashes match manifests. | Reproducible single-seed comparison. | passed |
| R1M2-A02 | major | Target-only uses 2,500 fully supervised target CE steps, unlike B5 target CE weight 0. | Treat only as a target-supervised upper/reference row. | constrained |
| R1M2-A03 | informational | Centralized Source-only, FedProx and DS do not isolate every GAPS mechanism. | Use them as coverage baselines, not causal ablations. | passed with scope |
| R1M2-A04 | informational | FedAvg+same adapter holds target calibration and DA settings, disables client GAPS losses and selective aggregation by design. | Closest R1-M2 mechanism comparator. | passed |
| R1M2-A05 | minor | Target-only and DS executed at head d6881d4 with an as-run wrapper later captured in 61a5d18; the only wrapper change was repository import-path plumbing. | Preserve both provenance identifiers. | documented |
| R1M2-A06 | informational | B5 commit compatibility diff changes only default-inert FedProx/profile plumbing in effective paths. | Existing B5 checkpoint remains code-compatible for this comparison. | passed |
| R1M2-A07 | major | Only one seed was authorized. | No stability, CI, or significance claim. | constrained |
| R1M2-A08 | blocking for the original broad claim | FedAvg+same target adapter exceeds B5 by +1.022 Macro-F1 percentage points. | The seed42 evidence does not support superiority of GAPS client mechanisms/selective aggregation beyond matched target adaptation. | manuscript claim must be narrowed |
| R1M2-A09 | minor | The FedProx result manifest retains a generic `ce_stats` statistics-payload note although its locked client profile is `ce_only`. | Numerical metrics and exact model-payload accounting are unaffected; treat FedProx as having no extra statistics payload. | documented amendment |
| R1M2-A10 | minor | Distributed model payload bytes are exact, but adapter `ce_stats` JSON wire bytes were not instrumented. | Communication comparison is usable only with an explicit `model bytes + unmeasured small statistics payload` qualifier for the adapter row. | constrained |

## Leakage assessment

No manifest reports target-test use for training, calibration, selection, stopping, or hyperparameter tuning. DS alpha was selected inside C5 calibration folds before the test split was opened.

## Verdict

The experiment set is approved for a seed42-only descriptive baseline table. The original broad superiority/attribution claim is blocked: it must be narrowed to the demonstrated value of target-assisted adaptation and to GAPS outperforming DS and no-target source baselines. Not approved for statistical superiority or stability claims.
