# GAPS Project Research Skills — Version 1 Validation

Validated on 2026-07-20 in the GAPS repository.

## Installed structure

```text
.agents/skills/
├── _shared/{contracts,references}/
├── experiment-planner/{SKILL.md,agents,assets,references,scripts}/
├── experiment-registry/{SKILL.md,agents,assets,references,scripts}/
├── result-analysis/{SKILL.md,agents,assets,references,scripts}/
├── experiment-audit/{SKILL.md,agents,assets,references,scripts}/
├── claim-evidence/{SKILL.md,agents,assets,references,scripts}/
├── number-consistency-audit/{SKILL.md,agents,assets,references,scripts}/
└── gaps-research-orchestrator/{SKILL.md,agents,assets,references,scripts}/
```

## Responsibilities and contracts

| Skill | Trigger focus | Standard outputs |
|---|---|---|
| `experiment-planner` | hypothesis, baseline, ablation, metrics, Expected Evidence | `EXPERIMENT_PLAN.md`, `EXPERIMENT_MATRIX.csv`, `ABLATION_PLAN.md` |
| `experiment-registry` | experiment identity, config, split, checkpoint, seed, provenance | `experiment_registry.csv` candidate/update proposal |
| `result-analysis` | confirmed CSV/JSON statistics, uncertainty, effects, anomalies, table/figure proposals | `RESULT_ANALYSIS.md` |
| `experiment-audit` | completeness, fairness, split/checkpoint consistency, baselines, leakage, reproducibility | `EXPERIMENT_AUDIT.md` |
| `claim-evidence` | mapping audited Evidence to scoped manuscript claims | `CLAIMS_EVIDENCE.md` |
| `number-consistency-audit` | numeric consistency across manuscript locations and approved sources | `NUMBER_AUDIT.md` |
| `gaps-research-orchestrator` | current stage, largest Evidence Gap, next Skill | `PROJECT_STATUS.md`, `NEXT_ACTIONS.md`, routing recommendation |

The canonical experiment contract includes `experiment_id`, `source_clients`, `target_clients`, `split_protocol`, `model`, `checkpoint`, `DA`, `calibration`, `QC`, `seed`, `result_path`, `metrics`, `status`, and `notes`, plus recommended provenance fields.

## Call relationship

```text
Research Brief
→ experiment-planner
→ experiment-registry
→ experiment execution (outside these Skills)
→ result-analysis
→ experiment-audit
→ claim-evidence
→ research-writing-skill (existing personal Skill)
→ number-consistency-audit
```

`gaps-research-orchestrator` reads artifact state and routes to one next Skill. It does not perform child analysis.

## Minimal trigger validation

Each Skill contains three positive trigger scenarios and one negative counterexample in `references/trigger-cases.json`: 21 positive and 7 negative cases total. Cases cover GAPS-specific split, DA, model, calibration, QC, metric, Evidence, manuscript, and routing language. Negative cases explicitly route to the correct neighboring Skill, including `research-writing-skill` where appropriate.

Repository contract test:

```text
python -m pytest tests/project_skills/test_project_research_skills.py -q
.......... [100%]
10 passed in 0.08s
```

The test validates shared fields, exactly seven triggerable directories, frontmatter descriptions, UI metadata, templates, trigger counts, CSV headers, read-only/collision language, analysis/audit separation, Evidence approval gates, number safety, orchestrator boundaries, and relative links.

Official validator command:

```powershell
python -X utf8 C:\Users\HUANGJUNHUA\.codex\skills\.system\skill-creator\scripts\quick_validate.py .agents\skills\<skill-name>
```

Result: all seven returned `Skill is valid!`. `-X utf8` is required on this Windows environment because the validator otherwise reads UTF-8 Skill files with the GBK default locale.

## Read-only guarantees and limitations

- Existing `results/`, CSV, JSON, Markdown, configuration, dataset, and checkpoint assets remain untouched.
- Version 1 does not scan `results/`, infer experiment semantics, load checkpoints, execute experiments, write canonical registries, or overwrite reports.
- Every output requires an explicit destination; collisions stop with a diff/update/new-name proposal.
- Missing facts remain `unknown`; traceable disagreements remain `conflict`.
- `scripts/INTERFACE.md` reserves narrow future interfaces but contains no executable automation.
- Validation is deterministic and fixture-based. Live forward-testing with independent subagents was not used because this task did not authorize subagent delegation.

## Recommended version 2 scripts

1. **Registry candidate collection:** read only explicitly selected config/manifest/git/result paths; emit candidate rows to a new file; never update the canonical registry automatically.
2. **Stable-schema Result Analysis:** support one confirmed CSV/JSON schema with seed aggregation, confidence intervals, effect sizes, and table drafts.
3. **Registry-based Experiment Audit:** check missing fields, split/checkpoint/seed/baseline consistency, provenance, and leakage declarations without recomputing metrics.
4. After those stabilize, consider a read-only `list-results-candidates` inventory that lists candidate directories and `.csv`, `.json`, `.md`, `.pth` paths without interpreting them.
