# GAPS Project Research Skills Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create and validate seven read-only, contract-first project research Skills for the GAPS experiment-to-paper workflow.

**Architecture:** Store discoverable Skills under `.agents/skills/<skill-name>/` and place non-triggerable shared contracts under `.agents/skills/_shared/`. Each Skill owns its workflow, templates, references, trigger cases, and a scripts interface note; a repository test validates structure, frontmatter, shared fields, trigger coverage, cross-references, and handoff behavior without reading or modifying real experiment outputs.

**Tech Stack:** Markdown, CSV templates, JSON trigger fixtures, Python 3 standard library, pytest.

## Global Constraints

- First version is “workflow instructions + fixed templates + shared data contracts + validation cases.”
- Existing `results/`, Markdown, CSV, JSON, configuration, and checkpoint assets are read-only.
- Do not infer `split_protocol`, `DA`, `calibration`, `QC`, or checkpoint identity from directory names.
- Missing facts use `unknown`; conflicting sources use `conflict` with provenance.
- Do not implement large-scale result scanning or automatic registry/report writes.
- Do not overwrite existing output files; require an explicit destination and stop on collision.
- `gaps-research-orchestrator` routes work only and does not reproduce child Skill logic.
- Preserve all unrelated user changes in the dirty worktree.

---

### Task 1: Shared contracts and validation skeleton

**Files:**
- Create: `.agents/skills/_shared/contracts/experiment-record.md`
- Create: `.agents/skills/_shared/contracts/metric-record.md`
- Create: `.agents/skills/_shared/contracts/evidence-record.md`
- Create: `.agents/skills/_shared/contracts/handoff-protocol.md`
- Create: `.agents/skills/_shared/references/gaps-taxonomy.md`
- Create: `.agents/skills/_shared/references/read-only-policy.md`
- Create: `.agents/skills/_shared/references/skill-boundaries.md`
- Create: `tests/project_skills/test_project_research_skills.py`

**Interfaces:**
- Consumes: Design specification in `docs/superpowers/specs/2026-07-20-gaps-project-research-skills-design.md`.
- Produces: Canonical experiment, metric, evidence, status, handoff, and safety definitions referenced by all seven Skills.

- [ ] **Step 1: Write the failing shared-contract test**

Create a pytest module that defines `SKILLS_ROOT`, the seven expected Skill names, the required experiment fields, and tests that all shared files exist and `experiment-record.md` contains every field:

```python
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SKILLS_ROOT = REPO_ROOT / ".agents" / "skills"
EXPECTED_SKILLS = {
    "experiment-planner",
    "experiment-registry",
    "result-analysis",
    "experiment-audit",
    "claim-evidence",
    "number-consistency-audit",
    "gaps-research-orchestrator",
}
REQUIRED_EXPERIMENT_FIELDS = {
    "experiment_id", "source_clients", "target_clients", "split_protocol",
    "model", "checkpoint", "DA", "calibration", "QC", "seed",
    "result_path", "metrics", "status", "notes",
}

def test_shared_contracts_exist_and_define_required_fields():
    contract = SKILLS_ROOT / "_shared" / "contracts" / "experiment-record.md"
    assert contract.is_file()
    text = contract.read_text(encoding="utf-8")
    assert REQUIRED_EXPERIMENT_FIELDS <= set(text.replace("`", "").split())
```

- [ ] **Step 2: Run the test and verify RED**

Run: `python -m pytest tests/project_skills/test_project_research_skills.py -q`

Expected: FAIL because `.agents/skills/_shared/contracts/experiment-record.md` does not exist.

- [ ] **Step 3: Create the canonical shared documents**

Define exact field meanings, allowed `unknown`/`conflict` handling, status transitions (`draft`, `registered`, `completed`, `audited`, `approved`, `blocked`, `conflict`), metric provenance, Evidence approval gates, handoff fields, GAPS terms, read-only collision behavior, and the seven-way responsibility table. Do not include executable scanning code.

- [ ] **Step 4: Run the shared-contract test**

Run: `python -m pytest tests/project_skills/test_project_research_skills.py -q`

Expected: PASS for the shared-contract test; later Skill tests are not yet present.

- [ ] **Step 5: Commit shared contracts**

```powershell
git add -- .agents/skills/_shared tests/project_skills/test_project_research_skills.py
git commit -m "feat: define GAPS research skill contracts"
```

### Task 2: Experiment Planner and Experiment Registry

**Files:**
- Create: `.agents/skills/experiment-planner/SKILL.md`
- Create: `.agents/skills/experiment-planner/agents/openai.yaml`
- Create: `.agents/skills/experiment-planner/assets/EXPERIMENT_PLAN.template.md`
- Create: `.agents/skills/experiment-planner/assets/EXPERIMENT_MATRIX.template.csv`
- Create: `.agents/skills/experiment-planner/assets/ABLATION_PLAN.template.md`
- Create: `.agents/skills/experiment-planner/references/trigger-cases.json`
- Create: `.agents/skills/experiment-planner/scripts/INTERFACE.md`
- Create: `.agents/skills/experiment-registry/SKILL.md`
- Create: `.agents/skills/experiment-registry/agents/openai.yaml`
- Create: `.agents/skills/experiment-registry/assets/experiment_registry.template.csv`
- Create: `.agents/skills/experiment-registry/references/trigger-cases.json`
- Create: `.agents/skills/experiment-registry/scripts/INTERFACE.md`
- Modify: `tests/project_skills/test_project_research_skills.py`

**Interfaces:**
- Consumes: Shared experiment record, read-only policy, boundaries, and handoff protocol.
- Produces: Planning artifacts accepted by Registry and registry records accepted by analysis/audit Skills.

- [ ] **Step 1: Add failing tests for frontmatter, templates, and trigger fixtures**

Add tests that parse the YAML-like frontmatter without external packages, assert `name` equals the directory name, assert `description` mentions positive trigger terms, assert each JSON fixture has at least two positive cases and one negative case, and assert CSV headers include all required experiment fields.

- [ ] **Step 2: Run targeted tests and verify RED**

Run: `python -m pytest tests/project_skills/test_project_research_skills.py -q`

Expected: FAIL listing missing Planner and Registry files.

- [ ] **Step 3: Implement Experiment Planner**

Write a focused `SKILL.md` covering when to use, when not to use, required inputs, read-only workflow, hypothesis-to-evidence matrix, output contracts, unknown/conflict behavior, and handoff to Registry. Add three GAPS-positive trigger cases and one writing-only negative case. Templates must cover hypotheses H1–H3, baseline, ablation, metric, expected evidence, budget, and acceptance criteria without embedding unverified result numbers.

- [ ] **Step 4: Implement Experiment Registry**

Write a focused `SKILL.md` covering explicit provenance, ID assignment, candidate-record workflow, collision safety, unknown/conflict behavior, and handoff to Result Analysis or Audit. The CSV template must use the canonical required fields plus recommended provenance fields. Add three positive trigger cases and one statistical-analysis negative case.

- [ ] **Step 5: Run tests and verify GREEN**

Run: `python -m pytest tests/project_skills/test_project_research_skills.py -q`

Expected: PASS.

- [ ] **Step 6: Commit Planner and Registry**

```powershell
git add -- .agents/skills/experiment-planner .agents/skills/experiment-registry tests/project_skills/test_project_research_skills.py
git commit -m "feat: add GAPS experiment planning and registry skills"
```

### Task 3: Result Analysis and Experiment Audit

**Files:**
- Create: `.agents/skills/result-analysis/SKILL.md`
- Create: `.agents/skills/result-analysis/agents/openai.yaml`
- Create: `.agents/skills/result-analysis/assets/RESULT_ANALYSIS.template.md`
- Create: `.agents/skills/result-analysis/references/trigger-cases.json`
- Create: `.agents/skills/result-analysis/references/statistical-reporting.md`
- Create: `.agents/skills/result-analysis/scripts/INTERFACE.md`
- Create: `.agents/skills/experiment-audit/SKILL.md`
- Create: `.agents/skills/experiment-audit/agents/openai.yaml`
- Create: `.agents/skills/experiment-audit/assets/EXPERIMENT_AUDIT.template.md`
- Create: `.agents/skills/experiment-audit/references/trigger-cases.json`
- Create: `.agents/skills/experiment-audit/references/audit-checklist.md`
- Create: `.agents/skills/experiment-audit/scripts/INTERFACE.md`
- Modify: `tests/project_skills/test_project_research_skills.py`

**Interfaces:**
- Consumes: Registry records, confirmed result schemas, metric contract, experiment plan, and provenance.
- Produces: Separate analysis and audit reports; Audit status gates Evidence approval.

- [ ] **Step 1: Add failing separation tests**

Assert Result Analysis declares statistical summaries but disclaims fairness decisions; assert Experiment Audit declares split/checkpoint/seed/baseline/leakage checks but disclaims statistical recomputation; assert templates contain provenance and unresolved-issue sections.

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m pytest tests/project_skills/test_project_research_skills.py -q`

Expected: FAIL for missing analysis and audit Skill files.

- [ ] **Step 3: Implement Result Analysis**

Document schema confirmation, descriptive statistics, assumption checks, effect size/uncertainty, multiple-comparison handling, anomaly reporting, and figure/table proposals. Require explicit labeling of reported versus recomputed values and route experiment-comparability questions to `experiment-audit`.

- [ ] **Step 4: Implement Experiment Audit**

Document completeness, fair comparison, split/checkpoint/seed/baseline/provenance checks, leakage risk, conflict recording, severity, and `approved` gate rules. Do not compute replacement metrics or edit experimental assets.

- [ ] **Step 5: Run tests and verify GREEN**

Run: `python -m pytest tests/project_skills/test_project_research_skills.py -q`

Expected: PASS.

- [ ] **Step 6: Commit analysis and audit Skills**

```powershell
git add -- .agents/skills/result-analysis .agents/skills/experiment-audit tests/project_skills/test_project_research_skills.py
git commit -m "feat: add GAPS result analysis and experiment audit skills"
```

### Task 4: Claim–Evidence and Number Consistency Audit

**Files:**
- Create: `.agents/skills/claim-evidence/SKILL.md`
- Create: `.agents/skills/claim-evidence/agents/openai.yaml`
- Create: `.agents/skills/claim-evidence/assets/CLAIMS_EVIDENCE.template.md`
- Create: `.agents/skills/claim-evidence/references/trigger-cases.json`
- Create: `.agents/skills/claim-evidence/references/claim-strength.md`
- Create: `.agents/skills/claim-evidence/scripts/INTERFACE.md`
- Create: `.agents/skills/number-consistency-audit/SKILL.md`
- Create: `.agents/skills/number-consistency-audit/agents/openai.yaml`
- Create: `.agents/skills/number-consistency-audit/assets/NUMBER_AUDIT.template.md`
- Create: `.agents/skills/number-consistency-audit/references/trigger-cases.json`
- Create: `.agents/skills/number-consistency-audit/references/number-comparison-rules.md`
- Create: `.agents/skills/number-consistency-audit/scripts/INTERFACE.md`
- Modify: `tests/project_skills/test_project_research_skills.py`

**Interfaces:**
- Consumes: Audited Evidence, manuscript claims, approved tables/figures, metric definitions, and provenance.
- Produces: Claim–Evidence matrix and non-mutating numeric discrepancy report.

- [ ] **Step 1: Add failing Evidence-gate and number-audit tests**

Assert Claim–Evidence refuses to mark unaudited experimental Evidence as approved; assert Number Audit covers Abstract/Results/Table/Figure/Conclusion locations, distinguishes exact mismatch from allowed display rounding, and never silently edits manuscript numbers.

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m pytest tests/project_skills/test_project_research_skills.py -q`

Expected: FAIL for missing claim and number audit files.

- [ ] **Step 3: Implement Claim–Evidence**

Define claim IDs, claim scope, evidence IDs, comparison, source, audit status, support strength, limitations, provenance, and manuscript locations. Route citation truth/entailment to a future citation audit and prose drafting to `research-writing-skill`.

- [ ] **Step 4: Implement Number Consistency Audit**

Define canonical source priority, metric identity keys, units, sample scope, aggregation, precision, rounding tolerance, discrepancy severity, and report-only corrections. Include GAPS metrics without collapsing distinct RMSE or Coverage variants.

- [ ] **Step 5: Run tests and verify GREEN**

Run: `python -m pytest tests/project_skills/test_project_research_skills.py -q`

Expected: PASS.

- [ ] **Step 6: Commit Evidence and number audit Skills**

```powershell
git add -- .agents/skills/claim-evidence .agents/skills/number-consistency-audit tests/project_skills/test_project_research_skills.py
git commit -m "feat: add GAPS claim evidence and number audit skills"
```

### Task 5: GAPS Research Orchestrator

**Files:**
- Create: `.agents/skills/gaps-research-orchestrator/SKILL.md`
- Create: `.agents/skills/gaps-research-orchestrator/agents/openai.yaml`
- Create: `.agents/skills/gaps-research-orchestrator/assets/PROJECT_STATUS.template.md`
- Create: `.agents/skills/gaps-research-orchestrator/assets/NEXT_ACTIONS.template.md`
- Create: `.agents/skills/gaps-research-orchestrator/references/trigger-cases.json`
- Create: `.agents/skills/gaps-research-orchestrator/references/stage-gates.md`
- Create: `.agents/skills/gaps-research-orchestrator/scripts/INTERFACE.md`
- Modify: `tests/project_skills/test_project_research_skills.py`

**Interfaces:**
- Consumes: Presence, completeness, status, and unresolved gaps from all standard artifacts.
- Produces: Project stage, largest Evidence Gap, and one prioritized next-Skill recommendation.

- [ ] **Step 1: Add failing routing tests**

Define a synthetic routing fixture in the pytest module and assert missing plan routes to Planner, missing metadata routes to Registry, comparability conflict routes to Audit, audited results without claims route to Claim–Evidence, and manuscript number conflict routes to Number Audit. Assert the orchestrator description explicitly disclaims child analysis.

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m pytest tests/project_skills/test_project_research_skills.py -q`

Expected: FAIL for missing orchestrator files.

- [ ] **Step 3: Implement Orchestrator**

Define stage gates, Evidence Gap ranking, stop conditions, routing table, output templates, and the three mandatory answers: current stage, largest Evidence Gap, and next Skill. Require one primary next action plus blocked prerequisites; do not perform child work.

- [ ] **Step 4: Run tests and verify GREEN**

Run: `python -m pytest tests/project_skills/test_project_research_skills.py -q`

Expected: PASS.

- [ ] **Step 5: Commit Orchestrator**

```powershell
git add -- .agents/skills/gaps-research-orchestrator tests/project_skills/test_project_research_skills.py
git commit -m "feat: add GAPS research orchestrator skill"
```

### Task 6: Full validation and installation report

**Files:**
- Create: `.agents/skills/VALIDATION.md`
- Modify: `tests/project_skills/test_project_research_skills.py`

**Interfaces:**
- Consumes: All seven Skills, shared contracts, templates, trigger fixtures, and validation output.
- Produces: Reproducible validation report and confirmed project-level installation.

- [ ] **Step 1: Complete full validation tests**

Add checks for exactly seven triggerable Skill directories, valid names/descriptions, existing relative references, at least two positive and one negative trigger per Skill, canonical template fields, read-only wording, no complex executable files under Skill `scripts/`, and a synthetic end-to-end handoff record.

- [ ] **Step 2: Run repository validation**

Run: `python -m pytest tests/project_skills/test_project_research_skills.py -q`

Expected: all tests PASS.

- [ ] **Step 3: Run the installed Skill validator**

Run this command once per Skill directory, replacing `<skill-name>` with each of the seven names:

```powershell
python 'C:\Users\HUANGJUNHUA\.codex\skills\.system\skill-creator\scripts\quick_validate.py' ".agents/skills/<skill-name>"
```

Expected: exit code 0 and `Skill is valid!` for every Skill. Record the exact command and per-Skill result in `VALIDATION.md`.

- [ ] **Step 4: Inspect repository safety**

Run:

```powershell
git diff --check
git status --short
git diff --name-only HEAD~5..HEAD
```

Expected: no whitespace errors; only the new plan, shared contracts, seven Skill trees, validation tests, and validation report are part of this feature's commits. Existing unrelated dirty files remain unstaged and unchanged.

- [ ] **Step 5: Write validation report**

Document directory structure, each Skill's description and contract, call graph, positive/negative trigger counts, structure/contract/handoff test results, read-only guarantees, known first-version limitations, and the exact second-phase priorities: Registry candidate collection, stable-schema Result Analysis, then Registry-based Experiment Audit.

- [ ] **Step 6: Run final tests and commit**

Run: `python -m pytest tests/project_skills/test_project_research_skills.py -q`

Expected: all tests PASS.

```powershell
git add -- .agents/skills/VALIDATION.md tests/project_skills/test_project_research_skills.py
git commit -m "test: validate GAPS project research skills"
```
