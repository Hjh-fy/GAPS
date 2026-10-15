# C5 Low-Label Commissioning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute and audit six canonical-v1 C5 low-label A0T/A4 Flower experiments using one frozen nested 15%/10%/5% calibration family.

**Architecture:** A preparation module indexes the immutable 320-window C5 calibration pool and writes calibration-only budget directories plus manifests. A runner clones the existing canonical A0T/A4 command builders, changes only experiment identity and calibration directory, deploys each immutable budget directory to ECS, and executes six complete Flower runs. A sealed evaluator gates all endpoints before reading the unchanged canonical test and an analyzer merges the reused 20% rows with the six new results.

**Tech Stack:** Python 3, NumPy, PyTorch, Flower orchestration, pytest, SSH/SCP, CSV/JSON/Markdown, SHA-256.

## Global Constraints

- Dataset and preprocessing remain canonical-v1 / `HZ5_MEAN_W10S`, 50×8.
- Budget counts are exactly 240/160/80, with 6/4/2 windows in each of 40 strata.
- `5% ⊂ 10% ⊂ 15% ⊂ 20%`; test identity and bytes remain unchanged.
- Six runs use seed42, 25 rounds, local_epochs=1, batch size 32, Adam `5e-4`, and 100 adaptation steps per round.
- No 20% target-adapted checkpoint is reused; fixed endpoint is round25.
- No test selection, hyperparameter search, C3/C4 expansion, multi-seed execution, regression, or QC.

---

### Task 1: Nested budget preparation and audit

**Files:**
- Create: `scripts/prepare_iotj_c5_label_budget.py`
- Create: `tests/test_prepare_iotj_c5_label_budget.py`
- Output: `results/iotj_canonical_v1_c5_budget_20260810/`

**Interfaces:**
- Produces `build_nested_indices(info: list[dict]) -> dict[int, list[int]]` and `prepare(source: Path, output: Path) -> dict`.
- Produces calibration-only directories `budget_data/client_5_budget_{15,10,05}` and the four required manifest CSVs/audits.

- [ ] Write failing tests asserting 40 strata, exact 8→6→4→2 allocation, identity nesting, deterministic raw-file/repeat-diverse ordering, no duplicates, no test arrays in budget directories, and zero calibration/test identity overlap.
- [ ] Run `python -m pytest tests/test_prepare_iotj_c5_label_budget.py -q` and confirm failure before implementation.
- [ ] Implement manifest parsing, deterministic per-stratum ordering, array indexing, audit Markdown/JSON, and SHA-256 output without changing source files.
- [ ] Run the preparation test and canonical dataset hash verification.

### Task 2: Frozen six-run command and deployment layer

**Files:**
- Create: `scripts/run_iotj_c5_label_budget.py`
- Create: `tests/test_run_iotj_c5_label_budget.py`
- Output: `results/iotj_canonical_v1_c5_budget_20260810/PRE_RUN_FREEZE.json`

**Interfaces:**
- Consumes budget directories and manifest SHA-256 values from Task 1.
- Produces `build_budget_commands(method: str, budget: int) -> dict`, `audit_commands() -> dict`, remote deployment verification, and sequential resumable execution.

- [ ] Write failing tests for exactly six unique IDs, fresh initialization/no checkpoint tokens, exact A0T and A4 setting parity, only calibration-path/identity changes, fixed round25/LE1/seed42/100 steps, no target-test token, and six-run stop rule.
- [ ] Run `python -m pytest tests/test_run_iotj_c5_label_budget.py -q` and confirm failure.
- [ ] Implement command cloning, immutable freeze, ECS-only budget-data deployment with remote hash verification, completion markers, and progress state.
- [ ] Run runner tests and protocol audit; fail closed unless all checks pass.

### Task 3: Unified sealed evaluation

**Files:**
- Create: `scripts/evaluate_iotj_c5_label_budget.py`
- Create: `tests/test_evaluate_iotj_c5_label_budget.py`

**Interfaces:**
- Consumes all six `fixed_endpoint_complete.json`, run manifests, checkpoint SHA-256 values, unchanged canonical C5 test, and C1/C2 source tests.
- Produces A0T/A4 metric CSVs, source-retention CSV, confusion matrices, and evaluation manifest.

- [ ] Write failing tests that evaluation rejects any missing endpoint, non-round25 endpoint, hash mismatch, test-opened marker, wrong budget/stratum hash, or altered dataset hash.
- [ ] Run the evaluator tests and confirm failure.
- [ ] Implement one-time gated evaluation with Accuracy/Macro-F1/NLL/ECE, per-class precision/recall/F1, confusion matrices, and FedAvg source-retention delta.
- [ ] Run evaluator tests and verify output schemas.

### Task 4: Analysis, strict audit, and Git evidence publication

**Files:**
- Create: `scripts/analyze_iotj_c5_label_budget.py`
- Create: `tests/test_analyze_iotj_c5_label_budget.py`
- Create after execution: `docs/experiments/iotj_canonical_v1_final/C5_LABEL_BUDGET_ANALYSIS.md`
- Create after execution: `docs/experiments/iotj_canonical_v1_final/c5_label_budget_comparison.csv`
- Create after execution: `docs/experiments/iotj_canonical_v1_final/c5_label_budget_strata_coverage.csv`
- Create after execution: `docs/experiments/iotj_canonical_v1_final/c5_label_budget_manifest_summary.json`

**Interfaces:**
- Consumes reused 20% canonical comparison rows plus six new fixed-endpoint rows.
- Produces the required 20/15/10/5 comparison, degradation, gap, source-retention, coverage, scientific decision, multi-seed proposal gate, and evidence hash index.

- [ ] Write failing tests for 20% reuse, ordered budgets, A4−A0T calculation, degradation calculation, the ≥1 percentage-point multi-seed proposal gate, and strict-boundary wording.
- [ ] Run analysis tests and confirm failure.
- [ ] Implement deterministic CSV/Markdown/JSON analysis and evidence hashing.
- [ ] Run six experiments only after pre-run freeze passes, then run the common evaluator and analysis exactly once.
- [ ] Run relevant pytest, `python -m compileall`, dataset/checkpoint/manifest/test hashes, and strict audit.
- [ ] Commit only code, compact manifests, summary CSVs, analysis, and scientific evidence; do not commit large checkpoints or predictions.
- [ ] Push `codex/iotj-final-classification-le1` and stop.
