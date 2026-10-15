# GAPS Next Method Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Execute the frozen Phase 1–4 validation without changing canonical data, algorithms, hyperparameters, or test-selection boundaries.

**Architecture:** One fail-closed runner per phase writes a pre-run freeze, endpoint locks, analysis outputs, audit, and SHA index. Phase transitions consume only the predecessor decision artifact and verified immutable endpoints.

**Tech Stack:** Python, PyTorch, Flower, NumPy, pytest, PowerShell/SSH orchestration, Git.

## Global Constraints

- Use the existing `codex/iotj-final-classification-le1` worktree and branch.
- Follow `EXPERIMENT_PLAN.md` exactly; no target-test tuning or matrix expansion.
- Write tests before runner behavior and observe RED then GREEN.
- Commit and push after each phase.

### Task 1: Phase 1 multi-seed runner

**Files:** create `scripts/run_iotj_s4_dg_multiseed.py`; create `tests/test_iotj_s4_dg_multiseed.py`.

**Interfaces:** `build_multiseed_commands(method: str, seed: int)`, `decide_multiseed(rows)`, `verify_endpoint_locks(root)`, and `run(args)`.

- [ ] Write tests proving seeds are exactly 41/42/43, seed42 is reuse, commands contain C1–C4 and no C5, DG differs only by frozen prototype settings, and the three decision branches match the plan.
- [ ] Run the new tests and confirm failure because the runner does not exist.
- [ ] Implement the runner by reusing Gate A orchestration/evaluation helpers and adding an explicit seed argument; do not alter Gate A results.
- [ ] Run relevant tests and compileall; write `PRE_RUN_FREEZE.json` and endpoint registry before training.
- [ ] Execute four new fixed round25 runs sequentially, evaluate only after all locks, generate `S4_DG_MULTI_SEED.csv` and `S4_DG_MULTI_SEED_REPORT.md`, audit hashes, commit, and push.

### Task 2: Phase 2 commissioning bridge

**Files:** create `scripts/run_iotj_dg_commissioning_bridge.py`; create `tests/test_iotj_dg_commissioning_bridge.py`.

**Interfaces:** `audit_source_initializations()`, `load_budget_manifest(budget)`, `adapt_endpoint(initialization, budget)`, `decide_bridge(rows)`, and `run(args)`.

- [ ] Write tests for exact I0/I1/I2 fingerprints, B20/B05 nested identities and counts, source reload per endpoint, fixed 100-step A0T, sealed-test lock, and decision priority.
- [ ] Confirm RED, then implement only the five non-reusable endpoints; audit and reuse I0+B20.
- [ ] Evaluate six endpoints on the same C5 test and source-pooled retention scopes; generate `dg_commissioning_bridge.csv` and `DG_TO_COMMISSIONING_REPORT.md`.
- [ ] Run tests/compileall/hash audit, commit, and push.

### Task 3: Phase 3 post-hoc R84 baseline

**Files:** create `scripts/run_iotj_posthoc_r84_baseline.py`; create `tests/test_iotj_posthoc_r84_baseline.py`.

**Interfaces:** `select_source_identity(phase2_decision, rows)`, `fit_frozen_r84(calibration_rows)`, and `evaluate_argmax_baseline()`.

- [ ] Write tests for the method-identity selection rule, unchanged H1 SHA, frozen C5 alphas, calibration-only R84 fit, and four regression scopes.
- [ ] Confirm RED, implement the single selected B20 endpoint baseline, and lock R84 before test evaluation.
- [ ] Generate `POSTHOC_R84_BASELINE.md`, predictions, manifests, and SHA index; test, compile, commit, and push.

### Task 4: Phase 4 cost-aware direct test

**Files:** create `scripts/run_iotj_posthoc_cost_aware_routing.py`; create `tests/test_iotj_posthoc_cost_aware_routing.py`.

**Interfaces:** `build_calibration_cost_matrix()`, `expected_cost_routes(probabilities, matrix)`, `grouped_file_bootstrap()`, and `decide_cost_router()`.

- [ ] Write tests for calibration-only matrix construction, diagonal/off-diagonal semantics, parameter-free route selection, paired metrics, grouped bootstrap, and guardrail decisions.
- [ ] Confirm RED, implement the matrix lock before test access, then compare argmax versus expected cost on the identical test rows.
- [ ] Generate all requested CSV/Markdown outputs and `SAFE_TOP2_ROUTING_PROPOSAL.md` only when the registered classification-cost branch occurs.
- [ ] Run tests/compileall/hash audit, commit, and push.

### Task 5: final story audit

**Files:** create `NEXT_METHOD_STORY_DECISION.md` in the requested method-breakthrough docs/results roots.

- [ ] Map Phase 1/2/4 decisions deterministically to Story A–E and the registered next action.
- [ ] Verify every required output, all SHA indexes, local/remote branch equality, and that no forbidden experiment directory was created.
- [ ] Commit/push the final audit, stop all runners/monitors, and report the final A–E format.

