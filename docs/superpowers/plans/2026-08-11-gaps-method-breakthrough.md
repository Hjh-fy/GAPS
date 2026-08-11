# GAPS Method-Breakthrough Implementation Plan

> **Execution:** Use `superpowers:executing-plans`; implement behavior changes with strict red-green TDD and verify before every commit.

**Goal:** Complete Gate A, Gate B, and Gate C in order, publish audited evidence, and stop before Gate D/E/F.

**Architecture:** New scripts create explicit derived inputs and new result roots while existing canonical results remain read-only. Gate A extends the proven remote Flower orchestration to four source clients. Gate B extends the post-hoc API with two localized parameter scopes. Gate C consumes frozen regression endpoints and generates a calibration-only route-cost matrix plus test-only diagnostic bootstrap.

**Tech stack:** Python 3, PyTorch, Flower, NumPy, CSV/JSON, pytest, SHA-256, SSH/SCP.

### Task 1: Freeze planning, registry, and audit

**Files:**
- Create `docs/experiments/iotj_canonical_v1_final/method_breakthrough_20260811/EXPERIMENT_PLAN.md`
- Create `docs/experiments/iotj_canonical_v1_final/method_breakthrough_20260811/EXPERIMENT_MATRIX.csv`
- Create `docs/experiments/iotj_canonical_v1_final/method_breakthrough_20260811/ABLATION_PLAN.md`
- Create `docs/experiments/iotj_canonical_v1_final/method_breakthrough_20260811/EXPERIMENT_REGISTRY.csv`
- Create `docs/experiments/iotj_canonical_v1_final/method_breakthrough_20260811/PRE_RUN_EXPERIMENT_AUDIT.md`
- Create `docs/experiments/iotj_canonical_v1_final/method_breakthrough_20260811/MANUSCRIPT_METHOD_CHANGELOG.md`

- [ ] Record immutable source/checkpoint/split hashes and every reused endpoint.
- [ ] Resolve S4 role semantics through a derived physical-identity view.
- [ ] Mark Gate D/E/F out of scope.
- [ ] Run baseline focused tests, commit, and push the pre-run freeze.

### Task 2: Gate A derived role-view builder

**Files:**
- Create `tools/build_iotj_canonical_v1_s4_role_view.py`
- Create `tests/test_iotj_method_breakthrough_gate_a.py`

- [ ] RED: test that C5 identities/hashes equal canonical-v1 and no canonical file is written.
- [ ] RED: test that C1-C4 each expose train/calibration/test and use the frozen source role map.
- [ ] GREEN: build the new role view with exact canonical preprocessing.
- [ ] Verify aggregate and per-split identity manifests.

### Task 3: Gate A four-client Flower runner and analysis

**Files:**
- Create `scripts/run_iotj_method_breakthrough_gate_a.py`
- Extend `tests/test_iotj_method_breakthrough_gate_a.py`
- Create under `results/iotj_canonical_v1_method_breakthrough_20260811/gate_a_source_diversity/`

- [ ] RED: commands require exactly C1-C4, reject C5 paths, fix round25/LE1/seed42, and isolate FedAvg versus exact G2 GAPS-DG-P.
- [ ] GREEN: support C3/C4 source clients in the existing remote orchestration without changing the S2 endpoints.
- [ ] Freeze and run S4 FedAvg, then S4 GAPS-DG-P.
- [ ] Evaluate each source test, pooled source, C5, per-class and representation diagnostics.
- [ ] Emit the registered Gate-A decision and eight-section report.
- [ ] Audit, pytest, compileall, hash, commit, and push.

### Task 4: Gate B lightweight commissioning

**Files:**
- Modify `gaps_flower/posthoc_commissioning.py`
- Create `scripts/run_iotj_method_breakthrough_gate_b.py`
- Create `tests/test_iotj_method_breakthrough_gate_b.py`
- Create under `results/iotj_canonical_v1_method_breakthrough_20260811/gate_b_lightweight_posthoc/`

- [ ] RED: classifier-only changes no encoder parameter.
- [ ] RED: rank-4 adapter has exactly the registered trainable state and its folded checkpoint is numerically equivalent.
- [ ] GREEN: implement B2 and the clean foldable B4 path.
- [ ] Audit and reuse B0/B1/B3 without rerunning them.
- [ ] Lock B2/B4, then evaluate C1/C2/pooled/C5 once.
- [ ] Emit the registered Gate-B decision and eight-section report.
- [ ] Audit, pytest, compileall, hash, commit, and push.

### Task 5: Gate C calibration-only downstream cost audit

**Files:**
- Create `scripts/analyze_iotj_method_breakthrough_gate_c.py`
- Create `tests/test_iotj_method_breakthrough_gate_c.py`
- Create under `results/iotj_canonical_v1_method_breakthrough_20260811/gate_c_routing_cost/`

- [ ] RED: primary cost uses calibration only, diagonal zero, off-diagonal clip, and no test argument can reach matrix construction.
- [ ] RED: grouped bootstrap resamples filenames/experiments, not windows.
- [ ] GREEN: force four frozen R84 routes for every C5 calibration row.
- [ ] Lock matrix, then open existing test streams for actual misroute diagnostics.
- [ ] Produce paired grouped-bootstrap CIs with at least 2,000 replicates.
- [ ] Emit Gate-C decision and eight-section report.

### Task 6: Cross-gate conclusion and stop

**Files:**
- Create `docs/experiments/iotj_canonical_v1_final/method_breakthrough_20260811/METHOD_BREAKTHROUGH_DECISION.md`
- Update only the new `MANUSCRIPT_METHOD_CHANGELOG.md`

- [ ] Summarize Gate A/B/C without modifying manuscript prose.
- [ ] Record either `GO_GATE_D` as a future authorized action or `STOP_COST_AWARE`; do not execute D.
- [ ] Verify full relevant pytest, compileall, hashes, git diff, and remote push.
- [ ] Stop without Gate D/E/F.

