# Canonical-v1 Final Scientific Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the eight canonical-v1 scientific claims with audited existing evidence, only the minimum missing comparator/robustness runs, and a final submission-readiness verdict.

**Architecture:** The work is split into an immutable Phase-0 audit, read-only analyses, preregistered missing experiments, and a final claim gate. Existing canonical artifacts are never edited. New code writes to a new `results/iotj_canonical_v1_scientific_validation_20260809/` root and publishes only compact evidence to `docs/experiments/iotj_canonical_v1_final/`.

**Tech Stack:** Python 3, PyTorch/Flower, NumPy/pandas/scikit-learn, pytest, existing GAPS canonical runners, PowerShell/SSH for the frozen three-machine topology.

## Global Constraints

- Freeze `HZ5_MEAN_W10S`, C1/C2 source, C3/C4/C5 target, A4, 25 rounds, local_epochs=1, R84_FED_H1, seed42.
- Never use target test for split, hyperparameter, threshold, stopping, checkpoint, or method selection.
- Never delete C5 methane 225 ppm repeat1.
- Do not add FedNova, FedDyn, FedBN, FedProto, Ditto, a new transformer, regression head, QC policy, or preprocessing candidate.
- Existing canonical results, checkpoints, manifests, thresholds, and deployment package are read-only.
- Preserve raw-file grouping in uncertainty calculations; do not use window-i.i.d. bootstrap as the only CI.
- Commit a pre-run freeze before any missing experiment starts.

---

### Task 1: Phase-0 claim/evidence audit and missing-run freeze

**Files:**
- Create: `results/iotj_canonical_v1_scientific_validation_20260809/FINAL_CLAIM_EVIDENCE_AUDIT.md`
- Create: `docs/experiments/iotj_canonical_v1_final/FINAL_CLAIM_EVIDENCE_AUDIT.md`
- Create: `docs/experiments/iotj_canonical_v1_final/A0T_REQUIRED_RUN_PLAN.md`
- Create: `docs/experiments/iotj_canonical_v1_final/SCIENTIFIC_VALIDATION_PRE_RUN_FREEZE.json`
- Create: `docs/experiments/iotj_canonical_v1_final/SCIENTIFIC_VALIDATION_EXPERIMENT_MATRIX.csv`
- Test: `tests/test_iotj_canonical_v1_scientific_validation.py`

**Interfaces:**
- Consumes: final evidence bundle, canonical manifests, historical comparator manifests.
- Produces: `audit_existing_evidence()` and a frozen list of only missing executable configurations.

- [ ] **Step 1: Write the failing audit-contract test**

```python
def test_phase0_marks_legacy_comparators_noncanonical():
    audit = audit_existing_evidence(ROOT)
    assert audit["FedAvg"] == "CANONICAL_COMPARATOR_MISSING"
    assert audit["A0T"] == "BLOCKED_NOT_RUN"
```

- [ ] **Step 2: Run the test and verify it fails**

Run: `python -m pytest -q tests/test_iotj_canonical_v1_scientific_validation.py`

- [ ] **Step 3: Implement manifest-only audit and fail-closed freeze generation**

The audit must compare dataset hash, preprocessing, source/target roles, backbone, rounds, local epochs, seed, checkpoint policy, and evaluation scope. Legacy matches remain historical even when method names match.

- [ ] **Step 4: Run the audit test and inspect every planned row**

Run: `python -m pytest -q tests/test_iotj_canonical_v1_scientific_validation.py`

- [ ] **Step 5: Commit the Phase-0 audit and pre-run freeze**

```powershell
git add scripts tests docs/experiments/iotj_canonical_v1_final
git commit -m "audit: freeze canonical scientific validation"
git push origin codex/iotj-final-classification-le1
```

### Task 2: Routing-error propagation analysis

**Files:**
- Create: `scripts/analyze_iotj_canonical_v1_scientific_claims.py`
- Create: `results/iotj_canonical_v1_scientific_validation_20260809/routing_scope_summary.csv`
- Create: `results/iotj_canonical_v1_scientific_validation_20260809/ROUTING_ERROR_PROPAGATION_ANALYSIS.md`
- Test: `tests/test_iotj_canonical_v1_scientific_claims.py`

**Interfaces:**
- Consumes: canonical `test_records.csv`, per-gas summaries, and route-decomposition artifacts.
- Produces: `routing_scope_summary(records) -> list[dict]` for ALL/C3/C4/C5 and per-gas rows.

- [ ] **Step 1: Write a failing synthetic routing-gap test**

```python
def test_routing_gap_definitions():
    row = routing_gap_row(s_all=12.0, s_cc=9.0, oracle=8.0)
    assert row["routing_gap"] == 3.0
    assert row["oracle_gap"] == 1.0
```

- [ ] **Step 2: Run the test and verify it fails**

- [ ] **Step 3: Implement S_ALL/S_CC/oracle aggregation without retraining**

- [ ] **Step 4: Generate ALL, target, and per-gas outputs; retain empty/low-N slices**

- [ ] **Step 5: Run tests and commit**

### Task 3: FedRidge raw-file-grouped bootstrap

**Files:**
- Modify: `scripts/analyze_iotj_canonical_v1_scientific_claims.py`
- Create: `results/iotj_canonical_v1_scientific_validation_20260809/fedridge_bootstrap_summary.csv`
- Create: `results/iotj_canonical_v1_scientific_validation_20260809/FEDRIDGE_GROUPED_BOOTSTRAP.md`
- Test: `tests/test_iotj_canonical_v1_scientific_claims.py`

**Interfaces:**
- Consumes paired R83/R84 predictions with target/raw filename/gas group IDs.
- Produces `grouped_paired_rmse_bootstrap(..., repeats=5000, seed=20260809)` with DeltaRMSE defined as RMSE84 minus RMSE83.

- [ ] **Step 1: Write a failing test proving whole groups are resampled and pairing is preserved**
- [ ] **Step 2: Run the test and verify it fails**
- [ ] **Step 3: Implement deterministic 5,000-replicate grouped bootstrap**
- [ ] **Step 4: Generate ALL, C3/C4/C5, and per-gas CIs**
- [ ] **Step 5: Test, audit CI interpretation, and commit**

### Task 4: QC risk/coverage and error-capture validation

**Files:**
- Modify: `scripts/analyze_iotj_canonical_v1_scientific_claims.py`
- Create: `results/iotj_canonical_v1_scientific_validation_20260809/qc_risk_coverage_final.csv`
- Create: `results/iotj_canonical_v1_scientific_validation_20260809/qc_error_capture_summary.csv`
- Create: `results/iotj_canonical_v1_scientific_validation_20260809/QC_CLAIM_VALIDATION.md`
- Test: `tests/test_iotj_canonical_v1_scientific_claims.py`

**Interfaces:**
- Consumes fixed canonical QC scores, thresholds, true errors, and route correctness.
- Produces fixed descending-risk capture curves, misroute/>40 ppm capture, and a documented AURC definition.

- [ ] **Step 1: Write failing capture-rate and AURC tests**
- [ ] **Step 2: Run tests and verify failure**
- [ ] **Step 3: Implement read-only fixed-score calculations**
- [ ] **Step 4: Generate outputs without changing thresholds**
- [ ] **Step 5: Test and commit**

### Task 5: Calibration, system, and figure/table claim maps

**Files:**
- Create: `docs/experiments/iotj_canonical_v1_final/CALIBRATION_CLAIM_RESTRICTION.md`
- Create: `docs/experiments/iotj_canonical_v1_final/SYSTEM_CLAIM_VALIDATION.md`
- Create: `docs/experiments/iotj_canonical_v1_final/FINAL_FIGURE_TABLE_SCIENTIFIC_MAP.md`

**Interfaces:**
- Consumes: deployment/package hashes, Pi 5 environment/summary, model-size audit, protocol and communication logs.
- Produces: claim-safe language and figure/table evidence sources. No new benchmark or plot.

- [ ] **Step 1: Verify the package/Pi hash chain and parameter semantics**
- [ ] **Step 2: Recompute communication formulas from 22,765 actual parameters where logs support it**
- [ ] **Step 3: Record unknown timing fields instead of inventing them**
- [ ] **Step 4: Map every proposed figure/table to canonical CSV/JSON/script/hash**
- [ ] **Step 5: Run Markdown/path checks and commit**

### Task 6: Deterministic strict grouped non-overlap split

**Files:**
- Create: `scripts/build_iotj_canonical_v1_strict_nonoverlap.py`
- Create: `docs/experiments/iotj_canonical_v1_final/STRICT_NON_OVERLAP_PROTOCOL.md`
- Create: `docs/experiments/iotj_canonical_v1_final/strict_non_overlap_split_manifest.csv`
- Test: `tests/test_iotj_canonical_v1_strict_nonoverlap.py`

**Interfaces:**
- Consumes canonical metadata/raw identities only; does not edit `dataset/iotj_canonical_v1`.
- Produces a new strict dataset root with zero exact-window, raw-time, and (if file grouping is used) raw-file overlap.

- [ ] **Step 1: Write failing overlap, coverage, determinism, and target-test-independent tests**
- [ ] **Step 2: Run tests and verify failure**
- [ ] **Step 3: Implement a single preregistered grouped split rule**
- [ ] **Step 4: Build and hash the new dataset; fail closed on label-coverage loss**
- [ ] **Step 5: Commit the split protocol/manifest before model execution**

### Task 7: Execute only the missing canonical runs

**Files:**
- Reuse: `scripts/run_iotj_canonical_v1_a0t.py`
- Reuse/extend: `scripts/run_iotj_canonical_v1_classification.py`
- Reuse: `scripts/run_iotj_canonical_v1_r84.py`
- Create: `scripts/run_iotj_canonical_v1_missing_comparators.py`
- Test: comparator and A0T protocol tests.

**Interfaces:**
- Consumes the committed Phase-0 matrix and strict split manifest.
- Produces fixed endpoints for canonical FedAvg, FedProx, canonical-SGD SCAFFOLD, MMD, A0T, and strict-split A4/R84 only.

- [ ] **Step 1: Write protocol tests for method-specific target-information gates and optimizer identity**
- [ ] **Step 2: Run tests and verify failure for the new orchestrator**
- [ ] **Step 3: Implement fail-closed commands with no search and fixed seed42**
- [ ] **Step 4: Run pre-execution audits and commit exact commands/hashes**
- [ ] **Step 5: Execute A0T C3/C4/C5, strict A4/R84 C3/C4/C5, and the minimal canonical comparator matrix**
- [ ] **Step 6: Evaluate only after all fixed endpoints exist; never replace a failed/low result**
- [ ] **Step 7: Run SCAFFOLD sanity audit and commit compact results**

### Task 8: Final scientific validation report and reproducibility closure

**Files:**
- Create: `results/iotj_canonical_v1_scientific_validation_20260809/FINAL_SCIENTIFIC_VALIDATION_REPORT.md`
- Create: `docs/experiments/iotj_canonical_v1_final/FINAL_SCIENTIFIC_VALIDATION_REPORT.md`
- Create: `docs/experiments/iotj_canonical_v1_final/scientific_validation_sha256_index.json`

**Interfaces:**
- Consumes all audited Phase-0 through Phase-5 outputs.
- Produces the ten-question decision gate with statuses limited to PASS, PASS_WITH_LIMITATION, BLOCKED, NOT_REQUIRED.

- [ ] **Step 1: Assemble the claim/evidence/protocol/canonical/result/risk/status table**
- [ ] **Step 2: Answer all ten required questions without stronger wording than evidence permits**
- [ ] **Step 3: Run relevant pytest and `python -m compileall`**
- [ ] **Step 4: Verify dataset/checkpoint/split/package/new-artifact SHA256 values**
- [ ] **Step 5: Commit and push only compact code/docs/tables, not large checkpoints/raw predictions**

## Self-review

- Spec coverage: all eight claims, strict non-overlap, canonical comparators, A0T, bootstrap, QC, calibration, system, figure map, final gate, tests, hashes, commit/push are mapped above.
- No target-test-driven selection is permitted in any task.
- The canonical main results are never replaced by strict-split sensitivity results.
- The only planned new methods are explicitly required comparators; no algorithm expansion is included.
