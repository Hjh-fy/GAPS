# B5 C5 Deployment and Pi Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build one frozen B5 C1/C2-to-C5 deployment bundle, prove 1360 C5 test rows reproduce the offline reference, and measure real Pi/PC inference latency and resources.

**Architecture:** A new C5-only builder consumes the frozen B5 classifier and formal R4/H8/HC90 assets. The same runtime entry point drives offline parity and device benchmark, so the measured runtime cannot drift from the validated one.

**Tech Stack:** Python, NumPy, PyTorch, scikit-learn, `gaps_deploy`, pytest, SSH/SCP.

## Global Constraints

- B5 is the only deployment mainline; B2 remains separate representative-system evidence.
- C1/C2 -> C5 only. C3/C4, R3aK16, H8+C4 rescue and P4 leakage paths are forbidden.
- Runtime chain: classifier route -> H1/H2/H3 -> C5-calibrated R4/H8 -> HC90 QC.
- Bundle includes model/regression/QC/schema/class map/normalisation/manifest/SHA-256.
- Exactly 1360 parity rows. Class/profile/QC exact; max absolute ppm delta <= 1e-6.
- Benchmark both Pi and PC: batch 1 primary, batch 32 auxiliary; 30 warm-up plus at least 100 measurements; report load/classification/regression/QC/total p50/p95/p99, throughput, peak/steady RSS and average/peak CPU.
- No training run or overwrite of existing evidence. Missing artifact, schema mismatch or parity failure stops the workflow.

### Task 1: Bind deployment inputs — code complete; asset rebuild pending

**Files:** create `scripts/inspect_b5_c5_deployment_inputs.py`; create `tests/test_inspect_b5_c5_deployment_inputs.py`; create `results/iotj_b5_c5_deployment_p1_20260721/input_audit.json`.

- [x] Write failing test and implement the audit (`6d3b8a9`).

```python
def test_input_audit_rejects_legacy_path(tmp_path):
    result = inspect_inputs(tmp_path)
    assert result["status"] == "blocked"
    assert "legacy_forbidden" in result["reasons"]
```

- [ ] Run `python -m pytest -q tests/test_inspect_b5_c5_deployment_inputs.py`; expect missing `inspect_inputs` failure.
- [ ] Implement `inspect_inputs(repo_root: Path) -> dict[str, object]`: bind only explicit B5/C5 inputs; hash them; return `blocked` for missing/legacy fields without guessing path semantics.
- [x] Run the audit. It is correctly `blocked`: current B5 classifier SHA-256 is bound, while the ten C5 deployment assets must be rebuilt against that classifier instead of being borrowed from a historical classifier with a different SHA-256.
- [ ] Commit only this task's files with `feat: audit B5 C5 deployment inputs`.

### Task 2: Build C5-only B5 bundle — packager complete; source assets pending

**Files:** create `scripts/build_iotj_b5_c5_deployment_bundle.py`; create `tests/test_build_iotj_b5_c5_deployment_bundle.py`; create `results/iotj_b5_c5_deployment_p1_20260721/bundle/manifest.json`.

- [x] Write failing test and implement the packager (`64abcb7`).

```python
def test_bundle_rejects_legacy_input_audit(tmp_path):
    with pytest.raises(ValueError, match="forbidden legacy"):
        build_bundle(legacy_input_audit, tmp_path / "bundle")
```

- [ ] Run `python -m pytest -q tests/test_build_iotj_b5_c5_deployment_bundle.py`; expect missing `build_bundle` failure.
- [ ] Implement `build_bundle(input_audit: Path, output_dir: Path) -> dict[str, object]`: refuse non-ready audit, copy only bound B5/C5 assets, emit schema/class map/QC/normalisation/file hashes.
- [x] Verify the packager rejects legacy/non-ready input and rechecks source hashes before copying. Real bundle creation remains blocked until the C5-only rebuild produces a `ready` audit.

**Current external state (2026-07-21):** The C5-only rebuild controller was stopped before it could create an ECS output because the local machine could not reach the configured ECS SSH port (`121.40.139.213:22`). No remote partial output was created. The fixed R3aK16 checkpoint was recovered with SHA-256 `790fc6ff…0f83` only for reproducing historical offline comparison fields; the bundle audit/packager explicitly forbid it from the final deployment package.
- [ ] Commit only this task's files with `feat: build B5 C5 deployment bundle`.

### Task 3: Gate exact offline/runtime parity

**Files:** create `scripts/validate_iotj_b5_c5_runtime_parity.py`; create `tests/test_validate_iotj_b5_c5_runtime_parity.py`; create `results/iotj_b5_c5_deployment_p1_20260721/runtime_parity_report.json`; create `results/iotj_b5_c5_deployment_p1_20260721/runtime_parity_rows.csv`.

- [ ] Write failing test:

```python
def test_parity_rejects_one_qc_mismatch(tmp_path):
    report = validate_parity(bundle, reference_with_one_changed_qc)
    assert report["status"] == "failed"
    assert report["qc_decision_mismatches"] == 1
```

- [ ] Run `python -m pytest -q tests/test_validate_iotj_b5_c5_runtime_parity.py`; expect missing `validate_parity` failure.
- [ ] Implement `validate_parity(bundle: Path, reference: Path) -> dict[str, object]`: require 1360 unique keys; compare class/profile/QC exactly and ppm delta in float64; fail above 1e-6.
- [ ] Run parity on frozen B5 C5 reference. Only `status=equivalent` unlocks device benchmark.
- [ ] Commit only this task's files with `feat: gate B5 C5 runtime parity`.

### Task 4: Execute actual Pi and PC benchmark

**Files:** create `scripts/benchmark_iotj_b5_c5_runtime.py`; create `tests/test_benchmark_iotj_b5_c5_runtime.py`; create `results/iotj_b5_c5_deployment_p1_20260721/edge_runtime_benchmark.csv`; create `results/iotj_b5_c5_deployment_p1_20260721/system_resource_summary.csv`.

- [ ] Write failing test:

```python
def test_benchmark_rejects_fewer_than_100_measurements(tmp_path):
    with pytest.raises(ValueError, match="at least 100"):
        benchmark_runtime(bundle, "cpu", warmup=30, repeats=99)
```

- [ ] Run `python -m pytest -q tests/test_benchmark_iotj_b5_c5_runtime.py`; expect missing `benchmark_runtime` failure.
- [ ] Implement `benchmark_runtime(bundle, device, warmup=30, repeats=100)`: record load/classification/regression/QC/total stage latencies, throughput, RSS, CPU and device/OS/Python/PyTorch metadata for batch 1 and 32.
- [ ] Transfer the immutable parity-approved bundle to Pi, run Pi and PC commands, recover raw logs and summarize. Never use the training resource sampler as inference data.
- [ ] Commit only this task's files with `feat: benchmark B5 C5 runtime on edge devices`.

### Task 5: Publish only parity-gated deployment evidence

**Files:** modify `results/iotj_advisor_metrics_20260721/build_advisor_workbook_v3.mjs`; create `results/iotj_b5_c5_deployment_p1_20260721/deployment_summary.md`; create `results/iotj_b5_c5_deployment_p1_20260721/claim_to_evidence_map.md`.

- [ ] Write failing test that `build_summary(parity_status="failed")` raises `ValueError("parity must be equivalent before publishing deployment metrics")`.
- [ ] Run the focused test; expect missing `build_summary` failure.
- [ ] Implement `build_summary` so only `equivalent` parity and measured Pi/PC CSV rows populate the workbook; otherwise retain `unknown`/`blocked` cells.
- [ ] Run all new deployment tests, inspect formula errors, inspect key workbook ranges, render each sheet if the runtime supports rendering, and run `git diff --check`.
- [ ] Commit with `docs: report B5 C5 deployment evidence`.

## Self-review

- Tasks 1-2 freeze and package B5/C5 assets; Task 3 blocks unsafe device measurement; Task 4 measures the actual Pi/PC runtime; Task 5 prevents unsupported manuscript values.
- No task changes classifier training, regression fitting, data splits, loss or legacy deployment routes.
- The user approved inline execution in this worktree. Do not dispatch subagents because every task depends on the same evolving runtime contract.
