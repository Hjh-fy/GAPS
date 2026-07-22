# C5 H8 Runtime-Parity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a versioned, manifest-validated C1/C2->C5 B5/H8 runtime and fail-closed 1360-row HC95/HC90 parity verifier without training or legacy routing.

**Architecture:** `gaps_deploy/c5_h8_bundle.py` owns immutable manifest, hash, schema, asset-role, and workpoint validation. `gaps_deploy/c5_h8_runtime.py` uses that contract with existing `qc_policy.py` and `package_contract.py` to run B5 classifier -> fixed H8/R4 -> deployment-visible risk -> selected QC workpoint. The existing parity validator gains a strict full-field C5-H8 mode.

**Tech Stack:** Python 3, NumPy, PyTorch, pytest, existing deployment modules.

## Global Constraints

- Do not train, refit, re-export, or overwrite B5/C5/QC assets or historical results.
- Accept only ready `iotj.b5_c5_deployment_bundle.v1` bundles with the exact `RUNTIME_ASSET_KEYS`, their SHA-256 values, and the bound parity reference.
- Reject C3, C4, R3aK16, H8+C4, P4, missing/extra roles, bad hash/schema, unknown class, NaN/Inf, bad workpoint, and duplicate/missing key.
- Fixed route: B5 classifier -> predicted class -> fixed H8/R4 -> deployment-visible risk -> HC95 or HC90 -> accept/review/reject. C4 rescue remains disabled.
- HC95 is default; HC90 is an explicit frozen-policy selection. Reuse QC/package primitives; do not call or modify `final_runtime.py`.
- A pass compares exactly 1360 rows and means deployment parity only, not a new training result or claim promotion.

---

### Task 1: Strict bundle loader

**Files:**
- Create: `gaps_deploy/c5_h8_bundle.py`
- Create: `tests/test_c5_h8_bundle.py`

**Interfaces:**
- Consumes: `manifest.json`, `RUNTIME_ASSET_KEYS`, `DeploymentPackageError`.
- Produces: `C5H8BundleError`, immutable `C5H8Bundle`, `load_c5_h8_bundle(bundle_dir: Path) -> C5H8Bundle`.

- [ ] **Step 1: Write failing contract tests**

```python
def test_ready_hashed_bundle_loads_with_hc95(tmp_path: Path) -> None:
    bundle = _write_bundle(tmp_path, workpoints={"HC95": _workpoint()})
    loaded = load_c5_h8_bundle(bundle)
    assert loaded.default_workpoint == "HC95"

@pytest.mark.parametrize("mutation", ["schema", "hash", "forbidden", "missing_role"])
def test_bad_bundle_fails_closed(tmp_path: Path, mutation: str) -> None:
    with pytest.raises(C5H8BundleError):
        load_c5_h8_bundle(_write_bad_bundle(tmp_path, mutation))
```

- [ ] **Step 2: Verify red**

Run: `python -m pytest tests/test_c5_h8_bundle.py -q --basetemp .tmp_pytest_c5_h8_bundle_red`

Expected: FAIL because `gaps_deploy.c5_h8_bundle` does not exist.

- [ ] **Step 3: Implement the loader**

```python
@dataclass(frozen=True)
class C5H8Bundle:
    root: Path
    manifest: Mapping[str, Any]
    asset_paths: Mapping[str, Path]
    parity_reference: Path
    risk_policy: Mapping[str, Any]
    default_workpoint: str

def load_c5_h8_bundle(bundle_dir: Path) -> C5H8Bundle:
    root = Path(bundle_dir).resolve()
    manifest = _read_ready_manifest(root / "manifest.json")
    paths = _verify_manifest_assets(root, manifest)
    reference = _verify_parity_reference(manifest["parity_reference"])
    policy = _load_and_verify_r4_policy(paths["r4_policy"])
    risk = _load_and_verify_risk_policy(paths["qc_risk_policy"])
    return C5H8Bundle(root, manifest, paths, reference, risk, "HC95")
```

Require `HC95` and validate finite ordered thresholds for HC95 plus optional HC90. Reject `FULL` as a deployment workpoint.

- [ ] **Step 4: Verify green**

Run: `python -m pytest tests/test_c5_h8_bundle.py tests/test_b5_c5_bundle_asset_roles.py -q --basetemp .tmp_pytest_c5_h8_bundle_green`

Expected: PASS.

- [ ] **Step 5: Commit**

Run: `git add gaps_deploy/c5_h8_bundle.py tests/test_c5_h8_bundle.py; git commit -m "feat: validate C5 H8 runtime bundles"`

### Task 2: Fixed-H8 runtime and QC adapter

**Files:**
- Create: `gaps_deploy/c5_h8_runtime.py`
- Create: `tests/test_c5_h8_runtime.py`
- Modify: `gaps_deploy/c5_h8_bundle.py`

**Interfaces:**
- Consumes: `C5H8Bundle`, strict B5 checkpoint loader, serialized R4 source/target heads, normalization, QC assets, existing `TwoThresholdDecider`.
- Produces: `C5H8Runtime.from_bundle(bundle_dir, device="cpu", workpoint=None)` and `predict_batch(windows) -> list[dict[str, Any]]`.

- [ ] **Step 1: Write failing runtime tests**

```python
def test_runtime_defaults_hc95_and_emits_h8_only(valid_bundle: Path) -> None:
    row = C5H8Runtime.from_bundle(valid_bundle).predict_batch(np.zeros((1, 100, 8), np.float32))[0]
    assert row["workpoint"] == "HC95"
    assert row["selected_profile"] == "H8_R4"

def test_runtime_rejects_invalid_workpoint_and_window(valid_bundle: Path) -> None:
    with pytest.raises(C5H8RuntimeError, match="workpoint"):
        C5H8Runtime.from_bundle(valid_bundle, workpoint="FULL")
    with pytest.raises(C5H8RuntimeError, match="finite"):
        C5H8Runtime.from_bundle(valid_bundle).predict_batch(np.full((1, 100, 8), np.nan, np.float32))
```

- [ ] **Step 2: Verify red**

Run: `python -m pytest tests/test_c5_h8_runtime.py -q --basetemp .tmp_pytest_c5_h8_runtime_red`

Expected: FAIL because `gaps_deploy.c5_h8_runtime` does not exist.

- [ ] **Step 3: Implement the minimal strict route**

```python
class C5H8Runtime:
    @classmethod
    def from_bundle(cls, bundle_dir: str | Path, device: str = "cpu", workpoint: str | None = None) -> "C5H8Runtime":
        bundle = load_c5_h8_bundle(Path(bundle_dir))
        selected = bundle.select_workpoint(workpoint or bundle.default_workpoint)
        return cls(bundle=bundle, workpoint=selected, device=device)

    def predict_batch(self, windows: np.ndarray) -> list[dict[str, Any]]:
        values = self._validate_windows(windows)
        logits, classes = self._classify(values)
        ppm = self._predict_fixed_h8(values, classes)
        risks = self._deployment_risk(values, logits, classes, ppm)
        return [self._row(cls_id, value, risk) for cls_id, value, risk in zip(classes, ppm, risks)]
```

Use `load_checkpoint_state` and `load_state_dict_strict`; no random fallback. Fail on non-finite logits/features/ppm/risk. Set `auto_output_ppm` to H8 ppm only for `accept`, otherwise `""`.

- [ ] **Step 4: Verify green**

Run: `python -m pytest tests/test_c5_h8_runtime.py tests/test_deploy_qc_fail_closed.py tests/test_deployment_package_contract.py -q --basetemp .tmp_pytest_c5_h8_runtime_green`

Expected: PASS.

- [ ] **Step 5: Commit**

Run: `git add gaps_deploy/c5_h8_runtime.py gaps_deploy/c5_h8_bundle.py tests/test_c5_h8_runtime.py; git commit -m "feat: add C5 fixed H8 runtime"`

### Task 3: Full-field parity validator

**Files:**
- Modify: `scripts/validate_iotj_b5_c5_runtime_parity.py`
- Modify: `tests/test_validate_iotj_b5_c5_runtime_parity.py`

**Interfaces:**
- Consumes: keyed C5-H8 streams with `sample_index`, `pred_class`, `h8_ppm`, `deployment_risk_full`, `qc_decision`, `auto_output_ppm`.
- Produces: `validate_c5_h8_parity(reference_path, runtime_path, workpoint) -> dict[str, Any]`.

- [ ] **Step 1: Write failing mismatch tests**

```python
def test_c5_h8_parity_rejects_risk_and_auto_output_mismatch(tmp_path: Path) -> None:
    reference, runtime = _write_c5_h8_rows(tmp_path, changed_risk=True, changed_auto=True)
    report = validate_c5_h8_parity(reference, runtime, "HC95")
    assert report["status"] == "failed"
    assert report["risk_mismatches"] == report["auto_output_mismatches"] == 1
```

- [ ] **Step 2: Verify red**

Run: `python -m pytest tests/test_validate_iotj_b5_c5_runtime_parity.py -q --basetemp .tmp_pytest_c5_h8_parity_red`

Expected: FAIL because `validate_c5_h8_parity` does not exist.

- [ ] **Step 3: Add strict mode without weakening legacy mode**

```python
C5_H8_FIELDS = ("sample_index", "pred_class", "h8_ppm", "deployment_risk_full", "qc_decision", "auto_output_ppm")

def validate_c5_h8_parity(reference_path: Path, runtime_path: Path, workpoint: str) -> dict[str, Any]:
    _require_workpoint(workpoint)
    reference = _read_indexed_fields(reference_path, C5_H8_FIELDS)
    runtime = _read_indexed_fields(runtime_path, C5_H8_FIELDS)
    return _compare_c5_h8_rows(reference, runtime, workpoint)
```

Reject workpoints other than HC95/HC90; preserve `validate_parity` unchanged for existing five-field references.

- [ ] **Step 4: Verify green**

Run: `python -m pytest tests/test_validate_iotj_b5_c5_runtime_parity.py tests/test_iotj_b5_c5_canonical_replay.py -q --basetemp .tmp_pytest_c5_h8_parity_green`

Expected: PASS.

- [ ] **Step 5: Commit**

Run: `git add scripts/validate_iotj_b5_c5_runtime_parity.py tests/test_validate_iotj_b5_c5_runtime_parity.py; git commit -m "feat: verify C5 H8 runtime parity"`

### Task 4: Frozen B5 execution and audit-safe output

**Files:**
- Create: `scripts/run_iotj_b5_c5_h8_parity.py`
- Create: `tests/test_run_iotj_b5_c5_h8_parity.py`
- Create after green only: `results/iotj_b5_c5_deployment_p1_20260722/c5_h8_runtime_parity_hc95/`

**Interfaces:**
- Consumes: frozen bundle, canonical keyed windows, workpoint-matched frozen reference.
- Produces: non-overwriting runtime stream and JSON report, nonzero failure for any contract/parity mismatch.

- [ ] **Step 1: Write failing non-overwrite test**

```python
def test_runner_refuses_nonempty_output(tmp_path: Path) -> None:
    output = tmp_path / "existing"; output.mkdir(); (output / "old.json").write_text("{}")
    with pytest.raises(FileExistsError, match="overwrite"):
        run_c5_h8_parity(bundle_dir=tmp_path, input_path=tmp_path / "x", reference_path=tmp_path / "r", output_dir=output)
```

- [ ] **Step 2: Verify red**

Run: `python -m pytest tests/test_run_iotj_b5_c5_h8_parity.py -q --basetemp .tmp_pytest_c5_h8_runner_red`

Expected: FAIL because runner module does not exist.

- [ ] **Step 3: Implement runner**

```python
def run_c5_h8_parity(*, bundle_dir: Path, input_path: Path, reference_path: Path, output_dir: Path, workpoint: str = "HC95") -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite parity output: {output_dir}")
    runtime = C5H8Runtime.from_bundle(bundle_dir, workpoint=workpoint)
    rows = runtime.predict_batch(load_keyed_windows(input_path))
    write_runtime_stream(output_dir / "runtime_rows.csv", rows)
    report = validate_c5_h8_parity(reference_path, output_dir / "runtime_rows.csv", workpoint)
    if report["status"] != "equivalent": raise RuntimeError("C5 H8 runtime parity failed")
    return report
```

Require exactly 1360 unique input IDs. Record workpoint, B5/C1/C2/C5 provenance, code commit, and bundle/reference/checkpoint hashes. Missing provenance yields `blocked`, never `equivalent`.

- [ ] **Step 4: Verify focused tests, canonical replay, then formal HC95 parity**

Run: `python -m pytest tests/test_c5_h8_bundle.py tests/test_c5_h8_runtime.py tests/test_validate_iotj_b5_c5_runtime_parity.py tests/test_run_iotj_b5_c5_h8_parity.py -q --basetemp .tmp_pytest_c5_h8_final`

Expected: PASS.

Run: `python scripts/verify_iotj_b5_c5_canonical_replay.py --root results/iotj_b5_c5_deployment_p1_20260722`

Expected: `status=ready`, `runtime_rows=1360`.

Run: `python scripts/run_iotj_b5_c5_h8_parity.py --bundle results/iotj_b5_c5_deployment_p1_20260722/bundle_candidate --workpoint HC95 --output-dir results/iotj_b5_c5_deployment_p1_20260722/c5_h8_runtime_parity_hc95`

Expected: `status=equivalent`, 1360 rows, zero mismatches. HC90 runs only with an explicitly bound HC90 reference and never replaces HC95.

- [ ] **Step 5: Commit code and evidence separately**

Run: `git add scripts/run_iotj_b5_c5_h8_parity.py tests/test_run_iotj_b5_c5_h8_parity.py; git commit -m "feat: run B5 C5 H8 parity"`

Run: `git add results/iotj_b5_c5_deployment_p1_20260722/c5_h8_runtime_parity_hc95; git commit -m "test: record B5 C5 HC95 runtime parity"`

## Plan self-review

- Task 1 covers schema, assets, hashes, forbidden legacy tokens, and workpoint contracts.
- Task 2 covers strict B5 loading, fixed R4/H8 route, risk, QC reuse, finite-input rejection, and HC95/HC90 behavior.
- Task 3 covers row key, class, H8, risk, QC, and auto-output parity while preserving the existing validator.
- Task 4 covers non-overwriting B5 execution and provenance/evidence boundaries; no task trains or alters frozen results.
