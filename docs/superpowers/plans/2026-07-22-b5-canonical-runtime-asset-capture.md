# B5 Canonical Runtime Asset Capture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persist the exact B5 canonical R4 and H23 fitted objects, establish 1360-row runtime parity, and then measure the parity-approved C5 bundle on PC and Raspberry Pi.

**Architecture:** R4 and H23 evaluators retain the model objects that already produce their prediction streams and optionally serialize those same objects. The regression-suite controller passes explicit output paths, QC consumes the regenerated streams, and the existing bundle/parity tooling remains fail-closed. Device benchmarking is an external stage unlocked only by an equivalent parity report.

**Tech Stack:** Python, NumPy, scikit-learn, PyTorch, pytest, SSH/SCP, existing `gaps_deploy` tools.

## Global Constraints

- B5 C1/C2 -> C5 is the sole deployment mainline; B2 remains separate evidence.
- No model, loss, data split, optimizer, Flower configuration, or training hyperparameter changes.
- C3/C4, R3aK16, H8+C4 rescue, P4, and C5-test fitting are forbidden from the bundle.
- The runtime must reproduce exactly 1360 rows: class/profile/QC exactly and max absolute ppm delta <= `1e-6`.
- Existing results are immutable; all rerun outputs are placed in a new dated result directory.
- Pi/PC benchmark runs only after an equivalent parity report.

---

### Task 1: Expose exact final H23 objects without changing existing callers

**Files:**
- Modify: `run_h2_3_plus_fusion_profile.py:249-393`
- Modify: `scripts/run_iotj_c5_h23_plus.py:74-150`
- Modify: `tests/test_iotj_c5_regression_suite.py`

**Interfaces:**
- `fit_ridge_family(..., return_final_models: bool = False)` returns the existing three-tuple, or a fourth `dict[tuple[str, int], RidgeHead]` when requested.
- `fit_mlp_family(..., return_final_models: bool = False)` returns the existing three-tuple, or a fourth `dict[tuple[str, int], MLPHead]` when requested.
- `run_iotj_c5_h23_plus.run(args)` accepts optional `args.runtime_reference_output` and writes a C5-only JSON reference from the exact final models used for test prediction.

- [ ] **Step 1: Write the failing tests**

```python
def test_h23_family_can_return_the_same_final_models_used_for_test_predictions():
    values = fit_ridge_family(..., return_final_models=True)
    _validation, test_rows, _audit, models = values
    assert set(models) == {("C5", 0), ("C5", 1), ("C5", 2), ("C5", 3)}
    assert models[("C5", 0)].predict([test_rows[0]])[0] == pytest.approx(
        test_rows[0]["regfeat_ridge_ppm"]
    )
```

```python
def test_h23_runtime_export_is_optional_and_c5_only(tmp_path):
    result = run(namespace_with_runtime_reference_output(tmp_path / "h23.json"))
    payload = json.loads((tmp_path / "h23.json").read_text())
    assert payload["h23_reference_policy"]["target_client"] == "C5"
    assert len(payload["h23_reference_policy"]["mlp_models"]) == 4
```

- [ ] **Step 2: Run focused tests and verify RED**

Run: `python -m pytest -q tests/test_iotj_c5_regression_suite.py -k "h23 and final_models"`

Expected: failure because `return_final_models` and `runtime_reference_output` do not exist.

- [ ] **Step 3: Implement the minimal optional return and export**

```python
if return_final_models:
    return validation_rows, test_rows, fit_audit, final_models
return validation_rows, test_rows, fit_audit
```

In `run_iotj_c5_h23_plus.py`, request `return_final_models=True`, serialize each `MLPHead` with `serialize_mlp_head`, each Ridge with `to_json()`, and call `build_h23_payload(...)`. Do not re-fit from CSV/audit files.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run: `python -m pytest -q tests/test_iotj_c5_regression_suite.py -k "h23 and final_models"`

Expected: PASS.

- [ ] **Step 5: Commit Task 1**

```bash
git add run_h2_3_plus_fusion_profile.py scripts/run_iotj_c5_h23_plus.py tests/test_iotj_c5_regression_suite.py
git commit -m "feat: capture canonical H23 runtime assets"
```

### Task 2: Export the exact in-process R4 policy and wire it through the suite

**Files:**
- Modify: `run_source_augmented_target_ridge_eval.py:1-430`
- Modify: `scripts/run_iotj_c5_regression_suite.py:12-160`
- Modify: `tests/test_source_augmented_target_ridge_validation.py`
- Modify: `tests/test_iotj_c5_regression_suite.py`

**Interfaces:**
- `build_r4_policy_payload(ridge_models, mlp_models, shared_model, target_models, feature_names, classifier_sha256) -> dict[str, Any]` builds only C5/all-four-route R4 assets.
- The source-augmented evaluator adds optional `--runtime-policy-output` and `--classifier-checkpoint`; the pair is required together.
- `build_suite_commands(...)` supplies `--runtime-policy-output h8_no_rescue/r4_policy.json` and the frozen B5 classifier path.

- [ ] **Step 1: Write the failing tests**

```python
def test_r4_runtime_policy_serializes_the_same_fitted_models(tmp_path):
    payload = build_r4_policy_payload(
        ridge_models=constant_ridges(), mlp_models=constant_mlps(),
        shared_model=constant_shared(), target_models=constant_targets(),
        feature_names=["x"], classifier_sha256="a" * 64,
    )
    assert payload["source_aug_target_ridge_policy"]["switch_rule"]["class_ids"] == [0, 1, 2, 3]
    assert len(payload["source_aug_target_ridge_policy"]["models"]) == 4
```

```python
def test_regression_suite_passes_in_process_runtime_asset_paths(tmp_path):
    commands = build_suite_commands(..., output_root=tmp_path)
    assert "--runtime-policy-output" in commands[2]
    assert "--runtime-reference-output" in commands[1]
```

- [ ] **Step 2: Run focused tests and verify RED**

Run: `python -m pytest -q tests/test_source_augmented_target_ridge_validation.py tests/test_iotj_c5_regression_suite.py -k "runtime_policy or runtime_asset_paths"`

Expected: failure because the policy builder and flags are absent.

- [ ] **Step 3: Implement the exact-object export**

Use the already-fitted `ridge_models`, `mlp_models`, `shared_model`, and `aug_models` in `run_source_augmented_target_ridge_eval.main`; serialize them before return. Add a SHA-256 helper for the supplied classifier path and reject a one-sided optional argument. Do not call `fit_source_heads`, `fit_select_refit`, or any export script a second time during export.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run: `python -m pytest -q tests/test_source_augmented_target_ridge_validation.py tests/test_iotj_c5_regression_suite.py -k "runtime_policy or runtime_asset_paths"`

Expected: PASS.

- [ ] **Step 5: Commit Task 2**

```bash
git add run_source_augmented_target_ridge_eval.py scripts/run_iotj_c5_regression_suite.py tests/test_source_augmented_target_ridge_validation.py tests/test_iotj_c5_regression_suite.py
git commit -m "feat: capture canonical R4 runtime policy"
```

### Task 3: Rebuild one B5 canonical regression/QC evidence directory on ECS

**Files:**
- Create: `results/iotj_b5_c5_deployment_p1_20260722/canonical_replay_manifest.json`
- Create: `results/iotj_b5_c5_deployment_p1_20260722/r4_policy.json`
- Create: `results/iotj_b5_c5_deployment_p1_20260722/h23_reference.json`
- Create: `results/iotj_b5_c5_deployment_p1_20260722/high_coverage_qc/`
- Modify: `docs/experiments/iotj_system_experiment_notebook.md`

**Interfaces:**
- ECS command consumes frozen B5 checkpoint, unchanged `client_data_c1234src_c5tgt_2080_timeaware_60_170_window_fullgrid`, R3aK16 only as historical input-builder reference, and emits R4/H23 assets inside the new output root.
- The recovered manifest records command, source archive SHA-256, checkpoint SHA-256, input hashes, and output hashes.

- [ ] **Step 1: Write the failing evidence-contract test**

```python
def test_canonical_replay_requires_both_in_process_runtime_assets(tmp_path):
    with pytest.raises(FileNotFoundError, match="r4_policy"):
        verify_canonical_replay(tmp_path)
```

- [ ] **Step 2: Run it and verify RED**

Run: `python -m pytest -q tests/test_iotj_b5_c5_canonical_replay.py`

Expected: FAIL because `verify_canonical_replay` is absent.

- [ ] **Step 3: Implement the narrow evidence verifier and run ECS replay**

`verify_canonical_replay(root: Path)` must require nonempty `h23_plus`, `h8_no_rescue/r4_policy.json`, `h23_plus/h23_reference.json`, HC90 QC JSON assets, 1360 unique B5 test rows, and no forbidden runtime asset path.

Run on ECS:

```bash
/root/gaps_env/bin/python scripts/run_iotj_c5_regression_suite.py \
  --classifier-id B5canonical --classifier-checkpoint results/iotj_reg_checkpoints_20260713/B5canonical.pth \
  --regression-checkpoint results/iotj_reg_checkpoints_20260713/R3aK16_source_regression.pt \
  --data-root dataset/client_data_c1234src_c5tgt_2080_timeaware_60_170_window_fullgrid \
  --output-root results/iotj_b5_c5_deployment_p1_20260722/canonical_replay --device cuda --seed 42 --n-random 1000
```

- [ ] **Step 4: Recover outputs and verify**

Run: `python -m pytest -q tests/test_iotj_b5_c5_canonical_replay.py`

Expected: PASS with hashes and required asset roles present.

- [ ] **Step 5: Commit code/docs only**

```bash
git add scripts/verify_iotj_b5_c5_canonical_replay.py tests/test_iotj_b5_c5_canonical_replay.py docs/experiments/iotj_system_experiment_notebook.md
git commit -m "feat: verify canonical B5 deployment replay"
```

### Task 4: Build, gate, and benchmark only an equivalent bundle

**Files:**
- Create: `results/iotj_b5_c5_deployment_p1_20260722/runtime_parity_rows.csv`
- Create: `results/iotj_b5_c5_deployment_p1_20260722/runtime_parity_report.json`
- Create: `results/iotj_b5_c5_deployment_p1_20260722/edge_runtime_benchmark.csv`
- Create: `results/iotj_b5_c5_deployment_p1_20260722/system_resource_summary.csv`
- Modify: `docs/experiments/iotj_system_experiment_notebook.md`

**Interfaces:**
- Existing `validate_iotj_b5_c5_runtime_parity.py` consumes two CSVs with `sample_index,pred_class,selected_profile,qc_decision,final_ppm`.
- Existing benchmark entry point must reject a missing or failed parity report.

- [ ] **Step 1: Write the failing benchmark-unlock test**

```python
def test_benchmark_refuses_non_equivalent_parity(tmp_path):
    with pytest.raises(ValueError, match="equivalent"):
        require_equivalent_parity(tmp_path / "failed.json")
```

- [ ] **Step 2: Run it and verify RED**

Run: `python -m pytest -q tests/test_benchmark_iotj_b5_c5_runtime.py -k parity`

Expected: FAIL because the guard is absent.

- [ ] **Step 3: Implement guard, build bundle, and run parity**

Use the canonical replay R4/H23/QC assets to build a fresh bundle. Run the actual B5 runtime on all C5 test windows; validate against the new HC90 external reference. Stop if any field differs or ppm delta exceeds `1e-6`.

- [ ] **Step 4: On equivalent parity only, run the real benchmark**

Transfer the immutable bundle to Pi. On PC and Pi, run batch 1 and batch 32 with 30 warm-up and 100 measured repetitions. Record model-load, classification, R4 regression, QC, end-to-end p50/p95/p99, throughput, steady/peak RSS, average/peak CPU, device, OS, Python, and PyTorch versions.

- [ ] **Step 5: Verify and commit code/docs only**

Run: `python -m pytest -q tests/test_validate_iotj_b5_c5_runtime_parity.py tests/test_benchmark_iotj_b5_c5_runtime.py`

Expected: PASS. Commit scripts/tests/notebook summaries; keep raw benchmark traces and bulky assets in local/ECS evidence storage.

## Self-review

- Task 1 avoids post-hoc H23 refit by retaining the exact final models used for the stream.
- Task 2 avoids post-hoc R4 refit and binds the same B5 classifier SHA-256.
- Task 3 is a regression/QC evaluation replay, not Flower training or an algorithm change.
- Task 4 keeps device metrics blocked until exact parity succeeds.
- No task expands Spec A or permits legacy runtime dependencies.
