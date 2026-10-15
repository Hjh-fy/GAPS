# System Safety Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make deployment, QC, Flower domain adaptation, specialist regression, and regression evaluation fail closed while preserving the frozen B1-B5 and formal R0-R7 experiment definitions.

**Architecture:** Add small shared validation seams at production boundaries, then reuse them from builders, validators, and runtime loaders. Keep each reviewed failure in its own red-green cycle, with package/QC fixes first, phase/calibration second, Flower DA third, specialist selection fourth, and regression aggregation/evaluation last.

**Tech Stack:** Python 3, PyTorch, Flower, NumPy, pytest, Git.

## Global Constraints

- Follow `docs/superpowers/specs/2026-07-13-system-safety-hardening-design.md` exactly.
- Do not change generic legacy MMD, stage-alignment, or Wasserstein defaults.
- Do not change frozen B1-B5 manifests, formal R0-R7 definitions, completed result files, thresholds, or training budgets.
- Use TDD for every production behavior change: focused RED, minimal GREEN, then subsystem regression tests.
- Keep the user's modified paper files and unrelated untracked research files untouched.
- Stage only the files named by the current task.
- Production imports and CLIs must work from a clean Git checkout.

---

### Task 1: Make QC Decisions Fail Closed

**Files:**
- Create: `tests/test_deploy_qc_fail_closed.py`
- Modify: `gaps_deploy/qc_policy.py:47-110`
- Modify: `gaps_deploy/qc_policy.py:115-455`
- Modify: `gaps_deploy/qc_policy.py:463-570`
- Modify: `gaps_deploy/inference.py:70-115`
- Modify: `gaps_deploy/inference.py:740-820`
- Modify: `gaps_deploy/inference.py:940-965`
- Modify: `gaps_deploy/final_runtime.py:145-170`

**Interfaces:**
- Produces `validate_qc_policy(policy: QCPolicy) -> None`.
- Produces `validate_calibration_refs(refs: Mapping[int, Any], num_classes: int) -> None` for response-dependent scores.
- Changes `QCDecision.risk_ratio` to `float | None`.
- Produces stable reasons in `QCDecision.risk_reasons`.

- [ ] **Step 1: Write focused failing tests**

```python
def test_no_policy_rejects_without_numeric_risk_ratio():
    decision = TwoThresholdDecider().decide({"classifier_uncertainty": 0.01})
    assert decision.decision == "reject"
    assert decision.risk_ratio is None
    assert decision.risk_reasons == ["qc_policy_missing"]

@pytest.mark.parametrize("scores", [{}, {"classifier_uncertainty": float("nan")}])
def test_required_score_must_be_available(scores):
    decider = TwoThresholdDecider()
    decider.load_policy(valid_policy())
    decision = decider.decide(scores)
    assert decision.decision == "reject"
    assert decision.risk_ratio is None
    assert decision.risk_reasons[0].startswith("qc_score_")

def test_final_runtime_never_auto_outputs_reject():
    row = FinalDeployRuntime._public_row(deploy_result(qc_status="reject", risk_score=None), 42.0)
    assert row["auto_output_ppm"] == ""
    assert row["risk_score"] is None

def test_response_policy_rejects_when_response_reference_is_unavailable():
    decider = TwoThresholdDecider()
    decider.load_policy(valid_response_policy())
    scores = RiskScoreComputer(calib_refs={}).compute(logits(), 50.0, 0, features())
    assert "response_signature_norm" not in scores
    assert decider.decide(scores).decision == "reject"
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `python -m pytest tests/test_deploy_qc_fail_closed.py -q --basetemp .tmp_pytest_deploy_qc_red`

Expected: failures show no-policy/missing-score paths still accept and `float(None)` is unsupported in the public row.

- [ ] **Step 3: Implement policy validation and fail-closed decisions**

```python
def validate_qc_policy(policy: QCPolicy) -> None:
    if not policy.scores or len(set(policy.scores)) != len(policy.scores):
        raise ValueError("QC policy scores must be non-empty and unique")
    if set(policy.thresholds) != set(policy.scores):
        raise ValueError("QC thresholds must exactly match scores")
    if not np.isfinite(policy.low_ratio) or not np.isfinite(policy.high_ratio):
        raise ValueError("QC ratio bounds must be finite")
    if not 0.0 <= policy.low_ratio < policy.high_ratio:
        raise ValueError("QC ratio bounds must satisfy 0 <= low < high")
    for name in policy.scores:
        threshold = float(policy.thresholds[name])
        if not np.isfinite(threshold) or threshold <= 0.0:
            raise ValueError(f"invalid QC threshold: {name}")
```

`TwoThresholdDecider.load_policy` calls the validator. `decide` returns reject/null for absent policy or unavailable required evidence. `FinalDeployRuntime._public_row` preserves JSON-null risk values and only fills `auto_output_ppm` for accept.

Define a supported-score registry with explicit response-reference dependencies. `RiskScoreComputer.compute` omits unavailable response scores and any composite that depends on them instead of inserting zero. Validate non-empty finite centers/scales/signatures, equal dimensions, positive finite normalization scales, and coverage for classes `0..num_classes-1` whenever a loaded policy selects a response-dependent score.

- [ ] **Step 4: Run focused and valid-path controls**

Run: `python -m pytest tests/test_deploy_qc_fail_closed.py -q --basetemp .tmp_pytest_deploy_qc_green`

Expected: PASS, including a valid low-risk policy accepting and a valid high-risk policy rejecting.

- [ ] **Step 5: Commit the QC contract**

```powershell
git add gaps_deploy/qc_policy.py gaps_deploy/inference.py gaps_deploy/final_runtime.py tests/test_deploy_qc_fail_closed.py
git commit -m "fix: make deployment QC fail closed"
```

### Task 2: Make Deployment Packages And Checkpoints Strict

**Files:**
- Create: `gaps_deploy/package_contract.py`
- Create: `tests/test_deployment_package_contract.py`
- Modify: `gaps_deploy/deploy_config.py:19-170`
- Modify: `gaps_deploy/inference.py:224-555`
- Modify and add to Git: `gaps_deploy/build_package.py`
- Modify and add to Git: `gaps_deploy/validate_deployment_packages.py`
- Modify and add to Git: `gaps_deploy/build_per_client_packages.py`
- Modify: `scripts/build_final_deployment_package.py`

**Interfaces:**
- Produces `DeploymentPackageError(ValueError)`.
- Produces `load_json_object(path: Path, label: str) -> dict[str, Any]`.
- Produces `extract_state_dict(checkpoint: Any, path: Path) -> dict[str, torch.Tensor]`.
- Produces `load_state_dict_strict(model: nn.Module, state: Mapping[str, Tensor], path: Path) -> None`.
- Produces `normalize_and_validate_routing_config(raw: Mapping[str, Any], num_classes: int) -> dict[str, Any]`.
- Produces `validate_package_layout(package: Path) -> dict[str, Path]`.

- [ ] **Step 1: Write empty-package and bad-checkpoint RED tests**

```python
def test_empty_package_never_constructs_random_models(tmp_path):
    with pytest.raises(DeploymentPackageError, match="deploy_config.json"):
        DeployPredictor.from_package(str(tmp_path))

def test_state_dict_shape_mismatch_is_rejected(tmp_path):
    package = minimal_valid_package(tmp_path)
    torch.save({"model_state": {"wrong.weight": torch.ones(1)}}, package / "models/classification_model.pth")
    with pytest.raises(DeploymentPackageError, match="classification_model.pth"):
        DeployPredictor.from_package(str(package))
```

- [ ] **Step 2: Verify package tests are RED**

Run: `python -m pytest tests/test_deployment_package_contract.py -q --basetemp .tmp_pytest_deploy_package_red`

Expected: empty package returns a randomly initialized predictor and mismatched checkpoints are tolerated.

- [ ] **Step 3: Implement shared package/checkpoint validation**

```python
class DeploymentPackageError(ValueError):
    pass

def load_state_dict_strict(model, state, path):
    expected = model.state_dict()
    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))
    bad_shapes = sorted(
        key for key in set(expected) & set(state)
        if tuple(expected[key].shape) != tuple(state[key].shape)
    )
    if missing or unexpected or bad_shapes:
        raise DeploymentPackageError(
            f"{path}: missing={missing}, unexpected={unexpected}, shape_mismatch={bad_shapes}"
        )
    model.load_state_dict(state, strict=True)
```

`from_package` and `from_config` require configs/checkpoints/QC/routing assets before constructing a predictor. Required full/specialist assets follow selected modes. No production path uses `strict=False`.

- [ ] **Step 4: Add builder and standalone-validator RED tests**

The builder test omits `--model-config` and `--qc-policy` and expects a validation error rather than default files. The validator test supplies malformed QC JSON and a shape-mismatched checkpoint and expects `errors`, never `status=pass`.

Run: `python -m pytest tests/test_deployment_package_contract.py -q --basetemp .tmp_pytest_deploy_builder_validator_red`

Expected: builder writes default model/QC configuration and validator checks only file presence.

- [ ] **Step 5: Make builder and validator share the strict contract**

Require an explicit trusted model-config JSON when checkpoint metadata lacks the full core architecture. Reject explicit/checkpoint disagreement. Propagate required arguments through `build_per_client_packages.py`. Make `normalize_and_validate_routing_config` reject missing/extra/colliding class keys, unknown modes, and missing selected assets. Make `validate_deployment_packages.py` parse all JSON, validate routing/QC/reference semantics, deserialize checkpoints, construct configured models, and verify exact key/shape compatibility.

Add `package_contract.py` to `scripts/build_final_deployment_package.py::copy_runtime_source`.

- [ ] **Step 6: Run package contract and runtime controls**

Run: `python -m pytest tests/test_deployment_package_contract.py tests/test_deploy_qc_fail_closed.py -q --basetemp .tmp_pytest_deploy_package_green`

Expected: PASS.

- [ ] **Step 7: Commit the strict package boundary**

```powershell
git add gaps_deploy/package_contract.py gaps_deploy/deploy_config.py gaps_deploy/inference.py gaps_deploy/build_package.py gaps_deploy/build_per_client_packages.py gaps_deploy/validate_deployment_packages.py scripts/build_final_deployment_package.py tests/test_deployment_package_contract.py
git commit -m "fix: validate deployment packages and checkpoints"
```

### Task 3: Enforce Calibration Schema And Phase Parity

**Files:**
- Create: `tests/test_deploy_calibration_contract.py`
- Create: `tests/test_deploy_phase_consistency.py`
- Modify: `gaps_deploy/calibration.py:150-235`
- Modify: `gaps_deploy/calibration.py:270-425`
- Modify: `gaps_deploy/inference.py:600-925`

**Interfaces:**
- Produces `normalize_phase_ids(phase, n_samples: int, num_phases: int) -> tuple[np.ndarray, np.ndarray]` returning raw and model phase arrays.
- `RegressionCalibrator.load_routing_config` reuses `normalize_and_validate_routing_config` and requires complete parameters for every selected mode.

- [ ] **Step 1: Write calibration-schema RED tests**

```python
@pytest.mark.parametrize("mode", ["unknown", "affine_only", "phase_affine_only"])
def test_selected_calibration_mode_requires_known_complete_parameters(mode):
    config = {"selected_modes": {str(i): "none" for i in range(4)}}
    config["selected_modes"]["0"] = mode
    with pytest.raises(ValueError):
        RegressionCalibrator().load_routing_config(config)

def test_routing_config_requires_every_class_exactly_once():
    with pytest.raises(ValueError, match="selected_modes"):
        RegressionCalibrator().load_routing_config({"selected_modes": {"0": "none"}})
```

- [ ] **Step 2: Verify calibration tests are RED**

Run: `python -m pytest tests/test_deploy_calibration_contract.py -q --basetemp .tmp_pytest_deploy_calibration_red`

Expected: incomplete and unknown modes load without error.

- [ ] **Step 3: Implement strict routing/calibration validation**

Normalize keys while detecting collisions such as `"0"` and `0`. Accept only `none`, `bias_only`, `affine_only`, `phase_affine_only`, `full`, `specialist`, and `specialist_full`. Require matching affine or phase-affine parameters for parameterized modes and validate finite coefficients/phase coverage.

- [ ] **Step 4: Write and verify phase-parity RED tests**

```python
def test_unknown_phase_uses_zero_for_batch_and_generator_but_preserves_raw_value():
    batch = predictor.predict_batch(features, phase=-1)
    streamed = list(predictor.predict_generator(loader_with_phase(-1)))
    assert batch[0].phase == -1
    assert streamed[0].phase == -1
    assert batch[0].final_ppm == pytest.approx(streamed[0].final_ppm)
    assert recorder.model_phases == [0, 0]

@pytest.mark.parametrize("phase", [-2, 3, 1.5])
def test_invalid_phase_is_rejected(phase):
    with pytest.raises(ValueError, match="phase"):
        predictor.predict_batch(features, phase=phase)
```

Run: `python -m pytest tests/test_deploy_phase_consistency.py -q --basetemp .tmp_pytest_deploy_phase_red`

Expected: generator and batch predictions differ for phase -1; invalid phase values are not consistently rejected.

- [ ] **Step 5: Implement one phase-normalization path and verify GREEN**

Use the returned model phase for the base/full/specialist regression heads and calibrator in both batch and generator paths. Preserve the raw array in `DeployResult.phase` and diagnostics.

Run: `python -m pytest tests/test_deploy_calibration_contract.py tests/test_deploy_phase_consistency.py -q --basetemp .tmp_pytest_deploy_calibration_phase_green`

Expected: PASS.

- [ ] **Step 6: Commit calibration and phase parity**

```powershell
git add gaps_deploy/calibration.py gaps_deploy/inference.py tests/test_deploy_calibration_contract.py tests/test_deploy_phase_consistency.py
git commit -m "fix: validate calibration routes and normalize phase"
```

### Task 4: Validate Flower DA Inputs And Handle No Shared Class

**Files:**
- Create: `gaps_flower/domain_adaptation_inputs.py`
- Modify: `gaps_flower/server_app.py:58-240`
- Modify: `gaps_flower/strategy.py:315-399`
- Modify: `gaps_flower/strategy.py:1083-1193`
- Modify: `gaps_flower/domain_adaptation.py:964-1081`
- Modify: `tests/test_flower_classification_contract.py`
- Modify: `tests/test_flower_da_v3_corrections.py`

**Interfaces:**
- Produces `validate_domain_adaptation_request(strategy, use_domain_adapt, server_val_data, server_calib_data) -> tuple[tuple[Path, ...], tuple[Path, ...]]`.
- Produces `load_domain_adaptation_arrays(data_dirs_spec, *, strict, expected_window_shape=(100, 8), num_classes=4, num_phases=3) -> tuple[np.ndarray, np.ndarray, np.ndarray]`.

- [ ] **Step 1: Write request/split RED tests**

Tests reject FedAvg plus DA, missing source/target specs, overlapping source/target directories, missing prefixed labels, missing prefixed phases, feature shapes other than `(N, 100, 8)`, mismatched rows, non-finite features, non-integer or out-of-range labels, and empty arrays. A valid pair returns merged arrays with exact row counts.

Run: `python -m pytest tests/test_flower_classification_contract.py -q --basetemp .tmp_pytest_flower_da_input_red`

Expected: current strategy accepts missing inputs and strict split falls back to root labels or -1 phases.

- [ ] **Step 2: Implement and reuse the input contract**

Call request validation after DA preset application but before `save_run_config` in the server. Repeat it in `GapsStrategy.__init__`. Replace the private fallback loader logic with the shared array loader and construct loaders only from validated arrays.

- [ ] **Step 3: Write no-shared-class RED test**

```python
def test_class_conditional_adversarial_no_shared_class_returns_connected_zero():
    source = torch.randn(4, 8, requires_grad=True)
    target = torch.randn(4, 8, requires_grad=True)
    loss = trainer._compute_adversarial_loss(source, target, all_zero, all_one)
    assert loss.item() == 0.0
    assert loss.requires_grad
    loss.backward()
    assert source.grad is not None
    assert target.grad is not None
```

Run: `python -m pytest tests/test_flower_da_v3_corrections.py::test_class_conditional_adversarial_no_shared_class_returns_connected_zero -q --basetemp .tmp_pytest_flower_da_shared_red`

Expected: RuntimeError because the constant discriminator loss has no grad function.

- [ ] **Step 4: Skip the invalid critic comparison and return connected zero**

Clear critic gradients with `zero_grad(set_to_none=True)`, detect the shared-class set before critic iterations, and return `(feat_s.sum() + feat_t.sum()) * 0.0` when it is empty.

- [ ] **Step 5: Verify Flower contracts without changing frozen semantics**

Run: `python -m pytest tests/test_flower_classification_contract.py tests/test_flower_da_v3_corrections.py -q --basetemp .tmp_pytest_flower_da_green`

Expected: PASS. Existing tests that construct DA without paths are updated to provide minimal valid calibration arrays; production validation is not weakened for test seams.

- [ ] **Step 6: Commit Flower DA safety**

```powershell
git add gaps_flower/domain_adaptation_inputs.py gaps_flower/server_app.py gaps_flower/strategy.py gaps_flower/domain_adaptation.py tests/test_flower_classification_contract.py tests/test_flower_da_v3_corrections.py
git commit -m "fix: validate Flower domain adaptation inputs"
```

### Task 5: Make Specialist Selection Leakage-Safe And Deployable

**Files:**
- Create: `tests/test_specialist_regression_safety.py`
- Modify: `gaps_flower/specialist_calibration_fit.py:302-359`
- Modify: `gaps_flower/specialist_calibration_fit.py:476-581`
- Modify: `gaps_flower/specialist_calibration_fit.py:671-885`
- Modify: `gaps_flower/specialist_calibration_fit.py:888-1097`
- Modify: `gaps_flower/specialist_calibration_fit.py:1175-1315`

**Interfaces:**
- Produces `InsufficientValidationDataError(ValueError)`.
- Produces `_collect_deployable_predictions(classifier, regressor, loader, device) -> tuple[true_ppm, pred_ppm, true_class, route_class, phase]`.
- Changes `_score` to `float | None`.

- [ ] **Step 1: Write split and finite-selection RED tests**

```python
def test_class_concentration_split_is_disjoint_and_complete():
    train, val = _split_loader(unique_concentration_loader(), 2, 0.25, 42, "class_concentration")
    train_ids = subset_indices(train)
    val_ids = subset_indices(val)
    assert train_ids.isdisjoint(val_ids)
    assert train_ids | val_ids == set(range(4))

def test_missing_or_tied_metrics_keep_none():
    routing, diagnostics = _build_auto_v2_routing(all_missing_or_tied_metrics(), {}, {}, {}, "R2", 0.0, 4)
    assert set(routing["selected_modes"].values()) == {"none"}
    assert diagnostics["selection_available"] == {str(i): False for i in range(4)}
```

- [ ] **Step 2: Verify RED and implement disjoint/finite selection**

Run: `python -m pytest tests/test_specialist_regression_safety.py -q --basetemp .tmp_pytest_specialist_split_red`

Expected: duplicate train indices overlap validation and missing/tied metrics select `full`.

Rebuild fallback indices from scratch, reject datasets with fewer than two rows, return `None` for missing/non-finite metrics, and replace candidates only for finite strict improvement `score > best + min_delta`. Gate equality and unavailable guardrail metrics fail closed.

- [ ] **Step 3: Write predicted-route RED test**

Use a dummy classifier that always predicts class 1 while true labels are class 0. Assert the regressor receives class 1 and full/specialist/calibration selection masks use route class 1, while the concentration truth is read from true class 0.

Run: `python -m pytest tests/test_specialist_regression_safety.py -q --basetemp .tmp_pytest_specialist_route_red`

Expected: the regressor receives the true class and the loaded classifier is unused.

- [ ] **Step 4: Thread predicted routes through deployable evaluation**

Use `route_class` for regression conditioning, model selection, calibration parameters, and route-specific candidate grouping. Keep oracle evaluation in explicitly named diagnostics only. Load classifier/regression checkpoints strictly.

- [ ] **Step 5: Freeze pre-refit independent metrics**

Add a test that monkeypatches the refit result to a deliberately bad model and proves gate decisions plus `selected_pre_refit_val_metrics` do not change. Do not calculate an independent-validation claim after refitting on the combined calibration+validation loader. Record `selection_metrics_source=pre_refit_independent_validation` and `deployment_models_refit_on_full_calibration`.

- [ ] **Step 6: Verify specialist and formal regression controls**

Run: `python -m pytest tests/test_specialist_regression_safety.py tests/test_iotj_c5_regression_suite.py tests/test_iotj_c5_regression_inputs.py -q --basetemp .tmp_pytest_specialist_green`

Expected: PASS.

- [ ] **Step 7: Commit specialist safety**

```powershell
git add gaps_flower/specialist_calibration_fit.py tests/test_specialist_regression_safety.py
git commit -m "fix: make specialist selection deployable and leakage safe"
```

### Task 6: Use Checkpoint Weights And Require Evaluation Assets

**Files:**
- Create: `tests/test_regression_runtime_safety.py`
- Modify: `gaps_flower/regression_task.py:600-630`
- Modify: `gaps_flower/regression_server.py:145-325`
- Modify: `gaps_flower/evaluate_regression_pipeline.py:158-217`
- Modify: `gaps_flower/evaluate_regression_pipeline.py:640-665`

**Interfaces:**
- Produces `_checkpoint_n_samples(checkpoint, client_id: int, path: Path) -> int`.
- Produces local strict state loading for base/full/specialist evaluation assets.
- Changes `load_full_model(..., required: bool = False)`.
- Adds `verify_live_sample_counts: bool = False` to regression aggregation and a matching CLI flag.

- [ ] **Step 1: Write aggregation-weight RED tests**

Tests reject absent, Boolean, non-integral, zero, and negative `n_samples`. An integration test provides checkpoint counts `{1: 3, 2: 7}` and different live counts, then asserts mismatch raises and live counts never reach `fedavg_regression_states`. Add a producer test proving the legacy regression task saves `n_samples`.

Run: `python -m pytest tests/test_regression_runtime_safety.py -q --basetemp .tmp_pytest_regression_weights_red`

Expected: server aggregation uses reconstructed live counts and the legacy producer omits `n_samples`.

- [ ] **Step 2: Implement checkpoint-owned weights**

```python
def _checkpoint_n_samples(checkpoint, client_id, path):
    value = checkpoint.get("n_samples") if isinstance(checkpoint, dict) else None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{path}: client {client_id} has invalid n_samples={value!r}")
    return value
```

Persist `n_samples=int(sample_counts[cid])` in every producer. Build aggregation weights only from accepted checkpoint metadata. Only when `verify_live_sample_counts=True`, reconstruct counts from the supplied data root and raise on disagreement; never replace checkpoint weights.

- [ ] **Step 3: Write evaluator-asset RED tests**

Selected specialist with no directory/file and selected full with no full checkpoint each raise. Missing/unexpected/shape-mismatched state keys raise with the asset path. A valid checkpoint control loads.

- [ ] **Step 4: Require selected assets and strict states**

Make `load_specialist_models` reject missing selected specialist assets. Call `load_full_model(required=True)` whenever any selected mode is `full`. Use strict state validation for base, full, and specialist models; never continue with the base model after a selected asset failure.

- [ ] **Step 5: Verify regression runtime safety**

Run: `python -m pytest tests/test_regression_runtime_safety.py tests/test_iotj_c5_regression_suite.py tests/test_iotj_c5_regression_inputs.py -q --basetemp .tmp_pytest_regression_runtime_green`

Expected: PASS.

- [ ] **Step 6: Commit regression runtime safety**

```powershell
git add gaps_flower/regression_task.py gaps_flower/regression_server.py gaps_flower/evaluate_regression_pipeline.py tests/test_regression_runtime_safety.py
git commit -m "fix: enforce regression aggregation and asset contracts"
```

### Task 7: Close Git Tracking And Verify The Full Hardening

**Files:**
- Create: `tests/test_deployment_git_closure.py`
- Add to Git: `gaps_deploy/r4a_residual.py`
- Add to Git: `scripts/validate_final_deployment_bundle.py`
- Modify: `代码文件介绍.md`
- Modify: `docs/experiments/iotj_system_experiment_notebook.md`

**Interfaces:**
- A clean Git archive can import `gaps_deploy.inference` and show help for the builder/validator CLIs.

- [ ] **Step 1: Add a clean-checkout smoke test**

Create `tests/test_deployment_git_closure.py`. It copies exactly the paths returned by `git ls-files` into a temporary directory, sets `PYTHONPATH` to that directory, and then runs:

```powershell
python -c "import gaps_deploy.inference; import gaps_deploy.r4a_residual"
python -m gaps_deploy.build_package --help
python -m gaps_deploy.validate_deployment_packages --help
```

The test must fail before missing tracked dependencies are added and pass afterward.

- [ ] **Step 2: Audit and stage the minimal production closure**

Use `git ls-files` and Python import tracing to confirm the exact production closure. Stage `gaps_deploy/r4a_residual.py` and `scripts/validate_final_deployment_bundle.py` alongside the already staged test. Do not add exploratory QC analyses, temporary outputs, local archives, `validate_package_runtime.py`, or the user's untracked regression-mainline test.

Run: `python -m pytest tests/test_deployment_git_closure.py -q --basetemp .tmp_pytest_deploy_git_closure`

Expected: PASS from the temporary tracked-only tree.

- [ ] **Step 3: Run focused subsystem suites**

Run:

```powershell
python -m pytest tests/test_deploy_qc_fail_closed.py tests/test_deploy_calibration_contract.py tests/test_deploy_phase_consistency.py tests/test_deployment_package_contract.py -q --basetemp .tmp_pytest_hardening_deploy
python -m pytest tests/test_flower_classification_contract.py tests/test_flower_da_v3_corrections.py -q --basetemp .tmp_pytest_hardening_flower
python -m pytest tests/test_specialist_regression_safety.py tests/test_regression_runtime_safety.py tests/test_iotj_c5_regression_suite.py tests/test_iotj_c5_regression_inputs.py -q --basetemp .tmp_pytest_hardening_regression
```

Expected: all focused tests pass.

- [ ] **Step 4: Run frozen-contract and broad regression verification**

Run the exact pre-change baseline selection:

```powershell
python -m pytest tests/test_flower_da_v3_corrections.py tests/test_flower_classification_contract.py tests/test_iotj_c5_regression_suite.py tests/test_iotj_c5_regression_inputs.py tests/test_iotj_cross_direction_classification.py tests/test_iotj_classification_summary.py tests/test_iotj_high_coverage_qc.py tests/test_profile_qc_coverage_audit.py -q --basetemp .tmp_pytest_hardening_contracts
```

Expected baseline before implementation: `98 passed, 2 warnings`. Also run `git diff --check`, `python -m compileall` on every changed Python module, and the tracked-only smoke test. Record exact pass/fail counts and warnings; do not reinterpret unrelated failures.

- [ ] **Step 5: Update progress documentation without touching paper drafts**

Record the corrected code revision, hardening tests, and the already completed F1 B2/B5 status in `代码文件介绍.md` and the experiment notebook. Do not edit the two user-modified paper files during this task.

- [ ] **Step 6: Request independent code review and resolve findings**

Review the full branch diff against the approved design. Fix Critical/Important findings with new RED-GREEN cycles, rerun affected suites, then rerun the full verification gate.

- [ ] **Step 7: Commit tracking and verification records**

```powershell
git add gaps_deploy/r4a_residual.py scripts/validate_final_deployment_bundle.py tests/test_deployment_git_closure.py
git add 代码文件介绍.md docs/experiments/iotj_system_experiment_notebook.md
git commit -m "chore: close deployment runtime tracking"
```
