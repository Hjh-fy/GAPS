# Canonical FedRidge R0-v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and pre-run-freeze an independently versioned, numerically stable canonical-v1 source-only FedRidge reconstruction protocol without executing the formal R0-v2 endpoint.

**Architecture:** Add a versioned FedRidge v2 numerical core using mergeable float64 central moments, deterministic C1-to-C2 aggregation, unchanged Ridge semantics, and preregistered multi-gate diagnostics. Add a fail-closed runner that freshly reconstructs canonical source feature caches only after a separately authorized freeze commit; the implementation phase exercises synthetic tests and static preflight only and must not create the formal result root.

**Tech Stack:** Python 3, NumPy float64, pytest, existing canonical-v1 feature extractor/cache contracts, CSV/JSON/Markdown evidence artifacts, Git SHA256 provenance.

## Global Constraints

- Frozen design: `docs/superpowers/specs/2026-08-12-canonical-fedridge-r0-v2-design.md` at commit `b41fee1d5bd64a19d6fefcad5fde610183856202`.
- C0 remains `V1_INTERLEAVED_RETAINED`; never rerun C0 or add classification experiments.
- Original R0 remains `R0_EXACT_RECOVERY_NOT_ESTABLISHED`; do not edit or reinterpret its code, result root, thresholds, or audit.
- Dataset is `dataset/iotj_canonical_v1`, aggregate SHA256 `2f810d7e93cae5f361923184e9dc87d5ae59e0f59be9f52aff7e14f9f33e94f6`, 5 Hz, 10 s, 5 s stride, 50x8.
- Ordered source clients are exactly `C1,C2`; target C3/C4/C5 inputs are structurally forbidden.
- Fresh C1/C2 train/calibration 83D/104D caches must be generated under the new formal result root; no old R0 cache reuse.
- All numerical computation is float64; population variance is `M2/n`; safe-scale condition remains `raw_scale < 1e-9`.
- Alpha grid remains `(0.0, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)`; first-in-grid tie break; `numpy.linalg.pinv`; intercept unregularized.
- Hard tolerances remain `tau_moment=64*gamma(1340)`, `tau_residual=128*gamma(105)`, functional metrics `1e-6 ppm`, condition gate `kappa*epsilon<1`.
- No target cache, target label, R1/R2, QC/Q1, new regression model, solver search, alpha search, tolerance search, manuscript edit, or formal execution is authorized.
- The pre-run implementation may run synthetic tests and read static dataset manifests/counts, but it must not create `results/iotj_canonical_v1_final/canonical_fedridge_r0_v2_20260812/`.
- Preserve unrelated watcher logs, preprocessing logs, and `.tmp_pytest_*` directories; never stage or delete them.

---

## File Structure

### Create

- `gaps_flower/canonical_fedridge_v2.py`: versioned central moments, scaler, normal equations, Ridge model, alpha selection, diagnostics, and decision gates.
- `scripts/run_iotj_canonical_fedridge_r0_v2.py`: fail-closed preflight/run/audit controller and evidence writer.
- `tests/test_iotj_canonical_fedridge_r0_v2.py`: synthetic numerical, protocol, access, runner, and audit tests.
- `docs/experiments/iotj_canonical_v1_final/canonical_fedridge_r0_v2_20260812/PROTOCOL.md`: human-readable frozen protocol.
- `docs/experiments/iotj_canonical_v1_final/canonical_fedridge_r0_v2_20260812/protocol_manifest.json`: machine-readable freeze with `formal_execution_started=false`.
- `docs/experiments/iotj_canonical_v1_final/canonical_fedridge_r0_v2_20260812/EXPERIMENT_PLAN.md`: falsifiable question, controls, gates, risks, and stop rule.
- `docs/experiments/iotj_canonical_v1_final/canonical_fedridge_r0_v2_20260812/EXPERIMENT_MATRIX.csv`: one planned R0-v2 configuration.
- `docs/experiments/iotj_canonical_v1_final/canonical_fedridge_r0_v2_20260812/EXPERIMENT_REGISTRY.csv`: registered workflow/provenance row.
- `docs/experiments/iotj_canonical_v1_final/canonical_fedridge_r0_v2_20260812/NEAR_CONSTANT_SCALE_POLICY.md`: reuse of `1e-9` without performance tuning.
- `docs/experiments/iotj_canonical_v1_final/canonical_fedridge_r0_v2_20260812/R0_V2_NUMERICAL_TOLERANCE_JUSTIFICATION.md`: formula-derived tolerances.
- `docs/experiments/iotj_canonical_v1_final/canonical_fedridge_r0_v2_20260812/FEDRIDGE_NUMERICAL_STABILITY_MANUSCRIPT_NOTE.md`: future minimal wording only.

### Modify

- `gaps_flower/canonical_quantitative_features.py`: add an optional expected study ID to cache build/load validation while preserving the old default behavior byte-for-byte at the API level.
- `代码文件介绍.md`: register R0-v2 as pre-run frozen and explicitly not executed.

### Must remain unchanged

- `gaps_flower/canonical_fedridge.py`
- `scripts/run_iotj_canonical_regression_reconstruction_r0.py`
- `results/iotj_canonical_v1_final/canonical_regression_reconstruction_qc_20260811/C0/`
- `results/iotj_canonical_v1_final/canonical_regression_reconstruction_qc_20260811/R0/`

---

### Task 1: Version the canonical feature-cache provenance contract

**Files:**
- Modify: `gaps_flower/canonical_quantitative_features.py`
- Create/Test: `tests/test_iotj_canonical_fedridge_r0_v2.py`

**Interfaces:**
- Consumes: existing `build_feature_cache`, `load_feature_cache`, `validate_cache_manifest` behavior.
- Produces:
  - `validate_cache_manifest(manifest, *, expected_dataset_sha256: str, expected_study_id: str = STUDY_ID) -> None`
  - `build_feature_cache(..., study_id: str = STUDY_ID) -> dict[str, Any]`
  - `load_feature_cache(..., expected_study_id: str = STUDY_ID) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], dict[str, Any]]`

- [ ] **Step 1: Write failing tests for explicit study identity and legacy default compatibility**

```python
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol

import numpy as np
import pytest

R0_V2_STUDY_ID = "CAN-V1-FEDRIDGE-R0V2-20260812"

def canonical_cache_manifest_fixture(*, study_id: str) -> dict[str, object]:
    return {
        "study_id": study_id,
        "sampling_rate_hz": 5,
        "window_shape": [50, 8],
        "dataset_aggregate_sha256": "a" * 64,
        "source_array_sha256": "b" * 64,
        "metadata_sha256": "c" * 64,
        "extractor_file_sha256": "d" * 64,
        "ordered_h1_feature_names_sha256": "e" * 64,
        "ordered_sensor_feature_names_sha256": "f" * 64,
        "h1_dimensions": 104,
        "sensor_dimensions": 83,
        "created_from_canonical_arrays": True,
        "legacy_cache_reused": False,
    }

def test_cache_manifest_requires_explicit_r0_v2_study_identity():
    manifest = canonical_cache_manifest_fixture(study_id=R0_V2_STUDY_ID)
    validate_cache_manifest(
        manifest,
        expected_dataset_sha256="a" * 64,
        expected_study_id=R0_V2_STUDY_ID,
    )
    with pytest.raises(RuntimeError, match="canonical cache provenance"):
        validate_cache_manifest(
            {**manifest, "study_id": "CAN-V1-CRRQ-20260811"},
            expected_dataset_sha256="a" * 64,
            expected_study_id=R0_V2_STUDY_ID,
        )

def test_cache_manifest_old_default_study_identity_is_unchanged():
    manifest = canonical_cache_manifest_fixture(study_id="CAN-V1-CRRQ-20260811")
    validate_cache_manifest(manifest, expected_dataset_sha256="a" * 64)
```

- [ ] **Step 2: Run the two tests and verify RED**

Run:

```powershell
python -m pytest -q tests/test_iotj_canonical_fedridge_r0_v2.py -k 'cache_manifest'
```

Expected: FAIL because `expected_study_id` and `study_id` are not accepted.

- [ ] **Step 3: Add backward-compatible study-ID parameters**

Implement the signatures above. Set `manifest["study_id"] = str(study_id)` in `build_feature_cache`; pass the same value to validation. Make `load_feature_cache` validate `expected_study_id`. Do not change extractor math, ordered names, shapes, dtype, hashes, or the old default constant.

- [ ] **Step 4: Run RED tests and existing cache tests**

```powershell
python -m pytest -q tests/test_iotj_canonical_fedridge_r0_v2.py -k 'cache_manifest'
python -m pytest -q tests/test_iotj_canonical_regression_reconstruction.py -k 'feature_cache or cache_manifest'
```

Expected: PASS.

- [ ] **Step 5: Commit the cache-contract change**

```powershell
git add -- gaps_flower/canonical_quantitative_features.py tests/test_iotj_canonical_fedridge_r0_v2.py
git commit -m "feat: version canonical quantitative cache provenance"
```

---

### Task 2: Implement stable mergeable central moments and safe scaling

**Files:**
- Create: `gaps_flower/canonical_fedridge_v2.py`
- Modify/Test: `tests/test_iotj_canonical_fedridge_r0_v2.py`

**Interfaces:**
- Consumes: finite float64 feature matrices and exact client/gas/role metadata.
- Produces:
  - `LocalCentralMomentsV2`
  - `StableGlobalScalerV2`
  - `local_central_moments(client_id, gas_id, role, values) -> LocalCentralMomentsV2`
  - `merge_central_moments(records, *, expected_client_order=("C1", "C2")) -> StableGlobalScalerV2`
  - `feature_numerical_audit_rows(records, scaler, feature_names) -> list[dict[str, Any]]`

- [ ] **Step 1: Write failing tests for constant, near-constant, cancellation, population variance, and ordering**

```python
def test_v2_constant_and_near_constant_features_use_registered_safe_scale():
    x1 = np.array([[10.0, 10.0], [10.0, 10.0 + 1e-14]], dtype=np.float64)
    x2 = np.array([[10.0, 10.0 - 1e-14], [10.0, 10.0]], dtype=np.float64)
    scaler = merge_central_moments([
        local_central_moments("C2", 3, "refit", x2),
        local_central_moments("C1", 3, "refit", x1),
    ])
    assert scaler.aggregation_order == ("C1", "C2")
    assert np.array_equal(scaler.safe_scale_mask, [True, True])
    assert np.array_equal(scaler.scale, [1.0, 1.0])

def test_v2_large_offset_small_variance_avoids_raw_moment_cancellation():
    x1 = np.array([[1e12], [1e12 + 1e-3]], dtype=np.float64)
    x2 = np.array([[1e12 + 2e-3], [1e12 + 3e-3]], dtype=np.float64)
    scaler = merge_central_moments([
        local_central_moments("C1", 1, "refit", x1),
        local_central_moments("C2", 1, "refit", x2),
    ])
    pooled = np.vstack([x1, x2])
    assert scaler.raw_scale[0] == pytest.approx(np.std(pooled[:, 0], ddof=0), rel=1e-10)
    assert scaler.raw_scale[0] > 0.0

def test_v2_safe_scale_boundary_is_strictly_less_than_one_e_minus_nine():
    exactly = np.array([[-1e-9], [1e-9]], dtype=np.float64)
    scaler = merge_central_moments([
        local_central_moments("C1", 0, "refit", exactly),
        local_central_moments("C2", 0, "refit", exactly),
    ])
    assert scaler.raw_scale[0] == pytest.approx(1e-9)
    assert not scaler.safe_scale_mask[0]
```

Also add tests for `M2/n` versus sample variance, float64 outputs, duplicate/missing clients, mismatched gas/role/dimensions, and nonfinite input rejection.

- [ ] **Step 2: Run central-moment tests and verify RED**

```powershell
python -m pytest -q tests/test_iotj_canonical_fedridge_r0_v2.py -k 'constant or cancellation or population or ordering or boundary'
```

Expected: FAIL because the v2 module does not exist.

- [ ] **Step 3: Implement immutable records and central-moment merge**

```python
FLOAT64_EPS = np.finfo(np.float64).eps
SCALE_FLOOR = 1e-9
CLIENT_ORDER = ("C1", "C2")

@dataclass(frozen=True)
class LocalCentralMomentsV2:
    client_id: str
    gas_id: int
    role: str
    n: int
    mean: np.ndarray
    m2: np.ndarray
    minimum: np.ndarray
    maximum: np.ndarray
    provenance_sha256: str

@dataclass(frozen=True)
class StableGlobalScalerV2:
    gas_id: int
    role: str
    n: int
    mean: np.ndarray
    variance: np.ndarray
    raw_scale: np.ndarray
    scale: np.ndarray
    safe_scale_mask: np.ndarray
    minimum: np.ndarray
    maximum: np.ndarray
    aggregation_order: tuple[str, ...]
```

Compute each local mean first, then `m2=np.sum((x-mean)**2, axis=0, dtype=np.float64)`. Merge the records using the frozen central-moment equations and `variance=np.maximum(m2/n, 0.0)`. Sort by `expected_client_order`, reject missing/extra/duplicate clients, and retain the order in the output.

- [ ] **Step 4: Implement the 104D numerical-audit row builder**

Emit per-feature fields:

```text
gas_id, role, feature_index, feature_name, n, minimum, maximum,
mean, population_variance, raw_scale, dynamic_range,
safe_scale_floor, safe_scale_applied, canonical_scale,
aggregation_order, dtype
```

- [ ] **Step 5: Run v2 moment tests and original R0 tests**

```powershell
python -m pytest -q tests/test_iotj_canonical_fedridge_r0_v2.py -k 'moment or scale or constant or cancellation or population or ordering or boundary'
python -m pytest -q tests/test_iotj_canonical_regression_reconstruction.py -k 'population_scaler or exactly_recovers'
```

Expected: all selected tests PASS; original module tests remain unchanged.

- [ ] **Step 6: Commit stable moments**

```powershell
git add -- gaps_flower/canonical_fedridge_v2.py tests/test_iotj_canonical_fedridge_r0_v2.py
git commit -m "feat: add stable mergeable FedRidge moments"
```

---

### Task 3: Implement v2 normal equations, Ridge model, and source-only alpha selection

**Files:**
- Modify: `gaps_flower/canonical_fedridge_v2.py`
- Modify/Test: `tests/test_iotj_canonical_fedridge_r0_v2.py`

**Interfaces:**
- Consumes: `StableGlobalScalerV2`, ordered source mappings, 104 feature names.
- Produces:
  - `LocalNormalEquationsV2`
  - `AggregatedNormalEquationsV2`
  - `CanonicalRidgeModelV2`
  - `local_normal_equations_v2(...) -> LocalNormalEquationsV2`
  - `aggregate_normal_equations_v2(..., expected_client_order=CLIENT_ORDER) -> AggregatedNormalEquationsV2`
  - `reconstruct_ridge_v2(equations, scaler, feature_names, alpha) -> CanonicalRidgeModelV2`
  - `pooled_reference_fit_v2(values, targets, ..., alpha) -> tuple[CanonicalRidgeModelV2, AggregatedNormalEquationsV2]`
  - `select_source_alpha_v2(source_train, source_calibration, ..., alphas=RIDGE_ALPHAS) -> tuple[float, list[dict[str, Any]]]`
  - `select_pooled_alpha_v2(source_train, source_calibration, ..., alphas=RIDGE_ALPHAS) -> tuple[float, list[dict[str, Any]]]`

- [ ] **Step 1: Write failing A/b, intercept, prediction, and alpha tests**

```python
def synthetic_two_client_regression() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    return {
        "C1": (
            np.array([[0.0, 1.0], [2.0, 1.0], [4.0, 1.0]], dtype=np.float64),
            np.array([1.0, 5.0, 9.0], dtype=np.float64),
        ),
        "C2": (
            np.array([[1.0, 1.0], [3.0, 1.0], [5.0, 1.0]], dtype=np.float64),
            np.array([3.0, 7.0, 11.0], dtype=np.float64),
        ),
    }

def stable_scaler_for(
    client_data: Mapping[str, tuple[np.ndarray, np.ndarray]],
) -> StableGlobalScalerV2:
    return merge_central_moments([
        local_central_moments(client, 0, "refit", values)
        for client, (values, _targets) in client_data.items()
    ])

def standardized_pooled_design(
    client_data: Mapping[str, tuple[np.ndarray, np.ndarray]],
    scaler: StableGlobalScalerV2,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.vstack([client_data[client][0] for client in ("C1", "C2")])
    targets = np.concatenate([client_data[client][1] for client in ("C1", "C2")])
    z = (values - scaler.mean) / scaler.scale
    return np.column_stack([np.ones(len(z), dtype=np.float64), z]), targets

def test_v2_normal_equations_match_same_standardized_pooled_rows():
    client_data = synthetic_two_client_regression()
    scaler = stable_scaler_for(client_data)
    local = [local_normal_equations_v2(client, 0, "refit", x, y, scaler)
             for client, (x, y) in reversed(list(client_data.items()))]
    fed = aggregate_normal_equations_v2(local)
    pooled_design, pooled_y = standardized_pooled_design(client_data, scaler)
    assert fed.aggregation_order == ("C1", "C2")
    assert np.allclose(fed.a, pooled_design.T @ pooled_design)
    assert np.allclose(fed.b, pooled_design.T @ pooled_y)

def test_v2_intercept_is_unregularized_and_alpha_grid_is_exact():
    client_data = synthetic_two_client_regression()
    scaler = stable_scaler_for(client_data)
    equations = aggregate_normal_equations_v2([
        local_normal_equations_v2(client, 0, "refit", values, targets, scaler)
        for client, (values, targets) in client_data.items()
    ])
    model = reconstruct_ridge_v2(equations, scaler, ["x0", "x1"], alpha=1000.0)
    assert model.intercept_regularized is False
    assert RIDGE_ALPHAS == (0.0, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
```

Add a source-role test that rejects any `train_role`, `validation_role`, or mapping key containing target/test semantics during alpha selection.

- [ ] **Step 2: Run equation/model tests and verify RED**

```powershell
python -m pytest -q tests/test_iotj_canonical_fedridge_r0_v2.py -k 'normal_equation or intercept or alpha or source_role'
```

Expected: FAIL because v2 equation/model functions are absent.

- [ ] **Step 3: Implement v2 equation and model records**

```python
@dataclass(frozen=True)
class LocalNormalEquationsV2:
    client_id: str
    gas_id: int
    role: str
    n: int
    a: np.ndarray
    b: np.ndarray
    y_y: float
    y_min: float
    y_max: float
    provenance_sha256: str

@dataclass(frozen=True)
class AggregatedNormalEquationsV2:
    gas_id: int
    role: str
    n: int
    a: np.ndarray
    b: np.ndarray
    y_y: float
    y_min: float
    y_max: float
    aggregation_order: tuple[str, ...]

@dataclass(frozen=True)
class CanonicalRidgeModelV2:
    gas_id: int
    role: str
    alpha: float
    feature_names: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    clip_min: float
    clip_max: float
    intercept_regularized: bool = False
```

Use design matrix `[1, (X-mean)/scale]`, `regularizer[0,0]=0`, and `np.linalg.pinv(A+alpha*P) @ b`. Implement `predict_matrix(values, *, clip: bool = True) -> np.ndarray` and `to_json() -> dict[str, Any]`; `clip=False` is the raw functional-parity path.

- [ ] **Step 4: Implement the source-only federated and pooled alpha loops**

Require exact roles `source_train` and `source_calibration`, exact alpha tuple, sorted C1/C2 mappings, clipped distributed SSE/count, and first-in-grid tie behavior. The pooled audit concatenates exactly the same C1-then-C2 rows. Record `target_input_accessed=false` and `source_test_accessed=false` in every audit row.

- [ ] **Step 5: Run v2 and original equation/alpha tests**

```powershell
python -m pytest -q tests/test_iotj_canonical_fedridge_r0_v2.py -k 'normal_equation or ridge or intercept or alpha or source_role'
python -m pytest -q tests/test_iotj_canonical_regression_reconstruction.py -k 'intercept or alpha_selection'
```

Expected: PASS.

- [ ] **Step 6: Commit v2 Ridge semantics**

```powershell
git add -- gaps_flower/canonical_fedridge_v2.py tests/test_iotj_canonical_fedridge_r0_v2.py
git commit -m "feat: add canonical FedRidge v2 equations"
```

---

### Task 4: Implement preregistered diagnostics and the R0-v2 decision gate

**Files:**
- Modify: `gaps_flower/canonical_fedridge_v2.py`
- Modify/Test: `tests/test_iotj_canonical_fedridge_r0_v2.py`

**Interfaces:**
- Consumes: federated and pooled scaler/equation/model records plus shared source-test arrays.
- Produces:
  - `NumericalTolerancesV2`
  - `registered_tolerances_v2() -> NumericalTolerancesV2`
  - `scaler_diagnostics_v2(...) -> dict[str, Any]`
  - `normal_equation_diagnostics_v2(...) -> dict[str, Any]`
  - `system_diagnostics_v2(...) -> dict[str, Any]`
  - `functional_diagnostics_v2(...) -> dict[str, Any]`
  - `decide_gas_equivalence_v2(...) -> dict[str, Any]`
  - `decide_r0_v2(gas_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]`

- [ ] **Step 1: Write failing tolerance and independent-gate tests**

```python
def passing_gas_diagnostic(
    gas_id: int = 0,
    **overrides: object,
) -> dict[str, object]:
    row: dict[str, object] = {
        "gas_id": gas_id,
        "alpha_equal": True,
        "scaler_pass": True,
        "safe_scale_mask_equal": True,
        "normal_equations_pass": True,
        "condition_pass": True,
        "fed_residual_pass": True,
        "pooled_residual_pass": True,
        "raw_prediction_pass": True,
        "clipped_prediction_pass": True,
        "rmse_parity_pass": True,
        "mae_parity_pass": True,
        "finite_pass": True,
        "relative_beta_difference": 0.0,
    }
    row.update(overrides)
    return row

def test_v2_tolerances_are_formula_derived_and_frozen():
    t = registered_tolerances_v2()
    eps = np.finfo(np.float64).eps
    gamma = lambda m: (m * eps) / (1.0 - m * eps)
    assert t.tau_moment == pytest.approx(64.0 * gamma(1340), rel=0, abs=0)
    assert t.tau_residual == pytest.approx(128.0 * gamma(105), rel=0, abs=0)
    assert t.tau_functional_ppm == 1e-6

@pytest.mark.parametrize("field", [
    "alpha_equal", "scaler_pass", "safe_scale_mask_equal",
    "normal_equations_pass", "condition_pass", "fed_residual_pass",
    "pooled_residual_pass", "raw_prediction_pass", "clipped_prediction_pass",
    "rmse_parity_pass", "mae_parity_pass", "finite_pass",
])
def test_each_registered_hard_gate_can_fail_r0_v2(field):
    rows = [passing_gas_diagnostic(gas_id=g) for g in range(4)]
    rows[0][field] = False
    assert decide_r0_v2(rows)["decision"] == "R0_V2_FAILED"

def test_coefficient_difference_is_diagnostic_not_a_hard_gate():
    rows = [passing_gas_diagnostic(gas_id=g, relative_beta_difference=1.0) for g in range(4)]
    assert decide_r0_v2(rows)["decision"] == (
        "FEDRIDGE_ALGEBRAIC_EXACT_NUMERICAL_EQUIVALENCE_ESTABLISHED"
    )
```

Also test `kappa` nonfinite and `kappa*epsilon>=1`, zero diagnostic denominators, safe-scale mask mismatch, wrong gas set, duplicate gas, NaN/Inf, and exact alpha mismatch.

- [ ] **Step 2: Run diagnostic tests and verify RED**

```powershell
python -m pytest -q tests/test_iotj_canonical_fedridge_r0_v2.py -k 'tolerance or gate or diagnostic or condition or coefficient'
```

Expected: FAIL because the diagnostic API is absent.

- [ ] **Step 3: Implement exact tolerance formulas and scaler/A/b diagnostics**

```python
@dataclass(frozen=True)
class NumericalTolerancesV2:
    epsilon: float
    n_max: int
    feature_dimensions: int
    design_dimensions: int
    tau_moment: float
    tau_residual: float
    tau_functional_ppm: float
```

For scaler coordinate `j`, use `S_j=max(1,max_abs_j,dynamic_range_j,abs(mean_pool_j),scale_pool_j)`. Require both mean and scale coordinate errors at most `tau_moment*S_j` and exact safe-scale-mask identity.

Use Frobenius relative A discrepancy and L2 relative b discrepancy with no zero-denominator substitution.

- [ ] **Step 4: Implement system, coefficient, and functional diagnostics**

Use:

```text
relative_residual = ||M beta-b||_2 / (||M||_2||beta||_2 + ||b||_2)
beta_forward_envelope = kappa*(2*tau_moment + tau_residual)
```

Report beta envelope membership without including it in hard-pass conjunction. Evaluate raw and clipped prediction max difference and clipped RMSE/MAE differences on identical source-test rows.

- [ ] **Step 5: Implement exact decision vocabulary**

Require exactly one row for each gas ID 0,1,2,3 and every hard Boolean true. Return only:

```text
FEDRIDGE_ALGEBRAIC_EXACT_NUMERICAL_EQUIVALENCE_ESTABLISHED
R0_V2_FAILED
```

- [ ] **Step 6: Run all v2 numerical tests**

```powershell
python -m pytest -q tests/test_iotj_canonical_fedridge_r0_v2.py
```

Expected: PASS.

- [ ] **Step 7: Commit diagnostics and gate**

```powershell
git add -- gaps_flower/canonical_fedridge_v2.py tests/test_iotj_canonical_fedridge_r0_v2.py
git commit -m "feat: preregister FedRidge v2 numerical gates"
```

---

### Task 5: Freeze protocol, experiment records, near-constant policy, and manuscript note

**Files:**
- Create all files under: `docs/experiments/iotj_canonical_v1_final/canonical_fedridge_r0_v2_20260812/`
- Modify/Test: `tests/test_iotj_canonical_fedridge_r0_v2.py`

**Interfaces:**
- Consumes: approved design constants and v2 public interfaces.
- Produces: complete immutable protocol bundle with one registered experiment row and no formal result.

- [ ] **Step 1: Write failing protocol-artifact tests**

```python
ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_ROOT = (
    ROOT
    / "docs/experiments/iotj_canonical_v1_final"
    / "canonical_fedridge_r0_v2_20260812"
)
PROTOCOL_MANIFEST = PROTOCOL_ROOT / "protocol_manifest.json"
EXPERIMENT_MATRIX = PROTOCOL_ROOT / "EXPERIMENT_MATRIX.csv"
EXPERIMENT_REGISTRY = PROTOCOL_ROOT / "EXPERIMENT_REGISTRY.csv"

def test_r0_v2_protocol_manifest_is_pre_run_and_target_free():
    manifest = json.loads(PROTOCOL_MANIFEST.read_text(encoding="utf-8"))
    assert manifest["study_id"] == "CAN-V1-FEDRIDGE-R0V2-20260812"
    assert manifest["status"] == "DESIGN_FREEZE_READY_FORMAL_NOT_STARTED"
    assert manifest["formal_execution_started"] is False
    assert manifest["source_clients"] == ["C1", "C2"]
    assert manifest["target_clients"] == []
    assert manifest["target_access"] == {
        "calibration_x": False, "calibration_labels": False,
        "test_x": False, "test_labels": False,
    }
    assert manifest["numerical_gates"]["functional_ppm"] == 1e-6

def test_r0_v2_matrix_registers_exactly_one_unexecuted_configuration():
    rows = list(csv.DictReader(EXPERIMENT_MATRIX.open(encoding="utf-8")))
    assert len(rows) == 1
    assert rows[0]["experiment_id"] == "CAN-V1-FEDRIDGE-R0V2-20260812"
    assert rows[0]["status"] == "registered"
    assert rows[0]["target_clients"] == ""
```

Also assert that every required Markdown file exists, contains the two decision terms, preserves original R0, and contains no target/QC execution instruction.

- [ ] **Step 2: Run protocol tests and verify RED**

```powershell
python -m pytest -q tests/test_iotj_canonical_fedridge_r0_v2.py -k 'protocol or matrix or registry or manuscript or near_constant'
```

Expected: FAIL because protocol artifacts do not exist.

- [ ] **Step 3: Create `protocol_manifest.json` and `PROTOCOL.md`**

The manifest must include canonical hashes/counts, feature hashes, design commit, new output root, frozen numerical formulas, alpha/solver/intercept semantics, access sequence, expected files, stop rules, and:

```json
{
  "status": "DESIGN_FREEZE_READY_FORMAL_NOT_STARTED",
  "formal_execution_started": false,
  "execution_commit_policy": "CLI authorized freeze commit must equal current Git HEAD",
  "original_R0_decision": "R0_EXACT_RECOVERY_NOT_ESTABLISHED",
  "C0_decision": "V1_INTERLEAVED_RETAINED"
}
```

- [ ] **Step 4: Create plan, matrix, and registry**

Register one configuration with exact source roles, canonical split, no target, no DA/calibration/QC, seed `42` with the note `deterministic numeric reconstruction; seed unused`, new result path, status `registered`, evidence `blocked_pending_execution`, and provenance to the design/protocol.

- [ ] **Step 5: Create numerical-policy notes**

`NEAR_CONSTANT_SCALE_POLICY.md` must state that `1e-9` is reused and not data/performance selected. `R0_V2_NUMERICAL_TOLERANCE_JUSTIFICATION.md` must derive epsilon/gamma values and list every hard gate. `FEDRIDGE_NUMERICAL_STABILITY_MANUSCRIPT_NOTE.md` must propose only:

```text
global mean/variance is reconstructed using numerically stable mergeable moments
```

and explicitly prohibit a bitwise-exact or novel-algorithm claim.

- [ ] **Step 6: Run protocol tests**

```powershell
python -m pytest -q tests/test_iotj_canonical_fedridge_r0_v2.py -k 'protocol or matrix or registry or manuscript or near_constant'
```

Expected: PASS.

- [ ] **Step 7: Commit the protocol bundle**

```powershell
git add -- docs/experiments/iotj_canonical_v1_final/canonical_fedridge_r0_v2_20260812 tests/test_iotj_canonical_fedridge_r0_v2.py
git commit -m "docs: freeze canonical FedRidge R0-v2 protocol"
```

---

### Task 6: Implement the fail-closed R0-v2 runner without formal execution

**Files:**
- Create: `scripts/run_iotj_canonical_fedridge_r0_v2.py`
- Modify/Test: `tests/test_iotj_canonical_fedridge_r0_v2.py`

**Interfaces:**
- Consumes: v2 module, versioned cache API, frozen protocol, canonical hash verifier.
- Produces:
  - `build_r0_v2_execution_plan() -> list[str]`
  - `protocol_freeze_hash() -> str`
  - `preflight(data_root: Path, output: Path, authorized_freeze_commit: str) -> dict[str, Any]`
  - `run(data_root: Path, output: Path, authorized_freeze_commit: str) -> dict[str, Any]`
  - `audit(output: Path) -> dict[str, Any]`
  - `build_parser() -> argparse.ArgumentParser`

- [ ] **Step 1: Write failing parser, access-order, output, and commit-lock tests**

```python
from scripts import run_iotj_canonical_fedridge_r0_v2 as runner

DATA_ROOT = ROOT / "dataset/iotj_canonical_v1"

def test_r0_v2_parser_has_no_target_or_qc_argument():
    parser = runner.build_parser()
    options = {option for action in parser._actions for option in action.option_strings}
    assert "--target" not in options
    assert "--target-data" not in options
    assert "--qc" not in options

def test_r0_v2_execution_plan_locks_model_before_source_test():
    plan = runner.build_r0_v2_execution_plan()
    assert plan.index("write_source_alpha_and_model_locks") < plan.index("open_source_test")
    assert all("target" not in step.lower() for step in plan)

def test_r0_v2_preflight_rejects_wrong_commit_and_existing_output(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "_git_head", lambda: "actual")
    with pytest.raises(RuntimeError, match="authorized freeze commit"):
        runner.preflight(DATA_ROOT, tmp_path / "out", "different")
    output = tmp_path / "occupied"
    output.mkdir()
    (output / "partial.txt").write_text("evidence", encoding="utf-8")
    with pytest.raises(FileExistsError, match="output"):
        runner.preflight(DATA_ROOT, output, "actual")
```

Also assert old C0/R0 roots are read-only prerequisites, not descendants of the v2 output; changed protocol hash, dataset hash, extractor hash, or original decision fails closed.

- [ ] **Step 2: Run runner-contract tests and verify RED**

```powershell
python -m pytest -q tests/test_iotj_canonical_fedridge_r0_v2.py -k 'parser or execution_plan or preflight or commit or output'
```

Expected: FAIL because the runner is absent.

- [ ] **Step 3: Implement pure preflight and parser**

Parser stages:

```text
preflight
run
audit
```

Require `--authorized-freeze-commit` for preflight/run. Preflight verifies current HEAD equality, protocol hash/status, formal-execution false, canonical aggregate hash, feature extractor hashes, source counts, original C0/R0 decisions/audits, absent output, no target path, and no existing completion marker. Preflight must not create the output root.

- [ ] **Step 4: Write failing synthetic pipeline and failure-preservation tests**

Inject a `SourceDataProvider` protocol into an internal `_execute_source_only(provider, output, protocol)` function so tests can supply synthetic C1/C2 train/calibration/test mappings without canonical formal execution.

```python
@dataclass(frozen=True)
class SourceRequest:
    client: str
    split: str
    gas_id: int | None

class SourceDataProvider(Protocol):
    def build_fresh_cache(self, client: str, split: str) -> Mapping[str, Any]: ...
    def gas_data(self, client: str, split: str, gas_id: int) -> tuple[np.ndarray, np.ndarray]: ...

def synthetic_four_gas_source_data(
) -> dict[tuple[str, str, int], tuple[np.ndarray, np.ndarray]]:
    rows: dict[tuple[str, str, int], tuple[np.ndarray, np.ndarray]] = {}
    split_offsets = {"train": 0.0, "calibration": 0.25, "test": 0.5}
    for gas_id in range(4):
        for client_index, client in enumerate(("C1", "C2")):
            for split, split_offset in split_offsets.items():
                n = 6 if split == "train" else 4
                signal = np.arange(n, dtype=np.float64) + client_index + split_offset
                x = np.zeros((n, 104), dtype=np.float64)
                x[:, 0] = signal
                x[:, 1] = 1.0
                x[:, 2] = gas_id
                y = 2.0 * signal + float(gas_id)
                rows[(client, split, gas_id)] = (x, y)
    return rows

def frozen_protocol_fixture() -> dict[str, object]:
    return {
        "study_id": R0_V2_STUDY_ID,
        "source_clients": ["C1", "C2"],
        "target_clients": [],
        "feature_names": [f"f{i}" for i in range(104)],
        "alpha_grid": [0.0, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
        "formal_execution_started": False,
        "numerical_gates": asdict(registered_tolerances_v2()),
    }

def force_synthetic_alpha(monkeypatch) -> None:
    def fixed_selection(*args, **kwargs):
        return 0.1, [{
            "alpha": 0.1,
            "source_calibration_RMSE": 0.0,
            "target_input_accessed": False,
            "source_test_accessed": False,
        }]
    monkeypatch.setattr(runner, "select_source_alpha_v2", fixed_selection)
    monkeypatch.setattr(runner, "select_pooled_alpha_v2", fixed_selection)

class RecordingSyntheticProvider:
    def __init__(self) -> None:
        self.requests: list[SourceRequest] = []
        self.data = synthetic_four_gas_source_data()

    def build_fresh_cache(self, client: str, split: str) -> Mapping[str, Any]:
        self.requests.append(SourceRequest(client, split, None))
        return {"client": client, "split": split, "study_id": R0_V2_STUDY_ID}

    def gas_data(self, client: str, split: str, gas_id: int) -> tuple[np.ndarray, np.ndarray]:
        self.requests.append(SourceRequest(client, split, gas_id))
        return self.data[(client, split, gas_id)]

def expected_source_only_request_sequence() -> list[SourceRequest]:
    expected: list[SourceRequest] = []
    for split in ("train", "calibration"):
        for client in ("C1", "C2"):
            expected.append(SourceRequest(client, split, None))
    for gas_id in range(4):
        for split in ("train", "calibration"):
            for client in ("C1", "C2"):
                expected.append(SourceRequest(client, split, gas_id))
    for client in ("C1", "C2"):
        expected.append(SourceRequest(client, "test", None))
    for gas_id in range(4):
        for client in ("C1", "C2"):
            expected.append(SourceRequest(client, "test", gas_id))
    return expected

def test_synthetic_runner_writes_all_diagnostics_and_never_requests_target(
    tmp_path, monkeypatch
):
    force_synthetic_alpha(monkeypatch)
    provider = RecordingSyntheticProvider()
    result = runner._execute_source_only(provider, tmp_path / "run", frozen_protocol_fixture())
    assert result["decision"] == "FEDRIDGE_ALGEBRAIC_EXACT_NUMERICAL_EQUIVALENCE_ESTABLISHED"
    assert provider.requests == expected_source_only_request_sequence()
    assert not any(request.client in {"C3", "C4", "C5"} for request in provider.requests)

def test_failed_synthetic_runner_preserves_partial_evidence_without_completion(
    tmp_path, monkeypatch
):
    force_synthetic_alpha(monkeypatch)
    output = tmp_path / "run"
    provider = RecordingSyntheticProvider()
    original = runner.functional_diagnostics_v2
    def forced_failure(*args, **kwargs):
        row = original(*args, **kwargs)
        row["raw_prediction_pass"] = False
        return row
    monkeypatch.setattr(runner, "functional_diagnostics_v2", forced_failure)
    result = runner._execute_source_only(provider, output, frozen_protocol_fixture())
    assert result["decision"] == "R0_V2_FAILED"
    assert (output / "R0_V2_DECISION.json").is_file()
    assert (output / "R0_V2_EXPERIMENT_AUDIT.md").is_file()
    assert not (output / "fixed_endpoint_complete.json").exists()
```

- [ ] **Step 5: Implement source-only pipeline and evidence writers**

The production provider calls the versioned `build_feature_cache(..., study_id=R0_V2_STUDY_ID)` for C1/C2 train/calibration, and only after model locks for C1/C2 test. It must write every formal file in the spec, environment/BLAS metadata, an access-event audit, and SHA256 index.

The PASS path writes `fixed_endpoint_complete.json` with `R1_released=true` but does not call R1. The FAIL path writes decision/failure audit/hash index and returns/stops without completion marker.

- [ ] **Step 6: Implement audit-only mode**

Audit recomputes every indexed SHA256, validates protocol and decision vocabularies, validates access events, requires all expected files for PASS, and never regenerates metrics.

- [ ] **Step 7: Run all v2 tests and original R0 regression tests**

```powershell
python -m pytest -q tests/test_iotj_canonical_fedridge_r0_v2.py
python -m pytest -q tests/test_iotj_canonical_regression_reconstruction.py
```

Expected: PASS with zero failures.

- [ ] **Step 8: Verify preflight does not create formal output**

Do not run preflight with an intentionally non-authorized dummy commit. Instead test the preflight through pytest monkeypatching and confirm directly:

```powershell
Test-Path 'results/iotj_canonical_v1_final/canonical_fedridge_r0_v2_20260812'
```

Expected: `False`.

- [ ] **Step 9: Commit the runner**

```powershell
git add -- scripts/run_iotj_canonical_fedridge_r0_v2.py tests/test_iotj_canonical_fedridge_r0_v2.py
git commit -m "feat: add fail-closed canonical R0-v2 runner"
```

---

### Task 7: Complete pre-run audit, handoff update, verification, and freeze commit

**Files:**
- Modify: `代码文件介绍.md`
- Modify if generated by static audit only: `docs/experiments/iotj_canonical_v1_final/canonical_fedridge_r0_v2_20260812/protocol_manifest.json`
- Verify all files from Tasks 1-6

**Interfaces:**
- Consumes: complete code, tests, and protocol bundle.
- Produces: one reported pre-run freeze commit with no formal result root.

- [ ] **Step 1: Update the Chinese handoff guide**

Add R0-v2 as:

```text
DESIGN_FREEZE_READY_FORMAL_NOT_STARTED
```

Record the new module/runner/protocol paths, frozen gates, old R0 preservation, no target/QC, and exact future run command shape without claiming execution.

- [ ] **Step 2: Run focused test suites**

```powershell
python -m pytest -q tests/test_iotj_canonical_fedridge_r0_v2.py
python -m pytest -q tests/test_iotj_canonical_regression_reconstruction.py
python -m pytest -q tests/test_iotj_canonical_v1_dataset.py
```

Expected: all PASS.

- [ ] **Step 3: Run compilation and canonical hash verification**

```powershell
python -m compileall -q gaps_flower scripts tools tests
python tools/verify_iotj_canonical_v1_hashes.py dataset/iotj_canonical_v1
```

Expected: compile exit 0; canonical verifier reports `status=PASS`, 71 checked files, and the frozen aggregate SHA256.

- [ ] **Step 4: Run static access and artifact audits**

```powershell
rg -n "C3|C4|C5|target_|QC|q0|q1" gaps_flower/canonical_fedridge_v2.py scripts/run_iotj_canonical_fedridge_r0_v2.py
rg -n "sum_x2\s*/|sum_x2.*mean" gaps_flower/canonical_fedridge_v2.py
git diff --check
```

Review every match. Allowed matches are explicit rejection/audit text only; no target loader/path and no raw-moment variance formula may exist.

- [ ] **Step 5: Verify old assets and absent formal result root**

Recheck original C0/R0 SHA indexes and verify:

```powershell
Test-Path 'results/iotj_canonical_v1_final/canonical_fedridge_r0_v2_20260812'
```

Expected: `False`.

- [ ] **Step 6: Review intended diff and exclude unrelated files**

```powershell
git -c core.quotepath=false status --short
git -c core.quotepath=false diff --name-only
git diff --check
```

Do not stage watcher logs, preprocessing logs, or `.tmp_pytest_*` directories.

- [ ] **Step 7: Create the pre-run freeze commit**

```powershell
git add -- gaps_flower/canonical_fedridge_v2.py gaps_flower/canonical_quantitative_features.py scripts/run_iotj_canonical_fedridge_r0_v2.py tests/test_iotj_canonical_fedridge_r0_v2.py docs/experiments/iotj_canonical_v1_final/canonical_fedridge_r0_v2_20260812 '代码文件介绍.md' docs/superpowers/plans/2026-08-12-canonical-fedridge-r0-v2.md
git commit -m "experiment: freeze canonical FedRidge R0-v2 pre-run"
git push origin codex/iotj-final-classification-le1
```

- [ ] **Step 8: Verify local/remote identity and stop**

```powershell
git rev-parse HEAD
git ls-remote origin refs/heads/codex/iotj-final-classification-le1
```

Require identical hashes. Report protocol summary, near-constant policy, numerical gates, tests, canonical hash, pre-run commit, and `formal_execution_started=false`. Do not run `scripts/run_iotj_canonical_fedridge_r0_v2.py run`.

---

## Execution Stop Condition

This plan is complete when the versioned code, tests, protocol bundle, handoff guide, and pre-run freeze commit are pushed and the formal result root is still absent. Formal R0-v2 execution requires a separate user authorization that names the reported pre-run freeze commit.
