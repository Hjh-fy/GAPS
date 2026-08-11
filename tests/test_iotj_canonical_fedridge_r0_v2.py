from dataclasses import FrozenInstanceError
from typing import Mapping

import numpy as np
import pytest

from gaps_flower.canonical_fedridge_v2 import (
    RIDGE_ALPHAS,
    LocalCentralMomentsV2,
    LocalNormalEquationsV2,
    SCALE_FLOOR,
    StableGlobalScalerV2,
    aggregate_normal_equations_v2,
    feature_numerical_audit_rows,
    local_central_moments,
    local_normal_equations_v2,
    merge_central_moments,
    pooled_reference_fit_v2,
    reconstruct_ridge_v2,
    select_pooled_alpha_v2,
    select_source_alpha_v2,
)
from gaps_flower.canonical_quantitative_features import validate_cache_manifest


R0_V2_STUDY_ID = "CAN-V1-FEDRIDGE-R0V2-20260812"


def synthetic_two_client_regression() -> dict[
    str, tuple[np.ndarray, np.ndarray]
]:
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
    *,
    role: str = "refit",
) -> StableGlobalScalerV2:
    return merge_central_moments(
        [
            local_central_moments(client, 0, role, values)
            for client, (values, _targets) in client_data.items()
        ]
    )


def standardized_pooled_design(
    client_data: Mapping[str, tuple[np.ndarray, np.ndarray]],
    scaler: StableGlobalScalerV2,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.vstack([client_data[client][0] for client in ("C1", "C2")])
    targets = np.concatenate(
        [client_data[client][1] for client in ("C1", "C2")]
    )
    z = (values - scaler.mean) / scaler.scale
    return np.column_stack([np.ones(len(z), dtype=np.float64), z]), targets


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


def test_cache_manifest_requires_explicit_r0_v2_study_identity() -> None:
    """Catches validating a cache against the legacy study identity."""
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


def test_cache_manifest_old_default_study_identity_is_unchanged() -> None:
    """Catches changing the legacy default cache-study contract."""
    manifest = canonical_cache_manifest_fixture(study_id="CAN-V1-CRRQ-20260811")
    validate_cache_manifest(manifest, expected_dataset_sha256="a" * 64)


def test_v2_constant_and_near_constant_features_use_registered_safe_scale() -> None:
    """Catches input-order merging or retaining a sub-floor raw scale."""
    x1 = np.array([[10.0, 10.0], [10.0, 10.0 + 1e-14]], dtype=np.float64)
    x2 = np.array([[10.0, 10.0 - 1e-14], [10.0, 10.0]], dtype=np.float64)
    scaler = merge_central_moments(
        [
            local_central_moments("C2", 3, "refit", x2),
            local_central_moments("C1", 3, "refit", x1),
        ]
    )

    assert scaler.aggregation_order == ("C1", "C2")
    assert np.array_equal(scaler.safe_scale_mask, [True, True])
    assert np.array_equal(scaler.scale, [1.0, 1.0])


def test_v2_large_offset_small_variance_avoids_raw_moment_cancellation() -> None:
    """Catches replacing central moments with E[x^2] - E[x]^2."""
    base = np.float64(2**40)
    step = np.float64(2**-10)
    x1 = np.array([[base], [base + step]], dtype=np.float64)
    x2 = np.array([[base + 2 * step], [base + 3 * step]], dtype=np.float64)
    scaler = merge_central_moments(
        [
            local_central_moments("C1", 1, "refit", x1),
            local_central_moments("C2", 1, "refit", x2),
        ]
    )
    pooled = np.vstack([x1, x2])
    naive_variance = np.mean(pooled[:, 0] ** 2) - np.mean(pooled[:, 0]) ** 2

    assert naive_variance == 0.0
    assert scaler.raw_scale[0] == pytest.approx(
        np.std(pooled[:, 0], ddof=0), rel=1e-10
    )
    assert scaler.raw_scale[0] > 0.0


def test_v2_population_variance_divides_merged_m2_by_n() -> None:
    """Catches accidentally applying the sample-variance n-1 denominator."""
    scaler = merge_central_moments(
        [
            local_central_moments(
                "C1", 2, "refit", np.array([[0.0]], dtype=np.float64)
            ),
            local_central_moments(
                "C2", 2, "refit", np.array([[2.0]], dtype=np.float64)
            ),
        ]
    )

    assert scaler.n == 2
    assert scaler.mean[0] == 1.0
    assert scaler.variance[0] == 1.0
    assert scaler.raw_scale[0] == 1.0


def test_v2_safe_scale_boundary_is_strictly_less_than_one_e_minus_nine() -> None:
    """Catches flooring a scale equal to the registered boundary."""
    exactly = np.array([[-1e-9], [1e-9]], dtype=np.float64)
    scaler = merge_central_moments(
        [
            local_central_moments("C1", 0, "refit", exactly),
            local_central_moments("C2", 0, "refit", exactly),
        ]
    )

    assert scaler.raw_scale[0] == pytest.approx(1e-9)
    assert not scaler.safe_scale_mask[0]
    assert scaler.scale[0] == scaler.raw_scale[0]


def test_v2_local_and_merged_numeric_outputs_are_immutable_float64() -> None:
    """Catches dtype drift or mutable arrays in the frozen audit records."""
    x = np.array([[1.0, 3.0], [2.0, 5.0]], dtype=np.float32)
    first = local_central_moments("C1", 4, "refit", x)
    second = local_central_moments("C2", 4, "refit", x)
    scaler = merge_central_moments([first, second])

    for values in (first.mean, first.m2, first.minimum, first.maximum):
        assert values.dtype == np.float64
        assert not values.flags.writeable
    for values in (
        scaler.mean,
        scaler.variance,
        scaler.raw_scale,
        scaler.scale,
        scaler.minimum,
        scaler.maximum,
    ):
        assert values.dtype == np.float64
        assert not values.flags.writeable
    assert scaler.safe_scale_mask.dtype == np.bool_
    assert not scaler.safe_scale_mask.flags.writeable
    with pytest.raises(ValueError, match="WRITEABLE"):
        first.mean.setflags(write=True)
    with pytest.raises(FrozenInstanceError):
        first.n = 99  # type: ignore[misc]


def test_v2_direct_local_record_construction_normalizes_immutable_arrays() -> None:
    """Catches bypassing immutable storage through the public dataclass API."""
    writable = np.array([1.0], dtype=np.float64)
    record = LocalCentralMomentsV2(
        client_id="C1",
        gas_id=1,
        role="refit",
        n=1,
        mean=writable,
        m2=writable,
        minimum=writable,
        maximum=writable,
        provenance_sha256="a" * 64,
    )

    for values in (record.mean, record.m2, record.minimum, record.maximum):
        assert not values.flags.writeable
        with pytest.raises(ValueError, match="WRITEABLE"):
            values.setflags(write=True)


@pytest.mark.parametrize(
    ("client_ids", "match"),
    [
        (("C1",), "missing"),
        (("C1", "C1"), "duplicate"),
        (("C1", "C2", "C3"), "extra"),
    ],
)
def test_v2_merge_rejects_noncanonical_client_sets(
    client_ids: tuple[str, ...], match: str
) -> None:
    """Catches accepting incomplete, duplicated, or expanded client sets."""
    records = [
        local_central_moments(
            client_id, 1, "refit", np.array([[float(index)]], dtype=np.float64)
        )
        for index, client_id in enumerate(client_ids)
    ]

    with pytest.raises(ValueError, match=match):
        merge_central_moments(records)


@pytest.mark.parametrize("mismatch", ["gas", "role", "dimension"])
def test_v2_merge_rejects_gas_role_and_dimension_mismatches(mismatch: str) -> None:
    """Catches combining records from incompatible canonical partitions."""
    first = local_central_moments(
        "C1", 1, "refit", np.array([[1.0, 2.0]], dtype=np.float64)
    )
    gas_id = 2 if mismatch == "gas" else 1
    role = "calibration" if mismatch == "role" else "refit"
    values = (
        np.array([[3.0]], dtype=np.float64)
        if mismatch == "dimension"
        else np.array([[3.0, 4.0]], dtype=np.float64)
    )
    second = local_central_moments("C2", gas_id, role, values)

    with pytest.raises(ValueError, match=mismatch):
        merge_central_moments([first, second])


@pytest.mark.parametrize(
    "values",
    [
        np.empty((0, 2), dtype=np.float64),
        np.empty((2, 0), dtype=np.float64),
        np.array([[np.nan]], dtype=np.float64),
        np.array([[np.inf]], dtype=np.float64),
    ],
)
def test_v2_local_moments_reject_empty_or_nonfinite_input(values: np.ndarray) -> None:
    """Catches allowing unusable matrices into irreversible local summaries."""
    with pytest.raises(ValueError, match="non-empty finite"):
        local_central_moments("C1", 1, "refit", values)


def test_v2_local_moments_avoid_mean_overflow_for_finite_constant_input() -> None:
    """Catches overflowing the mean while reducing finite same-sign values."""
    maximum = np.finfo(np.float64).max
    record = local_central_moments(
        "C1", 1, "refit", np.array([[maximum], [maximum]], dtype=np.float64)
    )

    assert record.mean[0] == maximum
    assert record.m2[0] == 0.0


@pytest.mark.parametrize("rows", [3, 7])
def test_v2_local_moments_avoid_scaled_sum_overflow_for_maximum_constants(
    rows: int,
) -> None:
    """Catches fallback overflow from summing repeated maximum/n values."""
    maximum = np.finfo(np.float64).max
    values = np.full((rows, 1), maximum, dtype=np.float64)

    record = local_central_moments("C1", 1, "refit", values)

    assert record.mean[0] == maximum
    assert record.m2[0] == 0.0


def test_v2_local_moments_preserve_constant_smallest_subnormal_mean() -> None:
    """Catches underflowing inputs by dividing before the mean reduction."""
    smallest = np.nextafter(np.float64(0.0), np.float64(1.0))
    record = local_central_moments(
        "C1", 1, "refit", np.array([[smallest], [smallest]], dtype=np.float64)
    )

    assert record.mean[0] == smallest
    assert record.m2[0] == 0.0


def test_v2_merge_rejects_variance_that_overflows_float64() -> None:
    """Catches publishing non-finite scaler fields from finite local records."""
    maximum = np.finfo(np.float64).max
    records = [
        local_central_moments(
            "C1", 1, "refit", np.array([[maximum]], dtype=np.float64)
        ),
        local_central_moments(
            "C2", 1, "refit", np.array([[-maximum]], dtype=np.float64)
        ),
    ]

    with pytest.raises(ValueError, match="overflow"):
        merge_central_moments(records)


def test_v2_feature_numerical_audit_emits_exact_104d_rows() -> None:
    """Catches missing, renamed, or incorrectly typed numerical-audit fields."""
    feature_names = tuple(f"h1_{index:03d}" for index in range(104))
    x1 = np.arange(104, dtype=np.float64).reshape(1, 104)
    x2 = (np.arange(104, dtype=np.float64) + 2.0).reshape(1, 104)
    records = [
        local_central_moments("C1", 3, "refit", x1),
        local_central_moments("C2", 3, "refit", x2),
    ]
    scaler = merge_central_moments(records)
    rows = feature_numerical_audit_rows(records, scaler, feature_names)

    assert len(rows) == 104
    assert tuple(rows[0]) == (
        "gas_id",
        "role",
        "feature_index",
        "feature_name",
        "n",
        "minimum",
        "maximum",
        "mean",
        "population_variance",
        "raw_scale",
        "dynamic_range",
        "safe_scale_floor",
        "safe_scale_applied",
        "canonical_scale",
        "aggregation_order",
        "dtype",
    )
    assert rows[0] == {
        "gas_id": 3,
        "role": "refit",
        "feature_index": 0,
        "feature_name": "h1_000",
        "n": 2,
        "minimum": 0.0,
        "maximum": 2.0,
        "mean": 1.0,
        "population_variance": 1.0,
        "raw_scale": 1.0,
        "dynamic_range": 2.0,
        "safe_scale_floor": SCALE_FLOOR,
        "safe_scale_applied": False,
        "canonical_scale": 1.0,
        "aggregation_order": ("C1", "C2"),
        "dtype": "float64",
    }


def test_v2_feature_numerical_audit_rejects_feature_dimension_mismatch() -> None:
    """Catches silently truncating the registered feature-name order."""
    x = np.ones((1, 104), dtype=np.float64)
    records = [
        local_central_moments("C1", 1, "refit", x),
        local_central_moments("C2", 1, "refit", x),
    ]
    scaler = merge_central_moments(records)

    with pytest.raises(ValueError, match="feature-name dimension"):
        feature_numerical_audit_rows(records, scaler, ["only_one"])


def test_v2_feature_numerical_audit_requires_exactly_104_dimensions() -> None:
    """Catches emitting a numerically aligned but non-canonical audit table."""
    x = np.ones((1, 2), dtype=np.float64)
    records = [
        local_central_moments("C1", 1, "refit", x),
        local_central_moments("C2", 1, "refit", x),
    ]
    scaler = merge_central_moments(records)

    with pytest.raises(ValueError, match="exactly 104"):
        feature_numerical_audit_rows(records, scaler, ["one", "two"])


def test_v2_normal_equations_match_same_standardized_pooled_rows() -> None:
    """Catches input-order aggregation or a design without the intercept."""
    client_data = synthetic_two_client_regression()
    scaler = stable_scaler_for(client_data)
    local = [
        local_normal_equations_v2(client, 0, "refit", x, y, scaler)
        for client, (x, y) in reversed(list(client_data.items()))
    ]
    federated = aggregate_normal_equations_v2(local)
    pooled_design, pooled_y = standardized_pooled_design(client_data, scaler)

    assert federated.aggregation_order == ("C1", "C2")
    assert np.allclose(federated.a, pooled_design.T @ pooled_design)
    assert np.allclose(federated.b, pooled_design.T @ pooled_y)
    assert federated.y_y == pytest.approx(float(pooled_y @ pooled_y))


def test_v2_ridge_uses_an_unregularized_intercept_and_exact_alpha_grid() -> None:
    """Catches shrinking the intercept or changing the registered grid."""
    client_data = synthetic_two_client_regression()
    scaler = stable_scaler_for(client_data)
    equations = aggregate_normal_equations_v2(
        [
            local_normal_equations_v2(
                client, 0, "refit", values, targets, scaler
            )
            for client, (values, targets) in client_data.items()
        ]
    )
    model = reconstruct_ridge_v2(
        equations, scaler, ["x0", "x1"], alpha=1000.0
    )

    expected_regularizer = np.eye(3, dtype=np.float64) * 1000.0
    expected_regularizer[0, 0] = 0.0
    expected_coef = np.linalg.pinv(equations.a + expected_regularizer) @ equations.b
    assert np.allclose(model.coef, expected_coef)
    assert model.intercept_regularized is False
    assert model.to_json()["intercept_regularized"] is False
    assert model.to_json()["solver"] == "numpy.linalg.pinv"
    assert RIDGE_ALPHAS == (0.0, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)


def test_v2_ridge_prediction_exposes_raw_and_clipped_parity_paths() -> None:
    """Catches clipping the raw path or failing to clip to source-train labels."""
    client_data = synthetic_two_client_regression()
    scaler = stable_scaler_for(client_data)
    equations = aggregate_normal_equations_v2(
        [
            local_normal_equations_v2(
                client, 0, "refit", values, targets, scaler
            )
            for client, (values, targets) in client_data.items()
        ]
    )
    model = reconstruct_ridge_v2(equations, scaler, ("x0", "x1"), 0.0)
    extrapolation = np.array([[100.0, 1.0]], dtype=np.float64)

    raw = model.predict_matrix(extrapolation, clip=False)
    clipped = model.predict_matrix(extrapolation)
    assert raw[0] > equations.y_max
    assert clipped[0] == equations.y_max
    assert model.clip_min == equations.y_min
    assert model.clip_max == equations.y_max


def test_v2_pooled_reference_fit_returns_equations_for_its_exact_rows() -> None:
    """Catches fitting pooled coefficients from rows unlike its returned A/b."""
    client_data = synthetic_two_client_regression()
    values = np.vstack([client_data[client][0] for client in ("C1", "C2")])
    targets = np.concatenate(
        [client_data[client][1] for client in ("C1", "C2")]
    )
    model, equations = pooled_reference_fit_v2(
        values,
        targets,
        gas_id=0,
        role="source_train",
        feature_names=("x0", "x1"),
        alpha=0.1,
    )

    design = np.column_stack(
        [np.ones(len(values), dtype=np.float64), (values - model.mean) / model.scale]
    )
    assert equations.aggregation_order == ("POOLED",)
    assert np.allclose(equations.a, design.T @ design)
    assert np.allclose(equations.b, design.T @ targets)


def test_v2_source_alpha_loops_use_exact_grid_and_first_tie() -> None:
    """Catches a changed grid, un-clipped score, or later-alpha tie break."""
    train = {
        "C2": (
            np.array([[2.0], [3.0]], dtype=np.float64),
            np.array([3.0, 3.0], dtype=np.float64),
        ),
        "C1": (
            np.array([[0.0], [1.0]], dtype=np.float64),
            np.array([3.0, 3.0], dtype=np.float64),
        ),
    }
    calibration = {
        "C2": (np.array([[20.0]], dtype=np.float64), np.array([3.0])),
        "C1": (np.array([[-20.0]], dtype=np.float64), np.array([3.0])),
    }

    federated_alpha, federated_audit = select_source_alpha_v2(
        train, calibration, gas_id=0, feature_names=("x0",)
    )
    pooled_alpha, pooled_audit = select_pooled_alpha_v2(
        train, calibration, gas_id=0, feature_names=("x0",)
    )

    assert federated_alpha == pooled_alpha == RIDGE_ALPHAS[0]
    assert [row["alpha"] for row in federated_audit] == list(RIDGE_ALPHAS)
    assert [row["alpha"] for row in pooled_audit] == list(RIDGE_ALPHAS)
    assert all(row["source_calibration_N"] == 2 for row in federated_audit)
    assert all(row["source_calibration_N"] == 2 for row in pooled_audit)
    assert all(row["target_input_accessed"] is False for row in federated_audit)
    assert all(row["source_test_accessed"] is False for row in federated_audit)
    assert all(row["target_input_accessed"] is False for row in pooled_audit)
    assert all(row["source_test_accessed"] is False for row in pooled_audit)
    assert np.allclose(
        [row["source_calibration_RMSE"] for row in federated_audit],
        [row["source_calibration_RMSE"] for row in pooled_audit],
    )


@pytest.mark.parametrize(
    ("train_role", "validation_role", "train_keys", "calibration_keys"),
    [
        ("target_train", "source_calibration", ("C1", "C2"), ("C1", "C2")),
        ("source_train", "source_test", ("C1", "C2"), ("C1", "C2")),
        ("source_train", "source_calibration", ("C1_target", "C2"), ("C1", "C2")),
        ("source_train", "source_calibration", ("C1", "C2"), ("C1_test", "C2")),
    ],
)
def test_v2_source_role_gate_rejects_target_or_test_semantics(
    train_role: str,
    validation_role: str,
    train_keys: tuple[str, str],
    calibration_keys: tuple[str, str],
) -> None:
    """Catches target/test data entering either alpha-selection loop."""
    base = synthetic_two_client_regression()
    train = {key: base[client] for key, client in zip(train_keys, ("C1", "C2"))}
    calibration = {
        key: base[client]
        for key, client in zip(calibration_keys, ("C1", "C2"))
    }

    for selector in (select_source_alpha_v2, select_pooled_alpha_v2):
        with pytest.raises(RuntimeError, match="source-only"):
            selector(
                train,
                calibration,
                gas_id=0,
                feature_names=("x0", "x1"),
                train_role=train_role,
                validation_role=validation_role,
            )


def test_v2_equation_and_model_arrays_are_immutable_float64() -> None:
    """Catches dtype drift or reversible write flags in Task 3 records."""
    client_data = synthetic_two_client_regression()
    scaler = stable_scaler_for(client_data)
    local = [
        local_normal_equations_v2(client, 0, "refit", values, targets, scaler)
        for client, (values, targets) in client_data.items()
    ]
    equations = aggregate_normal_equations_v2(local)
    model = reconstruct_ridge_v2(equations, scaler, ("x0", "x1"), 0.1)

    for values in (local[0].a, local[0].b, equations.a, equations.b, model.coef):
        assert values.dtype == np.float64
        assert not values.flags.writeable
        with pytest.raises(ValueError, match="WRITEABLE"):
            values.setflags(write=True)


def test_v2_normal_equations_fail_closed_when_finite_products_overflow() -> None:
    """Catches publishing infinite y'y from finite local labels."""
    values = np.array([[0.0], [1.0]], dtype=np.float64)
    scaler = merge_central_moments(
        [
            local_central_moments("C1", 0, "refit", values),
            local_central_moments("C2", 0, "refit", values),
        ]
    )

    with pytest.raises(ValueError, match="overflow"):
        local_normal_equations_v2(
            "C1",
            0,
            "refit",
            values,
            np.full(2, np.finfo(np.float64).max, dtype=np.float64),
            scaler,
        )


def test_v2_aggregation_rejects_malformed_direct_record_shapes() -> None:
    """Catches an IndexError escape from a scalar b in the public record API."""
    malformed = LocalNormalEquationsV2(
        client_id="C1",
        gas_id=0,
        role="refit",
        n=1,
        a=np.eye(2, dtype=np.float64),
        b=np.array(1.0, dtype=np.float64),
        y_y=1.0,
        y_min=1.0,
        y_max=1.0,
        provenance_sha256="a" * 64,
    )
    valid = LocalNormalEquationsV2(
        client_id="C2",
        gas_id=0,
        role="refit",
        n=1,
        a=np.eye(2, dtype=np.float64),
        b=np.ones(2, dtype=np.float64),
        y_y=1.0,
        y_min=1.0,
        y_max=1.0,
        provenance_sha256="b" * 64,
    )

    with pytest.raises(ValueError, match="dimension"):
        aggregate_normal_equations_v2([malformed, valid])


def test_v2_alpha_contract_rejects_nonexact_role_and_boolean_grid() -> None:
    """Catches non-string roles or bool-as-1.0 passing source-only selection."""
    data = synthetic_two_client_regression()
    boolean_grid = (0.0, 0.01, 0.1, True, 10.0, 100.0, 1000.0)

    for selector in (select_source_alpha_v2, select_pooled_alpha_v2):
        with pytest.raises(RuntimeError, match="source-only"):
            selector(
                data,
                data,
                gas_id=0,
                feature_names=("x0", "x1"),
                train_role=None,  # type: ignore[arg-type]
            )
        with pytest.raises(RuntimeError, match="grid"):
            selector(
                data,
                data,
                gas_id=0,
                feature_names=("x0", "x1"),
                alphas=boolean_grid,
            )
        with pytest.raises(RuntimeError, match="grid"):
            selector(
                data,
                data,
                gas_id=0,
                feature_names=("x0", "x1"),
                alphas=None,  # type: ignore[arg-type]
            )
