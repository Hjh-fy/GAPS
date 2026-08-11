import csv
import json
import re
from dataclasses import FrozenInstanceError, asdict
from pathlib import Path
from typing import Mapping

import numpy as np
import pytest

from gaps_flower.canonical_fedridge_v2 import (
    AggregatedNormalEquationsV2,
    CanonicalRidgeModelV2,
    RIDGE_ALPHAS,
    LocalCentralMomentsV2,
    LocalNormalEquationsV2,
    SCALE_FLOOR,
    StableGlobalScalerV2,
    aggregate_normal_equations_v2,
    decide_gas_equivalence_v2,
    decide_r0_v2,
    feature_numerical_audit_rows,
    functional_diagnostics_v2,
    local_central_moments,
    local_normal_equations_v2,
    merge_central_moments,
    normal_equation_diagnostics_v2,
    pooled_reference_fit_v2,
    registered_tolerances_v2,
    reconstruct_ridge_v2,
    scaler_diagnostics_v2,
    select_pooled_alpha_v2,
    select_source_alpha_v2,
    system_diagnostics_v2,
)
from gaps_flower.canonical_quantitative_features import validate_cache_manifest
from scripts import run_iotj_canonical_fedridge_r0_v2 as runner


R0_V2_STUDY_ID = "CAN-V1-FEDRIDGE-R0V2-20260812"
ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_ROOT = (
    ROOT
    / "docs/experiments/iotj_canonical_v1_final"
    / "canonical_fedridge_r0_v2_20260812"
)
PROTOCOL_MANIFEST = PROTOCOL_ROOT / "protocol_manifest.json"
EXPERIMENT_MATRIX = PROTOCOL_ROOT / "EXPERIMENT_MATRIX.csv"
EXPERIMENT_REGISTRY = PROTOCOL_ROOT / "EXPERIMENT_REGISTRY.csv"
FORMAL_RESULT_ROOT = (
    ROOT
    / "results/iotj_canonical_v1_final"
    / "canonical_fedridge_r0_v2_20260812"
)
DATA_ROOT = ROOT / "dataset/iotj_canonical_v1"

PLANNER_FIELDS = [
    "experiment_id",
    "source_clients",
    "target_clients",
    "split_protocol",
    "model",
    "checkpoint",
    "DA",
    "calibration",
    "QC",
    "seed",
    "result_path",
    "metrics",
    "status",
    "notes",
    "code_commit",
    "config_path",
    "dataset_path",
    "created_at",
    "evidence_status",
    "provenance",
    "hypothesis_id",
    "baseline_id",
    "ablation_factor",
    "expected_evidence",
    "acceptance_criterion",
]
REGISTRY_FIELDS = PLANNER_FIELDS[:20]
REQUIRED_PROTOCOL_MARKDOWN = (
    "PROTOCOL.md",
    "EXPERIMENT_PLAN.md",
    "NEAR_CONSTANT_SCALE_POLICY.md",
    "R0_V2_NUMERICAL_TOLERANCE_JUSTIFICATION.md",
    "FEDRIDGE_NUMERICAL_STABILITY_MANUSCRIPT_NOTE.md",
)
REQUIRED_BUNDLE_FILES = REQUIRED_PROTOCOL_MARKDOWN + (
    "protocol_manifest.json",
    "EXPERIMENT_MATRIX.csv",
    "EXPERIMENT_REGISTRY.csv",
)
REGISTERED_METRIC_REFERENCES = (
    "protocol_manifest.json#/numerical_gates"
    "|future:r0_v2_scaler_diagnostics.csv"
    "|future:r0_v2_normal_equation_diagnostics.csv"
    "|future:r0_v2_system_diagnostics.csv"
    "|future:r0_v2_functional_equivalence.csv"
    "|future:R0_V2_DECISION.json"
)
REQUIRED_FIELD_PROVENANCE = {
    "experiment_id",
    "source_clients",
    "target_clients",
    "split_protocol",
    "model",
    "checkpoint",
    "DA",
    "calibration",
    "QC",
    "seed",
    "result_path",
    "metrics",
    "status",
    "evidence_status",
    "code_commit",
    "config_path",
    "dataset_path",
}
EXECUTION_INSTRUCTION = re.compile(
    r"\b(?:open|load|evaluate|run|execute|conduct|perform|apply)\b"
    r"[^.\n]{0,80}\b(?:target|C3|C4|C5|QC)\b",
    flags=re.IGNORECASE,
)
EXPLICIT_PROHIBITION = re.compile(
    r"\b(?:no|not|never|forbidden|prohibited|must\s+not|do\s+not|"
    r"shall\s+not|unavailable|false|empty)\b",
    flags=re.IGNORECASE,
)


def unauthorized_target_qc_instructions(text: str) -> list[str]:
    findings: list[str] = []
    for line in text.splitlines():
        for match in EXECUTION_INSTRUCTION.finditer(line):
            if not EXPLICIT_PROHIBITION.search(line[: match.start()]):
                findings.append(match.group(0))
    return findings


class AccessRecordingMapping(
    Mapping[str, tuple[np.ndarray, np.ndarray]]
):
    def __init__(
        self, data: Mapping[str, tuple[np.ndarray, np.ndarray]]
    ) -> None:
        self._data = dict(data)
        self.value_reads: list[str] = []

    def __getitem__(self, key: str) -> tuple[np.ndarray, np.ndarray]:
        self.value_reads.append(key)
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)


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


@pytest.mark.parametrize("forbidden_mapping", ["train", "calibration"])
def test_v2_source_role_gate_scans_all_keys_before_any_value_access(
    forbidden_mapping: str,
) -> None:
    """Catches dereferencing one valid mapping before validating both key sets."""
    base = synthetic_two_client_regression()
    forbidden = {"C1_target": base["C1"], "C2": base["C2"]}

    for selector in (select_source_alpha_v2, select_pooled_alpha_v2):
        train = AccessRecordingMapping(
            forbidden if forbidden_mapping == "train" else base
        )
        calibration = AccessRecordingMapping(
            forbidden if forbidden_mapping == "calibration" else base
        )

        with pytest.raises(RuntimeError, match="source-only"):
            selector(
                train,
                calibration,
                gas_id=0,
                feature_names=("x0", "x1"),
            )

        assert train.value_reads == []
        assert calibration.value_reads == []


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


def diagnostic_scaler(
    *,
    mean: tuple[float, ...] = (0.0,),
    scale: tuple[float, ...] = (1.0,),
    minimum: tuple[float, ...] = (-1.0,),
    maximum: tuple[float, ...] = (1.0,),
    safe_scale_mask: tuple[bool, ...] = (False,),
) -> StableGlobalScalerV2:
    """Build a complete real scaler record for Task 4 diagnostic tests."""
    scale_array = np.asarray(scale, dtype=np.float64)
    return StableGlobalScalerV2(
        gas_id=0,
        role="source_refit",
        n=8,
        mean=np.asarray(mean, dtype=np.float64),
        variance=scale_array * scale_array,
        raw_scale=scale_array,
        scale=scale_array,
        safe_scale_mask=np.asarray(safe_scale_mask, dtype=np.bool_),
        minimum=np.asarray(minimum, dtype=np.float64),
        maximum=np.asarray(maximum, dtype=np.float64),
        aggregation_order=("C1", "C2"),
    )


def diagnostic_equations(
    *,
    a: np.ndarray | None = None,
    b: np.ndarray | None = None,
) -> AggregatedNormalEquationsV2:
    """Build a complete equation record; malformed numerics remain observable."""
    matrix = np.diag([4.0, 2.0]) if a is None else a
    vector = np.array([2.0, 1.0], dtype=np.float64) if b is None else b
    return AggregatedNormalEquationsV2(
        gas_id=0,
        role="source_refit",
        n=8,
        a=matrix,
        b=vector,
        y_y=5.0,
        y_min=0.0,
        y_max=10.0,
        aggregation_order=("C1", "C2"),
    )


def diagnostic_model(
    *,
    alpha: float = 0.0,
    coef: tuple[float, float] = (0.5, 0.5),
    clip_min: float = 0.0,
    clip_max: float = 10.0,
) -> CanonicalRidgeModelV2:
    return CanonicalRidgeModelV2(
        gas_id=0,
        role="source_refit",
        alpha=alpha,
        feature_names=("x0",),
        mean=np.array([0.0], dtype=np.float64),
        scale=np.array([1.0], dtype=np.float64),
        coef=np.asarray(coef, dtype=np.float64),
        clip_min=clip_min,
        clip_max=clip_max,
    )


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


def test_v2_tolerances_are_formula_derived_and_frozen() -> None:
    """Catches literal or result-tuned replacements for registered formulas."""
    tolerances = registered_tolerances_v2()
    epsilon = np.finfo(np.float64).eps

    def gamma(m: int) -> float:
        return (m * epsilon) / (1.0 - m * epsilon)

    assert tolerances.epsilon == epsilon
    assert tolerances.n_max == 1340
    assert tolerances.feature_dimensions == 104
    assert tolerances.design_dimensions == 105
    assert tolerances.tau_moment == pytest.approx(
        64.0 * gamma(1340), rel=0, abs=0
    )
    assert tolerances.tau_residual == pytest.approx(
        128.0 * gamma(105), rel=0, abs=0
    )
    assert tolerances.tau_functional_ppm == 1e-6


def test_v2_scaler_diagnostic_uses_registered_coordinate_scale() -> None:
    """Catches using only pooled mean/scale instead of the full S_j formula."""
    tolerances = registered_tolerances_v2()
    pooled = diagnostic_scaler(
        mean=(1.0,),
        scale=(2.0,),
        minimum=(-1e12,),
        maximum=(1e12,),
    )
    coordinate_scale = 2e12
    passing = diagnostic_scaler(
        mean=(1.0 + 0.5 * tolerances.tau_moment * coordinate_scale,),
        scale=(2.0 + 0.5 * tolerances.tau_moment * coordinate_scale,),
        minimum=(-1e12,),
        maximum=(1e12,),
    )
    failing = diagnostic_scaler(
        mean=(1.0 + 2.0 * tolerances.tau_moment * coordinate_scale,),
        scale=(2.0,),
        minimum=(-1e12,),
        maximum=(1e12,),
    )

    passing_row = scaler_diagnostics_v2(passing, pooled)
    failing_row = scaler_diagnostics_v2(failing, pooled)

    assert passing_row["coordinate_scale_max"] == coordinate_scale
    assert passing_row["mean_pass"] is True
    assert passing_row["scale_pass"] is True
    assert passing_row["scaler_pass"] is True
    assert failing_row["mean_pass"] is False
    assert failing_row["scaler_pass"] is False


def test_v2_scaler_diagnostic_requires_exact_safe_scale_mask_identity() -> None:
    """Catches tolerating a safe-scale-mask mismatch despite close scales."""
    pooled = diagnostic_scaler()
    federated = diagnostic_scaler(safe_scale_mask=(True,))

    row = scaler_diagnostics_v2(federated, pooled)

    assert row["safe_scale_mask_equal"] is False
    assert row["scaler_pass"] is False


def test_v2_normal_equation_diagnostic_uses_frobenius_and_l2_ratios() -> None:
    """Catches elementwise norms or a shared denominator for A and b."""
    pooled_a = np.array([[3.0, 1.0], [1.0, 2.0]], dtype=np.float64)
    fed_a = pooled_a + np.array([[0.5, 0.0], [0.0, -0.25]])
    pooled_b = np.array([2.0, -1.0], dtype=np.float64)
    fed_b = pooled_b + np.array([0.25, 0.5])

    row = normal_equation_diagnostics_v2(
        diagnostic_equations(a=fed_a, b=fed_b),
        diagnostic_equations(a=pooled_a, b=pooled_b),
    )

    assert row["relative_a_discrepancy"] == pytest.approx(
        np.linalg.norm(fed_a - pooled_a, ord="fro")
        / np.linalg.norm(pooled_a, ord="fro")
    )
    assert row["relative_b_discrepancy"] == pytest.approx(
        np.linalg.norm(fed_b - pooled_b) / np.linalg.norm(pooled_b)
    )
    assert row["normal_equations_pass"] is False


@pytest.mark.parametrize("zero_field", ["a", "b"])
def test_v2_normal_equation_zero_denominator_fails_closed(
    zero_field: str,
) -> None:
    """Catches substituting one for a zero diagnostic denominator."""
    pooled_a = np.zeros((2, 2)) if zero_field == "a" else np.eye(2)
    pooled_b = np.zeros(2) if zero_field == "b" else np.ones(2)
    row = normal_equation_diagnostics_v2(
        diagnostic_equations(a=pooled_a.copy(), b=pooled_b.copy()),
        diagnostic_equations(a=pooled_a, b=pooled_b),
    )

    assert row[f"{zero_field}_denominator_positive"] is False
    assert row["normal_equations_pass"] is False
    assert row["finite_pass"] is False


def test_v2_system_diagnostic_requires_exact_alpha_identity() -> None:
    """Catches approximate alpha comparison at the locked-model boundary."""
    equations = diagnostic_equations()
    row = system_diagnostics_v2(
        equations,
        equations,
        diagnostic_model(alpha=0.1),
        diagnostic_model(alpha=np.nextafter(0.1, 1.0)),
    )

    assert row["alpha_equal"] is False
    assert row["finite_pass"] is True


@pytest.mark.parametrize(
    "smallest_diagonal",
    [0.0, np.finfo(np.float64).eps / 2.0],
)
def test_v2_condition_gate_rejects_nonfinite_or_epsilon_unsafe_kappa(
    smallest_diagonal: float,
) -> None:
    """Catches accepting singular or kappa*epsilon >= 1 systems."""
    matrix = np.diag([1.0, smallest_diagonal]).astype(np.float64)
    coef = np.array([1.0, 1.0], dtype=np.float64)
    equations = diagnostic_equations(a=matrix, b=matrix @ coef)
    row = system_diagnostics_v2(
        equations,
        equations,
        diagnostic_model(coef=(1.0, 1.0)),
        diagnostic_model(coef=(1.0, 1.0)),
    )

    assert row["condition_pass"] is False


def test_v2_system_residual_uses_exact_registered_denominator() -> None:
    """Catches omitting ||b|| or replacing the spectral matrix norm."""
    matrix = np.array([[4.0, 1.0], [1.0, 3.0]], dtype=np.float64)
    vector = np.array([3.0, 2.0], dtype=np.float64)
    coef = np.array([0.5, 0.25], dtype=np.float64)
    equations = diagnostic_equations(a=matrix, b=vector)
    row = system_diagnostics_v2(
        equations,
        equations,
        diagnostic_model(coef=tuple(coef)),
        diagnostic_model(coef=tuple(coef)),
    )
    expected = np.linalg.norm(matrix @ coef - vector) / (
        np.linalg.norm(matrix, ord=2) * np.linalg.norm(coef)
        + np.linalg.norm(vector)
    )

    assert row["fed_relative_residual"] == pytest.approx(expected)
    assert row["pooled_relative_residual"] == pytest.approx(expected)


def test_v2_zero_system_and_beta_denominators_fail_closed() -> None:
    """Catches residual or coefficient denominator substitution."""
    zero_equations = diagnostic_equations(a=np.zeros((2, 2)), b=np.zeros(2))
    zero_model = diagnostic_model(coef=(0.0, 0.0))

    row = system_diagnostics_v2(
        zero_equations, zero_equations, zero_model, zero_model
    )

    assert row["fed_residual_denominator_positive"] is False
    assert row["pooled_residual_denominator_positive"] is False
    assert row["beta_denominator_positive"] is False
    assert row["fed_residual_pass"] is False
    assert row["pooled_residual_pass"] is False
    assert row["finite_pass"] is False


def test_v2_beta_forward_envelope_is_reported_but_not_a_system_hard_gate() -> None:
    """Catches promoting the coefficient envelope into the hard conjunction."""
    equations = diagnostic_equations()
    row = system_diagnostics_v2(
        equations,
        equations,
        diagnostic_model(coef=(0.5, 0.5)),
        diagnostic_model(coef=(0.5, 0.500001)),
    )

    assert row["beta_within_forward_envelope"] is False
    assert row["condition_pass"] is True
    assert row["fed_residual_pass"] is True
    assert row["pooled_residual_pass"] is False


def test_v2_functional_diagnostic_checks_raw_and_clipped_metrics_separately() -> None:
    """Catches reusing clipped predictions for the registered raw gate."""
    values = np.array([[0.0], [1.0]], dtype=np.float64)
    targets = np.ones(2, dtype=np.float64)
    row = functional_diagnostics_v2(
        diagnostic_model(coef=(100.0, 0.0), clip_max=1.0),
        diagnostic_model(coef=(101.0, 0.0), clip_max=1.0),
        values,
        targets,
    )

    assert row["max_abs_raw_prediction_difference"] == 1.0
    assert row["max_abs_clipped_prediction_difference"] == 0.0
    assert row["raw_prediction_pass"] is False
    assert row["clipped_prediction_pass"] is True
    assert row["rmse_parity_pass"] is True
    assert row["mae_parity_pass"] is True


@pytest.mark.parametrize("nonfinite", [np.nan, np.inf, -np.inf])
def test_v2_diagnostics_fail_closed_on_nonfinite_input(nonfinite: float) -> None:
    """Catches any NaN/Inf comparison silently becoming a passing gate."""
    bad_scaler = diagnostic_scaler(mean=(nonfinite,))
    scaler_row = scaler_diagnostics_v2(bad_scaler, diagnostic_scaler())
    bad_equations = diagnostic_equations(b=np.array([nonfinite, 1.0]))
    equation_row = normal_equation_diagnostics_v2(
        bad_equations, diagnostic_equations()
    )
    functional_row = functional_diagnostics_v2(
        diagnostic_model(),
        diagnostic_model(),
        np.array([[nonfinite]], dtype=np.float64),
        np.array([0.0], dtype=np.float64),
    )

    assert scaler_row["finite_pass"] is False
    assert scaler_row["scaler_pass"] is False
    assert equation_row["finite_pass"] is False
    assert equation_row["normal_equations_pass"] is False
    assert functional_row["finite_pass"] is False
    assert functional_row["raw_prediction_pass"] is False


def test_v2_scaler_diagnostic_rejects_nonfinite_secondary_statistics() -> None:
    """Catches ignoring nonfinite variance/raw-scale fields in input records."""
    bad = StableGlobalScalerV2(
        gas_id=0,
        role="source_refit",
        n=8,
        mean=np.array([0.0]),
        variance=np.array([np.nan]),
        raw_scale=np.array([np.inf]),
        scale=np.array([1.0]),
        safe_scale_mask=np.array([False]),
        minimum=np.array([-1.0]),
        maximum=np.array([1.0]),
        aggregation_order=("C1", "C2"),
    )

    row = scaler_diagnostics_v2(bad, diagnostic_scaler())

    assert row["finite_pass"] is False
    assert row["scaler_pass"] is False


def test_v2_equation_diagnostic_rejects_nonfinite_target_statistics() -> None:
    """Catches ignoring nonfinite y'y/min/max carried by equation records."""
    bad = AggregatedNormalEquationsV2(
        gas_id=0,
        role="source_refit",
        n=8,
        a=np.diag([4.0, 2.0]),
        b=np.array([2.0, 1.0]),
        y_y=np.nan,
        y_min=0.0,
        y_max=10.0,
        aggregation_order=("C1", "C2"),
    )

    row = normal_equation_diagnostics_v2(bad, diagnostic_equations())

    assert row["finite_pass"] is False
    assert row["normal_equations_pass"] is False


def test_v2_decide_gas_equivalence_combines_registered_hard_fields() -> None:
    """Catches dropping a diagnostic family before the four-gas decision."""
    row = decide_gas_equivalence_v2(
        {
            "gas_id": 0,
            "scaler_pass": True,
            "safe_scale_mask_equal": True,
            "finite_pass": True,
        },
        {"gas_id": 0, "normal_equations_pass": True, "finite_pass": True},
        {
            "gas_id": 0,
            "alpha_equal": True,
            "condition_pass": True,
            "fed_residual_pass": True,
            "pooled_residual_pass": True,
            "relative_beta_difference": 0.0,
            "finite_pass": True,
        },
        {
            "gas_id": 0,
            "raw_prediction_pass": True,
            "clipped_prediction_pass": True,
            "rmse_parity_pass": True,
            "mae_parity_pass": True,
            "finite_pass": True,
        },
    )

    assert row == passing_gas_diagnostic()


@pytest.mark.parametrize(
    "field",
    [
        "alpha_equal",
        "scaler_pass",
        "safe_scale_mask_equal",
        "normal_equations_pass",
        "condition_pass",
        "fed_residual_pass",
        "pooled_residual_pass",
        "raw_prediction_pass",
        "clipped_prediction_pass",
        "rmse_parity_pass",
        "mae_parity_pass",
        "finite_pass",
    ],
)
def test_each_registered_hard_gate_can_fail_r0_v2(field: str) -> None:
    """Catches omission of any preregistered Boolean from the conjunction."""
    rows = [passing_gas_diagnostic(gas_id=gas) for gas in range(4)]
    rows[0][field] = False

    assert decide_r0_v2(rows)["decision"] == "R0_V2_FAILED"


def test_coefficient_difference_is_diagnostic_not_a_hard_gate() -> None:
    """Catches rejecting an otherwise passing run on beta discrepancy alone."""
    rows = [
        passing_gas_diagnostic(gas_id=gas, relative_beta_difference=1.0)
        for gas in range(4)
    ]

    assert decide_r0_v2(rows)["decision"] == (
        "FEDRIDGE_ALGEBRAIC_EXACT_NUMERICAL_EQUIVALENCE_ESTABLISHED"
    )


@pytest.mark.parametrize(
    "gas_ids",
    [
        (0, 1, 2),
        (0, 1, 2, 4),
        (0, 1, 2, 3, 4),
        (0, 1, 2, 2),
        (False, 1, 2, 3),
    ],
)
def test_v2_decision_requires_exactly_one_row_for_each_registered_gas(
    gas_ids: tuple[object, ...],
) -> None:
    """Catches missing, extra, duplicate, or bool-coerced gas identities."""
    rows = [passing_gas_diagnostic(gas_id=gas) for gas in gas_ids]  # type: ignore[arg-type]

    assert decide_r0_v2(rows)["decision"] == "R0_V2_FAILED"


@pytest.mark.parametrize("invalid", [1, None, "true", np.bool_(True)])
def test_v2_decision_requires_exact_builtin_boolean_hard_fields(
    invalid: object,
) -> None:
    """Catches truthiness coercion at the final registered decision gate."""
    rows = [passing_gas_diagnostic(gas_id=gas) for gas in range(4)]
    rows[0]["condition_pass"] = invalid

    assert decide_r0_v2(rows)["decision"] == "R0_V2_FAILED"


def test_v2_decision_fails_when_a_registered_hard_field_is_missing() -> None:
    """Catches treating absent evidence as an implicit pass."""
    rows = [passing_gas_diagnostic(gas_id=gas) for gas in range(4)]
    del rows[0]["mae_parity_pass"]

    assert decide_r0_v2(rows)["decision"] == "R0_V2_FAILED"


@pytest.mark.parametrize("nonfinite", [np.nan, np.inf, -np.inf])
def test_v2_decision_fails_closed_on_nonfinite_diagnostic_values(
    nonfinite: float,
) -> None:
    """Catches coefficient diagnostics bypassing the global finite gate."""
    rows = [passing_gas_diagnostic(gas_id=gas) for gas in range(4)]
    rows[0]["relative_beta_difference"] = nonfinite

    assert decide_r0_v2(rows)["decision"] == "R0_V2_FAILED"


def test_r0_v2_protocol_manifest_is_pre_run_target_free_and_canonical() -> None:
    """Catches executing early, enabling target access, or drifting source data."""
    manifest = json.loads(PROTOCOL_MANIFEST.read_text(encoding="utf-8"))

    assert manifest["study_id"] == R0_V2_STUDY_ID
    assert manifest["status"] == "DESIGN_FREEZE_READY_FORMAL_NOT_STARTED"
    assert manifest["formal_execution_started"] is False
    assert manifest["execution_commit_policy"] == (
        "CLI authorized freeze commit must equal current Git HEAD"
    )
    assert manifest["source_clients"] == ["C1", "C2"]
    assert manifest["target_clients"] == []
    assert manifest["target_access"] == {
        "calibration_x": False,
        "calibration_labels": False,
        "test_x": False,
        "test_labels": False,
    }
    assert manifest["canonical_data"] == {
        "dataset_path": "dataset/iotj_canonical_v1",
        "dataset_aggregate_sha256": (
            "2f810d7e93cae5f361923184e9dc87d5ae59e0f59be9f52aff7e14f9f33e94f6"
        ),
        "preprocessing": "HZ5_MEAN_W10S",
        "sampling_rate_hz": 5,
        "window_seconds": 10,
        "stride_seconds": 5,
        "window_shape": [50, 8],
        "source_split_counts_per_client": {
            "train": 2360,
            "calibration": 320,
            "test": 680,
        },
        "per_gas_refit_count_per_client": 670,
        "per_gas_refit_count_pooled": 1340,
        "per_gas_test_count_per_client": 170,
        "per_gas_test_count_pooled": 340,
    }
    assert manifest["feature_protocol"]["sensor_dimensions"] == 83
    assert manifest["feature_protocol"]["h1_dimensions"] == 104
    assert manifest["feature_protocol"]["ordered_h1_feature_names_sha256"] == (
        "df696d3cfbe43eff40b515f6f1a7bb51c9cd11900dba93e231a3ded0755c3259"
    )
    assert manifest["feature_protocol"]["ordered_sensor_feature_names_sha256"] == (
        "4cb9e621b39cf726b18d0102d2ec395ba11b109b6ffcabb381c729dd44f26248"
    )
    assert manifest["feature_protocol"]["extractor_file_sha256"] == (
        "7627b72ee4e1823d24c374d41a6c931f66b5efedd6eaf4a839c62e7b5b1fa72a"
    )
    assert manifest["design_provenance"]["commit"] == (
        "b41fee1d5bd64a19d6fefcad5fde610183856202"
    )
    assert manifest["implementation_provenance"]["task_base_commit"] == (
        "6668dc5db83428a2d957d962d6a5fa4bb5dc2430"
    )
    assert manifest["C0_decision"] == "V1_INTERLEAVED_RETAINED"
    assert manifest["original_R0_decision"] == "R0_EXACT_RECOVERY_NOT_ESTABLISHED"
    assert manifest["formal_result_root"] == (
        "results/iotj_canonical_v1_final/canonical_fedridge_r0_v2_20260812"
    )
    assert not FORMAL_RESULT_ROOT.exists()


def test_r0_v2_protocol_manifest_freezes_exact_numerical_semantics() -> None:
    """Catches changing formulas, solver behavior, or the hard gate set."""
    manifest = json.loads(PROTOCOL_MANIFEST.read_text(encoding="utf-8"))
    numerical = manifest["numerical_protocol"]
    gates = manifest["numerical_gates"]

    assert numerical == {
        "dtype": "float64",
        "local_moment": "M2_k=sum_i((x_i-mean_k)^2)",
        "merge_mean": "mean_A+delta*n_B/(n_A+n_B)",
        "merge_M2": "M2_A+M2_B+delta^2*n_A*n_B/(n_A+n_B)",
        "variance": "max(M2/n,0)",
        "population_variance_denominator": "n",
        "safe_scale_condition": "raw_scale < 1e-9",
        "safe_scale_replacement": 1.0,
        "aggregation_order": ["C1", "C2"],
        "alpha_grid": [0.0, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
        "alpha_selection": "source train fit; source calibration SSE/count only",
        "alpha_tie_break": "first_in_registered_grid",
        "solver": "numpy.linalg.pinv",
        "intercept_regularized": False,
    }
    assert gates["epsilon"] == 2.220446049250313e-16
    assert gates["gamma_formula"] == "gamma(m)=m*epsilon/(1-m*epsilon)"
    assert gates["tau_moment_formula"] == "64*gamma(1340)"
    assert gates["tau_moment"] == 1.9042545318376352e-11
    assert gates["tau_residual_formula"] == "128*gamma(105)"
    assert gates["tau_residual"] == 2.9842794901924903e-12
    assert gates["functional_ppm"] == 1e-6
    assert gates["condition"] == "finite(kappa) and kappa*epsilon < 1"
    assert gates["hard_boolean_fields"] == [
        "alpha_equal",
        "scaler_pass",
        "safe_scale_mask_equal",
        "normal_equations_pass",
        "condition_pass",
        "fed_residual_pass",
        "pooled_residual_pass",
        "raw_prediction_pass",
        "clipped_prediction_pass",
        "rmse_parity_pass",
        "mae_parity_pass",
        "finite_pass",
    ]
    assert gates["coefficient_difference_hard_gate"] is False
    assert manifest["decision_vocabulary"] == [
        "FEDRIDGE_ALGEBRAIC_EXACT_NUMERICAL_EQUIVALENCE_ESTABLISHED",
        "R0_V2_FAILED",
    ]


def test_r0_v2_matrix_registers_exactly_one_unexecuted_configuration() -> None:
    """Catches multiplying rows, omitting planner fields, or claiming Evidence."""
    with EXPERIMENT_MATRIX.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    assert reader.fieldnames == PLANNER_FIELDS
    assert len(rows) == 1
    row = rows[0]
    assert row["experiment_id"] == R0_V2_STUDY_ID
    assert row["source_clients"] == "C1;C2"
    assert row["target_clients"] == ""
    assert row["DA"] == row["calibration"] == row["QC"] == "none"
    assert row["seed"] == "42"
    assert row["status"] == "registered"
    assert row["evidence_status"] == "blocked_pending_execution"
    assert row["hypothesis_id"] == "H-R0V2-NUM"
    assert row["metrics"] == REGISTERED_METRIC_REFERENCES
    assert row["expected_evidence"] == (
        "four per-gas gate records and one registered PASS/FAIL decision"
    )
    assert "deterministic numeric reconstruction; seed unused" in row["notes"]
    assert row["result_path"] == (
        "results/iotj_canonical_v1_final/canonical_fedridge_r0_v2_20260812"
    )


def test_r0_v2_registry_has_one_registered_record_and_planner_handoff() -> None:
    """Catches registry schema drift, premature completion, or missing handoff."""
    with EXPERIMENT_REGISTRY.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    assert reader.fieldnames == REGISTRY_FIELDS
    assert len(rows) == 1
    row = rows[0]
    assert row["experiment_id"] == R0_V2_STUDY_ID
    assert row["source_clients"] == "C1;C2"
    assert row["target_clients"] == ""
    assert row["status"] == "registered"
    assert row["evidence_status"] == "blocked_pending_execution"
    assert row["checkpoint"].startswith("not_created_pre_run")
    assert row["metrics"] == REGISTERED_METRIC_REFERENCES
    assert "separately named freeze commit" in row["notes"]


def test_r0_v2_protocol_markdown_preserves_boundaries_and_decisions() -> None:
    """Catches missing freeze notes or instructions that cross access scope."""
    for filename in REQUIRED_PROTOCOL_MARKDOWN:
        text = (PROTOCOL_ROOT / filename).read_text(encoding="utf-8")
        assert "FEDRIDGE_ALGEBRAIC_EXACT_NUMERICAL_EQUIVALENCE_ESTABLISHED" in text
        assert "R0_V2_FAILED" in text
        assert "R0_EXACT_RECOVERY_NOT_ESTABLISHED" in text
        assert "V1_INTERLEAVED_RETAINED" in text


def test_experiment_plan_hypothesis_is_neutral_before_execution() -> None:
    """Catches a preregistration that predicts the registered PASS outcome."""
    text = (PROTOCOL_ROOT / "EXPERIMENT_PLAN.md").read_text(encoding="utf-8")

    assert (
        "Formal execution will determine whether every preregistered gate "
        "passes; either registered decision is admissible, with no expected "
        "direction."
    ) in text
    assert (
        "the sufficient-statistics and pooled reconstructions satisfy every "
        "preregistered"
    ) not in text


def test_matrix_and_registry_use_metrics_not_workflow_evidence_state() -> None:
    """Catches storing evidence workflow state in the metrics field."""
    for path in (EXPERIMENT_MATRIX, EXPERIMENT_REGISTRY):
        with path.open(encoding="utf-8", newline="") as handle:
            row = next(csv.DictReader(handle))

        assert row["metrics"] == REGISTERED_METRIC_REFERENCES
        assert row["metrics"] != row["evidence_status"]
        assert row["evidence_status"] == "blocked_pending_execution"


def test_matrix_and_registry_provenance_is_field_addressable_json() -> None:
    """Catches free-text provenance that cannot trace each canonical field."""
    for path in (EXPERIMENT_MATRIX, EXPERIMENT_REGISTRY):
        with path.open(encoding="utf-8", newline="") as handle:
            row = next(csv.DictReader(handle))
        provenance = json.loads(row["provenance"])

        assert REQUIRED_FIELD_PROVENANCE <= set(provenance)
        assert all(
            isinstance(provenance[field], str) and provenance[field]
            for field in REQUIRED_FIELD_PROVENANCE
        )
        assert "task-5-brief.md" in provenance["experiment_id"]
        assert "b41fee1d5bd64a19d6fefcad5fde610183856202" in provenance[
            "source_clients"
        ]
        assert "canonical_fedridge_v2.py@6668dc5" in provenance["model"]
        assert "protocol_manifest.json#/numerical_gates" in provenance["metrics"]
        assert "dataset_sha256.json" in provenance["dataset_path"]
        assert "protocol_manifest.json" in provenance["config_path"]
        assert "task-5-brief.md" in provenance["status"]
        assert "task-5-brief.md" in provenance["evidence_status"]


def test_protocol_records_explicit_multi_source_reconciliation() -> None:
    """Catches asserting no conflict without documenting compared sources."""
    text = (PROTOCOL_ROOT / "PROTOCOL.md").read_text(encoding="utf-8")

    assert "## Explicit reconciliation check" in text
    for source in (
        "Approved design commit",
        "Task 4 code/constants",
        "Canonical manifests/data roles",
        "Planner/registry records",
    ):
        assert source in text
    assert "No traceable disagreement found" in text
    assert "conflict_fields=[]" in text


def test_target_qc_instruction_guard_detects_semantic_variants() -> None:
    """Catches unauthorized access phrased without the original six verbs."""
    for instruction in (
        "Open target calibration arrays.",
        "Load C3 labels.",
        "Evaluate C5 test rows.",
        "Run target inference.",
        "Execute QC thresholds.",
        "Conduct target evaluation.",
        "Perform QC filtering.",
        "Apply QC policy.",
    ):
        assert unauthorized_target_qc_instructions(instruction)

    for prohibition in (
        "Do not open target calibration arrays.",
        "Never load C3 labels.",
        "Target evaluation is prohibited.",
        "QC execution is unavailable.",
    ):
        assert unauthorized_target_qc_instructions(prohibition) == []


def test_protocol_bundle_has_no_authorized_target_or_qc_instruction() -> None:
    """Catches positive target/QC instructions anywhere in the eight files."""
    for filename in REQUIRED_BUNDLE_FILES:
        text = (PROTOCOL_ROOT / filename).read_text(encoding="utf-8")
        assert unauthorized_target_qc_instructions(text) == []


def test_near_constant_policy_reuses_strict_floor_without_selection() -> None:
    """Catches tuning the inherited scale floor or changing its boundary."""
    text = (PROTOCOL_ROOT / "NEAR_CONSTANT_SCALE_POLICY.md").read_text(
        encoding="utf-8"
    )

    assert "raw_scale < 1e-9" in text
    assert "raw_scale == 1e-9" in text
    assert "reused" in text
    assert "not selected or tuned" in text
    assert "performance" in text
    assert "target data" in text


def test_numerical_tolerance_justification_lists_every_hard_gate() -> None:
    """Catches an under-specified or result-tuned tolerance freeze."""
    text = (
        PROTOCOL_ROOT / "R0_V2_NUMERICAL_TOLERANCE_JUSTIFICATION.md"
    ).read_text(encoding="utf-8")

    for literal in (
        "2.220446049250313e-16",
        "gamma(m) = m*epsilon / (1 - m*epsilon)",
        "64*gamma(1340)",
        "1.9042545318376352e-11",
        "128*gamma(105)",
        "2.9842794901924903e-12",
        "1e-6 ppm",
        "alpha_equal",
        "scaler_pass",
        "safe_scale_mask_equal",
        "normal_equations_pass",
        "condition_pass",
        "fed_residual_pass",
        "pooled_residual_pass",
        "raw_prediction_pass",
        "clipped_prediction_pass",
        "rmse_parity_pass",
        "mae_parity_pass",
        "finite_pass",
    ):
        assert literal in text
    assert "not selected or adjusted from observed R0-v2 results" in text


def test_manuscript_note_limits_the_future_stability_claim() -> None:
    """Catches upgrading numerical engineering into an unsupported claim."""
    text = (
        PROTOCOL_ROOT / "FEDRIDGE_NUMERICAL_STABILITY_MANUSCRIPT_NOTE.md"
    ).read_text(encoding="utf-8")
    proposed = (
        "global mean/variance is reconstructed using numerically stable "
        "mergeable moments"
    )

    assert text.count(proposed) == 1
    assert "bitwise-exact claim is prohibited" in text
    assert "novel-algorithm claim is prohibited" in text
    assert "manuscript body is not edited by this protocol bundle" in text


def test_r0_v2_parser_has_only_preflight_run_and_audit_without_target_or_qc() -> None:
    """Catches exposing any target/QC execution surface or an unregistered stage."""
    parser = runner.build_parser()
    options = {
        option
        for action in parser._actions
        for option in action.option_strings
    }
    stage = next(action for action in parser._actions if action.dest == "stage")

    assert stage.choices == ("preflight", "run", "audit")
    assert "--target" not in options
    assert "--target-data" not in options
    assert "--qc" not in options


@pytest.mark.parametrize("stage", ("preflight", "run"))
def test_r0_v2_parser_requires_authorized_freeze_commit(stage: str) -> None:
    """Catches allowing a production-capable stage without the commit lock."""
    parser = runner.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args([stage])


def test_r0_v2_execution_plan_locks_model_before_source_test() -> None:
    """Catches source-test access before immutable alpha/model locks."""
    plan = runner.build_r0_v2_execution_plan()

    assert plan.index("write_source_alpha_and_model_locks") < plan.index(
        "open_source_test"
    )
    assert all("target" not in step.lower() for step in plan)


def test_r0_v2_preflight_rejects_wrong_commit_and_existing_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches bypassing authorization or reusing partial evidence."""
    monkeypatch.setattr(runner, "_git_head", lambda: "actual")
    with pytest.raises(RuntimeError, match="authorized freeze commit"):
        runner.preflight(DATA_ROOT, tmp_path / "out", "different")

    output = tmp_path / "occupied"
    output.mkdir()
    (output / "partial.txt").write_text("evidence", encoding="utf-8")
    with pytest.raises(FileExistsError, match="output"):
        runner.preflight(DATA_ROOT, output, "actual")


def test_source_dataset_preflight_never_opens_unregistered_client_artifacts(
    tmp_path: Path,
) -> None:
    """Catches an aggregate verifier traversing non-source array/label paths."""
    data_root = tmp_path / "canonical"
    data_root.mkdir()
    protocol = json.loads(PROTOCOL_MANIFEST.read_text(encoding="utf-8"))
    protocol["canonical_data"]["dataset_aggregate_sha256"] = "a" * 64
    artifacts: dict[str, str] = {}
    index_files: dict[str, str] = {}
    for client in ("C1", "C2"):
        directory = data_root / f"client_{client[1:]}"
        directory.mkdir()
        for split in ("train", "calibration", "test"):
            for kind, suffix in (
                ("features", "features.npy"),
                ("phase", "phase_labels.npy"),
                ("metadata", "experiment_info.json"),
            ):
                path = directory / f"{split}_{suffix}"
                path.write_text(f"{client}/{split}/{kind}", encoding="utf-8")
                artifacts[f"{client}_{split}_{kind}"] = runner.sha256_file(path)
                index_files[path.relative_to(data_root).as_posix()] = runner.sha256_file(
                    path
                )
            for label_kind in ("classification", "regression"):
                path = directory / f"{split}_{label_kind}_labels.npy"
                n = {"train": 2360, "calibration": 320, "test": 680}[split]
                if label_kind == "classification":
                    values = np.repeat(np.arange(4, dtype=np.int64), n // 4)
                else:
                    values = np.zeros((n, 4), dtype=np.float64)
                np.save(path, values, allow_pickle=False)
                index_files[path.relative_to(data_root).as_posix()] = runner.sha256_file(
                    path
                )
        stats_path = directory / "stats.json"
        stats_path.write_text(
            json.dumps(
                {
                    "client_id": client,
                    "role": "source",
                    "counts": {"train": 2360, "calibration": 320, "test": 680},
                }
            ),
            encoding="utf-8",
        )
        index_files[stats_path.relative_to(data_root).as_posix()] = runner.sha256_file(
            stats_path
        )
    protocol["canonical_source_artifact_sha256"] = artifacts
    canonical_manifest = data_root / "canonical_preprocessing_manifest.json"
    canonical_manifest.write_text(
        json.dumps(
            {
                "candidate_id": "HZ5_MEAN_W10S",
                "sampling_rate_hz": 5,
                "points_per_window": 50,
                "window_duration_s": 10.0,
            }
        ),
        encoding="utf-8",
    )
    index_files[canonical_manifest.name] = runner.sha256_file(canonical_manifest)
    (data_root / "dataset_sha256.json").write_text(
        json.dumps(
            {
                "aggregate_sha256": "a" * 64,
                "files": {
                    **index_files,
                    "client_3/unregistered_secret.npy": "b" * 64,
                },
            }
        ),
        encoding="utf-8",
    )

    result = runner._verify_source_dataset(data_root, protocol)

    assert result["status"] == "PASS"
    assert result["bad_files"] == []

    (data_root / "client_1/test_regression_labels.npy").write_text(
        "changed", encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="source hash"):
        runner._verify_source_dataset(data_root, protocol)

    np.save(
        data_root / "client_1/test_regression_labels.npy",
        np.zeros((680, 4), dtype=np.float64),
        allow_pickle=False,
    )
    class_path = data_root / "client_1/test_classification_labels.npy"
    classes = np.load(class_path, allow_pickle=False)
    classes[0] = 1
    np.save(class_path, classes, allow_pickle=False)
    index_path = data_root / "dataset_sha256.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    relative = class_path.relative_to(data_root).as_posix()
    index["files"][relative] = runner.sha256_file(class_path)
    index_path.write_text(json.dumps(index), encoding="utf-8")
    with pytest.raises(RuntimeError, match="per-gas source counts"):
        runner._verify_source_dataset(data_root, protocol)


def test_preflight_is_pure_and_accepts_only_the_frozen_source_prerequisites(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches preflight writing evidence or omitting a real prerequisite check."""
    output = tmp_path / "absent"
    monkeypatch.setattr(runner, "_git_head", lambda: "authorized")

    result = runner.preflight(DATA_ROOT, output, "authorized")

    assert result["status"] == "PASS"
    assert result["output_created"] is False
    assert result["source_clients"] == ["C1", "C2"]
    assert result["target_clients"] == []
    assert not output.exists()


def test_preflight_fails_closed_on_protocol_hash_and_status_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches accepting either byte drift or semantic execution-state drift."""
    changed = tmp_path / "protocol.json"
    original = PROTOCOL_MANIFEST.read_text(encoding="utf-8")
    changed.write_text(original + " ", encoding="utf-8")
    monkeypatch.setattr(runner, "PROTOCOL_MANIFEST", changed)
    with pytest.raises(RuntimeError, match="protocol hash"):
        runner._verify_protocol()

    payload = json.loads(original)
    payload["formal_execution_started"] = True
    changed.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(
        runner,
        "EXPECTED_PROTOCOL_FREEZE_SHA256",
        runner.sha256_file(changed),
    )
    with pytest.raises(RuntimeError, match="status or role"):
        runner._verify_protocol()

    payload = json.loads(original)
    payload["numerical_protocol"]["aggregation_order"] = ["C2", "C1"]
    changed.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(
        runner,
        "EXPECTED_PROTOCOL_FREEZE_SHA256",
        runner.sha256_file(changed),
    )
    with pytest.raises(RuntimeError, match="source order"):
        runner._verify_protocol()


def test_preflight_fails_closed_on_dataset_and_extractor_hash_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catches accepting changed canonical aggregate or extractor provenance."""
    protocol = json.loads(PROTOCOL_MANIFEST.read_text(encoding="utf-8"))
    changed_dataset = json.loads(json.dumps(protocol))
    changed_dataset["canonical_data"]["dataset_aggregate_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="dataset hash"):
        runner._verify_source_dataset(DATA_ROOT, changed_dataset)

    original_hash = runner.sha256_file

    def changed_extractor(path: str | Path) -> str:
        if Path(path).resolve() == runner.EXTRACTOR_PATH.resolve():
            return "0" * 64
        return original_hash(path)

    monkeypatch.setattr(runner, "sha256_file", changed_extractor)
    with pytest.raises(RuntimeError, match="extractor/schema"):
        runner._verify_source_dataset(DATA_ROOT, protocol)


def test_preflight_fails_closed_on_original_decision_and_path_conflicts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches rewriting/relocating the read-only C0 or original-R0 evidence."""
    c0_root = tmp_path / "old/C0"
    r0_root = tmp_path / "old/R0"
    c0_root.mkdir(parents=True)
    r0_root.mkdir(parents=True)
    (c0_root / "C0_DECISION.json").write_text(
        json.dumps({"decision": "CHANGED"}), encoding="utf-8"
    )
    (c0_root / "C0_EXPERIMENT_AUDIT.md").write_text("audit", encoding="utf-8")
    (c0_root / "C0_SHA256_INDEX.json").write_text(
        json.dumps(
            {
                name: runner.sha256_file(c0_root / name)
                for name in ("C0_DECISION.json", "C0_EXPERIMENT_AUDIT.md")
            }
        ),
        encoding="utf-8",
    )
    for name, payload in (
        ("canonical_fedridge_exact_recovery.json", '{"status":"FAIL_CLOSED"}'),
        ("R0_FAILURE_AUDIT.json", '{"status":"FAIL_CLOSED"}'),
        ("R0_EXPERIMENT_AUDIT.md", "audit"),
    ):
        (r0_root / name).write_text(payload, encoding="utf-8")
    (r0_root / "R0_SHA256_INDEX.json").write_text(
        json.dumps(
            {
                name: runner.sha256_file(r0_root / name)
                for name in (
                    "canonical_fedridge_exact_recovery.json",
                    "R0_FAILURE_AUDIT.json",
                    "R0_EXPERIMENT_AUDIT.md",
                )
            }
        ),
        encoding="utf-8",
    )
    original_protocol = tmp_path / "old_protocol.json"
    original_protocol.write_text(
        json.dumps({"R0": {"execution_result": {"status": "FAIL_CLOSED"}}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(runner, "ORIGINAL_C0_ROOT", c0_root)
    monkeypatch.setattr(runner, "ORIGINAL_R0_ROOT", r0_root)
    monkeypatch.setattr(runner, "ORIGINAL_PROTOCOL_MANIFEST", original_protocol)
    with pytest.raises(RuntimeError, match="original C0 decision"):
        runner._verify_original_prerequisites(tmp_path / "new")

    monkeypatch.setattr(runner, "ORIGINAL_C0_ROOT", tmp_path / "new/C0")
    with pytest.raises(RuntimeError, match="inside output"):
        runner._verify_original_prerequisites(tmp_path / "new")

    (c0_root / "C0_DECISION.json").write_text(
        json.dumps({"decision": "V1_INTERLEAVED_RETAINED"}), encoding="utf-8"
    )
    c0_index = json.loads(
        (c0_root / "C0_SHA256_INDEX.json").read_text(encoding="utf-8")
    )
    c0_index["C0_DECISION.json"] = runner.sha256_file(
        c0_root / "C0_DECISION.json"
    )
    (c0_root / "C0_SHA256_INDEX.json").write_text(
        json.dumps(c0_index), encoding="utf-8"
    )
    monkeypatch.setattr(runner, "ORIGINAL_C0_ROOT", c0_root)
    with pytest.raises(RuntimeError, match="overlaps output"):
        runner._verify_original_prerequisites(c0_root / "new_output")


@pytest.mark.parametrize(
    "protected_root",
    (DATA_ROOT, PROTOCOL_ROOT),
)
def test_preflight_rejects_output_beneath_read_only_inputs(
    protected_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches using a canonical input/protocol tree as an output parent."""
    output = protected_root / "uncreated_r0_v2_output"
    assert not output.exists()
    monkeypatch.setattr(runner, "_git_head", lambda: "authorized")

    with pytest.raises(RuntimeError, match="path separation"):
        runner.preflight(DATA_ROOT, output, "authorized")

    assert not output.exists()


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


def force_synthetic_alpha(monkeypatch: pytest.MonkeyPatch) -> None:
    def fixed_selection(*args: object, **kwargs: object) -> tuple[float, list[dict[str, object]]]:
        return 0.1, [
            {
                "alpha": 0.1,
                "source_calibration_RMSE": 0.0,
                "target_input_accessed": False,
                "source_test_accessed": False,
            }
        ]

    monkeypatch.setattr(runner, "select_source_alpha_v2", fixed_selection)
    monkeypatch.setattr(runner, "select_pooled_alpha_v2", fixed_selection)


class RecordingSyntheticProvider:
    def __init__(self) -> None:
        self.requests: list[runner.SourceRequest] = []
        self.data = synthetic_four_gas_source_data()

    def build_fresh_cache(self, client: str, split: str) -> Mapping[str, object]:
        self.requests.append(runner.SourceRequest(client, split, None))
        return {"client": client, "split": split, "study_id": R0_V2_STUDY_ID}

    def gas_data(
        self, client: str, split: str, gas_id: int
    ) -> tuple[np.ndarray, np.ndarray]:
        self.requests.append(runner.SourceRequest(client, split, gas_id))
        return self.data[(client, split, gas_id)]


def expected_source_only_request_sequence() -> list[runner.SourceRequest]:
    expected: list[runner.SourceRequest] = []
    for split in ("train", "calibration"):
        for client in ("C1", "C2"):
            expected.append(runner.SourceRequest(client, split, None))
    for gas_id in range(4):
        for split in ("train", "calibration"):
            for client in ("C1", "C2"):
                expected.append(runner.SourceRequest(client, split, gas_id))
    for client in ("C1", "C2"):
        expected.append(runner.SourceRequest(client, "test", None))
    for gas_id in range(4):
        for client in ("C1", "C2"):
            expected.append(runner.SourceRequest(client, "test", gas_id))
    return expected


EXPECTED_EXECUTION_FILES = {
    "canonical_feature_caches",
    "H1_CANONICAL_FEATURE_NUMERICAL_AUDIT.csv",
    "r0_v2_scaler_diagnostics.csv",
    "r0_v2_normal_equation_diagnostics.csv",
    "r0_v2_system_diagnostics.csv",
    "r0_v2_functional_equivalence.csv",
    "source_alpha_audit.csv",
    "source_alpha_lock.json",
    "model_lock.json",
    "DATA_ACCESS_AUDIT.md",
    "R0_V2_DECISION.json",
    "R0_V2_EXPERIMENT_AUDIT.md",
    "protocol_manifest_execution.json",
    "sha256_index.json",
}


def test_synthetic_runner_writes_all_diagnostics_and_never_requests_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches missing evidence, reordered source access, or target leakage."""
    force_synthetic_alpha(monkeypatch)
    provider = RecordingSyntheticProvider()
    output = tmp_path / "run"

    result = runner._execute_source_only(
        provider, output, frozen_protocol_fixture()
    )

    assert result["decision"] == (
        "FEDRIDGE_ALGEBRAIC_EXACT_NUMERICAL_EQUIVALENCE_ESTABLISHED"
    )
    assert provider.requests == expected_source_only_request_sequence()
    assert not any(
        request.client not in {"C1", "C2"} for request in provider.requests
    )
    assert EXPECTED_EXECUTION_FILES <= {path.name for path in output.iterdir()}
    assert (output / "fixed_endpoint_complete.json").is_file()
    assert (output / "sha256_index.json").stat().st_mtime_ns <= (
        output / "fixed_endpoint_complete.json"
    ).stat().st_mtime_ns
    marker = json.loads(
        (output / "fixed_endpoint_complete.json").read_text(encoding="utf-8")
    )
    assert marker["R1_released"] is True
    index = json.loads((output / "sha256_index.json").read_text(encoding="utf-8"))
    assert "fixed_endpoint_complete.json" not in index
    assert "sha256_index.json" not in index
    assert marker["sha256_index_sha256"] == runner.sha256_file(
        output / "sha256_index.json"
    )
    execution = json.loads(
        (output / "protocol_manifest_execution.json").read_text(encoding="utf-8")
    )
    assert execution["target_clients"] == []
    assert execution["global_key_audit"] == {
        "allowed_clients": ["C1", "C2"],
        "allowed_gases": [0, 1, 2, 3],
        "allowed_splits": ["train", "calibration", "test"],
        "exact_registered_sequence": True,
        "observed_request_count": len(expected_source_only_request_sequence()),
    }
    assert sum(
        1
        for _ in csv.DictReader(
            (output / "H1_CANONICAL_FEATURE_NUMERICAL_AUDIT.csv").open(
                encoding="utf-8", newline=""
            )
        )
    ) == 416
    for filename in (
        "r0_v2_scaler_diagnostics.csv",
        "r0_v2_normal_equation_diagnostics.csv",
        "r0_v2_system_diagnostics.csv",
        "r0_v2_functional_equivalence.csv",
    ):
        with (output / filename).open(encoding="utf-8", newline="") as handle:
            assert len(list(csv.DictReader(handle))) == 4


def test_failed_synthetic_runner_preserves_evidence_without_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches favorable diagnostics overriding one failed hard gate."""
    force_synthetic_alpha(monkeypatch)
    output = tmp_path / "run"
    provider = RecordingSyntheticProvider()
    original = runner.functional_diagnostics_v2

    def forced_failure(*args: object, **kwargs: object) -> dict[str, object]:
        row = original(*args, **kwargs)
        row["raw_prediction_pass"] = False
        return row

    monkeypatch.setattr(runner, "functional_diagnostics_v2", forced_failure)
    result = runner._execute_source_only(
        provider, output, frozen_protocol_fixture()
    )

    assert result["decision"] == "R0_V2_FAILED"
    assert (output / "R0_V2_DECISION.json").is_file()
    assert (output / "R0_V2_EXPERIMENT_AUDIT.md").is_file()
    assert (output / "sha256_index.json").is_file()
    assert not (output / "fixed_endpoint_complete.json").exists()
    audit_text = (output / "R0_V2_EXPERIMENT_AUDIT.md").read_text(
        encoding="utf-8"
    )
    assert "blocking" in audit_text.lower()
    assert "raw_prediction_pass" in audit_text


def test_provider_exception_and_output_collision_preserve_failure_evidence(
    tmp_path: Path,
) -> None:
    """Catches deleting partial evidence or overwriting a collided output."""
    class FailingProvider(RecordingSyntheticProvider):
        def gas_data(
            self, client: str, split: str, gas_id: int
        ) -> tuple[np.ndarray, np.ndarray]:
            super().gas_data(client, split, gas_id)
            raise OSError("synthetic provider failure")

    output = tmp_path / "failed"
    result = runner._execute_source_only(
        FailingProvider(), output, frozen_protocol_fixture()
    )

    assert result["decision"] == "R0_V2_FAILED"
    assert "synthetic provider failure" in json.loads(
        (output / "R0_V2_DECISION.json").read_text(encoding="utf-8")
    )["error"]
    assert runner.audit(output)["decision"] == "R0_V2_FAILED"
    assert not (output / "fixed_endpoint_complete.json").exists()

    occupied = tmp_path / "occupied"
    occupied.mkdir()
    evidence = occupied / "partial.txt"
    evidence.write_text("immutable", encoding="utf-8")
    with pytest.raises(FileExistsError, match="output"):
        runner._execute_source_only(
            RecordingSyntheticProvider(), occupied, frozen_protocol_fixture()
        )
    assert evidence.read_text(encoding="utf-8") == "immutable"


def test_production_provider_passes_versioned_study_id_to_fresh_cache_api(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches silently falling back to the old unversioned cache identity."""
    observed: dict[str, object] = {}

    def fake_build(
        dataset_root: Path,
        cache_root: Path,
        **kwargs: object,
    ) -> dict[str, object]:
        observed.update(
            {"dataset_root": dataset_root, "cache_root": cache_root, **kwargs}
        )
        return {
            "client": kwargs["client"],
            "split": kwargs["split"],
            "study_id": kwargs["study_id"],
        }

    monkeypatch.setattr(runner, "build_feature_cache", fake_build)
    provider = runner.CanonicalSourceDataProvider(
        tmp_path / "data", tmp_path / "output", "a" * 64
    )

    manifest = provider.build_fresh_cache("C1", "train")

    assert manifest["study_id"] == R0_V2_STUDY_ID
    assert observed["study_id"] == R0_V2_STUDY_ID
    assert observed["client"] == "C1"
    assert observed["split"] == "train"
    assert observed["cache_root"] == (tmp_path / "output/canonical_feature_caches")


def test_audit_only_validates_pass_and_fail_without_modifying_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches audit regenerating metrics or rejecting preserved failure evidence."""
    force_synthetic_alpha(monkeypatch)
    pass_output = tmp_path / "pass"
    fail_output = tmp_path / "fail"
    runner._execute_source_only(
        RecordingSyntheticProvider(), pass_output, frozen_protocol_fixture()
    )
    original = runner.functional_diagnostics_v2

    def forced_failure(*args: object, **kwargs: object) -> dict[str, object]:
        row = original(*args, **kwargs)
        row["mae_parity_pass"] = False
        return row

    monkeypatch.setattr(runner, "functional_diagnostics_v2", forced_failure)
    runner._execute_source_only(
        RecordingSyntheticProvider(), fail_output, frozen_protocol_fixture()
    )
    before = {
        path.relative_to(tmp_path).as_posix(): (path.stat().st_mtime_ns, path.read_bytes())
        for path in tmp_path.rglob("*")
        if path.is_file()
    }

    pass_audit = runner.audit(pass_output)
    fail_audit = runner.audit(fail_output)

    after = {
        path.relative_to(tmp_path).as_posix(): (path.stat().st_mtime_ns, path.read_bytes())
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    assert pass_audit["status"] == "PASS"
    assert fail_audit == {"status": "PASS", "decision": "R0_V2_FAILED", "bad": []}
    assert after == before


def test_audit_only_fails_closed_on_hash_decision_and_access_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches accepting altered diagnostics, vocabulary, or access events."""
    force_synthetic_alpha(monkeypatch)
    outputs = [
        tmp_path / name for name in ("hash", "decision", "access", "operation")
    ]
    for output in outputs:
        runner._execute_source_only(
            RecordingSyntheticProvider(), output, frozen_protocol_fixture()
        )

    (outputs[0] / "r0_v2_scaler_diagnostics.csv").write_text(
        "tampered\n", encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="hash"):
        runner.audit(outputs[0])

    decision_path = outputs[1] / "R0_V2_DECISION.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    decision["decision"] = "UNKNOWN"
    decision_path.write_text(json.dumps(decision), encoding="utf-8")
    index_path = outputs[1] / "sha256_index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["R0_V2_DECISION.json"] = runner.sha256_file(decision_path)
    index_path.write_text(json.dumps(index), encoding="utf-8")
    with pytest.raises(RuntimeError, match="decision vocabulary"):
        runner.audit(outputs[1])

    execution_path = outputs[2] / "protocol_manifest_execution.json"
    execution = json.loads(execution_path.read_text(encoding="utf-8"))
    execution["access_events"][0]["client"] = "C9"
    execution_path.write_text(json.dumps(execution), encoding="utf-8")
    index_path = outputs[2] / "sha256_index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["protocol_manifest_execution.json"] = runner.sha256_file(execution_path)
    index_path.write_text(json.dumps(index), encoding="utf-8")
    with pytest.raises(RuntimeError, match="access"):
        runner.audit(outputs[2])

    execution_path = outputs[3] / "protocol_manifest_execution.json"
    execution = json.loads(execution_path.read_text(encoding="utf-8"))
    execution["access_events"][0]["operation"] = "unregistered_operation"
    execution["access_events"][0]["sequence"] = 99
    execution_path.write_text(json.dumps(execution), encoding="utf-8")
    index_path = outputs[3] / "sha256_index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["protocol_manifest_execution.json"] = runner.sha256_file(execution_path)
    index_path.write_text(json.dumps(index), encoding="utf-8")
    marker_path = outputs[3] / "fixed_endpoint_complete.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["sha256_index_sha256"] = runner.sha256_file(index_path)
    marker_path.write_text(json.dumps(marker), encoding="utf-8")
    with pytest.raises(RuntimeError, match="access"):
        runner.audit(outputs[3])


def test_audit_rejects_incomplete_pass_and_fail_hash_indexes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches an omitted evidence file being outside audit hash coverage."""
    force_synthetic_alpha(monkeypatch)
    pass_output = tmp_path / "pass"
    fail_output = tmp_path / "fail"
    runner._execute_source_only(
        RecordingSyntheticProvider(), pass_output, frozen_protocol_fixture()
    )
    original = runner.functional_diagnostics_v2

    def forced_failure(*args: object, **kwargs: object) -> dict[str, object]:
        row = original(*args, **kwargs)
        row["raw_prediction_pass"] = False
        return row

    monkeypatch.setattr(runner, "functional_diagnostics_v2", forced_failure)
    runner._execute_source_only(
        RecordingSyntheticProvider(), fail_output, frozen_protocol_fixture()
    )
    for output in (pass_output, fail_output):
        index_path = output / "sha256_index.json"
        index = json.loads(index_path.read_text(encoding="utf-8"))
        del index["r0_v2_scaler_diagnostics.csv"]
        index_path.write_text(json.dumps(index), encoding="utf-8")
        marker_path = output / "fixed_endpoint_complete.json"
        if marker_path.exists():
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            marker["sha256_index_sha256"] = runner.sha256_file(index_path)
            marker_path.write_text(json.dumps(marker), encoding="utf-8")

        with pytest.raises(RuntimeError, match="index completeness"):
            runner.audit(output)


def test_audit_rejects_changed_exact_blocking_finding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches replacing the observed failed gate with a vague favorable record."""
    force_synthetic_alpha(monkeypatch)
    original = runner.functional_diagnostics_v2

    def forced_failure(*args: object, **kwargs: object) -> dict[str, object]:
        row = original(*args, **kwargs)
        row["mae_parity_pass"] = False
        return row

    monkeypatch.setattr(runner, "functional_diagnostics_v2", forced_failure)
    output = tmp_path / "fail"
    runner._execute_source_only(
        RecordingSyntheticProvider(), output, frozen_protocol_fixture()
    )
    decision_path = output / "R0_V2_DECISION.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    decision["blocking_findings"] = ["incomplete_or_unknown_gate_evidence"]
    decision_path.write_text(json.dumps(decision), encoding="utf-8")
    index_path = output / "sha256_index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["R0_V2_DECISION.json"] = runner.sha256_file(decision_path)
    index_path.write_text(json.dumps(index), encoding="utf-8")

    with pytest.raises(RuntimeError, match="blocking findings"):
        runner.audit(output)


def test_late_environment_failure_is_preserved_as_complete_failure_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches a finalizer retry colliding with already-written core evidence."""
    force_synthetic_alpha(monkeypatch)
    real_environment = runner._environment_metadata

    def unavailable_environment() -> dict[str, object]:
        raise RuntimeError("synthetic environment metadata failure")

    monkeypatch.setattr(runner, "_environment_metadata", unavailable_environment)
    output = tmp_path / "run"

    result = runner._execute_source_only(
        RecordingSyntheticProvider(), output, frozen_protocol_fixture()
    )

    assert result["decision"] == "R0_V2_FAILED"
    assert (output / "R0_V2_DECISION.json").is_file()
    assert (output / "R0_V2_EXPERIMENT_AUDIT.md").is_file()
    assert (output / "sha256_index.json").is_file()
    assert not (output / "fixed_endpoint_complete.json").exists()
    decision = json.loads(
        (output / "R0_V2_DECISION.json").read_text(encoding="utf-8")
    )
    assert "environment metadata failure" in decision["error"]
    assert runner.audit(output)["decision"] == "R0_V2_FAILED"

    monkeypatch.setattr(runner, "_environment_metadata", real_environment)
    real_write_index = runner._write_hash_index
    attempts = 0

    def fail_first_index_write(index_output: Path) -> dict[str, str]:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("synthetic late index failure")
        return real_write_index(index_output)

    monkeypatch.setattr(runner, "_write_hash_index", fail_first_index_write)
    retry_output = tmp_path / "late-index"
    retry_provider = RecordingSyntheticProvider()
    retry_result = runner._execute_source_only(
        retry_provider, retry_output, frozen_protocol_fixture()
    )

    assert attempts == 2
    assert retry_result["decision"] == (
        "FEDRIDGE_ALGEBRAIC_EXACT_NUMERICAL_EQUIVALENCE_ESTABLISHED"
    )
    assert retry_provider.requests == expected_source_only_request_sequence()
    assert runner.audit(retry_output)["status"] == "PASS"
