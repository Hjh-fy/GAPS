from __future__ import annotations

import math

import pytest

from scripts.finalize_iotj_regression_qc_closure import (
    COVERAGE_GRID,
    build_risk_fields,
    enrich_endpoint_split,
    exact_retention_mask,
    fit_locked_qc,
    grouped_bootstrap_rmse_delta,
    is_methane_225_repeat1,
    load_frozen_r84_models,
    resolve_posthoc_endpoint,
    same_count_risk_summary,
    trapezoidal_area,
)


def _row(index: int, filename: str, confidence: float, pred83: float, pred84: float):
    return {
        "sample_index": index,
        "filename": filename,
        "true_class": index % 2,
        "pred_class": index % 2,
        "true_ppm": 10.0 + index,
        "confidence": confidence,
        "pred_83d_ppm": pred83,
        "pred_84d_h1_ppm": pred84,
        "h1_ppm": pred84,
        "h2_ppm": pred84 + 1.0,
        "h3_ppm": pred84 - 1.0,
    }


def test_only_registered_c5_posthoc_endpoint_is_resolvable():
    assert resolve_posthoc_endpoint("C5").name == "retry3"
    with pytest.raises(RuntimeError, match="BLOCKED_MISSING_POSTHOC_ENDPOINT"):
        resolve_posthoc_endpoint("C3")
    with pytest.raises(RuntimeError, match="BLOCKED_MISSING_POSTHOC_ENDPOINT"):
        resolve_posthoc_endpoint("C4")


def test_qc_lock_uses_calibration_only_and_registered_coverage_grid():
    calibration = [
        build_risk_fields(_row(i, f"cal_{i // 2}", 0.8 + i / 100.0, 10 + i, 11 + i))
        for i in range(10)
    ]
    lock = fit_locked_qc(calibration)
    assert tuple(point["target_coverage"] for point in lock) == COVERAGE_GRID
    assert all(point["selection_split"] == "C5_calibration_x_only_risk" for point in lock)
    assert all(point["target_test_used_for_selection"] is False for point in lock)


def test_equal_mean_risk_contains_all_three_frozen_components():
    row = build_risk_fields(_row(0, "raw_a", 0.8, 10.0, 12.0))
    assert math.isclose(row["classification_uncertainty_risk"], 0.2)
    assert row["regression_disagreement_risk"] > 0
    assert row["source_prior_disagreement_risk"] > 0


def test_same_count_comparison_retains_exact_q3_count():
    rows = [
        build_risk_fields(_row(i, f"raw_{i // 2}", 0.70 + i / 100.0, 10 + i, 10.5 + i))
        for i in range(20)
    ]
    scales = {
        "classification_uncertainty_risk": 1.0,
        "regression_disagreement_risk": 1.0,
        "source_prior_disagreement_risk": 1.0,
    }
    result = same_count_risk_summary(rows, scales, retained_n=15, random_repeats=10, seed=42)
    assert {row["accepted_N"] for row in result} == {15}
    assert {row["method"] for row in result} == {"Q0_random", "Q1_confidence", "Q2_regression_disagreement", "Q3_equal_mean"}


def test_grouped_bootstrap_resamples_whole_raw_files():
    rows = [
        build_risk_fields(_row(i, f"raw_{i // 2}", 0.9, 9.0 + i, 10.0 + i))
        for i in range(8)
    ]
    summary = grouped_bootstrap_rmse_delta(rows, repeats=50, seed=42)
    assert summary["grouping_key"] == "filename"
    assert summary["group_N"] == 4
    assert summary["bootstrap_repeats"] == 50
    assert math.isfinite(summary["delta_rmse_m84_minus_m83"])


def test_frozen_r84_models_are_loaded_without_refitting():
    models = load_frozen_r84_models()
    assert set(models) == {0, 1, 2, 3}
    assert all(len(model.feature_names) == 84 for model in models.values())


def test_methane_225_repeat1_uses_canonical_numeric_metadata_semantics():
    assert is_methane_225_repeat1({"gas": "methane", "concentration": "225.0", "repeat_id": "1"})
    assert not is_methane_225_repeat1({"gas": "methane", "concentration": "225.0", "repeat_id": "2"})


def test_aurc_uses_numpy_version_compatible_trapezoid_rule():
    assert math.isclose(trapezoidal_area([2.0, 1.0], [0.5, 1.0]), 0.75)


def test_reconstructed_oracle_r84_matches_locked_endpoint_test_predictions():
    rows = enrich_endpoint_split("test")
    expected_rmse = math.sqrt(
        sum((float(row["pred_84d_oracle_ppm"]) - float(row["true_ppm"])) ** 2 for row in rows)
        / len(rows)
    )
    assert math.isclose(expected_rmse, 14.4488286458766, rel_tol=0, abs_tol=1e-9)


def test_risk_curve_comparators_use_identical_exact_retained_counts():
    values_a = [0.4, 0.1, 0.3, 0.2]
    values_b = [0.1, 0.4, 0.2, 0.3]
    mask_a = exact_retention_mask(values_a, 3)
    mask_b = exact_retention_mask(values_b, 3)
    assert int(mask_a.sum()) == int(mask_b.sum()) == 3
