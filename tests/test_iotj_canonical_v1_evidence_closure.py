from __future__ import annotations

import math

import pytest
from gaps_deploy.c5_h8_runtime import SerializedRidge
import numpy as np

from scripts.finalize_iotj_canonical_v1_evidence import (
    assign_quality_stratum,
    classify_qc_decision,
    combined_equal_mean_risk,
    engineering_claims,
    interval_overlap_seconds,
    merged_interval_length,
    predict_serialized_ridge,
    summarize_error_scope,
)


def test_quality_strata_are_predeclared_from_acquisition_metadata() -> None:
    assert assign_quality_stratum(
        observed_ratio=1.0,
        empty_bin_ratio=0.0,
        max_missing_run=0,
        short_gap_interpolated_ratio=0.0,
    ) == "Q0"
    assert assign_quality_stratum(
        observed_ratio=0.99,
        empty_bin_ratio=0.01,
        max_missing_run=1,
        short_gap_interpolated_ratio=0.01,
    ) == "Q1"
    assert assign_quality_stratum(
        observed_ratio=0.94,
        empty_bin_ratio=0.06,
        max_missing_run=3,
        short_gap_interpolated_ratio=0.04,
    ) == "Q2"
    assert assign_quality_stratum(
        observed_ratio=0.80,
        empty_bin_ratio=0.20,
        max_missing_run=7,
        short_gap_interpolated_ratio=0.10,
    ) == "Q3"


def test_qc_three_way_rule_preserves_frozen_accept_threshold() -> None:
    assert classify_qc_decision(0.20, accept_threshold=0.30, review_threshold=0.40) == "accepted"
    assert classify_qc_decision(0.35, accept_threshold=0.30, review_threshold=0.40) == "review"
    assert classify_qc_decision(0.50, accept_threshold=0.30, review_threshold=0.40) == "reject"
    with pytest.raises(ValueError, match="review threshold"):
        classify_qc_decision(0.2, accept_threshold=0.4, review_threshold=0.3)


def test_equal_mean_risk_clips_each_calibration_normalized_component() -> None:
    row = {
        "classification_uncertainty_risk": 0.5,
        "regression_disagreement_risk": 4.0,
        "source_prior_disagreement_risk": 0.0,
    }
    scales = {
        "classification_uncertainty_risk": 1.0,
        "regression_disagreement_risk": 2.0,
        "source_prior_disagreement_risk": 1.0,
    }
    assert combined_equal_mean_risk(row, scales) == pytest.approx((0.5 + 1.0 + 0.0) / 3.0)


def test_error_summary_reports_required_submission_metrics() -> None:
    rows = [
        {"true_ppm": 0.0, "pred_84d_h1_ppm": 1.0, "true_class": 0},
        {"true_ppm": 2.0, "pred_84d_h1_ppm": 0.0, "true_class": 0},
    ]
    summary = summarize_error_scope(rows)
    assert summary["N"] == 2
    assert summary["RMSE"] == pytest.approx(math.sqrt(2.5))
    assert summary["MAE"] == pytest.approx(1.5)
    assert summary["P90AE"] == pytest.approx(2.0)
    assert summary["Bias"] == pytest.approx(-0.5)
    assert "NRMSE_range" in summary


def test_serialized_ridge_prediction_uses_flat_runtime_features() -> None:
    model = SerializedRidge(
        feature_names=("x",),
        mean=np.asarray([0.0]),
        scale=np.asarray([1.0]),
        coef=np.asarray([0.0, 2.0]),
        clip_min=-100.0,
        clip_max=100.0,
    )
    assert predict_serialized_ridge(model, {"x": 3.0}) == pytest.approx(6.0)


def test_raw_time_overlap_uses_physical_interval_intersection() -> None:
    assert interval_overlap_seconds(60.0, 70.0, 65.0, 75.0) == pytest.approx(5.0)
    assert interval_overlap_seconds(60.0, 70.0, 70.0, 80.0) == 0.0
    assert interval_overlap_seconds(60.0, 70.0, 55.0, 65.0) == pytest.approx(5.0)
    assert merged_interval_length([(60.0, 70.0), (65.0, 75.0), (90.0, 100.0)]) == pytest.approx(25.0)


def test_engineering_claims_separate_input_and_parameter_payload() -> None:
    row = engineering_claims(points_old=100, points_new=50, channels=8, parameter_bytes=1234)
    assert row["legacy_input_tensor_bytes_fp32"] == 3200
    assert row["canonical_input_tensor_bytes_fp32"] == 1600
    assert row["temporal_input_reduction"] == pytest.approx(0.5)
    assert row["parameter_communication_reduction"] == 0.0
