"""Unit tests for the authoritative regression/QC consolidation helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from consolidate_current_authoritative_results import (  # noqa: E402
    calibration_threshold,
    scoped_prediction,
    metric_row,
    normalize_evidence_status,
    summarize_selective,
)


def test_metric_row_recomputes_standard_regression_metrics_and_percentiles() -> None:
    row = metric_row(
        truth=np.array([0.0, 10.0, 20.0]),
        prediction=np.array([1.0, 8.0, 23.0]),
        gas_range=20.0,
    )

    assert row["n"] == 3
    assert row["rmse"] == pytest.approx(np.sqrt(14.0 / 3.0))
    assert row["mae"] == pytest.approx(2.0)
    assert row["nrmse_range"] == pytest.approx(np.sqrt(14.0 / 3.0) / 20.0)
    assert row["r2"] == pytest.approx(0.93)
    assert row["bias"] == pytest.approx(2.0 / 3.0)
    assert row["p90_absolute_error"] == pytest.approx(2.8)


def test_calibration_threshold_uses_only_stable_calibration_order_for_ties() -> None:
    lock = calibration_threshold(
        risks=np.array([0.4, 0.1, 0.4, 0.2]),
        identities=["z", "b", "a", "c"],
        nominal=0.75,
    )

    assert lock["threshold"] == pytest.approx(0.4)
    assert lock["selected_identity"] == "a"
    assert lock["selected_index"] == 2
    assert lock["ties_at_threshold"] == 2
    assert lock["source"] == "calibration_only"


def test_selective_summary_accepts_threshold_ties_and_reports_population_metrics() -> None:
    row = summarize_selective(
        truth=np.array([0.0, 10.0, 20.0]),
        prediction=np.array([1.0, 14.0, 19.0]),
        risks=np.array([0.2, 0.4, 0.4]),
        threshold=0.4,
        gas_range=20.0,
    )

    assert row["accepted_n"] == 3
    assert row["rejected_n"] == 0
    assert row["coverage"] == pytest.approx(1.0)
    assert row["accepted_rmse"] == pytest.approx(np.sqrt(6.0))


def test_selective_summary_filters_rowwise_normalization_ranges() -> None:
    row = summarize_selective(
        truth=np.array([0.0, 10.0, 20.0]),
        prediction=np.array([1.0, 14.0, 19.0]),
        risks=np.array([0.2, 0.4, 0.9]),
        threshold=0.4,
        gas_range=np.array([10.0, 20.0, 40.0]),
    )

    assert row["accepted_nrmse_range"] == pytest.approx(np.sqrt(((1 / 10) ** 2 + (4 / 20) ** 2) / 2))


def test_pooled_micro_metric_is_not_macro_average() -> None:
    micro = metric_row(
        truth=np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
        prediction=np.array([0.0, 0.0, 0.0, 0.0, 10.0]),
        gas_range=10.0,
    )
    macro_rmse = (0.0 + 10.0) / 2.0

    assert micro["rmse"] == pytest.approx(np.sqrt(20.0))
    assert micro["rmse"] != pytest.approx(macro_rmse)


def test_metric_row_uses_per_sample_ranges_for_canonical_nrmse() -> None:
    row = metric_row(
        truth=np.array([0.0, 0.0]),
        prediction=np.array([10.0, 10.0]),
        gas_range=np.array([100.0, 200.0]),
    )

    assert row["nrmse_range"] == pytest.approx(np.sqrt((0.1**2 + 0.05**2) / 2.0))


def test_historical_and_invalid_evidence_statuses_remain_distinct() -> None:
    assert normalize_evidence_status("historical_superseded") == "HISTORICAL_SUPERSEDED"
    assert normalize_evidence_status("invalid_do_not_cite") == "INVALID_DO_NOT_CITE"
    with pytest.raises(ValueError):
        normalize_evidence_status("archived")


def test_scoped_prediction_uses_sealed_r1_route_columns_for_oracle_scopes() -> None:
    row = {
        "prediction": "12.0",
        "true_class": "2",
        "predicted_class": "1",
        "route_2": "21.0",
    }

    assert scoped_prediction(row, "S_ALL") == pytest.approx(12.0)
    assert scoped_prediction(row, "Oracle_ALL") == pytest.approx(21.0)
    assert scoped_prediction(row, "Oracle_CC") == pytest.approx(21.0)
