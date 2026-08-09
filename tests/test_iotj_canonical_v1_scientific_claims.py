from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.analyze_iotj_canonical_v1_scientific_claims import (
    _aurc,
    capture_summary,
    grouped_paired_rmse_bootstrap,
    risk_coverage_curve,
    routing_gap_row,
)


def test_routing_gap_definitions() -> None:
    row = routing_gap_row(s_all_rmse=12.0, s_cc_rmse=9.0, oracle_rmse=8.0)
    assert row["routing_gap_ppm"] == pytest.approx(3.0)
    assert row["oracle_gap_ppm"] == pytest.approx(1.0)


def test_grouped_bootstrap_preserves_pairs_and_is_deterministic() -> None:
    frame = pd.DataFrame({
        "group": ["a", "a", "b", "b"],
        "true_ppm": [0.0, 1.0, 10.0, 11.0],
        "pred_83d_ppm": [1.0, 2.0, 12.0, 13.0],
        "pred_84d_h1_ppm": [0.5, 1.5, 11.0, 12.0],
    })
    first = grouped_paired_rmse_bootstrap(frame, "group", repeats=200, seed=7)
    second = grouped_paired_rmse_bootstrap(frame, "group", repeats=200, seed=7)
    assert first == second
    assert first["group_count"] == 2
    assert first["delta_rmse_ppm"] < 0
    assert first["bootstrap_unit"] == "raw_file_group"


def test_risk_curve_accepts_lowest_risk_first() -> None:
    frame = pd.DataFrame({
        "qc_risk_score_final": [0.1, 0.2, 0.8, 0.9],
        "true_ppm": [0.0, 0.0, 0.0, 0.0],
        "pred_84d_h1_ppm": [1.0, 1.0, 10.0, 10.0],
        "true_class": [0, 0, 0, 0],
    })
    curve = risk_coverage_curve(frame, coverages=[0.5, 1.0])
    assert curve[0]["RMSE"] == pytest.approx(1.0)
    assert curve[1]["RMSE"] > curve[0]["RMSE"]


def test_error_capture_uses_nonaccepted_population() -> None:
    frame = pd.DataFrame({
        "HC90_decision": ["accepted", "review", "reject", "accepted"],
        "route_correct": [1, 0, 1, 0],
        "abs_error": [1.0, 2.0, 50.0, 60.0],
    })
    row = capture_summary(frame, "HC90")
    assert row["misroute_capture_rate"] == pytest.approx(0.5)
    assert row["error_gt40_capture_rate"] == pytest.approx(0.5)
    assert row["high_risk_N"] == 2


def test_aurc_is_compatible_with_workspace_numpy() -> None:
    rows = [
        {"scope": "ALL", "coverage": 0.5, "NRMSE_range": 0.1},
        {"scope": "ALL", "coverage": 1.0, "NRMSE_range": 0.2},
    ]
    assert _aurc(rows, "ALL") == pytest.approx(0.15)
