from pathlib import Path
import shutil

import pytest

from run_current_base_regression_story import (
    build_low_cal_summary,
    build_mainline_summary,
    build_post_qc_summary,
    build_route_gap_summary,
    write_story_report,
)


def test_build_mainline_summary_extracts_oracle_profiles():
    rows = [
        {"mode": "oracle-route_H2.3+", "scope": "ALL", "N": "10", "RMSE": "9.8", "NRMSE": "0.05"},
        {"mode": "oracle-route_H2.3+", "scope": "C3", "N": "4", "RMSE": "9.1", "NRMSE": "0.04"},
        {"mode": "oracle-route_H8+C4", "scope": "ALL", "N": "10", "RMSE": "9.0", "NRMSE": "0.051"},
        {"mode": "oracle-route_H8+C4", "scope": "C3", "N": "4", "RMSE": "9.2", "NRMSE": "0.041"},
        {"mode": "oracle-route_guarded_profile", "scope": "ALL", "N": "10", "RMSE": "9.1", "NRMSE": "0.049"},
        {"mode": "oracle-route_guarded_profile", "scope": "C3", "N": "4", "RMSE": "9.1", "NRMSE": "0.04"},
    ]
    h23_rows = [
        {"mode": "H2_3_oracle_route", "scope": "ALL", "N": "10", "RMSE": "10.5", "NRMSE": "0.056"},
        {"mode": "H2_3_oracle_route", "scope": "C3", "N": "4", "RMSE": "9.7", "NRMSE": "0.052"},
    ]

    out = build_mainline_summary(rows, h23_rows, scopes=["ALL", "C3"])

    assert out[0] == {
        "profile": "H2.3 oracle-route",
        "scope": "ALL",
        "N": 10,
        "RMSE": 10.5,
        "NRMSE": 0.056,
    }
    assert {"profile": "Guarded practical oracle-route", "scope": "ALL", "N": 10, "RMSE": 9.1, "NRMSE": 0.049} in out


def test_build_mainline_summary_requires_profile_scope_pair():
    rows = [{"mode": "oracle-route_H2.3+", "scope": "ALL", "N": "10", "RMSE": "9.8", "NRMSE": "0.05"}]

    with pytest.raises(ValueError, match="Missing metric row"):
        build_mainline_summary(rows, [], scopes=["ALL"])


def test_build_post_qc_summary_extracts_accepted_review_metrics():
    rows = [
        {
            "profile": "Guarded practical oracle-route",
            "scope": "ALL",
            "N": "10",
            "coverage_review": "0.75",
            "nonreject_N": "8",
            "coverage_review_RMSE": "6.3",
            "coverage_review_NRMSE": "0.035",
        }
    ]

    out = build_post_qc_summary(rows, scopes=["ALL"], profiles=["Guarded practical oracle-route"])

    assert out == [
        {
            "profile": "Guarded practical oracle-route",
            "scope": "ALL",
            "N": 10,
            "coverage_review": 0.75,
            "nonreject_N": 8,
            "coverage_review_RMSE": 6.3,
            "coverage_review_NRMSE": 0.035,
        }
    ]


def test_build_route_gap_summary_keeps_core_gap_fields():
    rows = [
        {
            "profile_family": "H2.3+",
            "scope": "C5",
            "N": "1360",
            "gap_full_RMSE": "27.09",
            "gap_full_NRMSE": "0.259",
            "gap_full_RMSE_pct_of_real": "0.69",
        }
    ]

    assert build_route_gap_summary(rows, scopes=["C5"], profile_families=["H2.3+"]) == [
        {
            "profile_family": "H2.3+",
            "scope": "C5",
            "N": 1360,
            "gap_full_RMSE": 27.09,
            "gap_full_NRMSE": 0.259,
            "gap_full_RMSE_pct_of_real": 0.69,
        }
    ]


def test_build_low_cal_summary_keeps_budget_96_modes():
    profile_choice_rows = [
        {
            "route": "oracle-route",
            "budget_per_client": "96",
            "client": "C5",
            "H2_3_plus_rate": "0.0",
            "H8_C4_rate": "1.0",
            "profile_mode": "H8+C4",
            "profile_mode_rate": "1.0",
        }
    ]
    blend_rows = [
        {
            "route": "oracle-route",
            "budget_per_client": "96",
            "client": "C5",
            "weight_mode": "0.25",
            "weight_mode_rate": "1.0",
        }
    ]

    out = build_low_cal_summary(profile_choice_rows, blend_rows, budget=96)

    assert out == [
        {
            "route": "oracle-route",
            "client": "C5",
            "budget_per_client": 96,
            "profile_mode": "H8+C4",
            "profile_mode_rate": 1.0,
            "H8_C4_rate": 1.0,
            "blend_weight_mode": 0.25,
            "blend_weight_mode_rate": 1.0,
        }
    ]


def test_write_story_report_contains_required_sections():
    out_dir = Path(".pytest_tmp_current_base_story")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    report = write_story_report(
        out_dir,
        mainline_rows=[
            {"profile": "Guarded practical oracle-route", "scope": "ALL", "N": 10, "RMSE": 9.1, "NRMSE": 0.049}
        ],
        post_qc_rows=[
            {
                "profile": "Guarded practical oracle-route",
                "scope": "ALL",
                "N": 10,
                "coverage_review": 0.75,
                "nonreject_N": 8,
                "coverage_review_RMSE": 6.3,
                "coverage_review_NRMSE": 0.035,
            }
        ],
        route_gap_rows=[
            {
                "profile_family": "H2.3+",
                "scope": "C5",
                "N": 1360,
                "gap_full_RMSE": 27.09,
                "gap_full_NRMSE": 0.259,
                "gap_full_RMSE_pct_of_real": 0.69,
            }
        ],
        low_cal_rows=[
            {
                "route": "oracle-route",
                "client": "C5",
                "budget_per_client": 96,
                "profile_mode": "H8+C4",
                "profile_mode_rate": 1.0,
                "H8_C4_rate": 1.0,
                "blend_weight_mode": 0.25,
                "blend_weight_mode_rate": 1.0,
            }
        ],
    )

    text = Path(report).read_text(encoding="utf-8")
    assert "Oracle-route Full" in text
    assert "Accepted+Review" in text
    assert "Route Gap" in text
    assert "R3aK16" in text
    assert "9.100 / 0.0490" in text
    assert "6.300 / 0.0350" in text
    shutil.rmtree(out_dir)
