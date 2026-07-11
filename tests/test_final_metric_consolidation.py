from pathlib import Path

from run_final_metric_consolidation_20260709 import (
    build_classification_correct_table,
    build_co_rescue_decomposition,
    build_primary_result_table,
    build_regression_slice_table,
    write_story_brief,
)


def metric_row(scope: str, qc_slice: str = "accepted_review", n: int = 10) -> dict[str, object]:
    return {
        "dataset": "test",
        "route": "real-route",
        "budget_per_client": "160",
        "repeat": "0",
        "scope": scope,
        "qc_slice": qc_slice,
        "N": n,
        "RMSE": 5.0,
        "NRMSE": 0.03,
        "MAE": 3.0,
        "P90AE": 8.0,
        "baseline_h23_RMSE": 6.0,
        "baseline_h23_NRMSE": 0.04,
        "h8_all_RMSE": 5.5,
        "h8_all_NRMSE": 0.035,
        "rmse_gain_vs_h23": 1.0,
        "nrmse_gain_vs_h23": 0.01,
        "h8_usage_rate": 0.2,
        "selected_threshold_labels": "0.05",
    }


def test_primary_result_table_reports_accepted_review_coverage():
    rows = [
        metric_row("ALL", "full", 20),
        metric_row("ALL", "accepted_review", 10),
        metric_row("C3", "full", 8),
        metric_row("C3", "accepted_review", 4),
        metric_row("C4", "full", 6),
        metric_row("C4", "accepted_review", 3),
        metric_row("C5", "full", 6),
        metric_row("C5", "accepted_review", 3),
    ]

    out = build_primary_result_table(rows)
    all_row = next(row for row in out if row["scope"] == "ALL")

    assert all_row["reporting_slice"] == "real-route Accepted+Review"
    assert all_row["coverage_review"] == 0.5
    assert all_row["rmse_gain_vs_h23"] == 1.0


def _legacy_classification_correct_table_filters_to_ar_and_correct_route():
    test_rows = [
        {
            "client": "C3",
            "split": "test",
            "qc_decision": "accept",
            "route_class": "1",
            "true_class": "1",
            "true_ppm": "100",
            "h23_ppm": "90",
            "h8_ppm": "100",
            "threshold_guard_ppm": "100",
        },
        {
            "client": "C3",
            "split": "test",
            "qc_decision": "review",
            "route_class": "0",
            "true_class": "1",
            "true_ppm": "100",
            "h23_ppm": "90",
            "h8_ppm": "100",
            "threshold_guard_ppm": "100",
        },
        {
            "client": "C4",
            "split": "test",
            "qc_decision": "reject",
            "route_class": "1",
            "true_class": "1",
            "true_ppm": "100",
            "h23_ppm": "90",
            "h8_ppm": "100",
            "threshold_guard_ppm": "100",
        },
        {
            "client": "C5",
            "split": "test",
            "qc_decision": "review",
            "route_class": "3",
            "true_class": "3",
            "true_ppm": "150",
            "h23_ppm": "140",
            "h8_ppm": "145",
            "threshold_guard_ppm": "140",
        },
    ]

    out = build_classification_correct_table(test_rows)
    p4_all = next(row for row in out if row["scope"] == "ALL" and row["profile"] == "P4 threshold guard")
    h23_all = next(row for row in out if row["scope"] == "ALL" and row["profile"] == "H2.3+ anchor")

    assert p4_all["condition"] == "S_AR ∩ S_CC"
    assert p4_all["N"] == 2
    assert p4_all["accepted_review_N"] == 3
    assert p4_all["class_correct_rate_within_AR"] == 2 / 3
    assert p4_all["RMSE"] < h23_all["RMSE"]


def test_classification_correct_slice_is_independent_of_qc():
    rows = [
        {
            "client": "C5",
            "split": "test",
            "qc_decision": "accept",
            "route_class": "1",
            "true_class": "1",
            "true_ppm": "100",
            "h23_ppm": "90",
            "threshold_guard_ppm": "100",
        },
        {
            "client": "C5",
            "split": "test",
            "qc_decision": "reject",
            "route_class": "3",
            "true_class": "3",
            "true_ppm": "150",
            "h23_ppm": "140",
            "threshold_guard_ppm": "145",
        },
        {
            "client": "C5",
            "split": "test",
            "qc_decision": "review",
            "route_class": "0",
            "true_class": "1",
            "true_ppm": "100",
            "h23_ppm": "90",
            "threshold_guard_ppm": "80",
        },
        {
            "client": "C5",
            "split": "test",
            "qc_decision": "reject",
            "route_class": "0",
            "true_class": "0",
            "true_ppm": "50",
            "h23_ppm": "45",
            "threshold_guard_ppm": "48",
        },
    ]

    out = build_regression_slice_table(rows, [("P4", "threshold_guard_ppm")])
    by_slice = {
        row["slice"]: row
        for row in out
        if row["scope"] == "ALL" and row["profile"] == "P4"
    }

    assert by_slice["all"]["N"] == 4
    assert by_slice["class_correct"]["condition"] == "S_CC"
    assert by_slice["class_correct"]["N"] == 3
    assert by_slice["accepted_review"]["N"] == 2
    assert by_slice["accepted_review_class_correct"]["N"] == 1
    assert by_slice["class_correct"]["N"] > by_slice["accepted_review_class_correct"]["N"]
    assert by_slice["accepted_review"]["N"] > by_slice["accepted_review_class_correct"]["N"]
    assert by_slice["accepted_review"]["parent_slice"] == "all"
    assert by_slice["accepted_review_class_correct"]["parent_slice"] == "class_correct"
    assert by_slice["class_correct"]["coverage"] == 3 / 4

    pure_cc = build_classification_correct_table(rows)
    p4_cc = next(
        row
        for row in pure_cc
        if row["scope"] == "ALL" and row["profile"] == "P4 threshold guard"
    )
    assert p4_cc["condition"] == "S_CC"
    assert p4_cc["N"] == 3


def test_co_rescue_decomposition_marks_nonco_as_guarded():
    rows = [
        metric_row("C5-CO"),
        {**metric_row("C5-nonCO"), "h8_usage_rate": 0.0, "rmse_gain_vs_h23": 0.0},
    ]

    out = build_co_rescue_decomposition(rows)
    by_scope = {row["scope"]: row for row in out}

    assert "CO rescue branch" in by_scope["C5-CO"]["interpretation"]
    assert "protected fallback" in by_scope["C5-nonCO"]["interpretation"]


def test_story_brief_contains_main_slices(tmp_path: Path):
    primary_rows = [metric_row("ALL"), metric_row("C5")]
    cc_rows = [
        {
            "scope": "ALL",
            "profile": "P4 threshold guard",
            "condition": "S_AR ∩ S_CC",
            "N": 8,
            "RMSE": 4.0,
            "NRMSE": 0.02,
            "rmse_gain_vs_h23": 1.0,
            "class_correct_rate_within_AR": 0.8,
        }
    ]
    co_rows = [build_co_rescue_decomposition([metric_row("C5-CO")])[0]]
    low_cal_rows = []
    path = tmp_path / "brief.zh.md"

    write_story_brief(path, primary_rows, cc_rows, co_rows, low_cal_rows)

    text = path.read_text(encoding="utf-8")
    assert "real-route Accepted+Review" in text
    assert "S_AR ∩ S_CC" in text
    assert "P4 threshold guard" in text
