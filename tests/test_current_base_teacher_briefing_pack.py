from pathlib import Path

from run_current_base_teacher_briefing_pack import (
    build_slide_outline,
    build_table_figure_sequence,
    p4_metric_table,
    write_f1_system_pipeline,
    write_teacher_briefing,
)


def sample_policy() -> dict[str, object]:
    return {
        "per_client_thresholds": {
            "C3": {"threshold_label": "0.015903"},
            "C4": {"threshold_label": "0.000000"},
            "C5": {"threshold_label": "0.050000"},
        }
    }


def sample_headline_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for scope, rmse, nrmse, baseline, h8_all, gain, usage in [
        ("ALL", 5.850, 0.0339, 7.049, 6.509, 1.199, 0.203),
        ("C3", 4.882, 0.0307, 5.797, 5.870, 0.915, 0.212),
        ("C4", 5.774, 0.0333, 6.336, 6.567, 0.562, 0.216),
        ("C5", 7.558, 0.0406, 9.695, 7.632, 2.137, 0.170),
    ]:
        rows.append(
            {
                "metric_id": f"P4_{scope}",
                "RMSE": rmse,
                "NRMSE": nrmse,
                "baseline_h23_RMSE": baseline,
                "h8_all_RMSE": h8_all,
                "rmse_gain_vs_h23": gain,
                "h8_usage_rate": usage,
            }
        )
    return rows


def sample_table_rows() -> list[dict[str, str]]:
    return [
        {"table_id": "T1", "title": "Real-route full main table", "source": "real_route_mainline_summary.csv"},
        {"table_id": "T2", "title": "Real-route Accepted+Review main table", "source": "real_route_post_qc_summary.csv"},
        {"table_id": "T3", "title": "P4 threshold guard deployment audit", "source": "threshold_guard_metrics.csv"},
        {"table_id": "T4", "title": "P4 selected thresholds", "source": "threshold_guard_selected_thresholds.csv"},
        {"table_id": "T5", "title": "P3 low-cal stress", "source": "selector_low_cal_metric_summary.csv"},
        {"table_id": "T6", "title": "P5 route-gap appendix", "source": "light_route_gap_appendix_table.csv"},
        {"table_id": "T7", "title": "Claim-evidence matrix", "source": "claim_evidence_matrix.csv"},
    ]


def sample_figure_rows() -> list[dict[str, str]]:
    return [
        {"figure_id": "F2", "title": "P4 threshold guard per-client gains", "source": "f2.csv"},
        {"figure_id": "F3", "title": "CO/nonCO safety panel", "source": "f3.csv"},
        {"figure_id": "F4", "title": "Low-cal budget stability", "source": "f4.csv"},
        {"figure_id": "F5", "title": "Route-gap appendix", "source": "f5.csv"},
    ]


def test_f1_system_pipeline_contains_runtime_nodes_and_thresholds(tmp_path: Path):
    path = tmp_path / "F1_system_pipeline.svg"

    write_f1_system_pipeline(path, sample_policy())

    text = path.read_text(encoding="utf-8")
    assert "F6 real-route" in text
    assert "Backbone freeze principle" in text
    assert "H2.3+ target profile" in text
    assert "H8+C4 formal rescue" in text
    assert "Per-client threshold guard" in text
    assert "C5: 0.050000" in text


def test_table_figure_sequence_prioritizes_system_story_and_p4_result():
    rows = build_table_figure_sequence(
        sample_table_rows(),
        sample_figure_rows(),
        {"F1": "figures/F1.svg", "F2": "figures/F2.svg", "F3": "figures/F3.svg", "F4": "figures/F4.svg", "F5": "figures/F5.svg"},
    )

    assert [row["item_id"] for row in rows] == ["F1", "T3", "F2", "F3", "T4", "F4", "T1", "T2", "T5", "T6", "F5", "T7"]
    assert rows[0]["placement"] == "opening_system_story"
    assert rows[1]["source"] == "threshold_guard_metrics.csv"
    assert rows[-1]["title"] == "Claim-evidence matrix"


def test_slide_outline_is_teacher_ready_eight_page_story():
    rows = build_slide_outline()

    assert len(rows) == 8
    assert rows[1]["assets"] == "F1"
    assert "F6 real-route" in rows[1]["content"]
    assert rows[-1]["assets"] == "T7"
    assert "P6" in rows[-1]["content"]


def test_p4_metric_table_formats_accepted_review_metrics():
    rows = p4_metric_table(sample_headline_rows())
    by_scope = {row["scope"]: row for row in rows}

    assert by_scope["ALL"]["RMSE/NRMSE"] == "5.850 / 0.0339"
    assert by_scope["C5"]["H2.3+ RMSE"] == "9.695"
    assert by_scope["C5"]["gain"] == "2.137"
    assert by_scope["C5"]["H8 usage"] == "17.0%"


def test_teacher_briefing_links_main_metrics_figures_and_sequence(tmp_path: Path):
    path = tmp_path / "teacher_briefing.zh.md"
    figure_paths = {
        "F1": "figures/F1_system_pipeline.svg",
        "F2": "figures/F2_threshold_guard_gains.svg",
        "F3": "figures/F3_co_nonco_safety.svg",
        "F4": "figures/F4_low_cal_stability.svg",
        "F5": "figures/F5_route_gap_appendix.svg",
    }
    sequence_rows = build_table_figure_sequence(sample_table_rows(), sample_figure_rows(), figure_paths)

    write_teacher_briefing(
        path,
        headline_rows=sample_headline_rows(),
        sequence_rows=sequence_rows,
        slide_rows=build_slide_outline(),
        figure_paths=figure_paths,
    )

    text = path.read_text(encoding="utf-8")
    assert "T1-T7 / F1-F5" in text
    assert "P4 threshold guard" in text
    assert "5.850 / 0.0339" in text
    assert "figures/F1_system_pipeline.svg" in text
    assert "threshold_guard_metrics.csv" in text
