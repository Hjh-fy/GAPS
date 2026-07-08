from pathlib import Path

from run_current_base_submission_docs_pack import (
    build_archive_inventory,
    build_document_index,
    build_method_chapter_sections,
    build_source_document_synthesis,
    summarize_archive_inventory,
    write_archive_plan,
    write_method_chapter,
    write_system_index,
)


def sample_headline_rows() -> list[dict[str, object]]:
    return [
        {
            "metric_id": "P4_ALL",
            "RMSE": 5.8496596735,
            "NRMSE": 0.0339497180,
            "baseline_h23_RMSE": 7.0488942382,
            "h8_all_RMSE": 6.5088327150,
            "rmse_gain_vs_h23": 1.1992345647,
            "h8_usage_rate": 0.2026963103,
        },
        {
            "metric_id": "P4_C5",
            "RMSE": 7.5578453781,
            "NRMSE": 0.0406048830,
            "baseline_h23_RMSE": 9.6945896864,
            "h8_all_RMSE": 7.6322970698,
            "rmse_gain_vs_h23": 2.1367443083,
            "h8_usage_rate": 0.1696428571,
        },
    ]


def sample_modules() -> list[dict[str, str]]:
    return [
        {
            "module_id": "M1",
            "module": "F6 real-route classification base",
            "why_use": "Use deployable route context.",
            "method_principle": "Predict route_class and risk.",
            "evidence": "F1",
            "paper_section": "System overview",
        },
        {
            "module_id": "M5",
            "module": "Per-client threshold guard",
            "why_use": "Use H8+C4 only where CO rescue is needed.",
            "method_principle": "route_class=CO and risk>=tau_c selects H8+C4.",
            "evidence": "T4",
            "paper_section": "Guarded profile selector",
        },
    ]


def sample_params() -> list[dict[str, str]]:
    return [
        {"name": "tau_C3", "value": "0.015903", "used_by": "P4 runtime", "meaning": "C3 threshold"},
        {"name": "tau_C4", "value": "0.000000", "used_by": "P4 runtime", "meaning": "C4 threshold"},
        {"name": "tau_C5", "value": "0.050000", "used_by": "P4 runtime", "meaning": "C5 threshold"},
    ]


def sample_sequence_rows() -> list[dict[str, str]]:
    return [
        {"item_id": "F1", "kind": "figure", "title": "System pipeline", "source": "figures/F1.svg", "placement": "opening_system_story"},
        {"item_id": "T3", "kind": "table", "title": "P4 threshold guard deployment audit", "source": "threshold_guard_metrics.csv", "placement": "main_result"},
        {"item_id": "F2", "kind": "figure", "title": "P4 gains", "source": "figures/F2.svg", "placement": "main_result_visual"},
        {"item_id": "T7", "kind": "table", "title": "Claim-evidence matrix", "source": "claim_evidence_matrix.csv", "placement": "claim_evidence"},
    ]


def sample_commands() -> list[dict[str, str]]:
    return [
        {"stage": "P4", "purpose": "Export threshold guard", "command": "python export_real_route_threshold_guard_deployment_candidate.py", "key_outputs": "threshold_guard_policy.json", "notes": "validation-selected"},
        {"stage": "Freeze", "purpose": "Freeze evidence", "command": "python run_current_base_evidence_freeze.py", "key_outputs": "frozen_headline_metrics.csv", "notes": "checks"},
    ]


def test_method_sections_keep_formula_metrics_and_reporting_boundaries():
    sections = build_method_chapter_sections(sample_headline_rows(), sample_modules(), sample_params(), sample_sequence_rows())
    by_id = {row["section_id"]: row for row in sections}

    assert "single-machine simulated federated continual learning" in by_id["S2"]["core_text"]
    assert "CLS-FlowerExpB-TimeAware2080" in by_id["S3"]["core_text"]
    assert "H8+C4" in by_id["S5"]["core_text"]
    assert "g_i" in by_id["S5"]["core_text"]
    assert "tau_c" in by_id["S5"]["core_text"]
    assert "5.850 / 0.0339" in by_id["S8"]["core_text"]
    assert "S_CC" in by_id["S9"]["core_text"]
    assert by_id["S9"]["role"] == "writing boundary"


def test_document_index_covers_teacher_method_system_and_figures():
    rows = build_document_index(
        freeze_dir=Path("results/current_base_evidence_freeze_20260708"),
        method_dir=Path("results/current_base_method_story_20260708"),
        teacher_dir=Path("results/current_base_teacher_briefing_pack_20260708"),
        output_dir=Path("results/current_base_submission_docs_pack_20260708"),
    )
    paths = {row["path"] for row in rows}

    assert "results/current_base_teacher_briefing_pack_20260708/teacher_briefing.zh.md" in paths
    assert "results/current_base_teacher_briefing_pack_20260708/figures/F1_system_pipeline.svg" in paths
    assert "results/current_base_submission_docs_pack_20260708/paper_method_chapter_draft.zh.md" in paths
    assert any(row["artifact_id"] == "TFS" for row in rows)


def test_source_document_synthesis_preserves_backbone_freeze_and_qc_v2_boundaries():
    rows = build_source_document_synthesis()
    by_id = {row["source_id"]: row for row in rows}

    assert "server_latest_adapted.pth + logits" in by_id["CLS"]["usable_takeaway"]
    assert "0.989444" in by_id["CLS"]["usable_takeaway"]
    assert "F6" in by_id["CLS"]["current_base_update"]
    assert "single-machine simulated federated continual learning" in by_id["FCL"]["usable_takeaway"]
    assert "candidate risk signal" in by_id["QCV2"]["current_base_update"]


def test_archive_inventory_marks_core_outputs_and_review_candidates(tmp_path: Path):
    (tmp_path / "run_current_base_submission_docs_pack.py").write_text("", encoding="utf-8")
    (tmp_path / "diagnose_old_probe.py").write_text("", encoding="utf-8")
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "current_base_submission_docs_pack_20260708").mkdir()
    (tmp_path / "results" / "old_probe_result").mkdir()

    rows = build_archive_inventory(tmp_path)
    by_path = {row["path"]: row for row in rows}

    assert by_path["run_current_base_submission_docs_pack.py"]["status"] == "keep_current_core"
    assert by_path["results/current_base_submission_docs_pack_20260708"]["status"] == "keep_current_core"
    assert by_path["diagnose_old_probe.py"]["status"] == "archive_candidate_unreviewed"
    assert by_path["results/old_probe_result"]["archive_bucket"] == "_local_archive_20260708/results_exploratory"


def test_archive_summary_counts_statuses():
    rows = [
        {"status": "keep_current_core"},
        {"status": "keep_current_core"},
        {"status": "archive_candidate_unreviewed"},
    ]

    summary = summarize_archive_inventory(rows)

    assert summary == [
        {"status": "archive_candidate_unreviewed", "count": 1},
        {"status": "keep_current_core", "count": 2},
    ]


def test_docs_are_written_with_expected_sections(tmp_path: Path):
    method_path = tmp_path / "method.zh.md"
    system_path = tmp_path / "system.zh.md"
    archive_path = tmp_path / "archive.zh.md"
    sections = build_method_chapter_sections(sample_headline_rows(), sample_modules(), sample_params(), sample_sequence_rows())
    doc_index = build_document_index(
        freeze_dir=Path("results/current_base_evidence_freeze_20260708"),
        method_dir=Path("results/current_base_method_story_20260708"),
        teacher_dir=Path("results/current_base_teacher_briefing_pack_20260708"),
        output_dir=tmp_path,
    )
    archive_rows = [
        {"path": "run_current_base_submission_docs_pack.py", "kind": "script", "status": "keep_current_core", "archive_bucket": "", "reason": "active", "recommended_action": "keep"},
        {"path": "diagnose_old_probe.py", "kind": "script", "status": "archive_candidate_unreviewed", "archive_bucket": "_local_archive_20260708/scripts_exploratory", "reason": "old", "recommended_action": "review_then_move"},
    ]

    write_method_chapter(
        method_path,
        sections=sections,
        modules=sample_modules(),
        params=sample_params(),
        sequence_rows=sample_sequence_rows(),
        source_rows=build_source_document_synthesis(),
    )
    write_system_index(
        system_path,
        document_rows=doc_index,
        commands=sample_commands(),
        params=sample_params(),
        sequence_rows=sample_sequence_rows(),
        source_rows=build_source_document_synthesis(),
    )
    write_archive_plan(archive_path, archive_rows=archive_rows, summary_rows=summarize_archive_inventory(archive_rows))

    assert "论文方法章节草稿" in method_path.read_text(encoding="utf-8")
    assert "g_i = I" in method_path.read_text(encoding="utf-8")
    assert "S_{AR}" in method_path.read_text(encoding="utf-8")
    assert "S_{CC}" in method_path.read_text(encoding="utf-8")
    assert "CLS-FlowerExpB-TimeAware2080" in method_path.read_text(encoding="utf-8")
    assert "系统文档目录化索引" in system_path.read_text(encoding="utf-8")
    assert "T1-T7 / F1-F5" in system_path.read_text(encoding="utf-8")
    assert "source document synthesis" in system_path.read_text(encoding="utf-8")
    assert "中间文件归档清单" in archive_path.read_text(encoding="utf-8")
    assert "只移动、不删除" in archive_path.read_text(encoding="utf-8")
