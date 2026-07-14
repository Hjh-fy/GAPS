from __future__ import annotations

import argparse
import copy
import subprocess
import sys
from pathlib import Path

import pytest

import scripts.evaluate_iotj_high_coverage_qc as qc
from scripts.evaluate_iotj_high_coverage_qc import (
    attach_oracle_prediction,
    apply_workpoint,
    evaluate_workpoint,
    fit_component_calibrator,
    fit_feature_reference,
    fit_workpoints,
    merge_aligned_streams,
    raw_deployment_risk,
    score_deployment_rows,
    select_score_family,
    evaluate_ranking_curve,
)


def calibration_row(
    sample_index: int,
    true_class: int,
    phase: int,
    features: tuple[float, float],
) -> dict[str, object]:
    return {
        "split": "calibration",
        "sample_index": sample_index,
        "true_class": true_class,
        "phase": phase,
        "cls_feat_000": features[0],
        "cls_feat_001": features[1],
    }


def deployment_row(sample_index: int, feature: tuple[float, float]) -> dict[str, object]:
    return {
        "split": "test",
        "sample_index": sample_index,
        "pred_class": 0,
        "route_class": 0,
        "phase": 1,
        "deployment_risk_classifier_entropy": 0.2,
        "deployment_risk_margin": 0.1,
        "h23_plus_ppm": 50.0,
        "target_ridge_plus_source_preds_ppm": 60.0,
        "H1_source_ridge_ppm": 52.0,
        "H2_source_per_gas_mlp_ppm": 58.0,
        "H3_source_shared_mlp_ppm": 55.0,
        "cls_feat_000": feature[0],
        "cls_feat_001": feature[1],
        "true_class": 0,
        "true_ppm": 50.0,
        "class_correct": 1,
        "route_correct": 1,
    }


def test_raw_risk_is_invariant_to_all_test_truth_fields() -> None:
    reference = fit_feature_reference(
        [
            calibration_row(0, 0, 1, (0.0, 0.0)),
            calibration_row(1, 0, 1, (0.2, 0.1)),
            calibration_row(2, 1, 1, (5.0, 5.0)),
            calibration_row(3, 1, 1, (5.2, 5.1)),
        ]
    )
    first = deployment_row(10, (0.1, 0.1))
    second = copy.deepcopy(first)
    second.update(
        {
            "true_class": 3,
            "true_ppm": 250.0,
            "class_correct": 0,
            "route_correct": 0,
            "abs_error_ppm": 999.0,
        }
    )

    assert raw_deployment_risk(first, reference) == raw_deployment_risk(second, reference)


def test_component_calibration_and_full_score_use_validation_only() -> None:
    raw_rows = [
        {
            "sample_index": index,
            "raw_risk_confidence": value,
            "raw_risk_prototype": value * 2.0,
            "raw_risk_support": value * 3.0,
            "raw_risk_expert_disagreement": value * 4.0,
            "raw_risk_source_spread": value * 5.0,
        }
        for index, value in enumerate((0.1, 0.2, 0.3, 0.4))
    ]
    calibrator = fit_component_calibrator(raw_rows)

    scored = score_deployment_rows(raw_rows, calibrator)

    assert scored[0]["deployment_risk_full"] < scored[-1]["deployment_risk_full"]
    assert scored[-1]["deployment_risk_full"] == 1.0
    assert calibrator["selection_split"] == "calibration_validation"

    mixed = dict(raw_rows[0])
    mixed["raw_risk_confidence"] = raw_rows[-1]["raw_risk_confidence"]
    mixed_score = score_deployment_rows([mixed], calibrator)[0]
    assert mixed_score["deployment_risk_confidence"] == 1.0
    assert mixed_score["deployment_risk_full"] < 1.0


def test_hc95_and_hc90_decisions_do_not_change_when_truth_changes() -> None:
    validation = [
        {"sample_index": index, "deployment_risk_full": float(index) / 99.0}
        for index in range(100)
    ]
    policy = fit_workpoints(validation, "deployment_risk_full")
    row = {
        "sample_index": 1000,
        "deployment_risk_full": 0.97,
        "true_class": 0,
        "true_ppm": 25.0,
        "class_correct": 1,
    }
    changed = dict(row)
    changed.update({"true_class": 3, "true_ppm": 250.0, "class_correct": 0})

    first = apply_workpoint([row], policy, "HC95")[0]
    second = apply_workpoint([changed], policy, "HC95")[0]

    assert first["qc_decision"] == second["qc_decision"] == "review"
    assert policy["workpoints"]["HC95"]["target_accept_coverage"] == 0.95
    assert policy["workpoints"]["HC90"]["target_accept_coverage"] == 0.90
    assert policy["workpoints"]["HC95"]["reject_threshold"] < 1.0
    assert policy["workpoints"]["HC90"]["reject_threshold"] < policy["workpoints"]["HC95"]["reject_threshold"]
    assert policy["workpoints"]["FULL"]["accept_threshold"] is None
    assert apply_workpoint([row], policy, "FULL")[0]["qc_decision"] == "accept"
    assert policy["selection_split"] == "calibration_validation"


def test_merge_aligned_streams_requires_every_h23_row_in_all_inputs() -> None:
    base = [
        {"split": "test", "sample_index": 0, "true_class": 0, "pred_class": 0},
        {"split": "test", "sample_index": 1, "true_class": 1, "pred_class": 1},
    ]
    h23 = [
        {"split": "test", "sample_index": 0, "h23_plus_ppm": 10.0},
        {"split": "test", "sample_index": 1, "h23_plus_ppm": 20.0},
    ]
    h8 = [
        {"split": "test", "sample_index": 0, "target_ridge_plus_source_preds_ppm": 11.0},
        {"split": "test", "sample_index": 1, "target_ridge_plus_source_preds_ppm": 21.0},
    ]
    features = [
        {"split": "test", "sample_index": 0, "cls_feat_000": 0.0},
        {"split": "test", "sample_index": 1, "cls_feat_000": 1.0},
    ]

    merged = merge_aligned_streams(base, h23, h8, features, split="test")

    assert len(merged) == 2
    assert merged[1]["h23_plus_ppm"] == 20.0
    assert merged[1]["target_ridge_plus_source_preds_ppm"] == 21.0
    assert merged[1]["cls_feat_000"] == 1.0

    try:
        merge_aligned_streams(base, h23, h8[:1], features, split="test")
    except ValueError as error:
        assert "missing H8 row" in str(error)
    else:
        raise AssertionError("missing aligned H8 row was accepted")


def test_workpoint_evaluation_reports_yield_error_capture_and_random_control() -> None:
    rows = [
        {"qc_decision": "accept", "true_class": 0, "pred_class": 0, "true_ppm": 10.0, "pred_ppm": 11.0},
        {"qc_decision": "accept", "true_class": 1, "pred_class": 1, "true_ppm": 100.0, "pred_ppm": 102.0},
        {"qc_decision": "review", "true_class": 2, "pred_class": 3, "true_ppm": 50.0, "pred_ppm": 100.0},
        {"qc_decision": "reject", "true_class": 3, "pred_class": 3, "true_ppm": 200.0, "pred_ppm": 250.0},
    ]

    report = evaluate_workpoint(rows, "pred_ppm", n_random=50, seed=42)

    assert report["automatic_yield"] == 0.5
    assert report["nonreject_coverage"] == 0.75
    assert report["route_wrong_total"] == 1
    assert report["route_wrong_flagged"] == 1
    assert report["route_wrong_recall"] == 1.0
    assert report["high_error_total"] == 2
    assert report["high_error_flagged"] == 2
    assert report["accept_metrics"]["RMSE"] < report["full_metrics"]["RMSE"]
    assert report["random_control"]["iterations"] == 50


def test_workpoint_reports_nonreject_and_oracle_metrics() -> None:
    rows = [
        {"client": "C5", "split": "test", "sample_index": 0, "true_class": 0, "pred_class": 1,
         "true_ppm": 10.0, "pred_ppm": 30.0, "oracle_ppm": 12.0, "qc_decision": "accept"},
        {"client": "C5", "split": "test", "sample_index": 1, "true_class": 0, "pred_class": 0,
         "true_ppm": 20.0, "pred_ppm": 24.0, "oracle_ppm": 21.0, "qc_decision": "review"},
        {"client": "C5", "split": "test", "sample_index": 2, "true_class": 0, "pred_class": 0,
         "true_ppm": 30.0, "pred_ppm": 90.0, "oracle_ppm": 31.0, "qc_decision": "reject"},
    ]

    report = evaluate_workpoint(rows, "pred_ppm", oracle_pred_key="oracle_ppm", n_random=0)

    assert report["nonreject_N"] == 2
    assert report["nonreject_metrics"]["RMSE"] == pytest.approx((416.0 / 2.0) ** 0.5)
    assert report["oracle_accept_metrics"]["RMSE"] == pytest.approx(2.0)
    assert report["oracle_nonreject_metrics"]["RMSE"] == pytest.approx((5.0 / 2.0) ** 0.5)


def test_workpoint_rejects_invalid_qc_decision_before_metrics() -> None:
    rows = [
        {"qc_decision": "hold", "true_class": 0, "pred_class": 0, "true_ppm": 10.0, "pred_ppm": 10.0},
    ]

    with pytest.raises(ValueError, match="invalid qc_decision"):
        evaluate_workpoint(rows, "pred_ppm", n_random=0)


def test_attach_oracle_prediction_keys_by_client_split_and_sample_index() -> None:
    rows = [
        {"client": "C5", "split": "test", "sample_index": 7},
        {"client": "C6", "split": "test", "sample_index": 7},
    ]
    oracle_rows = [
        {"client": "C5", "split": "test", "sample_index": 7, "oracle_ppm": 11.0},
        {"client": "C6", "split": "test", "sample_index": 7, "oracle_ppm": 22.0},
    ]

    attached = attach_oracle_prediction(rows, oracle_rows, "oracle_ppm", "attached_oracle_ppm")

    assert [row["attached_oracle_ppm"] for row in attached] == [11.0, 22.0]


def test_attach_oracle_prediction_requires_client_coordinates() -> None:
    rows = [{"client": None, "split": "test", "sample_index": 7}]
    oracle_rows = [
        {"client": None, "split": "test", "sample_index": 7, "oracle_ppm": 11.0}
    ]

    with pytest.raises(ValueError, match="oracle row key requires client"):
        attach_oracle_prediction(rows, oracle_rows, "oracle_ppm", "attached_oracle_ppm")


@pytest.mark.parametrize(
    ("oracle_rows", "message"),
    [
        (
            [
                {"client": "C5", "split": "test", "sample_index": 7, "oracle_ppm": 11.0},
                {"client": "C5", "split": "test", "sample_index": 7, "oracle_ppm": 12.0},
            ],
            "duplicate oracle row",
        ),
        (
            [{"client": "C6", "split": "test", "sample_index": 7, "oracle_ppm": 11.0}],
            "missing oracle prediction",
        ),
        (
            [{"client": "C5", "split": "test", "sample_index": 7, "oracle_ppm": float("nan")}],
            "non-finite oracle prediction",
        ),
    ],
)
def test_attach_oracle_prediction_rejects_invalid_oracle_rows(
    oracle_rows: list[dict[str, object]], message: str
) -> None:
    rows = [{"client": "C5", "split": "test", "sample_index": 7}]

    with pytest.raises(ValueError, match=message):
        attach_oracle_prediction(rows, oracle_rows, "oracle_ppm", "attached_oracle_ppm")


def test_qc_run_consumes_exact_upstream_oracle_ppm_column(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    oracle_key = "target_ridge_plus_source_preds_oracle_route_ppm"
    validation = [{"client": "C5", "split": "calibration", "sample_index": index} for index in range(80)]
    test = [{"client": "C5", "split": "test", "sample_index": index} for index in range(1360)]
    oracle = [
        {"client": "C5", "split": "test", "sample_index": index, oracle_key: float(index)}
        for index in range(1360)
    ]
    args = argparse.Namespace(
        output_dir=tmp_path,
        target_inputs="target_inputs",
        h23_validation="h23_validation",
        h23_test="h23_test",
        h8_validation="h8_validation",
        h8_test="h8_test",
        h8_test_oracle="h8_test_oracle",
        backbone_calibration="backbone_calibration",
        backbone_test="backbone_test",
        pred_key="pred_ppm",
        n_random=0,
        seed=42,
    )

    monkeypatch.setattr(qc, "_read_csv", lambda path: oracle if path == "h8_test_oracle" else [])
    monkeypatch.setattr(qc, "_attach_calibration_features", lambda *_: [{}] * 240)
    monkeypatch.setattr(
        qc,
        "merge_aligned_streams",
        lambda *_args, split: validation if split == "calibration" else test,
    )
    monkeypatch.setattr(qc, "fit_feature_reference", lambda _: {})
    monkeypatch.setattr(qc, "raw_deployment_risk", lambda *_: {})
    monkeypatch.setattr(qc, "fit_component_calibrator", lambda _: {})
    monkeypatch.setattr(qc, "score_deployment_rows", lambda rows, _: list(rows))
    monkeypatch.setattr(
        qc,
        "select_score_family",
        lambda *_: {"selected_policy": {}, "selected_score": "deployment_risk_full"},
    )
    monkeypatch.setattr(qc, "fit_workpoints", lambda *_: {})
    monkeypatch.setattr(
        qc,
        "apply_workpoint",
        lambda rows, _policy, workpoint: [
            {**row, "qc_workpoint": workpoint, "qc_decision": "accept"} for row in rows
        ],
    )
    monkeypatch.setattr(
        qc,
        "evaluate_workpoint",
        lambda *_args, **_kwargs: {
            "accept_N": 1360,
            "nonreject_N": 1360,
            "accept_metrics": {},
            "nonreject_metrics": {},
            "oracle_accept_metrics": {},
            "oracle_nonreject_metrics": {},
        },
    )
    monkeypatch.setattr(qc, "evaluate_ranking_curve", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(qc, "_write_csv", lambda *_: None)
    monkeypatch.setattr(qc, "_write_json", lambda *_: None)

    result = qc.run_high_coverage_qc(args)

    assert result["manifest"]["oracle_pred_key"] == oracle_key


def test_score_family_selection_uses_calibration_error_capture() -> None:
    rows = []
    for index in range(20):
        is_bad = index == 19
        rows.append(
            {
                "sample_index": index,
                "true_class": 0,
                "pred_class": 1 if is_bad else 0,
                "true_ppm": 50.0,
                "pred_ppm": 150.0 if is_bad else 50.0,
                "deployment_risk_confidence": 1.0 if index == 0 else index / 100.0,
                "deployment_risk_feature": index / 100.0,
                "deployment_risk_disagreement": index / 19.0,
                "deployment_risk_full": index / 19.0,
            }
        )

    selection = select_score_family(rows, "pred_ppm")

    assert selection["selected_score"] in {
        "deployment_risk_disagreement",
        "deployment_risk_full",
    }
    assert selection["selection_split"] == "calibration_validation"
    assert selection["selected_report"]["route_wrong_recall"] == 1.0


def test_fixed_coverage_curve_is_labeled_non_operational() -> None:
    rows = [
        {
            "sample_index": index,
            "true_class": 0,
            "pred_class": 0,
            "true_ppm": 50.0,
            "pred_ppm": 50.0 + index,
            "deployment_risk_full": index / 9.0,
        }
        for index in range(10)
    ]

    curve = evaluate_ranking_curve(
        rows,
        "deployment_risk_full",
        "pred_ppm",
        coverages=(1.0, 0.9),
        n_random=10,
    )

    assert curve[0]["target_coverage"] == 1.0
    assert curve[0]["report"]["automatic_yield"] == 1.0
    assert curve[1]["target_coverage"] == 0.9
    assert curve[1]["report"]["accept_N"] == 9
    assert all(row["operational_threshold"] is False for row in curve)


def test_high_coverage_qc_script_is_directly_executable() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "scripts/evaluate_iotj_high_coverage_qc.py", "--help"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "high-coverage C5 QC" in result.stdout
    assert "--h8-test-oracle" in result.stdout
