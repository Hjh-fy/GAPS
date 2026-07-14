from __future__ import annotations

import numpy as np
import pytest
import subprocess
import sys
import torch
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

from gaps_flower.evaluate_regression_pipeline import (
    collect_records,
    deployment_risk_components,
)
from scripts.run_iotj_c5_regression_suite import build_suite_commands, expected_outputs
from scripts.run_iotj_c5_regression_cloud import (
    ClassifierSpec,
    build_remote_suite_command,
    merge_cloud_manifest,
    parse_classifier_spec,
)
from run_source_augmented_target_ridge_eval import (
    apply_oracle_h8_predictions,
    attach_prediction_column,
    force_oracle_routes,
)
from scripts.summarize_iotj_c5_formal_regression import (
    _report,
    collect_qc_oracle_sources,
    flatten_operational_qc,
    validate_ladder_summary,
)
from scripts.assemble_iotj_c5_regression_ladder import (
    apply_ladder,
    select_r6_policy,
    summarize_ladder,
)


def test_deployment_risk_uses_predicted_route_range_only() -> None:
    risks = deployment_risk_components(
        probabilities=np.asarray([0.1, 0.7, 0.1, 0.1], dtype=np.float64),
        margin=0.6,
        route_class=1,
        base_raw_ppm=25.0,
        routed_ppm=70.0,
        calibrated_ppm=75.0,
    )

    assert risks["deployment_route_range_ppm"] == 225.0
    assert risks["deployment_risk_route_gap"] == pytest.approx(45.0 / 225.0)
    assert risks["deployment_risk_response_gap"] == pytest.approx(5.0 / 225.0)
    assert risks["deployment_risk_margin"] == pytest.approx(0.4)
    assert 0.0 <= risks["deployment_risk_classifier_entropy"] <= 1.0
    assert risks["deployment_risk_composite"] >= risks["deployment_risk_margin"]


def test_collected_record_separates_evaluation_and_deployment_ranges() -> None:
    class Classifier(torch.nn.Module):
        def forward(self, x: torch.Tensor):
            logits = torch.tensor([[0.0, 8.0, 0.0, 0.0]], dtype=x.dtype).repeat(
                x.size(0), 1
            )
            features = torch.zeros(x.size(0), 2, dtype=x.dtype)
            return logits, features, features

    class Regressor(torch.nn.Module):
        def forward(self, x: torch.Tensor):
            features = torch.zeros(x.size(0), 2, dtype=x.dtype)
            return torch.zeros(x.size(0), 4), features, features

        def forward_reg(self, features, y_cls, y_phase):
            return torch.full((features.size(0), 1), 0.5, dtype=features.dtype)

    loader = DataLoader(
        TensorDataset(
            torch.zeros(1, 2),
            torch.tensor([0]),
            torch.tensor([[50.0, 100.0, 50.0, 100.0]]),
            torch.tensor([1]),
            torch.tensor([5]),
            torch.tensor([7]),
        ),
        batch_size=1,
    )

    records = collect_records(
        Classifier(),
        Regressor(),
        None,
        {},
        loader,
        torch.device("cpu"),
        "predicted",
        {},
    )

    assert records[0]["true_class"] == 0
    assert records[0]["route_cls"] == 1
    assert records[0]["range_ppm"] == 112.5
    assert records[0]["deployment_route_range_ppm"] == 225.0


def test_regression_suite_commands_run_inputs_h23_h8_and_qc_in_order(tmp_path: Path) -> None:
    commands = build_suite_commands(
        classifier_checkpoint=Path("results/classifier.pth"),
        regression_checkpoint=Path("results/source_regression.pt"),
        data_root=Path("dataset/c12_c5"),
        output_root=tmp_path / "suite",
        device="cpu",
        seed=42,
        n_random=1000,
    )

    assert len(commands) == 5
    assert commands[0][1].endswith("build_iotj_c5_regression_inputs.py")
    assert commands[1][1].endswith("run_iotj_c5_h23_plus.py")
    assert commands[2][1].endswith("run_source_augmented_target_ridge_eval.py")
    assert "--disable-c4-rescue" in commands[2]
    assert commands[3][1].endswith("evaluate_iotj_high_coverage_qc.py")
    oracle_h8_test = commands[3][commands[3].index("--h8-test-oracle") + 1]
    assert Path(oracle_h8_test).as_posix().endswith(
        "h8_no_rescue/target_predictions_plus_source_preds_oracle_route.csv"
    )
    assert commands[3][commands[3].index("--n-random") + 1] == "1000"
    assert commands[4][1].endswith("assemble_iotj_c5_regression_ladder.py")
    risk_selection = Path(commands[4][commands[4].index("--risk-selection") + 1])
    assert risk_selection.as_posix().endswith("high_coverage_qc/risk_selection.json")
    outputs = expected_outputs(tmp_path / "suite")
    assert outputs[-2].as_posix().endswith("r0_r7/r0_r7_summary.csv")
    assert outputs[-1].as_posix().endswith("r0_r7/manifest.json")


def _ladder_row(
    index: int,
    *,
    split: str,
    true_class: int,
    pred_class: int,
    true_ppm: float,
    h23: float,
    h8: float,
    risk: float,
) -> dict[str, object]:
    return {
        "client": "C5",
        "split": split,
        "sample_index": index,
        "true_class": true_class,
        "pred_class": pred_class,
        "route_class": pred_class,
        "true_ppm": true_ppm,
        "baseline_final_ppm": true_ppm + 20.0,
        "target_ridge_rich_only_ppm": true_ppm + 10.0,
        "h23_anchor_ppm": true_ppm + 8.0,
        "h23_plus_ppm": h23,
        "target_ridge_plus_source_preds_ppm": h8,
        "deployment_risk_full": risk,
    }


def test_r0_r7_ladder_uses_validation_only_selector_and_labels_oracle() -> None:
    validation = [
        _ladder_row(0, split="calibration", true_class=1, pred_class=1, true_ppm=100, h23=101, h8=120, risk=0.1),
        _ladder_row(1, split="calibration", true_class=1, pred_class=1, true_ppm=100, h23=125, h8=101, risk=0.9),
        _ladder_row(2, split="calibration", true_class=0, pred_class=0, true_ppm=50, h23=51, h8=70, risk=0.8),
        _ladder_row(3, split="calibration", true_class=2, pred_class=2, true_ppm=50, h23=52, h8=60, risk=0.2),
    ]
    policy = select_r6_policy(validation, "deployment_risk_full")
    assert policy["selection_split"] == "calibration_validation"
    assert policy["selection_uses_test_labels"] is False
    assert 0.1 < float(policy["threshold"]) <= 0.9

    test = [
        _ladder_row(10, split="test", true_class=1, pred_class=1, true_ppm=100, h23=130, h8=102, risk=0.95),
        _ladder_row(11, split="test", true_class=0, pred_class=0, true_ppm=50, h23=51, h8=80, risk=0.99),
        _ladder_row(12, split="test", true_class=2, pred_class=1, true_ppm=50, h23=52, h8=90, risk=0.05),
    ]
    ladder = apply_ladder(test, policy)

    assert ladder[0]["R5_ppm"] == 102
    assert ladder[0]["R6_ppm"] == 102
    assert ladder[1]["R5_ppm"] == 51
    assert ladder[1]["R6_ppm"] == 51
    assert ladder[2]["R6_ppm"] == 52
    assert ladder[2]["R7_ppm"] == 52
    assert ladder[2]["R7_uses_test_truth"] == 1

    changed_truth = [dict(row) for row in test]
    changed_truth[0]["true_ppm"] = 999.0
    changed_truth[0]["true_class"] = 3
    changed = apply_ladder(changed_truth, policy)
    assert changed[0]["R6_ppm"] == ladder[0]["R6_ppm"]

    summary = summarize_ladder(ladder)
    assert {row["mode"] for row in summary} == {f"R{i}" for i in range(8)}
    assert {row["scope"] for row in summary} >= {"S_ALL", "S_CC", "S_CW", "gas_1"}
    assert all("coverage" in row for row in summary)


def test_h8_augmented_stream_receives_aligned_rich_only_prediction() -> None:
    augmented = [
        {"client": "C5", "split": "test", "sample_index": 1, "target_ridge_plus_source_preds_ppm": 20.0},
        {"client": "C5", "split": "test", "sample_index": 2, "target_ridge_plus_source_preds_ppm": 30.0},
    ]
    rich = [
        {"client": "C5", "split": "test", "sample_index": 2, "target_ridge_rich_only_ppm": 31.0},
        {"client": "C5", "split": "test", "sample_index": 1, "target_ridge_rich_only_ppm": 21.0},
    ]

    merged = attach_prediction_column(
        augmented,
        rich,
        "target_ridge_rich_only_ppm",
    )

    assert [row["target_ridge_rich_only_ppm"] for row in merged] == [21.0, 31.0]


def test_force_oracle_routes_copies_rows_and_updates_route_metadata() -> None:
    source = [
        {
            "client": "C5",
            "sample_index": 7,
            "pred_class": 1,
            "true_class": 3,
            "route_class": 1,
            "route_cls": 1,
            "route_gas": "CO",
            "route_correct": 0,
            "route_source": "predicted",
        }
    ]
    result = force_oracle_routes(source)
    assert result[0]["route_class"] == 3
    assert result[0]["route_cls"] == 3
    assert result[0]["route_gas"] == "Methane"
    assert result[0]["route_correct"] == 1
    assert result[0]["route_source"] == "oracle_true_class"
    assert result[0]["actual_route_class"] == 1
    assert result[0]["actual_route_gas"] == "CO"
    assert result[0]["pred_class"] == 1
    assert source[0]["route_class"] == 1


class _ConstantRidge:
    def __init__(self, value: float) -> None:
        self.value = value

    def predict(self, rows, clip=True):
        return np.full(len(rows), self.value, dtype=np.float64)


class _ConstantMlp:
    def __init__(self, value: float) -> None:
        self.value = value

    def predict(self, rows):
        return np.full(len(rows), self.value, dtype=np.float64)


class _RouteAwareSharedMlp:
    def predict(self, rows):
        return np.asarray(
            [3000.0 + float(row["route_class"]) for row in rows], dtype=np.float64
        )


def test_apply_oracle_h8_predictions_routes_every_head_by_true_class() -> None:
    rows = [
        {
            "client": "C5",
            "sample_index": 7,
            "pred_class": 1,
            "true_class": 3,
            "route_class": 1,
            "route_cls": 1,
            "route_gas": "CO",
            "route_correct": 0,
            "route_source": "predicted",
            "feature_dict": {"feature": 1.0},
            "final_ppm": 50.0,
        }
    ]
    ridge_models = {class_id: _ConstantRidge(1000.0 + class_id) for class_id in range(4)}
    mlp_models = {class_id: _ConstantMlp(2000.0 + class_id) for class_id in range(4)}
    target_models = {
        ("C5", class_id): _ConstantRidge(4000.0 + class_id)
        for class_id in range(4)
    }

    result = apply_oracle_h8_predictions(
        rows,
        ridge_models,
        mlp_models,
        _RouteAwareSharedMlp(),
        target_models,
    )

    assert result[0]["H1_source_ridge_ppm"] == 1003.0
    assert result[0]["H2_source_per_gas_mlp_ppm"] == 2003.0
    assert result[0]["H3_source_shared_mlp_ppm"] == 3003.0
    assert result[0]["target_ridge_plus_source_preds_oracle_route_ppm"] == 4003.0


def test_ladder_fails_closed_when_a_required_base_prediction_is_missing() -> None:
    row = _ladder_row(
        20,
        split="test",
        true_class=0,
        pred_class=0,
        true_ppm=50,
        h23=51,
        h8=52,
        risk=0.2,
    )
    del row["target_ridge_rich_only_ppm"]
    policy = {
        "score_key": "deployment_risk_full",
        "threshold": 0.5,
    }

    with pytest.raises(ValueError, match="R1"):
        apply_ladder([row], policy)


def test_cloud_runner_builds_isolated_remote_suite_commands() -> None:
    parsed = parse_classifier_spec("A6=results/a6/server_latest_adapted.pth")
    assert parsed == ClassifierSpec(
        classifier_id="A6",
        local_checkpoint=Path("results/a6/server_latest_adapted.pth"),
    )
    command = build_remote_suite_command(
        classifier_id="A6",
        remote_checkpoint=Path("/root/GAPS/results/iotj_reg_checkpoints/A6.pth"),
        remote_regression_checkpoint=Path("/root/GAPS/results/iotj_reg_checkpoints/R3aK16.pt"),
        remote_data_root=Path("/root/GAPS/dataset/c12_c5"),
        remote_output_root=Path("/root/GAPS/results/iotj_formal_regression/A6"),
        python_bin="/root/gaps_env/bin/python",
        device="cuda",
        seed=42,
        n_random=1000,
    )

    assert command[:2] == ["/root/gaps_env/bin/python", "scripts/run_iotj_c5_regression_suite.py"]
    assert command[command.index("--classifier-id") + 1] == "A6"
    assert command[command.index("--device") + 1] == "cuda"
    assert command[command.index("--n-random") + 1] == "1000"
    assert command[command.index("--output-root") + 1].endswith("iotj_formal_regression/A6")


def test_cloud_classifier_spec_rejects_ambiguous_or_duplicate_ids() -> None:
    with pytest.raises(ValueError, match="ID=PATH"):
        parse_classifier_spec("A6")
    with pytest.raises(ValueError, match="classifier ID"):
        parse_classifier_spec("=results/a6.pth")


def test_cloud_runner_is_directly_executable() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "scripts/run_iotj_c5_regression_cloud.py", "--help"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--classifier" in result.stdout


def test_cloud_manifest_append_preserves_prior_classifier_runs() -> None:
    existing = {
        "schema_version": 1,
        "training_location": "Alibaba Cloud ECS",
        "classifiers": ["A6", "B5"],
        "seed": 42,
        "n_random": 1000,
        "device": "cpu",
        "remote_output_base": "/root/GAPS/results/formal",
        "recovered": {"A6": "results/formal/A6", "B5": "results/formal/B5"},
    }
    update = {
        **existing,
        "classifiers": ["B2"],
        "recovered": {"B2": "results/formal/B2"},
    }

    merged = merge_cloud_manifest(existing, update)

    assert merged["classifiers"] == ["A6", "B5", "B2"]
    assert set(merged["recovered"]) == {"A6", "B5", "B2"}
    incompatible = dict(update, seed=43)
    with pytest.raises(ValueError, match="seed"):
        merge_cloud_manifest(existing, incompatible)


def test_formal_summary_validator_requires_complete_r0_r7_contract() -> None:
    rows = []
    for mode_index in range(8):
        for scope, n in (("S_ALL", 1360), ("S_CC", 1345), ("S_CW", 15), ("gas_0", 340), ("gas_1", 340), ("gas_2", 340), ("gas_3", 340)):
            rows.append(
                {
                    "mode": f"R{mode_index}",
                    "scope": scope,
                    "N": str(n),
                    "RMSE": "10.0",
                    "NRMSE": "0.1",
                    "MAE": "5.0",
                    "P90AE": "8.0",
                    "uses_test_truth_at_runtime": str(int(mode_index == 7)),
                }
            )

    validate_ladder_summary(rows, classifier_id="B5")
    rows[-1]["RMSE"] = ""
    with pytest.raises(ValueError, match="empty RMSE"):
        validate_ladder_summary(rows, classifier_id="B5")


def test_formal_qc_flattening_preserves_realized_coverage_and_random_control() -> None:
    workpoint = {
            "N": 1360,
            "accept_N": 1309,
            "nonreject_N": 1342,
            "review_N": 33,
            "reject_N": 18,
            "automatic_yield": 1309 / 1360,
            "nonreject_coverage": 1342 / 1360,
            "route_wrong_recall": 7 / 15,
            "high_error_recall": 0.2,
            "class_correct_false_flag_rate": 0.03,
            "full_metrics": {"N": 1360, "RMSE": 17.4},
            "accept_metrics": {"N": 1309, "RMSE": 15.9, "NRMSE": 0.12, "MAE": 7.2, "P90AE": 16.4},
            "nonreject_metrics": {"N": 1342, "RMSE": 16.2, "NRMSE": 0.13, "MAE": 7.5, "P90AE": 16.8},
            "review_metrics": {"N": 33, "RMSE": 20.0},
            "reject_metrics": {"N": 18, "RMSE": 30.0},
            "oracle_accept_metrics": {"N": 1309, "RMSE": 10.9, "NRMSE": 0.08, "MAE": 5.2, "P90AE": 11.4},
            "oracle_nonreject_metrics": {"N": 1342, "RMSE": 11.2, "NRMSE": 0.09, "MAE": 5.5, "P90AE": 11.8},
            "random_control": {
                "accept_RMSE": {"mean": 17.4},
                "route_wrong_recall": {"mean": 0.04},
                "high_error_recall": {"mean": 0.04},
            },
        }
    operational = {
        "FULL": dict(workpoint),
        "HC95": dict(workpoint),
        "HC90": dict(workpoint),
    }

    rows = flatten_operational_qc("B5", operational)
    hc95 = next(row for row in rows if row["workpoint"] == "HC95")

    assert len(rows) == 3
    assert hc95["classifier_id"] == "B5"
    assert hc95["automatic_yield"] == pytest.approx(1309 / 1360)
    assert hc95["random_accept_RMSE_mean"] == 17.4
    assert hc95["nonreject_N"] == 1342
    assert hc95["nonreject_RMSE"] == 16.2
    assert hc95["nonreject_NRMSE"] == 0.13
    assert hc95["oracle_accept_RMSE"] == 10.9
    assert hc95["oracle_accept_NRMSE"] == 0.08
    assert hc95["oracle_nonreject_RMSE"] == 11.2
    assert hc95["oracle_nonreject_NRMSE"] == 0.09

    report = _report([], rows)
    assert "Actual Accepted RMSE" in report
    assert "Actual Nonreject NRMSE" in report
    assert "Oracle Accepted RMSE" in report
    assert "Oracle Nonreject NRMSE" in report
    assert "forced-true-class routing diagnostic under frozen QC masks" in report


def test_formal_qc_flattening_requires_all_workpoints() -> None:
    with pytest.raises(ValueError, match="missing QC workpoints"):
        flatten_operational_qc("B5", {"HC95": {}})


def test_formal_qc_flattening_rejects_extra_workpoints() -> None:
    operational = {name: {} for name in ("FULL", "HC95", "HC90", "HC80")}
    with pytest.raises(ValueError, match="unexpected QC workpoints"):
        flatten_operational_qc("B5", operational)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda item: item.update(N=7), "expected N=1360"),
        (lambda item: item.update(reject_N=19), "decision counts do not sum"),
        (lambda item: item.update(nonreject_N=1300), "nonreject_N"),
        (
            lambda item: item["accept_metrics"].update(N=1308),
            "accept_metrics.N",
        ),
        (lambda item: item.update(automatic_yield=0.5), "automatic_yield"),
    ],
)
def test_formal_qc_flattening_validates_counts_and_metric_cardinality(
    mutation, message: str
) -> None:
    base = {
        "N": 1360,
        "accept_N": 1309,
        "review_N": 33,
        "reject_N": 18,
        "nonreject_N": 1342,
        "automatic_yield": 1309 / 1360,
        "nonreject_coverage": 1342 / 1360,
        "route_wrong_recall": 0.0,
        "high_error_recall": 0.0,
        "class_correct_false_flag_rate": 0.0,
        "full_metrics": {"N": 1360},
        "accept_metrics": {"N": 1309, "RMSE": 1.0, "NRMSE": 0.1, "MAE": 1.0, "P90AE": 1.0},
        "nonreject_metrics": {"N": 1342, "RMSE": 1.0, "NRMSE": 0.1},
        "review_metrics": {"N": 33},
        "reject_metrics": {"N": 18},
        "oracle_accept_metrics": {"N": 1309, "RMSE": 1.0, "NRMSE": 0.1},
        "oracle_nonreject_metrics": {"N": 1342, "RMSE": 1.0, "NRMSE": 0.1},
        "random_control": {
            "accept_RMSE": {"mean": 1.0},
            "route_wrong_recall": {"mean": 0.0},
            "high_error_recall": {"mean": 0.0},
        },
    }
    mutation(base)
    operational = {name: base for name in ("FULL", "HC95", "HC90")}

    with pytest.raises(ValueError, match=message):
        flatten_operational_qc("B5", operational)


def test_qc_oracle_source_manifest_requires_and_hashes_extension_files(
    tmp_path: Path,
) -> None:
    required = (
        "h8_no_rescue/target_predictions_plus_source_preds.csv",
        "h8_no_rescue/target_predictions_plus_source_preds_oracle_route.csv",
        "high_coverage_qc/manifest.json",
        "high_coverage_qc/operational_summary.json",
        "high_coverage_qc/risk_policy.json",
        "high_coverage_qc/risk_selection.json",
        "high_coverage_qc/test_full_records.csv",
        "high_coverage_qc/test_hc95_records.csv",
        "high_coverage_qc/test_hc90_records.csv",
    )
    for relative in required:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative, encoding="utf-8")

    result = collect_qc_oracle_sources(tmp_path)

    assert set(result) == set(required)
    assert all(len(item["sha256"]) == 64 for item in result.values())
    (tmp_path / required[1]).unlink()
    with pytest.raises(FileNotFoundError, match="oracle_route"):
        collect_qc_oracle_sources(tmp_path)
