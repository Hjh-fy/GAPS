"""Summarize the locked LE1 server-DA budget sensitivity experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any


EXPECTED_N = 1360
LEVELS = (100, 80, 50, 30)


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _finite(payload: Any) -> bool:
    if isinstance(payload, float):
        return math.isfinite(payload)
    if isinstance(payload, dict):
        return all(_finite(value) for value in payload.values())
    if isinstance(payload, list):
        return all(_finite(value) for value in payload)
    return True


def _errors(metrics: dict[str, Any]) -> int:
    confusion = metrics["confusion_matrix"]
    return EXPECTED_N - sum(int(confusion[i][i]) for i in range(4))


def _prediction_map(path: Path) -> dict[str, tuple[int, int]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    result = {
        row["row_key"]: (int(row["true_class"]), int(row["pred_class"]))
        for row in rows
    }
    if len(rows) != EXPECTED_N or len(result) != EXPECTED_N:
        raise RuntimeError(f"FAIL_CLOSED invalid prediction rows: {path}")
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-root",
        type=Path,
        default=Path("results/iotj_b5_server_da_budget_ablation_20260731"),
    )
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=Path(
            "results/iotj_b5_local_epoch_ablation_20260729/le1"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"REFUSE_TO_OVERWRITE: {args.output_dir}")

    inputs = {
        100: {
            "metrics": args.baseline_root
            / "classification_evaluation/le1_classification_metrics.json",
            "predictions": args.baseline_root
            / "classification_evaluation/le1_test_predictions.csv",
            "summary": args.baseline_root / "le1_training_summary.json",
            "status": "post_freeze_existing_LE1_reference",
        }
    }
    for steps in LEVELS[1:]:
        root = args.result_root / f"da{steps}"
        inputs[steps] = {
            "metrics": root
            / f"classification_evaluation/da{steps}_classification_metrics.json",
            "predictions": root
            / f"classification_evaluation/da{steps}_test_predictions.csv",
            "summary": (
                root / f"da{steps}_training_summary.json"
                if steps != 30
                else root / "da30_noncanonical_training_summary.json"
            ),
            "status": (
                "post_freeze_single_seed_budget_sensitivity"
                if steps != 30
                else "blocked_observability_contract_technical_result_only"
            ),
        }

    rows: list[dict[str, Any]] = []
    predictions: dict[int, dict[str, tuple[int, int]]] = {}
    for steps in LEVELS:
        item = inputs[steps]
        metrics_payload = _json(item["metrics"])
        summary = _json(item["summary"])
        metrics = metrics_payload["metrics"]
        if (
            int(metrics_payload["seed"]) != 42
            or int(metrics["N"]) != EXPECTED_N
            or int(metrics_payload["predicted_route_rows"]) != EXPECTED_N
            or int(metrics_payload["unique_row_keys"]) != EXPECTED_N
            or metrics_payload["test_used_for_training_selection_or_stopping"]
            or int(summary["da_total_steps"]) != 25 * steps
            or not _finite(metrics)
        ):
            raise RuntimeError(f"FAIL_CLOSED invalid DA{steps} evidence")
        predictions[steps] = _prediction_map(item["predictions"])
        timing = summary["timing"]
        wall_seconds = float(
            summary["training_wall_seconds"]
            if "training_wall_seconds" in summary
            else summary["attempt_wall_seconds"]
        )
        canonical_validator_accepted = bool(
            summary.get(
                "canonical_validator_accepted",
                summary["status"] == "canonical",
            )
        )
        row = {
            "variant": f"LE1_DA{steps}",
            "seed": 42,
            "rounds": 25,
            "local_epochs_per_round": 1,
            "server_da_steps_per_round": steps,
            "server_da_total_steps": 25 * steps,
            "accuracy": float(metrics["accuracy"]),
            "macro_f1": float(metrics["macro_f1"]),
            "nll": float(metrics["nll"]),
            "ece": float(metrics["ece"]),
            "error_count": _errors(metrics),
            "recall_class_0": float(metrics["per_class_recall"]["0"]),
            "recall_class_1": float(metrics["per_class_recall"]["1"]),
            "recall_class_2": float(metrics["per_class_recall"]["2"]),
            "recall_class_3": float(metrics["per_class_recall"]["3"]),
            "server_da_seconds_mean_per_round": float(
                timing["server_da_seconds"]["mean"]
            ),
            "training_wall_seconds": wall_seconds,
            "training_wall_hours": wall_seconds / 3600.0,
            "metrics_sha256": _sha256(item["metrics"]),
            "predictions_sha256": _sha256(item["predictions"]),
            "summary_sha256": _sha256(item["summary"]),
            "evidence_status": item["status"],
            "canonical_validator_accepted": canonical_validator_accepted,
            "formal_evidence_approved": canonical_validator_accepted,
        }
        rows.append(row)

    baseline = rows[0]
    baseline_predictions = predictions[100]
    for row in rows:
        steps = int(row["server_da_steps_per_round"])
        candidate = predictions[steps]
        if set(candidate) != set(baseline_predictions):
            raise RuntimeError(f"FAIL_CLOSED row-key mismatch for DA{steps}")
        baseline_only_correct = 0
        candidate_only_correct = 0
        disagreements = 0
        for key, (truth, baseline_pred) in baseline_predictions.items():
            candidate_truth, candidate_pred = candidate[key]
            if candidate_truth != truth:
                raise RuntimeError("FAIL_CLOSED true-class mismatch")
            disagreements += int(candidate_pred != baseline_pred)
            baseline_only_correct += int(
                baseline_pred == truth and candidate_pred != truth
            )
            candidate_only_correct += int(
                candidate_pred == truth and baseline_pred != truth
            )
        row["accuracy_delta_vs_da100"] = (
            row["accuracy"] - baseline["accuracy"]
        )
        row["macro_f1_delta_vs_da100"] = (
            row["macro_f1"] - baseline["macro_f1"]
        )
        row["nll_delta_vs_da100"] = row["nll"] - baseline["nll"]
        row["ece_delta_vs_da100"] = row["ece"] - baseline["ece"]
        row["wall_time_reduction_vs_da100"] = (
            baseline["training_wall_seconds"] - row["training_wall_seconds"]
        ) / baseline["training_wall_seconds"]
        row["prediction_disagreement_rows_vs_da100"] = disagreements
        row["da100_only_correct_rows"] = baseline_only_correct
        row["candidate_only_correct_rows"] = candidate_only_correct
        row["accuracy_retention_pass"] = (
            row["accuracy_delta_vs_da100"] >= -0.005
        )
        row["macro_f1_retention_pass"] = (
            row["macro_f1_delta_vs_da100"] >= -0.005
        )
        row["joint_retention_pass"] = (
            row["accuracy_retention_pass"]
            and row["macro_f1_retention_pass"]
        )

    args.output_dir.mkdir(parents=True)
    csv_path = args.output_dir / "b5_server_da_budget_comparison.csv"
    with csv_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    receipt = {
        "schema_version": "iotj.b5_server_da_budget_summary.v1",
        "verdict": "PARTIAL_FORMAL_EVIDENCE_DA30_BLOCKED",
        "comparison_identity": {
            "classifier": "B5",
            "seed": 42,
            "source_clients": ["C1", "C2"],
            "target_client": "C5",
            "rounds": 25,
            "local_epochs_per_round": 1,
            "only_intended_variable": "server_adaptation.steps",
        },
        "rows": rows,
        "engineering_retention_tolerance": {
            "accuracy_absolute_drop_max": 0.005,
            "macro_f1_absolute_drop_max": 0.005,
            "statistical_noninferiority_claim": False,
        },
        "test_role": "evaluation_only_after_each fixed protocol",
        "selection_rule": "descriptive only; no frozen B5 reselection",
        "evidence_boundary": (
            "Post-freeze single-seed server-DA compute-budget sensitivity. "
            "It does not replace the frozen five-seed B5 evidence. DA30 is a "
            "non-canonical technical result because the preregistered C2 "
            "resource-coverage validator did not pass."
        ),
    }
    json_path = args.output_dir / "b5_server_da_budget_summary.json"
    json_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    table = "\n".join(
        "| {variant} | {server_da_steps_per_round} | {accuracy:.6f} | "
        "{macro_f1:.6f} | {nll:.6f} | {ece:.6f} | {error_count} | "
        "{training_wall_hours:.3f} | {joint_retention_pass} | "
        "{canonical_validator_accepted} |".format(**row)
        for row in rows
    )
    (args.output_dir / "iotj_b5_server_da_budget_result_20260731.zh.md").write_text(
        f"""# B5 Server-DA 计算预算敏感性结果

| 配置 | DA steps/round | Accuracy | Macro-F1 | NLL | ECE | errors | wall (h) | 0.5 pp 保持门槛 | canonical |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
{table}

本结果为冻结后、seed 42 单种子敏感性分析。DA100 复用既有 LE1
结果；DA80/50/30 在配置锁定后顺序运行。C5 test 不参与拟合、早停、
checkpoint 选择或步数重选。0.5 个百分点为工程容差，不是统计非劣检验。

DA30 完成了 25/25 轮、750 个 DA steps、严格 checkpoint 评估和 1360 行
预测，但 C2 资源采样覆盖率为 0.948214，低于预注册 validator 下限 0.95，
因此仅作为 non-canonical 技术结果，不得作为通过正式观测审计的证据。
""",
        encoding="utf-8",
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
