from __future__ import annotations

import csv
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any


ROOT = Path("results/iotj_b5_multiseed_20260724")
SEEDS = (42, 43, 44, 45, 46)


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _summary(values: list[float]) -> dict[str, float | int]:
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "sample_std_ddof1": statistics.stdev(values),
        "min": min(values),
        "max": max(values),
        "median": statistics.median(values),
    }


def main() -> None:
    metric_payloads: dict[int, dict[str, Any]] = {}
    for seed in SEEDS:
        base = ROOT / ("seed42_reference" if seed == 42 else f"seed{seed}")
        path = (
            base
            / "classification_evaluation"
            / f"seed{seed}_classification_metrics.json"
        )
        payload = _json(path)
        if (
            payload["seed"] != seed
            or payload["predicted_route_rows"] != 1360
            or payload["unique_row_keys"] != 1360
            or payload["test_used_for_training_selection_or_stopping"]
        ):
            raise ValueError(f"invalid classification evidence for seed {seed}")
        metric_payloads[seed] = payload

    per_seed = []
    for seed, payload in metric_payloads.items():
        metrics = payload["metrics"]
        confusion = metrics["confusion_matrix"]
        reference = (
            ROOT / "seed42_reference" / "reference_manifest.json"
            if seed == 42
            else ROOT / f"seed{seed}" / f"seed{seed}_training_summary.json"
        )
        reference_payload = _json(reference)
        per_seed.append(
            {
                "seed": seed,
                "status": "canonical",
                "N": metrics["N"],
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "nll": metrics["nll"],
                "ece": metrics["ece"],
                "error_count": 1360
                - sum(confusion[index][index] for index in range(4)),
                "training_wall_seconds": reference_payload[
                    "training_wall_seconds"
                    if seed == 42
                    else "attempt_wall_seconds"
                ],
                "checkpoint_sha256": payload.get(
                    "checkpoint_sha256",
                    _sha256(Path(payload["checkpoint"])),
                ),
                "metrics_source": str(
                    (
                        ROOT
                        / ("seed42_reference" if seed == 42 else f"seed{seed}")
                        / "classification_evaluation"
                        / f"seed{seed}_classification_metrics.json"
                    )
                ),
            }
        )
    per_seed_path = ROOT / "per_seed_b5_classification_metrics.csv"
    _write_csv(per_seed_path, per_seed)

    summary_rows = []
    for metric in ("accuracy", "macro_f1", "nll", "ece", "error_count"):
        values = [float(row[metric]) for row in per_seed]
        summary_rows.append({"metric": metric, **_summary(values)})
    summary_path = ROOT / "b5_classification_multiseed_summary.csv"
    _write_csv(summary_path, summary_rows)

    confusion_rows: list[dict[str, Any]] = []
    for true_class in range(4):
        for pred_class in range(4):
            values = [
                float(
                    metric_payloads[seed]["metrics"]["confusion_matrix"][
                        true_class
                    ][pred_class]
                )
                for seed in SEEDS
            ]
            confusion_rows.append(
                {
                    "record_type": "confusion_cell",
                    "true_class": true_class,
                    "pred_class": pred_class,
                    **_summary(values),
                    "sum": sum(values),
                }
            )
    for class_id in range(4):
        values = [
            float(
                metric_payloads[seed]["metrics"]["per_class_recall"][
                    str(class_id)
                ]
            )
            for seed in SEEDS
        ]
        confusion_rows.append(
            {
                "record_type": "class_recall",
                "true_class": class_id,
                "pred_class": "",
                **_summary(values),
                "sum": "",
            }
        )
    confusion_path = ROOT / "b5_confusion_matrix_summary.csv"
    _write_csv(confusion_path, confusion_rows)

    manifest = {
        "schema_version": "iotj.b5_classification_multiseed_completion.v1",
        "status": "complete",
        "seed_set": list(SEEDS),
        "canonical_seeds": list(SEEDS),
        "newly_trained_seeds": [43, 44, 45, 46],
        "seed42_retrained": False,
        "all_test_rows": 1360,
        "all_row_keys_unique": True,
        "all_postflight_pass": True,
        "training_topology": "ecs_c2_pi_c1",
        "training_code_revision": "2ef7aea77b9dfabdd09da4f38742907a37c58c30",
        "runtime_v4_frozen_six": {
            "bundle_manifest": "a2514bd74ba0a98334d146af218922ee84884a53b93b0d4c44414723abee73b5",
            "c5_test_features": "7955cb70b24fa86ce109a52ca3b2231ad543b8ba8be0276781ffa03384143a82",
            "c5_test_metadata": "9b48459f52698b11fad66c0a2c63c9ede22292555e4bcaa71125e1f7e90097bf",
            "c5_test_phase_labels": "a69f333c8418fa3bf94c599a2d684cd122b4a46df2ff405bced227b68fcdb8b5",
            "hc95_reference": "33d04439376852bb976d9a4ed5f09235107b296c5f839c75ed667fdecc598860",
            "hc90_reference": "6051e7787915e0163ffd815dc089626e751906474c858072c5c0520c615dccb3",
        },
        "outputs": [
            str(per_seed_path),
            str(summary_path),
            str(confusion_path),
        ],
        "evidence_boundary": (
            "B5 classification five-seed stability only; no regression, QC, "
            "runtime v5, Pi benchmark, or low-calibration conclusion."
        ),
        "next_stage_ready": True,
        "next_stage_not_started": "B5 regression multi-seed",
    }
    manifest_path = ROOT / "b5_multiseed_completion_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    summary_by_metric = {row["metric"]: row for row in summary_rows}
    report_path = Path(
        "docs/experiments/iotj_b5_classification_multiseed_result_20260724.zh.md"
    )
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite {report_path}")
    table = "\n".join(
        "| {seed} | {accuracy:.6f} | {macro_f1:.6f} | {nll:.6f} | "
        "{ece:.6f} | {error_count} |".format(**row)
        for row in per_seed
    )
    report_path.write_text(
        f"""# IoT-J B5 classification five-seed 正式结果

五个 seed（42–46）均为 canonical；seed42 复用正式 checkpoint，seed43–46 在相同三机拓扑和冻结协议下顺序训练。所有新 seed 均通过 25/25 rounds、C1/C2 每轮参与、2500 DA steps、严格 checkpoint 加载和 1360 行唯一 predicted route 的 postflight。

| Seed | Accuracy | Macro-F1 | NLL | ECE | Errors |
|---:|---:|---:|---:|---:|---:|
{table}

## 五种子描述统计

- Accuracy：{summary_by_metric['accuracy']['mean']:.6f} ± {summary_by_metric['accuracy']['sample_std_ddof1']:.6f}，range [{summary_by_metric['accuracy']['min']:.6f}, {summary_by_metric['accuracy']['max']:.6f}]
- Macro-F1：{summary_by_metric['macro_f1']['mean']:.6f} ± {summary_by_metric['macro_f1']['sample_std_ddof1']:.6f}，range [{summary_by_metric['macro_f1']['min']:.6f}, {summary_by_metric['macro_f1']['max']:.6f}]
- NLL：{summary_by_metric['nll']['mean']:.6f} ± {summary_by_metric['nll']['sample_std_ddof1']:.6f}，range [{summary_by_metric['nll']['min']:.6f}, {summary_by_metric['nll']['max']:.6f}]
- ECE：{summary_by_metric['ece']['mean']:.6f} ± {summary_by_metric['ece']['sample_std_ddof1']:.6f}，range [{summary_by_metric['ece']['min']:.6f}, {summary_by_metric['ece']['max']:.6f}]
- Error count：{summary_by_metric['error_count']['mean']:.2f} ± {summary_by_metric['error_count']['sample_std_ddof1']:.2f}，range [{summary_by_metric['error_count']['min']:.0f}, {summary_by_metric['error_count']['max']:.0f}]

## 结论与边界

B5 在 seeds42–46 上保持稳定：最差 Accuracy 为 {summary_by_metric['accuracy']['min']:.6f}，最好为 {summary_by_metric['accuracy']['max']:.6f}；未出现训练、拓扑、checkpoint 或 row-map 异常。该结论只支持 B5 classification five-seed stability，不支持回归方法、QC、runtime v5、Pi 性能或 low-calibration 结论。

五种子分类 route 和 provenance 已齐备，因此具备在获得下一阶段授权后启动 B5 regression multi-seed 的输入条件；本次没有启动该阶段。
""",
        encoding="utf-8",
    )
    indexed_paths = [
        per_seed_path,
        summary_path,
        confusion_path,
        manifest_path,
        report_path,
    ]
    index_path = Path(
        "docs/experiments/"
        "iotj_b5_classification_multiseed_result_index_20260724.json"
    )
    if index_path.exists():
        raise FileExistsError(f"refusing to overwrite {index_path}")
    index = {
        "schema_version": "iotj.b5_classification_multiseed_result_index.v1",
        "status": "complete",
        "seed_set": list(SEEDS),
        "artifacts": [
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in indexed_paths
        ],
        "checkpoint_bodies_committed_to_git": False,
        "evidence_boundary": manifest["evidence_boundary"],
    }
    index_path.write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary_rows, indent=2))


if __name__ == "__main__":
    main()
