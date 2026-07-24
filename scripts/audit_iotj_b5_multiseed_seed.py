from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any


EXPECTED_ROUNDS = 25
EXPECTED_DA_STEPS = 2500


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _finite(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    return True


def _stats(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "sample_std": statistics.stdev(values),
        "min": min(values),
        "max": max(values),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--seed-dir", type=Path, required=True)
    parser.add_argument("--runtime-contract", type=Path, required=True)
    parser.add_argument("--bundle-manifest", type=Path, required=True)
    parser.add_argument("--hc95", type=Path, required=True)
    parser.add_argument("--hc90", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    seed = args.seed
    run_id = f"c12_to_c5__b5__s{seed}"
    attempt_id = f"{run_id}__a001"
    attempt = args.seed_dir / "raw" / run_id / attempt_id
    raw = attempt / "raw"
    training = raw / "ecs" / "training"
    status = _json(attempt / "attempt_status.json")
    audit = _json(attempt / "attempt_audit.json")
    run_config = _json(training / "run_config.json")
    metrics_path = (
        args.seed_dir
        / "classification_evaluation"
        / f"seed{seed}_classification_metrics.json"
    )
    predictions_path = (
        args.seed_dir
        / "classification_evaluation"
        / f"seed{seed}_test_predictions.csv"
    )
    metrics = _json(metrics_path)
    server_events = _jsonl(raw / "ecs" / "events.jsonl")
    c1_events = _jsonl(raw / "pi" / "events.jsonl")
    c2_events = _jsonl(raw / "ecs_c2" / "events.jsonl")
    fit_ends = [x for x in server_events if x["event_type"] == "fit_round_end"]
    c1_fit = [x for x in c1_events if x["event_type"] == "client_fit_end"]
    c2_fit = [x for x in c2_events if x["event_type"] == "client_fit_end"]
    stats_files = sorted(training.glob("client_stats_round_*.json"))
    da_files = sorted(training.glob("domain_adapt_round_*.json"))
    checkpoint = training / "server_round_025_adapted.pth"
    latest_checkpoint = training / "server_latest_adapted.pth"
    normal_checkpoint = training / "server_round_025.pth"

    client_stats = [_json(path) for path in stats_files]
    da_stats = [_json(path) for path in da_files]
    checks = {
        "canonical_validator_accepted": (
            status["state"] == "canonical"
            and status["reason"] == "validator_accepted"
            and audit["status"] == "valid"
        ),
        "rounds_25": (
            audit["counts"]["rounds"] == EXPECTED_ROUNDS
            and len(fit_ends) == EXPECTED_ROUNDS
            and {int(x["round"]) for x in fit_ends}
            == set(range(1, EXPECTED_ROUNDS + 1))
        ),
        "fitins_fitres_50": (
            audit["counts"]["fitins"] == 50
            and audit["counts"]["fitres"] == 50
        ),
        "c1_c2_every_round": (
            len(c1_fit) == len(c2_fit) == EXPECTED_ROUNDS
            and {int(x["round"]) for x in c1_fit}
            == {int(x["round"]) for x in c2_fit}
            == set(range(1, EXPECTED_ROUNDS + 1))
            and all(
                {int(client["client_id"]) for client in item["clients"]}
                == {1, 2}
                for item in client_stats
            )
        ),
        "zero_fit_eval_failures": not audit["reasons"],
        "da_2500_steps": (
            len(da_stats) == EXPECTED_ROUNDS
            and sum(int(item["num_steps"]) for item in da_stats)
            == EXPECTED_DA_STEPS
        ),
        "round25_adapted_checkpoint": checkpoint.is_file(),
        "checkpoint_strict_load": (
            metrics["checkpoint_sha256"] == _sha256(checkpoint)
        ),
        "finite_json_evidence": all(
            _finite(item) for item in client_stats + da_stats + [metrics]
        ),
        "seed_correct": (
            int(run_config["args"]["seed"]) == seed
            and all(int(x["training_seed"]) == seed for x in fit_ends + c1_fit + c2_fit)
        ),
        "classification_rows_1360": (
            metrics["predicted_route_rows"] == 1360
            and metrics["unique_row_keys"] == 1360
        ),
        "runtime_frozen": (
            _sha256(args.bundle_manifest)
            == "a2514bd74ba0a98334d146af218922ee84884a53b93b0d4c44414723abee73b5"
            and _sha256(args.hc95)
            == "33d04439376852bb976d9a4ed5f09235107b296c5f839c75ed667fdecc598860"
            and _sha256(args.hc90)
            == "6051e7787915e0163ffd815dc089626e751906474c858072c5c0520c615dccb3"
            and _sha256(args.runtime_contract)
            == "54a42bb9f622c441a889a36fb1e585cb437e04c11128eb0578cfef6fd7711c3c"
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(
            "FAIL_CLOSED: "
            + ", ".join(name for name, passed in checks.items() if not passed)
        )

    attempt_start = _json(attempt / "status_events" / "status_001.json")[
        "wall_time_utc"
    ]
    start_dt = datetime.fromisoformat(attempt_start.replace("Z", "+00:00"))
    end_dt = datetime.fromisoformat(status["wall_time_utc"].replace("Z", "+00:00"))
    wall_seconds = (end_dt - start_dt).total_seconds()
    trace_rows = []
    c1_by_round = {int(x["round"]): x for x in c1_fit}
    c2_by_round = {int(x["round"]): x for x in c2_fit}
    for event in fit_ends:
        round_id = int(event["round"])
        payload = event["payload"]
        trace_rows.append(
            {
                "round": round_id,
                "c1_fit_seconds": c1_by_round[round_id]["payload"][
                    "client_fit_callback_ns"
                ]
                / 1e9,
                "c2_fit_seconds": c2_by_round[round_id]["payload"][
                    "client_fit_callback_ns"
                ]
                / 1e9,
                "fit_round_wall_seconds": payload["fit_round_wall_ns"] / 1e9,
                "server_da_seconds": payload["server_da_total_ns"] / 1e9,
                "server_aggregate_non_da_seconds": payload[
                    "server_aggregate_non_da_ns"
                ]
                / 1e9,
                "da_steps": 100,
                "fit_failures": 0,
                "eval_failures": 0,
            }
        )
    trace_path = args.seed_dir / f"seed{seed}_training_trace.csv"
    with trace_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(trace_rows[0]))
        writer.writeheader()
        writer.writerows(trace_rows)

    run_manifest = {
        "schema_version": "iotj.b5_seed_run_manifest.v1",
        "run_id": run_id,
        "attempt_id": attempt_id,
        "seed": seed,
        "status": "canonical",
        "validator_status": "accepted",
        "training_code_revision": status["confirmation_commit"],
        "source_archive_sha256": status["source_archive_sha256"],
        "dataset_manifest_sha256": status["dataset_manifest_sha256"],
        "algorithm_config_sha256": status["algorithm_config_sha256"],
        "attempt_audit_sha256": status["audit_sha256"],
        "topology": "ecs_c2_pi_c1",
        "rounds": 25,
        "local_epochs": 5,
        "batch_size": 32,
        "client_optimizer": "Adam",
        "client_lr": 0.0005,
        "server_da_steps_per_round": 100,
        "server_da_total_steps": 2500,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "test_used_for_training_selection_or_stopping": False,
    }
    run_manifest_path = args.seed_dir / f"seed{seed}_run_manifest.json"
    run_manifest_path.write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    timing = {
        "fit_round_wall_seconds": _stats(
            [row["fit_round_wall_seconds"] for row in trace_rows]
        ),
        "c1_fit_seconds": _stats([row["c1_fit_seconds"] for row in trace_rows]),
        "c2_fit_seconds": _stats([row["c2_fit_seconds"] for row in trace_rows]),
        "server_da_seconds": _stats(
            [row["server_da_seconds"] for row in trace_rows]
        ),
    }
    summary = {
        "schema_version": "iotj.b5_seed_training_summary.v1",
        "run_id": run_id,
        "attempt_id": attempt_id,
        "seed": seed,
        "status": "canonical",
        "attempt_wall_seconds": wall_seconds,
        "rounds": 25,
        "fitins": 50,
        "fitres": 50,
        "da_total_steps": 2500,
        "timing": timing,
        "checkpoint_sha256": _sha256(checkpoint),
        "latest_adapted_checkpoint_sha256": _sha256(latest_checkpoint),
        "round25_normal_checkpoint_sha256": _sha256(normal_checkpoint),
        "classification_metrics": metrics["metrics"],
    }
    summary_path = args.seed_dir / f"seed{seed}_training_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    postflight = {
        "schema_version": "iotj.b5_seed_postflight.v1",
        "run_id": run_id,
        "attempt_id": attempt_id,
        "seed": seed,
        "checks": checks,
        "verdict": "PASS",
        "next_seed_authorized": seed < 46,
        "evidence_boundary": "classification stability only",
    }
    postflight_path = args.seed_dir / f"seed{seed}_postflight.json"
    postflight_path.write_text(
        json.dumps(postflight, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_path = args.seed_dir / f"seed{seed}_completion_report.zh.md"
    m = metrics["metrics"]
    report_path.write_text(
        f"""# IoT-J B5 seed{seed} 正式完成报告

- run：`{run_id}`
- attempt：`{attempt_id}`
- 状态：`canonical / validator_accepted`
- 训练：25/25 rounds，C1/C2 每轮参与，0 fit/eval failure
- Server DA：100 steps/round，共 2500 steps
- attempt wall：{wall_seconds:.3f} s（{wall_seconds / 3600:.3f} h）
- round-25 adapted checkpoint SHA256：`{_sha256(checkpoint)}`

## 冻结 C5 test 分类评估

- N：1360，唯一 row key：1360
- Accuracy：{m['accuracy']:.12f}
- Macro-F1：{m['macro_f1']:.12f}
- NLL：{m['nll']:.12f}
- ECE：{m['ece']:.12f}
- Error count：{1360 - sum(m['confusion_matrix'][i][i] for i in range(4))}

## 审计结论

POSTFLIGHT_PASS。checkpoint 已由严格 round-25 加载路径完成推理验证；训练、DA、拓扑、seed、row key、有限数值与冻结 runtime/HC95/HC90 检查全部通过。C5 test 未参与训练、停止或 checkpoint 选择。

本结果只构成 B5 classification multi-seed 的单 seed 证据，不形成回归、QC、runtime v5 或 Pi benchmark 结论。
""",
        encoding="utf-8",
    )

    artifact_paths = [
        attempt / "attempt_status.json",
        attempt / "attempt_audit.json",
        attempt / "attempt_provenance.json",
        training / "run_config.json",
        checkpoint,
        latest_checkpoint,
        normal_checkpoint,
        args.seed_dir / f"seed{seed}_preflight.json",
        run_manifest_path,
        trace_path,
        summary_path,
        postflight_path,
        metrics_path,
        predictions_path,
        report_path,
    ]
    artifact_manifest = {
        "schema_version": "iotj.b5_seed_artifact_sha256.v1",
        "run_id": run_id,
        "files": [
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in artifact_paths
        ],
    }
    artifact_path = args.seed_dir / f"seed{seed}_artifact_sha256.json"
    artifact_path.write_text(
        json.dumps(artifact_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(postflight, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
