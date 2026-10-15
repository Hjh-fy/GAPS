"""Create a blocked technical receipt for a completed, validator-rejected B5 run.

This command never promotes the attempt to canonical.  It only verifies that the
training/checkpoint/evaluation artifacts are internally complete and records the
independent observability-contract failure.
"""

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


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-dir", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--server-da-steps-per-round", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    attempt = args.raw_root / args.run_id / args.attempt_id
    raw = attempt / "raw"
    training = raw / "ecs" / "training"
    status = _json(attempt / "attempt_status.json")
    audit = _json(attempt / "attempt_audit.json")
    metrics_path = (
        args.seed_dir
        / "classification_evaluation"
        / f"{args.output_prefix}_classification_metrics.json"
    )
    predictions_path = (
        args.seed_dir
        / "classification_evaluation"
        / f"{args.output_prefix}_test_predictions.csv"
    )
    metrics_payload = _json(metrics_path)
    metrics = metrics_payload["metrics"]
    server_events = _jsonl(raw / "ecs" / "events.jsonl")
    c1_events = _jsonl(raw / "pi" / "events.jsonl")
    c2_events = _jsonl(raw / "ecs_c2" / "events.jsonl")
    fit_ends = [x for x in server_events if x["event_type"] == "fit_round_end"]
    c1_fit = [x for x in c1_events if x["event_type"] == "client_fit_end"]
    c2_fit = [x for x in c2_events if x["event_type"] == "client_fit_end"]
    da_stats = [
        _json(path)
        for path in sorted(training.glob("domain_adapt_round_*.json"))
    ]
    checkpoint = training / "server_round_025_adapted.pth"

    with predictions_path.open(newline="", encoding="utf-8") as handle:
        prediction_rows = list(csv.DictReader(handle))
    unique_keys = {row["row_key"] for row in prediction_rows}
    expected_total_steps = (
        EXPECTED_ROUNDS * args.server_da_steps_per_round
    )
    expected_reason = (
        len(audit["reasons"]) == 1
        and str(audit["reasons"][0]).startswith("C2 resource coverage ")
        and str(audit["reasons"][0]).endswith(" is below 0.95")
    )
    checks = {
        "validator_rejected_preserved": (
            status["state"] == "invalid"
            and status["reason"] == "validator_rejected"
            and audit["status"] == "invalid"
        ),
        "only_observability_coverage_reason": expected_reason,
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
        ),
        "da_expected_steps": (
            len(da_stats) == EXPECTED_ROUNDS
            and all(
                int(item["num_steps"]) == args.server_da_steps_per_round
                for item in da_stats
            )
            and sum(int(item["num_steps"]) for item in da_stats)
            == expected_total_steps
        ),
        "checkpoint_strict_evaluation_load": (
            checkpoint.is_file()
            and metrics_payload["checkpoint_sha256"] == _sha256(checkpoint)
        ),
        "evaluation_rows_1360_unique": (
            metrics_payload["predicted_route_rows"] == 1360
            and metrics_payload["unique_row_keys"] == 1360
            and len(prediction_rows) == len(unique_keys) == 1360
        ),
        "finite_evidence": _finite(metrics_payload) and _finite(da_stats),
        "test_not_used_for_training_selection_or_stopping": not metrics_payload[
            "test_used_for_training_selection_or_stopping"
        ],
    }
    if not all(checks.values()):
        raise RuntimeError(
            "FAIL_CLOSED technical-integrity receipt failed: "
            + ", ".join(name for name, passed in checks.items() if not passed)
        )

    c1_by_round = {int(x["round"]): x for x in c1_fit}
    c2_by_round = {int(x["round"]): x for x in c2_fit}
    trace_rows: list[dict[str, Any]] = []
    for event in fit_ends:
        round_id = int(event["round"])
        payload = event["payload"]
        trace_rows.append(
            {
                "round": round_id,
                "c1_fit_seconds": (
                    c1_by_round[round_id]["payload"]["client_fit_callback_ns"]
                    / 1e9
                ),
                "c2_fit_seconds": (
                    c2_by_round[round_id]["payload"]["client_fit_callback_ns"]
                    / 1e9
                ),
                "fit_round_wall_seconds": payload["fit_round_wall_ns"] / 1e9,
                "server_da_seconds": payload["server_da_total_ns"] / 1e9,
                "server_aggregate_non_da_seconds": (
                    payload["server_aggregate_non_da_ns"] / 1e9
                ),
                "da_steps": args.server_da_steps_per_round,
                "fit_failures": 0,
                "eval_failures": 0,
            }
        )

    trace_path = (
        args.seed_dir / f"{args.output_prefix}_noncanonical_training_trace.csv"
    )
    with trace_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(trace_rows[0]))
        writer.writeheader()
        writer.writerows(trace_rows)

    start_dt = datetime.fromisoformat(
        _json(attempt / "status_events/status_001.json")["wall_time_utc"].replace(
            "Z", "+00:00"
        )
    )
    end_dt = datetime.fromisoformat(
        status["wall_time_utc"].replace("Z", "+00:00")
    )
    summary = {
        "schema_version": "iotj.b5_rejected_observability_summary.v1",
        "run_id": args.run_id,
        "attempt_id": args.attempt_id,
        "seed": int(metrics_payload["seed"]),
        "status": "blocked_observability_contract",
        "canonical_validator_accepted": False,
        "training_integrity_verified": True,
        "formal_evidence_approved": False,
        "attempt_wall_seconds": (end_dt - start_dt).total_seconds(),
        "rounds": EXPECTED_ROUNDS,
        "fitins": int(audit["counts"]["fitins"]),
        "fitres": int(audit["counts"]["fitres"]),
        "da_total_steps": expected_total_steps,
        "checkpoint_sha256": _sha256(checkpoint),
        "classification_metrics": metrics,
        "timing": {
            "fit_round_wall_seconds": _stats(
                [row["fit_round_wall_seconds"] for row in trace_rows]
            ),
            "c1_fit_seconds": _stats(
                [row["c1_fit_seconds"] for row in trace_rows]
            ),
            "c2_fit_seconds": _stats(
                [row["c2_fit_seconds"] for row in trace_rows]
            ),
            "server_da_seconds": _stats(
                [row["server_da_seconds"] for row in trace_rows]
            ),
        },
        "observability_contract": {
            "verdict": "FAIL",
            "reasons": audit["reasons"],
            "resource": audit["resource"],
        },
        "checks": checks,
        "evidence_boundary": (
            "Non-canonical technical result only. The model training and "
            "evaluation artifacts are complete, but the preregistered "
            "observability validator did not pass."
        ),
    }
    summary_path = (
        args.seed_dir
        / f"{args.output_prefix}_noncanonical_training_summary.json"
    )
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    receipt = {
        "schema_version": "iotj.b5_rejected_observability_receipt.v1",
        "verdict": "BLOCKED_OBSERVABILITY_CONTRACT",
        "formal_postflight_pass": False,
        "technical_training_integrity_pass": True,
        "summary_path": str(summary_path),
        "summary_sha256": _sha256(summary_path),
        "trace_path": str(trace_path),
        "trace_sha256": _sha256(trace_path),
        "attempt_audit_path": str(attempt / "attempt_audit.json"),
        "attempt_audit_sha256": _sha256(attempt / "attempt_audit.json"),
        "metrics_path": str(metrics_path),
        "metrics_sha256": _sha256(metrics_path),
        "predictions_path": str(predictions_path),
        "predictions_sha256": _sha256(predictions_path),
    }
    receipt_path = (
        args.seed_dir
        / f"{args.output_prefix}_noncanonical_observability_receipt.json"
    )
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
