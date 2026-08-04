"""Assemble communication, Pi 5 deployment, and three-machine validation evidence."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.finalize_iotj_a4_end_to_end import write_json


def _flag_value(argv: Sequence[str], flag: str) -> str:
    index = list(argv).index(flag)
    return str(argv[index + 1])


def build_system_evidence(
    efficiency: Sequence[Mapping[str, Any]],
    fl_communication: Sequence[Mapping[str, Any]],
    h1_communication: Sequence[Mapping[str, Any]],
    locked: Mapping[str, Any],
    completed: Mapping[str, Any],
    run_manifest: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(fl_communication) != 1:
        raise RuntimeError("FAIL_CLOSED Flower communication summary must have one row")
    h1_total = next(row for row in h1_communication if row["direction"] == "TOTAL")
    system: list[dict[str, Any]] = [
        {
            "record_type": "communication",
            "label": "25-round Flower model exchange",
            "bytes": int(fl_communication[0]["measured_application_total_25round_bytes"]),
            "rounds": int(fl_communication[0]["rounds"]),
            "evidence_type": "measured_application_bytes",
        },
        {
            "record_type": "communication",
            "label": "One-shot federated H1 statistics",
            "bytes": int(h1_total["theoretical_serialized_exchange_bytes"]),
            "rounds": 1,
            "evidence_type": "theoretical_serialized_exchange_bytes",
        },
    ]
    for row in efficiency:
        system.append(
            {
                "record_type": "pi5_runtime",
                "label": row["runtime"],
                "pi_p50_ms": float(row["Pi_p50_ms"]),
                "pi_p95_ms": float(row["Pi_p95_ms"]),
                "pi_peak_rss_mib": float(row["Pi_peak_RSS_MiB"]),
                "pi_throughput_windows_per_s": float(row["Pi_throughput_windows_per_s"]),
                "deployment_status": row["deployment_status"],
            }
        )

    completed_rounds = int(completed["fixed_endpoint"]["round"])
    expected_rounds = int(_flag_value(locked["server"], "--rounds"))
    client_devices = [
        _flag_value(locked["client_c1"], "--device"),
        _flag_value(locked["client_c2"], "--device"),
    ]
    target_opened = bool(run_manifest.get("target_test_opened", True))
    status = (
        "PASS"
        if completed_rounds == expected_rounds == 25
        and client_devices == ["cpu", "cpu"]
        and not target_opened
        else "FAIL"
    )
    physical = [
        {
            "experiment_id": completed["experiment_id"],
            "topology": "client_C1__cloud_server__client_C2",
            "participating_machines": 3,
            "source_clients": "C1;C2",
            "client_devices": ";".join(client_devices),
            "target": locked["protocol"]["target"],
            "seed": int(locked["protocol"]["seed"]),
            "expected_rounds": expected_rounds,
            "completed_rounds": completed_rounds,
            "wall_seconds": float(run_manifest["wall_seconds"]),
            "target_test_opened_during_training": target_opened,
            "fixed_endpoint": True,
            "status": status,
            "physical_validation_scope": "real three-machine Flower execution plus independent Raspberry Pi 5 runtime benchmark",
        }
    ]
    if status != "PASS":
        raise RuntimeError(f"FAIL_CLOSED physical validation audit failed: {physical[0]}")
    return system, physical


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"FAIL_CLOSED JSON object required: {path}")
    return payload


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fields} for row in rows)


def run_system_evidence(system_root: Path, classification_run: Path, output: Path) -> None:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "efficiency": system_root / "paper_tables/table_system_efficiency.csv",
        "fl_communication": system_root / "system_metrics/b5_fl_communication_summary.csv",
        "h1_communication": system_root / "system_metrics/federated_h1_communication_summary.csv",
        "locked": classification_run / "locked_run_spec.json",
        "completed": classification_run / "fixed_endpoint_complete.json",
        "run_manifest": classification_run / "run_manifest.json",
    }
    system, physical = build_system_evidence(
        _read_csv(paths["efficiency"]),
        _read_csv(paths["fl_communication"]),
        _read_csv(paths["h1_communication"]),
        _read_json(paths["locked"]),
        _read_json(paths["completed"]),
        _read_json(paths["run_manifest"]),
    )
    _write_csv(output / "system_deployment_summary.csv", system)
    _write_csv(output / "physical_validation_audit.csv", physical)
    write_json(
        output / "protocol_manifest.json",
        {
            "schema_version": "iotj.final_a4_system.v1",
            "status": "complete",
            "source_files": {key: str(path.resolve()) for key, path in paths.items()},
            "communication_evidence_boundary": {
                "flower": "measured application payload; transport bytes unavailable",
                "federated_h1": "theoretical serialized one-shot exchange",
            },
            "physical_validation_boundary": "no hardware photograph; evidence is the audited three-machine Flower run and separate Pi 5 benchmark",
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--system-root", default="results/iotj_final_system_benchmark_20260725"
    )
    parser.add_argument(
        "--classification-run",
        default="results/iotj_final_classification_le1_20260804/FCL-E4-A4",
    )
    parser.add_argument(
        "--output", default="results/iotj_final_end_to_end_a4_20260804/system"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_system_evidence(
        Path(args.system_root), Path(args.classification_run), Path(args.output)
    )


if __name__ == "__main__":
    main()
