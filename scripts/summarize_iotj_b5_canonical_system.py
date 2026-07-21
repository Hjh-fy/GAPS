"""Create a read-only B5 canonical real-topology system summary from audit-bound raw evidence.

This intentionally summarizes one representative canonical B5 run; it does not
create multi-seed classification statistics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable


MIB = 1024 * 1024


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def exact(events: Iterable[dict[str, Any]], event_type: str, round_idx: int, client_id: str | None) -> dict[str, Any]:
    matches = [
        e for e in events
        if e.get("event_type") == event_type and e.get("round") == round_idx and e.get("client_id") == client_id
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one {event_type} round={round_idx} client={client_id}; got {len(matches)}")
    return matches[0]


def map_fitins_to_clients(server_events: Iterable[dict[str, Any]], round_idx: int) -> dict[str, dict[str, Any]]:
    """Resolve server FitIns proxy UUIDs through the paired logical FitRes events."""
    by_proxy: dict[str, dict[str, Any]] = {}
    for event in server_events:
        if event.get("event_type") == "flower_fitins_prepared" and event.get("round") == round_idx:
            proxy_id = event.get("payload", {}).get("proxy_id")
            if not isinstance(proxy_id, str) or proxy_id in by_proxy:
                raise RuntimeError("FitIns proxy identity is missing or duplicated")
            by_proxy[proxy_id] = event
    mapped: dict[str, dict[str, Any]] = {}
    for logical_id in ("C1", "C2"):
        fitres = exact(server_events, "flower_fitres_available", round_idx, logical_id)
        proxy_id = fitres.get("payload", {}).get("proxy_id")
        if not isinstance(proxy_id, str) or proxy_id not in by_proxy:
            raise RuntimeError(f"FitRes proxy has no paired FitIns: {logical_id}")
        mapped[logical_id] = by_proxy.pop(proxy_id)
    if by_proxy:
        raise RuntimeError("unpaired FitIns proxy remains")
    return mapped


def ns(event: dict[str, Any], field: str) -> int:
    value = event.get("payload", {}).get(field)
    if type(value) is not int or value < 0:
        raise RuntimeError(f"invalid {field}")
    return value


def q(values: list[float], p: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * p
    low, high = math.floor(index), math.ceil(index)
    return ordered[low] if low == high else ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def active_resource_payloads(events: list[dict[str, Any]], samples: list[dict[str, Any]], client_id: str) -> list[dict[str, Any]]:
    intervals = [
        (exact(events, "client_fit_start", r, client_id)["monotonic_ns"], exact(events, "client_fit_end", r, client_id)["monotonic_ns"])
        for r in range(1, 26)
    ]
    result = []
    for sample in samples:
        p = sample["payload"]
        start, end = p.get("sample_interval_start_monotonic_ns"), p.get("sample_interval_end_monotonic_ns")
        if type(start) is not int or type(end) is not int or end < start:
            raise RuntimeError("resource interval invalid")
        if any(start <= fit_end and end >= fit_start for fit_start, fit_end in intervals):
            result.append(p)
    if not result:
        raise RuntimeError(f"no active resource samples for {client_id}")
    return result


def summarize_resource(events: list[dict[str, Any]], samples: list[dict[str, Any]], audit: dict[str, Any], client_id: str) -> dict[str, Any]:
    active = active_resource_payloads(events, samples, client_id)
    payloads = [x["payload"] for x in samples]
    coverage = audit["resource"][client_id]
    if coverage["coverage"] < 0.95:
        raise RuntimeError(f"resource coverage below gate for {client_id}")
    temp = [p["cpu_temperature_c"] for p in active if p.get("cpu_temperature_available")]
    throttling = [p["throttled_bits"] for p in active if p.get("throttled_available")]
    return {
        "sample_count": len(payloads), "active_sample_count": len(active), "resource_coverage": coverage["coverage"],
        "expected_sample_points": coverage["expected_sample_points"], "covered_sample_points": coverage["covered_sample_points"],
        "rss_active_mean_mib": mean(p["rss_tree_bytes"] / MIB for p in active),
        "rss_peak_mib": max(p["rss_tree_peak_bytes"] / MIB for p in payloads),
        "cpu_host_mean_percent": mean(p["cpu_percent_tree_host_scale"] for p in active),
        "cpu_host_peak_percent": max(p["cpu_percent_tree_host_scale"] for p in active),
        "temperature_available": bool(temp), "temperature_mean_c": mean(temp) if temp else None,
        "temperature_peak_c": max(temp) if temp else None,
        "throttling_observed": any(x != 0 for x in throttling) if throttling else None,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def summarize(attempt_dir: Path, output_dir: Path) -> None:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    audit, status = load_json(attempt_dir / "attempt_audit.json"), load_json(attempt_dir / "attempt_status.json")
    if audit.get("status") != "valid" or status.get("state") != "canonical":
        raise RuntimeError("attempt must be audit-valid canonical evidence")
    if audit.get("counts") != {"fitins": 50, "fitres": 50, "rounds": 25}:
        raise RuntimeError("attempt count contract failed")
    raw = attempt_dir / "raw"
    server, c1, c2 = (load_jsonl(raw / host / "events.jsonl") for host in ("ecs", "pi", "ecs_c2"))
    pi_samples = [x for x in load_jsonl(raw / "pi" / "resource.jsonl") if x.get("event_type") == "resource_sample"]
    c2_samples = [x for x in load_jsonl(raw / "ecs_c2" / "resource.jsonl") if x.get("event_type") == "resource_sample"]
    timing_rows, communication_rows = [], []
    for r in range(1, 26):
        c1_train, c2_train = ns(exact(c1, "client_train_end", r, "C1"), "client_train_core_ns"), ns(exact(c2, "client_train_end", r, "C2"), "client_train_core_ns")
        aggregate = exact(server, "server_aggregate_end", r, None)
        total, da, non_da = ns(aggregate, "server_aggregate_fit_total_ns"), ns(aggregate, "server_da_total_ns"), ns(aggregate, "server_aggregate_non_da_ns")
        if total != da + non_da: raise RuntimeError("server timing decomposition mismatch")
        wall = ns(exact(server, "fit_round_end", r, None), "fit_round_wall_ns")
        timing_rows.append({"round": r, "round_wall_s": wall / 1e9, "pi_c1_train_s": c1_train / 1e9, "ecs_c2_train_s": c2_train / 1e9, "client_train_critical_path_s": max(c1_train, c2_train) / 1e9, "server_aggregate_s": total / 1e9, "server_da_s": da / 1e9, "server_non_da_s": non_da / 1e9})
        fitins = map_fitins_to_clients(server, r)
        for client_id in ("C1", "C2"):
            down = fitins[client_id]["payload"]["downlink_audit"]
            up = exact(server, "flower_fitres_available", r, client_id)["payload"]["uplink_audit"]
            communication_rows.append({"round": r, "client_id": client_id, "logical_downlink_bytes": down["logical"]["logical_downlink_total_bytes"], "logical_uplink_bytes": up["logical"]["logical_uplink_total_bytes"], "application_downlink_bytes": down["application_message_bytes"], "application_uplink_bytes": up["application_message_bytes"], "application_total_bytes": down["application_message_bytes"] + up["application_message_bytes"]})
    pi_resource, c2_resource = summarize_resource(c1, pi_samples, audit, "C1"), summarize_resource(c2, c2_samples, audit, "C2")
    closes = [load_json(raw / host / filename) for host, filename in (("ecs", "events.close.json"), ("pi", "resource.close.json"), ("ecs_c2", "resource.close.json"))]
    wall_total_ns = round(sum(x["round_wall_s"] for x in timing_rows) * 1e9)
    app_down, app_up = sum(x["application_downlink_bytes"] for x in communication_rows), sum(x["application_uplink_bytes"] for x in communication_rows)
    logical_down, logical_up = sum(x["logical_downlink_bytes"] for x in communication_rows), sum(x["logical_uplink_bytes"] for x in communication_rows)
    checkpoint = raw / "ecs" / "training" / "server_round_025_adapted.pth"
    summary = {
        "schema_version": "iotj.canonical_b5_representative_system.v1", "calculation_status": "recomputed_from_audit_bound_raw_records",
        "claim_boundary": "One canonical B5 seed-42 representative ECS+Pi+ECS-C2 run. It supports real-topology system cost evidence only, not B5 five-seed algorithm confirmation. B2 a006 remains a failed diagnostic and is not its canonical paired counterpart.",
        "run_id": status["run_id"], "attempt_id": status["attempt_id"], "group_id": "B5", "seed": 42, "attempt_state": status["state"], "audit_status": audit["status"], "audit_sha256": status["audit_sha256"], "algorithm_confirmation_commit": status["confirmation_commit"], "source_archive_sha256": status["source_archive_sha256"], "dataset_manifest_sha256": status["dataset_manifest_sha256"],
        "communication": {"logical_downlink_25round_total_bytes": logical_down, "logical_uplink_25round_total_bytes": logical_up, "application_downlink_25round_total_bytes": app_down, "application_uplink_25round_total_bytes": app_up, "application_25round_total_bytes": app_down + app_up, "application_25round_total_mib": (app_down + app_up) / MIB, "application_round_mean_bytes": (app_down + app_up) / 25, "transport_status": "not_collected"},
        "timing_seconds": {"round_wall_mean": mean(x["round_wall_s"] for x in timing_rows), "round_wall_p50": median(x["round_wall_s"] for x in timing_rows), "round_wall_p95": q([x["round_wall_s"] for x in timing_rows], .95), "round_wall_total": sum(x["round_wall_s"] for x in timing_rows), "pi_c1_train_mean": mean(x["pi_c1_train_s"] for x in timing_rows), "ecs_c2_train_mean": mean(x["ecs_c2_train_s"] for x in timing_rows), "server_da_mean": mean(x["server_da_s"] for x in timing_rows), "server_da_p95": q([x["server_da_s"] for x in timing_rows], .95), "server_non_da_mean": mean(x["server_non_da_s"] for x in timing_rows), "server_da_share_of_round_wall": sum(x["server_da_s"] for x in timing_rows) / sum(x["round_wall_s"] for x in timing_rows)},
        "resources": {"pi_c1": pi_resource, "ecs_c2": c2_resource},
        "observer": {"total_overhead_ms": sum(x["observer_total_ns"] for x in closes) / 1e6, "overhead_to_round_wall_ratio": sum(x["observer_total_ns"] for x in closes) / wall_total_ns, "event_bytes_written": sum(x["observer_event_bytes_written"] for x in closes)},
        "checkpoint": {"relative_path": "raw/ecs/training/server_round_025_adapted.pth", "size_bytes": checkpoint.stat().st_size, "sha256": sha256(checkpoint), "scope": "classifier training checkpoint only; not final C5 deployment bundle"},
    }
    output_dir.mkdir(parents=True)
    (output_dir / "b5_canonical_system_metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_csv(output_dir / "b5_canonical_communication_per_round.csv", communication_rows)
    write_csv(output_dir / "b5_canonical_round_timing.csv", timing_rows)
    (output_dir / "b5_canonical_system_metrics.md").write_text("# B5 canonical representative real-system metrics\n\n" + f"- Attempt: `{summary['attempt_id']}`; canonical, audit SHA-256 `{summary['audit_sha256']}`.\n" + f"- Application communication: {summary['communication']['application_25round_total_mib']:.4f} MiB; transport bytes not collected.\n" + f"- Mean round wall: {summary['timing_seconds']['round_wall_mean']:.2f} s; server DA share: {summary['timing_seconds']['server_da_share_of_round_wall']:.2%}.\n" + f"- Pi peak RSS: {pi_resource['rss_peak_mib']:.2f} MiB; Pi peak temperature: {pi_resource['temperature_peak_c']:.2f} C.\n" + f"- Boundary: {summary['claim_boundary']}\n", encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--attempt-dir", type=Path, required=True); p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args(); summarize(args.attempt_dir, args.output_dir); return 0


if __name__ == "__main__": raise SystemExit(main())
