from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pytest

from scripts.validate_iotj_confirmation_attempt import (
    read_events,
    validate_attempt,
)


SCHEMA_VERSION = "iotj.confirmation.observability.v1"
RUN_ID = "c12_to_c5__b2__s42"
ATTEMPT_ID = f"{RUN_ID}__a001"
COMMIT = "a" * 40
SOURCE_SHA = "b" * 64
DATASET_SHA = "c" * 64
ALGORITHM_SHA = "d" * 64
PROTOCOL_SHA = "e" * 64

COMMON_FIELDS = {
    "schema_version",
    "event_id",
    "event_type",
    "run_id",
    "attempt_id",
    "group_id",
    "training_seed",
    "round",
    "client_id",
    "host_id",
    "producer",
    "process_instance_id",
    "sequence",
    "wall_time_utc",
    "monotonic_ns",
    "confirmation_commit",
    "source_archive_sha256",
    "dataset_manifest_sha256",
    "algorithm_config_sha256",
    "status",
    "payload",
}

OVERHEAD_FIELDS = {
    "observed_event_id",
    "observer_flower_serialize_ns",
    "observer_event_encode_ns",
    "observer_io_write_ns",
    "observer_fsync_ns",
    "observer_total_ns",
    "observer_event_bytes_written",
    "observer_event_count",
}


@dataclass(frozen=True)
class AttemptFixture:
    path: Path
    protocol: dict[str, Any]


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _event(
    *,
    host_id: str,
    producer: str,
    process: str,
    sequence: int,
    event_type: str,
    round_idx: int | None,
    client_id: str | None,
    monotonic_ns: int,
    status: str = "succeeded",
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "event_id": (
            f"{ATTEMPT_ID}/{host_id}/{producer}/{process}/{sequence}"
        ),
        "event_type": event_type,
        "run_id": RUN_ID,
        "attempt_id": ATTEMPT_ID,
        "group_id": "B2",
        "training_seed": 42,
        "round": round_idx,
        "client_id": client_id,
        "host_id": host_id,
        "producer": producer,
        "process_instance_id": process,
        "sequence": sequence,
        "wall_time_utc": "2026-07-16T00:00:00.000000Z",
        "monotonic_ns": monotonic_ns,
        "confirmation_commit": COMMIT,
        "source_archive_sha256": SOURCE_SHA,
        "dataset_manifest_sha256": DATASET_SHA,
        "algorithm_config_sha256": ALGORITHM_SHA,
        "status": status,
        "payload": {} if payload is None else payload,
    }


class _Producer:
    def __init__(self, host_id: str, producer: str, process: str) -> None:
        self.host_id = host_id
        self.producer = producer
        self.process = process
        self.rows: list[dict[str, Any]] = []

    def emit(
        self,
        event_type: str,
        *,
        round_idx: int | None,
        client_id: str | None,
        monotonic_ns: int,
        status: str = "succeeded",
        payload: dict[str, Any] | None = None,
    ) -> None:
        domain_sequence = len(self.rows) + 1
        domain = _event(
            host_id=self.host_id,
            producer=self.producer,
            process=self.process,
            sequence=domain_sequence,
            event_type=event_type,
            round_idx=round_idx,
            client_id=client_id,
            monotonic_ns=monotonic_ns,
            status=status,
            payload=payload,
        )
        self.rows.append(domain)
        overhead_sequence = len(self.rows) + 1
        self.rows.append(
            _event(
                host_id=self.host_id,
                producer=self.producer,
                process=self.process,
                sequence=overhead_sequence,
                event_type="observer_overhead",
                round_idx=round_idx,
                client_id=client_id,
                monotonic_ns=monotonic_ns + 1,
                payload={
                    "observed_event_id": domain["event_id"],
                    "observer_flower_serialize_ns": (
                        7 if event_type.startswith("flower_") else 0
                    ),
                    "observer_event_encode_ns": 11,
                    "observer_io_write_ns": 13,
                    "observer_fsync_ns": 0,
                    "observer_total_ns": (
                        31 if event_type.startswith("flower_") else 24
                    ),
                    "observer_event_bytes_written": 127,
                    "observer_event_count": 1,
                },
            )
        )


def _message_audit(direction: str, round_idx: int, client_id: str) -> dict[str, Any]:
    if direction == "downlink":
        logical = {
            "logical_downlink_model_value_bytes": 1000 + round_idx,
            "logical_downlink_parameter_blob_bytes": 1100 + round_idx,
            "logical_downlink_semantic_proto_utf8_bytes": 20,
            "logical_downlink_other_config_value_bytes": 30,
            "logical_downlink_total_bytes": 1150 + round_idx,
        }
    else:
        logical = {
            "logical_uplink_model_value_bytes": 1000 + round_idx,
            "logical_uplink_parameter_blob_bytes": 1100 + round_idx,
            "logical_uplink_prototype_utf8_bytes": 20,
            "logical_uplink_prototype_var_utf8_bytes": 30,
            "logical_uplink_statistics_utf8_bytes": 40,
            "logical_uplink_diagnostic_value_bytes": 50,
            "logical_uplink_total_bytes": 1240 + round_idx,
        }
    return {
        "logical": logical,
        "application_message_bytes": 1300 + round_idx,
        "application_message_sha256": hashlib.sha256(
            f"{direction}/{round_idx}/{client_id}".encode("ascii")
        ).hexdigest(),
    }


def _server_rows() -> list[dict[str, Any]]:
    producer = _Producer("ecs", "server", "server-process")
    for round_idx in range(1, 26):
        base = round_idx * 10_000_000_000
        producer.emit(
            "fit_round_start",
            round_idx=round_idx,
            client_id=None,
            monotonic_ns=base,
            status="started",
        )
        for client_id in ("C1", "C2"):
            proxy_id = f"proxy-{client_id}-round-{round_idx}"
            producer.emit(
                "flower_fitins_prepared",
                round_idx=round_idx,
                client_id=proxy_id,
                monotonic_ns=base + 100,
                payload={
                    "proxy_id": proxy_id,
                    "downlink_audit": _message_audit(
                        "downlink", round_idx, client_id
                    ),
                },
            )
        producer.emit(
            "server_aggregate_start",
            round_idx=round_idx,
            client_id=None,
            monotonic_ns=base + 8_000_000_000,
            status="started",
        )
        for client_id in ("C1", "C2"):
            proxy_id = f"proxy-{client_id}-round-{round_idx}"
            producer.emit(
                "flower_fitres_available",
                round_idx=round_idx,
                client_id=client_id,
                monotonic_ns=base + 8_100_000_000,
                payload={
                    "proxy_id": proxy_id,
                    "uplink_audit": _message_audit(
                        "uplink", round_idx, client_id
                    ),
                },
            )
        timing = {
            "server_aggregate_fit_total_ns": 1_000_000_000,
            "server_da_total_ns": 0,
            "server_aggregate_non_da_ns": 1_000_000_000,
            "da_executed": False,
        }
        producer.emit(
            "server_aggregate_end",
            round_idx=round_idx,
            client_id=None,
            monotonic_ns=base + 9_000_000_000,
            payload=timing,
        )
        producer.emit(
            "fit_round_end",
            round_idx=round_idx,
            client_id=None,
            monotonic_ns=base + 9_100_000_000,
            payload={**timing, "fit_round_wall_ns": 9_100_000_000},
        )
    return producer.rows


def _client_rows(client_id: str, host_id: str, clock_offset: int) -> list[dict[str, Any]]:
    producer = _Producer(host_id, "client", f"{client_id.lower()}-process")
    for round_idx in range(1, 26):
        base = clock_offset + round_idx * 10_000_000_000
        producer.emit(
            "client_fit_start",
            round_idx=round_idx,
            client_id=client_id,
            monotonic_ns=base + 1_000_000_000,
            status="started",
        )
        producer.emit(
            "client_train_start",
            round_idx=round_idx,
            client_id=client_id,
            monotonic_ns=base + 2_000_000_000,
            status="started",
        )
        producer.emit(
            "client_train_end",
            round_idx=round_idx,
            client_id=client_id,
            monotonic_ns=base + 7_000_000_000,
            payload={"client_train_core_ns": 5_000_000_000},
        )
        producer.emit(
            "client_fit_end",
            round_idx=round_idx,
            client_id=client_id,
            monotonic_ns=base + 8_000_000_000,
            payload={"client_fit_callback_ns": 7_000_000_000},
        )
    return producer.rows


def _resource_rows(
    client_id: str, host_id: str, clock_offset: int
) -> list[dict[str, Any]]:
    producer = _Producer(
        host_id, "resource_sampler", f"{client_id.lower()}-sampler-process"
    )
    for round_idx in range(1, 26):
        base = clock_offset + round_idx * 10_000_000_000
        producer.emit(
            "resource_sample",
            round_idx=None,
            client_id=client_id,
            monotonic_ns=base + 5_100_000_000,
            payload={
                "root_pid": 1000 + round_idx,
                "sampler_pid_excluded": 2000 + round_idx,
                "pids": [1000 + round_idx],
                "process_identities": [
                    {
                        "pid": 1000 + round_idx,
                        "create_time": 1.0,
                        "identity_available": True,
                    }
                ],
                "rss_tree_bytes": 10_000,
                "rss_tree_peak_bytes": 20_000,
                "process_count_tree": 1,
                "thread_count_tree": 2,
                "cpu_time_tree_seconds": 1.0,
                "cpu_time_tree_delta_seconds": 0.1,
                "cpu_percent_tree_one_core_scale": 10.0,
                "cpu_percent_tree_host_scale": 2.5,
                "logical_cpu_count": 4,
                "sample_interval_start_monotonic_ns": base + 3_000_000_000,
                "sample_interval_end_monotonic_ns": base + 5_000_000_000,
                "sample_interval_wall_ns": 2_000_000_000,
                "sample_errors": [],
                "cpu_temperature_c": None,
                "cpu_temperature_available": False,
                "cpu_temperature_source": None,
                "vcgencmd_available": False,
                "throttled_raw": None,
                "throttled_bits": None,
                "throttled_available": False,
                "thermal_errors": [],
            },
        )
    producer.emit(
        "resource_sampler_end",
        round_idx=None,
        client_id=client_id,
        monotonic_ns=clock_offset + 260_000_000_000,
        payload={
            "root_pid": 1001,
            "sampler_pid": 2001,
            "shutdown_reason": "stop_file",
            "shutdown_error": None,
            "sample_count": 25,
            "sampler_cpu_user_seconds": 0.1,
            "sampler_cpu_system_seconds": 0.1,
            "sampler_rss_peak_bytes": 50_000,
            "sampler_rss_peak_available": True,
            "sampler_rss_peak_method": "test",
            "sampler_rss_peak_error": None,
            "observer_cost_values_scope": "before_resource_sampler_end_emit",
            "observer_event_encode_ns": 100,
            "observer_io_write_ns": 100,
            "observer_fsync_ns": 0,
            "observer_event_bytes_written": 1_000,
            "observer_event_count": 50,
            "observer_close_summary_path": "resource.close.json",
            "observer_close_summary_is_authoritative": True,
        },
    )
    return producer.rows


def _close_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    first = rows[0]
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": RUN_ID,
        "attempt_id": ATTEMPT_ID,
        "host_id": first["host_id"],
        "producer": first["producer"],
        "process_instance_id": first["process_instance_id"],
        "observer_flower_serialize_ns": 100,
        "observer_event_encode_ns": 200,
        "observer_io_write_ns": 300,
        "observer_fsync_ns": 0,
        "observer_total_ns": 600,
        "observer_event_bytes_written": 10_000,
        "observer_event_count": len(rows),
        "observer_reporting_tail_bytes": 100,
    }


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(_canonical_bytes(row) + b"\n" for row in rows))
    path.with_suffix(".close.json").write_bytes(
        _canonical_bytes(_close_summary(rows)) + b"\n"
    )


def _read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _rewrite_rows(path: Path, mutate: Callable[[list[dict[str, Any]]], None]) -> None:
    rows = _read_rows(path)
    mutate(rows)
    _write_rows(path, rows)


def _resequence(rows: list[dict[str, Any]]) -> None:
    id_map: dict[str, str] = {}
    for sequence, row in enumerate(rows, 1):
        old_id = row["event_id"]
        row["sequence"] = sequence
        row["event_id"] = (
            f"{row['attempt_id']}/{row['host_id']}/{row['producer']}/"
            f"{row['process_instance_id']}/{sequence}"
        )
        id_map[old_id] = row["event_id"]
    for row in rows:
        if row["event_type"] == "observer_overhead":
            observed = row["payload"]["observed_event_id"]
            if observed in id_map:
                row["payload"]["observed_event_id"] = id_map[observed]


def _remove_domain_and_overhead(
    rows: list[dict[str, Any]], predicate: Callable[[dict[str, Any]], bool]
) -> None:
    removed_ids = {
        row["event_id"]
        for row in rows
        if row["event_type"] != "observer_overhead" and predicate(row)
    }
    rows[:] = [
        row
        for row in rows
        if row["event_id"] not in removed_ids
        and not (
            row["event_type"] == "observer_overhead"
            and row["payload"].get("observed_event_id") in removed_ids
        )
    ]
    _resequence(rows)


def _build_fixture(root: Path) -> AttemptFixture:
    path = root / ATTEMPT_ID
    _write_rows(path / "raw" / "ecs" / "events.jsonl", _server_rows())
    _write_rows(
        path / "raw" / "pi" / "events.jsonl",
        _client_rows("C1", "pi-c1", 100_000_000_000),
    )
    _write_rows(
        path / "raw" / "pi" / "resource.jsonl",
        _resource_rows("C1", "pi-c1", 100_000_000_000),
    )
    # Deliberately use an unrelated host monotonic-clock origin. Cross-host
    # comparisons would reject this otherwise-valid evidence.
    _write_rows(
        path / "raw" / "pc" / "events.jsonl",
        _client_rows("C2", "pc-c2", 9_000_000_000_000),
    )
    _write_rows(
        path / "raw" / "pc" / "resource.jsonl",
        _resource_rows("C2", "pc-c2", 9_000_000_000_000),
    )
    protocol = {
        "schema_version": 1,
        "protocol_id": "iotj_main_direction_confirmation",
        "direction": "C1/C2 -> C5",
        "active_source_clients": ["C1", "C2"],
        "confirmation_commit": COMMIT,
        "source_archive_sha256": SOURCE_SHA,
        "dataset_manifest_sha256": DATASET_SHA,
        "protocol_manifest_sha256": PROTOCOL_SHA,
        "schedule": [
            {
                "run_id": RUN_ID,
                "group_id": "B2",
                "seed": 42,
                "algorithm_config_sha256": ALGORITHM_SHA,
                "transport_status": "not_collected",
            }
        ],
    }
    return AttemptFixture(path, protocol)


@pytest.fixture
def valid_attempt(tmp_path: Path) -> AttemptFixture:
    return _build_fixture(tmp_path)


def _reason_contains(audit: dict[str, Any], text: str) -> bool:
    return any(text.lower() in reason.lower() for reason in audit["reasons"])


def test_validator_requires_exact_25_by_2_message_matrix(
    valid_attempt: AttemptFixture,
) -> None:
    audit = validate_attempt(valid_attempt.path, valid_attempt.protocol)

    assert audit["status"] == "valid", audit["reasons"]
    assert audit["counts"]["rounds"] == 25
    assert audit["counts"]["fitins"] == 50
    assert audit["counts"]["fitres"] == 50
    assert audit["resource"]["C1"]["coverage"] >= 0.95
    assert audit["resource"]["C2"]["coverage"] >= 0.95
    assert list(audit["counts"]) == sorted(audit["counts"])
    assert audit["reasons"] == sorted(audit["reasons"])
    assert sorted(audit["inputs"]) == [
        "raw/ecs/events.close.json",
        "raw/ecs/events.jsonl",
        "raw/pc/events.close.json",
        "raw/pc/events.jsonl",
        "raw/pc/resource.close.json",
        "raw/pc/resource.jsonl",
        "raw/pi/events.close.json",
        "raw/pi/events.jsonl",
        "raw/pi/resource.close.json",
        "raw/pi/resource.jsonl",
    ]
    assert all(len(digest) == 64 for digest in audit["inputs"].values())


def test_duplicate_event_id_is_invalid(valid_attempt: AttemptFixture) -> None:
    path = valid_attempt.path / "raw" / "ecs" / "events.jsonl"

    def mutate(rows: list[dict[str, Any]]) -> None:
        rows[2]["event_id"] = rows[0]["event_id"]

    _rewrite_rows(path, mutate)
    audit = validate_attempt(valid_attempt.path, valid_attempt.protocol)
    assert audit["status"] == "invalid"
    assert _reason_contains(audit, "duplicate event_id")


def test_sequence_gap_is_invalid(valid_attempt: AttemptFixture) -> None:
    path = valid_attempt.path / "raw" / "ecs" / "events.jsonl"

    def mutate(rows: list[dict[str, Any]]) -> None:
        del rows[1]

    _rewrite_rows(path, mutate)
    audit = validate_attempt(valid_attempt.path, valid_attempt.protocol)
    assert audit["status"] == "invalid"
    assert _reason_contains(audit, "sequence")


def test_common_hash_mismatch_is_invalid(valid_attempt: AttemptFixture) -> None:
    path = valid_attempt.path / "raw" / "pi" / "events.jsonl"

    def mutate(rows: list[dict[str, Any]]) -> None:
        rows[0]["dataset_manifest_sha256"] = "0" * 64

    _rewrite_rows(path, mutate)
    audit = validate_attempt(valid_attempt.path, valid_attempt.protocol)
    assert audit["status"] == "invalid"
    assert _reason_contains(audit, "dataset_manifest_sha256")


def test_missing_c2_fitres_in_round_17_is_invalid(
    valid_attempt: AttemptFixture,
) -> None:
    path = valid_attempt.path / "raw" / "ecs" / "events.jsonl"

    def mutate(rows: list[dict[str, Any]]) -> None:
        _remove_domain_and_overhead(
            rows,
            lambda row: row["event_type"] == "flower_fitres_available"
            and row["round"] == 17
            and row["client_id"] == "C2",
        )

    _rewrite_rows(path, mutate)
    audit = validate_attempt(valid_attempt.path, valid_attempt.protocol)
    assert audit["status"] == "invalid"
    assert _reason_contains(audit, "round 17")
    assert _reason_contains(audit, "C2")
    assert _reason_contains(audit, "FitRes")


@pytest.mark.parametrize(
    ("field_path", "value", "reason"),
    [
        (("downlink_audit", "application_message_bytes"), -1, "nonnegative"),
        (("downlink_audit", "application_message_bytes"), float("inf"), "non-finite"),
    ],
)
def test_invalid_application_byte_value_fails_closed(
    valid_attempt: AttemptFixture,
    field_path: tuple[str, str],
    value: float,
    reason: str,
) -> None:
    path = valid_attempt.path / "raw" / "ecs" / "events.jsonl"

    def mutate(rows: list[dict[str, Any]]) -> None:
        event = next(row for row in rows if row["event_type"] == "flower_fitins_prepared")
        event["payload"][field_path[0]][field_path[1]] = value

    _rewrite_rows(path, mutate)
    audit = validate_attempt(valid_attempt.path, valid_attempt.protocol)
    assert audit["status"] == "invalid"
    assert _reason_contains(audit, reason)


def test_negative_phase_timing_is_invalid(valid_attempt: AttemptFixture) -> None:
    path = valid_attempt.path / "raw" / "pc" / "events.jsonl"

    def mutate(rows: list[dict[str, Any]]) -> None:
        event = next(row for row in rows if row["event_type"] == "client_train_end")
        event["payload"]["client_train_core_ns"] = -1

    _rewrite_rows(path, mutate)
    audit = validate_attempt(valid_attempt.path, valid_attempt.protocol)
    assert audit["status"] == "invalid"
    assert _reason_contains(audit, "client_train_core_ns")


def test_missing_application_message_sha_is_invalid(
    valid_attempt: AttemptFixture,
) -> None:
    path = valid_attempt.path / "raw" / "ecs" / "events.jsonl"

    def mutate(rows: list[dict[str, Any]]) -> None:
        event = next(row for row in rows if row["event_type"] == "flower_fitres_available")
        del event["payload"]["uplink_audit"]["application_message_sha256"]

    _rewrite_rows(path, mutate)
    audit = validate_attempt(valid_attempt.path, valid_attempt.protocol)
    assert audit["status"] == "invalid"
    assert _reason_contains(audit, "application_message_sha256")


def test_resource_sample_absent_from_one_client_round_is_invalid(
    valid_attempt: AttemptFixture,
) -> None:
    path = valid_attempt.path / "raw" / "pi" / "resource.jsonl"

    def mutate(rows: list[dict[str, Any]]) -> None:
        samples = [row for row in rows if row["event_type"] == "resource_sample"]
        missing = samples[16]
        _remove_domain_and_overhead(rows, lambda row: row is missing)

    _rewrite_rows(path, mutate)
    audit = validate_attempt(valid_attempt.path, valid_attempt.protocol)
    assert audit["status"] == "invalid"
    assert _reason_contains(audit, "C1 round 17")
    assert _reason_contains(audit, "resource sample")
    assert audit["resource"]["C1"]["coverage"] == pytest.approx(24 / 25)


def test_resource_coverage_below_point_95_is_invalid(
    valid_attempt: AttemptFixture,
) -> None:
    path = valid_attempt.path / "raw" / "pc" / "resource.jsonl"

    def mutate(rows: list[dict[str, Any]]) -> None:
        samples = [row for row in rows if row["event_type"] == "resource_sample"]
        removed = {id(samples[0]), id(samples[1])}
        _remove_domain_and_overhead(rows, lambda row: id(row) in removed)

    _rewrite_rows(path, mutate)
    audit = validate_attempt(valid_attempt.path, valid_attempt.protocol)
    assert audit["status"] == "invalid"
    assert audit["resource"]["C2"]["coverage"] == pytest.approx(23 / 25)
    assert _reason_contains(audit, "coverage")
    assert _reason_contains(audit, "0.95")


def test_unpaired_observer_overhead_is_invalid(
    valid_attempt: AttemptFixture,
) -> None:
    path = valid_attempt.path / "raw" / "pc" / "events.jsonl"

    def mutate(rows: list[dict[str, Any]]) -> None:
        domain = next(row for row in rows if row["event_type"] == "client_train_end")
        rows[:] = [
            row
            for row in rows
            if not (
                row["event_type"] == "observer_overhead"
                and row["payload"].get("observed_event_id") == domain["event_id"]
            )
        ]
        _resequence(rows)

    _rewrite_rows(path, mutate)
    audit = validate_attempt(valid_attempt.path, valid_attempt.protocol)
    assert audit["status"] == "invalid"
    assert _reason_contains(audit, "unpaired")


@pytest.mark.parametrize("transport_value", [None, "collected", 0])
def test_transport_status_must_be_explicit_not_collected(
    valid_attempt: AttemptFixture, transport_value: object
) -> None:
    protocol = copy.deepcopy(valid_attempt.protocol)
    row = protocol["schedule"][0]
    if transport_value is None:
        row.pop("transport_status")
    else:
        row["transport_status"] = transport_value

    audit = validate_attempt(valid_attempt.path, protocol)
    assert audit["status"] == "invalid"
    assert _reason_contains(audit, "transport_status")


def test_conflicting_top_level_transport_status_is_invalid(
    valid_attempt: AttemptFixture,
) -> None:
    protocol = copy.deepcopy(valid_attempt.protocol)
    protocol["transport_status"] = "collected"
    audit = validate_attempt(valid_attempt.path, protocol)
    assert audit["status"] == "invalid"
    assert _reason_contains(audit, "transport_status")


def test_proxy_identity_is_resolved_only_within_same_round(
    valid_attempt: AttemptFixture,
) -> None:
    path = valid_attempt.path / "raw" / "ecs" / "events.jsonl"

    def mutate(rows: list[dict[str, Any]]) -> None:
        fitins = next(
            row
            for row in rows
            if row["event_type"] == "flower_fitins_prepared"
            and row["round"] == 1
            and row["payload"]["proxy_id"] == "proxy-C1-round-1"
        )
        fitins["payload"]["proxy_id"] = "proxy-C1-round-2"
        fitins["client_id"] = "proxy-C1-round-2"

    _rewrite_rows(path, mutate)
    audit = validate_attempt(valid_attempt.path, valid_attempt.protocol)
    assert audit["status"] == "invalid"
    assert _reason_contains(audit, "round 1")
    assert _reason_contains(audit, "proxy")


@pytest.mark.parametrize(
    "bad_line",
    [
        b"{not-json}\n",
        b"[]\n",
        b'{"value":NaN}\n',
        b'{"value":Infinity}\n',
    ],
)
def test_read_events_rejects_malformed_non_object_and_non_finite_json(
    tmp_path: Path, bad_line: bytes
) -> None:
    path = tmp_path / "events.jsonl"
    path.write_bytes(bad_line)
    with pytest.raises(ValueError):
        read_events(path)


def test_unknown_or_duplicate_host_evidence_is_invalid(
    valid_attempt: AttemptFixture,
) -> None:
    extra = valid_attempt.path / "raw" / "unknown-host" / "events.jsonl"
    _write_rows(extra, _client_rows("C1", "pi-c1", 0))
    duplicate = valid_attempt.path / "raw" / "ecs" / "duplicate.jsonl"
    duplicate.write_bytes(
        (valid_attempt.path / "raw" / "ecs" / "events.jsonl").read_bytes()
    )
    duplicate.with_suffix(".close.json").write_bytes(
        (valid_attempt.path / "raw" / "ecs" / "events.close.json").read_bytes()
    )

    audit = validate_attempt(valid_attempt.path, valid_attempt.protocol)
    assert audit["status"] == "invalid"
    assert _reason_contains(audit, "unknown")
    assert _reason_contains(audit, "duplicate")


def test_symlink_evidence_is_rejected(valid_attempt: AttemptFixture) -> None:
    source = valid_attempt.path / "raw" / "ecs" / "events.jsonl"
    link = valid_attempt.path / "raw" / "ecs" / "linked.jsonl"
    try:
        os.symlink(source, link)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation is unavailable")

    audit = validate_attempt(valid_attempt.path, valid_attempt.protocol)
    assert audit["status"] == "invalid"
    assert _reason_contains(audit, "symlink")


def test_close_summary_count_mismatch_is_invalid(
    valid_attempt: AttemptFixture,
) -> None:
    path = valid_attempt.path / "raw" / "ecs" / "events.close.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    summary["observer_event_count"] -= 1
    path.write_bytes(_canonical_bytes(summary) + b"\n")

    audit = validate_attempt(valid_attempt.path, valid_attempt.protocol)
    assert audit["status"] == "invalid"
    assert _reason_contains(audit, "close summary")


def test_classification_outcomes_do_not_affect_or_enter_audit(
    valid_attempt: AttemptFixture,
) -> None:
    protocol = copy.deepcopy(valid_attempt.protocol)
    protocol["classification_accuracy"] = -999
    protocol["loss"] = "do not inspect"
    protocol["ranking"] = ["bad", "good"]

    audit = validate_attempt(valid_attempt.path, protocol)
    assert audit["status"] == "valid", audit["reasons"]
    serialized = json.dumps(audit, sort_keys=True).lower()
    assert "classification_accuracy" not in serialized
    assert '"loss"' not in serialized
    assert '"ranking"' not in serialized


def test_cli_writes_canonical_audit_and_refuses_overwrite(
    valid_attempt: AttemptFixture, tmp_path: Path
) -> None:
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_bytes(_canonical_bytes(valid_attempt.protocol) + b"\n")
    output = tmp_path / "attempt_audit.json"
    command = [
        sys.executable,
        "-m",
        "scripts.validate_iotj_confirmation_attempt",
        "--attempt-dir",
        str(valid_attempt.path),
        "--protocol-manifest",
        str(protocol_path),
        "--output",
        str(output),
    ]

    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    audit = json.loads(output.read_text(encoding="utf-8"))
    assert audit["status"] == "valid"
    assert output.read_bytes() == _canonical_bytes(audit) + b"\n"
    response = json.loads(completed.stdout)
    assert response == {
        "audit_sha256": hashlib.sha256(output.read_bytes()).hexdigest()
    }

    second = subprocess.run(command, check=False, capture_output=True, text=True)
    assert second.returncode not in {0, 2}
    assert json.loads(output.read_text(encoding="utf-8")) == audit


def test_cli_writes_invalid_audit_before_exit_two(
    valid_attempt: AttemptFixture, tmp_path: Path
) -> None:
    protocol = copy.deepcopy(valid_attempt.protocol)
    protocol["schedule"][0].pop("transport_status")
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_bytes(_canonical_bytes(protocol) + b"\n")
    output = tmp_path / "attempt_audit.json"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.validate_iotj_confirmation_attempt",
            "--attempt-dir",
            str(valid_attempt.path),
            "--protocol-manifest",
            str(protocol_path),
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    audit = json.loads(output.read_text(encoding="utf-8"))
    assert audit["status"] == "invalid"
    assert _reason_contains(audit, "transport_status")
