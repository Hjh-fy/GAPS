"""Append-only confirmation observability events with bounded self-accounting."""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO, Mapping


SCHEMA_VERSION = "iotj.confirmation.observability.v1"
DURABLE_EVENT_TYPES = {
    "round_end",
    "fit_round_end",
    "attempt_end",
    "attempt_failure",
    "producer_failure",
    "resource_sampler_end",
}

_RUN_ID_PATTERN = re.compile(r"^c12_to_c5__(b2|b5)__s(42|43|44|45|46)$")


def canonical_json_bytes(value: Any) -> bytes:
    """Encode a value as compact canonical UTF-8 JSON."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _require_hex(field_name: str, value: str, length: int) -> None:
    if not isinstance(value, str) or re.fullmatch(
        rf"[0-9a-fA-F]{{{length}}}", value
    ) is None:
        raise ValueError(f"{field_name} must be exactly {length} hex characters")


@dataclass(frozen=True)
class ObserverIdentity:
    run_id: str
    attempt_id: str
    group_id: str
    training_seed: int
    client_id: str | None
    host_id: str
    producer: str
    confirmation_commit: str
    source_archive_sha256: str
    dataset_manifest_sha256: str
    algorithm_config_sha256: str

    def __post_init__(self) -> None:
        run_match = _RUN_ID_PATTERN.fullmatch(self.run_id)
        if run_match is None:
            raise ValueError(f"invalid confirmation run_id: {self.run_id!r}")
        attempt_pattern = rf"{re.escape(self.run_id)}__a\d{{3}}"
        if re.fullmatch(attempt_pattern, self.attempt_id) is None:
            raise ValueError(
                f"attempt_id must match run_id plus '__aNNN': {self.attempt_id!r}"
            )
        expected_group_id = run_match.group(1).upper()
        if self.group_id != expected_group_id:
            raise ValueError(
                f"group_id must match run_id: expected {expected_group_id!r}"
            )
        expected_training_seed = int(run_match.group(2))
        if self.training_seed != expected_training_seed:
            raise ValueError(
                "training_seed must match run_id: "
                f"expected {expected_training_seed}"
            )
        _require_hex("confirmation_commit", self.confirmation_commit, 40)
        _require_hex("source_archive_sha256", self.source_archive_sha256, 64)
        _require_hex("dataset_manifest_sha256", self.dataset_manifest_sha256, 64)
        _require_hex("algorithm_config_sha256", self.algorithm_config_sha256, 64)


@dataclass(frozen=True)
class _ObserverCost:
    flower_serialize_ns: int = 0
    event_encode_ns: int = 0
    io_write_ns: int = 0
    fsync_ns: int = 0
    event_bytes_written: int = 0
    event_count: int = 0

    @property
    def total_ns(self) -> int:
        return (
            self.flower_serialize_ns
            + self.event_encode_ns
            + self.io_write_ns
            + self.fsync_ns
        )

    def __add__(self, other: "_ObserverCost") -> "_ObserverCost":
        return _ObserverCost(
            flower_serialize_ns=self.flower_serialize_ns
            + other.flower_serialize_ns,
            event_encode_ns=self.event_encode_ns + other.event_encode_ns,
            io_write_ns=self.io_write_ns + other.io_write_ns,
            fsync_ns=self.fsync_ns + other.fsync_ns,
            event_bytes_written=self.event_bytes_written
            + other.event_bytes_written,
            event_count=self.event_count + other.event_count,
        )


@dataclass(frozen=True)
class _PendingOverhead:
    observed_event_id: str
    round_idx: int | None
    client_id: str | None
    cost: _ObserverCost


class NullObserver:
    """Observer implementation for disabled instrumentation."""

    __slots__ = ()

    def emit(
        self,
        event_type: str,
        *,
        round_idx: int | None,
        client_id: str | None,
        status: str,
        payload: Mapping[str, Any],
        flower_serialize_ns: int = 0,
    ) -> None:
        return None

    def close(self) -> None:
        return None


class JsonlObserver:
    """Write canonical JSONL events and delayed observer-cost records."""

    def __init__(self, identity: ObserverIdentity, events_path: str | os.PathLike[str]):
        self.identity = identity
        self.events_path = Path(events_path)
        self.process_instance_id = uuid.uuid4().hex
        self._sequence = 0
        self._file: BinaryIO = self.events_path.open("xb")
        self._pending: _PendingOverhead | None = None
        self._accumulated_cost = _ObserverCost()
        self._closed = False

    def emit(
        self,
        event_type: str,
        *,
        round_idx: int | None,
        client_id: str | None,
        status: str,
        payload: Mapping[str, Any],
        flower_serialize_ns: int = 0,
    ) -> str:
        if self._closed:
            raise RuntimeError("cannot emit after observer is closed")

        payload_snapshot = dict(payload)
        preflight_started = time.perf_counter_ns()
        canonical_json_bytes(payload_snapshot)
        preflight_encode_ns = time.perf_counter_ns() - preflight_started

        reporting_cost = _ObserverCost()
        if self._pending is not None:
            reporting_cost = self._write_overhead(self._pending)

        event_id, event_cost = self._write_event(
            event_type,
            round_idx=round_idx,
            client_id=client_id,
            status=status,
            payload=payload_snapshot,
            flower_serialize_ns=flower_serialize_ns,
            extra_event_encode_ns=preflight_encode_ns,
        )
        self._pending = _PendingOverhead(
            observed_event_id=event_id,
            round_idx=round_idx,
            client_id=client_id,
            cost=reporting_cost + event_cost,
        )
        return event_id

    def close(self) -> None:
        if self._closed:
            return

        reporting_tail_bytes = 0
        if self._pending is not None:
            reporting_tail_bytes = self._write_overhead(
                self._pending
            ).event_bytes_written
            self._pending = None

        self._file.close()
        self._closed = True
        self._write_close_summary(reporting_tail_bytes)

    def _write_overhead(self, pending: _PendingOverhead) -> _ObserverCost:
        payload = {
            "observed_event_id": pending.observed_event_id,
            "observer_flower_serialize_ns": pending.cost.flower_serialize_ns,
            "observer_event_encode_ns": pending.cost.event_encode_ns,
            "observer_io_write_ns": pending.cost.io_write_ns,
            "observer_fsync_ns": pending.cost.fsync_ns,
            "observer_total_ns": pending.cost.total_ns,
            "observer_event_bytes_written": pending.cost.event_bytes_written,
            "observer_event_count": pending.cost.event_count,
        }
        _, cost = self._write_event(
            "observer_overhead",
            round_idx=pending.round_idx,
            client_id=pending.client_id,
            status="succeeded",
            payload=payload,
        )
        return cost

    def _write_event(
        self,
        event_type: str,
        *,
        round_idx: int | None,
        client_id: str | None,
        status: str,
        payload: Mapping[str, Any],
        flower_serialize_ns: int = 0,
        extra_event_encode_ns: int = 0,
    ) -> tuple[str, _ObserverCost]:
        sequence = self._sequence + 1
        event_id = (
            f"{self.identity.attempt_id}/{self.identity.host_id}/"
            f"{self.identity.producer}/{self.process_instance_id}/{sequence}"
        )
        event = {
            "schema_version": SCHEMA_VERSION,
            "event_id": event_id,
            "event_type": event_type,
            "run_id": self.identity.run_id,
            "attempt_id": self.identity.attempt_id,
            "group_id": self.identity.group_id,
            "training_seed": self.identity.training_seed,
            "round": round_idx,
            "client_id": client_id,
            "host_id": self.identity.host_id,
            "producer": self.identity.producer,
            "process_instance_id": self.process_instance_id,
            "sequence": sequence,
            "wall_time_utc": datetime.now(timezone.utc)
            .isoformat(timespec="microseconds")
            .replace("+00:00", "Z"),
            "monotonic_ns": time.perf_counter_ns(),
            "confirmation_commit": self.identity.confirmation_commit,
            "source_archive_sha256": self.identity.source_archive_sha256,
            "dataset_manifest_sha256": self.identity.dataset_manifest_sha256,
            "algorithm_config_sha256": self.identity.algorithm_config_sha256,
            "status": status,
            "payload": dict(payload),
        }

        encode_started = time.perf_counter_ns()
        event_bytes = canonical_json_bytes(event) + b"\n"
        event_encode_ns = (
            extra_event_encode_ns + time.perf_counter_ns() - encode_started
        )
        self._sequence = sequence

        write_started = time.perf_counter_ns()
        self._file.write(event_bytes)
        self._file.flush()
        io_write_ns = time.perf_counter_ns() - write_started

        fsync_ns = 0
        if event_type in DURABLE_EVENT_TYPES:
            fsync_started = time.perf_counter_ns()
            os.fsync(self._file.fileno())
            fsync_ns = time.perf_counter_ns() - fsync_started

        cost = _ObserverCost(
            flower_serialize_ns=flower_serialize_ns,
            event_encode_ns=event_encode_ns,
            io_write_ns=io_write_ns,
            fsync_ns=fsync_ns,
            event_bytes_written=len(event_bytes),
            event_count=1,
        )
        self._accumulated_cost = self._accumulated_cost + cost
        return event_id, cost

    def _write_close_summary(self, reporting_tail_bytes: int) -> None:
        cost = self._accumulated_cost
        summary = {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.identity.run_id,
            "attempt_id": self.identity.attempt_id,
            "host_id": self.identity.host_id,
            "producer": self.identity.producer,
            "process_instance_id": self.process_instance_id,
            "observer_flower_serialize_ns": cost.flower_serialize_ns,
            "observer_event_encode_ns": cost.event_encode_ns,
            "observer_io_write_ns": cost.io_write_ns,
            "observer_fsync_ns": cost.fsync_ns,
            "observer_total_ns": cost.total_ns,
            "observer_event_bytes_written": cost.event_bytes_written,
            "observer_event_count": cost.event_count,
            "observer_reporting_tail_bytes": reporting_tail_bytes,
        }
        close_path = self.events_path.with_suffix(".close.json")
        with close_path.open("xb") as close_file:
            close_file.write(canonical_json_bytes(summary) + b"\n")
            close_file.flush()


def load_observer(
    context_path: str | None, events_path: str | None
) -> NullObserver | JsonlObserver:
    """Load an observer identity from JSON, or return the disabled observer."""

    if context_path is None and events_path is None:
        return NullObserver()
    if context_path is None or events_path is None:
        raise ValueError("context_path and events_path must be provided together")

    with Path(context_path).open("r", encoding="utf-8") as context_file:
        context = json.load(context_file)
    return JsonlObserver(ObserverIdentity(**context), events_path)
