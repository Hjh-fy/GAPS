"""Fail-closed structural validation for one confirmation attempt."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import uuid
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "iotj.confirmation.observability.v1"
EXPECTED_ROUNDS = tuple(range(1, 26))
EXPECTED_CLIENTS = ("C1", "C2")
RESOURCE_COVERAGE_MINIMUM = 0.95

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

CLOSE_SUMMARY_FIELDS = {
    "schema_version",
    "run_id",
    "attempt_id",
    "host_id",
    "producer",
    "process_instance_id",
    "observer_flower_serialize_ns",
    "observer_event_encode_ns",
    "observer_io_write_ns",
    "observer_fsync_ns",
    "observer_total_ns",
    "observer_event_bytes_written",
    "observer_event_count",
    "observer_reporting_tail_bytes",
}

DOWNLINK_LOGICAL_FIELDS = {
    "logical_downlink_model_value_bytes",
    "logical_downlink_parameter_blob_bytes",
    "logical_downlink_semantic_proto_utf8_bytes",
    "logical_downlink_other_config_value_bytes",
    "logical_downlink_total_bytes",
}

UPLINK_LOGICAL_FIELDS = {
    "logical_uplink_model_value_bytes",
    "logical_uplink_parameter_blob_bytes",
    "logical_uplink_prototype_utf8_bytes",
    "logical_uplink_prototype_var_utf8_bytes",
    "logical_uplink_statistics_utf8_bytes",
    "logical_uplink_diagnostic_value_bytes",
    "logical_uplink_total_bytes",
}

RESOURCE_NUMERIC_FIELDS = {
    "root_pid",
    "sampler_pid_excluded",
    "rss_tree_bytes",
    "rss_tree_peak_bytes",
    "process_count_tree",
    "thread_count_tree",
    "cpu_time_tree_seconds",
    "cpu_time_tree_delta_seconds",
    "cpu_percent_tree_one_core_scale",
    "cpu_percent_tree_host_scale",
    "logical_cpu_count",
    "sample_interval_start_monotonic_ns",
    "sample_interval_end_monotonic_ns",
    "sample_interval_wall_ns",
}

SAMPLER_END_NUMERIC_FIELDS = {
    "root_pid",
    "sampler_pid",
    "sample_count",
    "sampler_cpu_user_seconds",
    "sampler_cpu_system_seconds",
    "observer_event_encode_ns",
    "observer_io_write_ns",
    "observer_fsync_ns",
    "observer_event_bytes_written",
    "observer_event_count",
}

SERVER_EVENT_TYPES = {
    "fit_round_start",
    "flower_fitins_prepared",
    "flower_fitres_available",
    "server_aggregate_start",
    "server_da_start",
    "server_da_end",
    "server_aggregate_end",
    "fit_round_end",
    "observer_overhead",
    "producer_failure",
}
CLIENT_EVENT_TYPES = {
    "client_fit_start",
    "client_train_start",
    "client_train_end",
    "client_fit_end",
    "observer_overhead",
    "producer_failure",
}
SAMPLER_EVENT_TYPES = {
    "resource_sample",
    "resource_sampler_end",
    "observer_overhead",
    "producer_failure",
}

_EXPECTED_EVIDENCE: dict[str, tuple[str, str, str | None]] = {
    "raw/ecs/events.jsonl": ("ecs", "server", None),
    "raw/pi/events.jsonl": ("pi-c1", "client", "C1"),
    "raw/pi/resource.jsonl": ("pi-c1", "resource_sampler", "C1"),
    "raw/pc/events.jsonl": ("pc-c2", "client", "C2"),
    "raw/pc/resource.jsonl": ("pc-c2", "resource_sampler", "C2"),
}

_RUN_RE = re.compile(r"^c12_to_c5__(b2|b5)__s(42|43|44|45|46)$")
_ATTEMPT_RE = re.compile(
    r"^(c12_to_c5__(?:b2|b5)__s(?:42|43|44|45|46))__a\d{3}$"
)
_COMMIT_RE = re.compile(r"^[0-9a-fA-F]{40}$")
_HASH_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_WALL_TIME_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$"
)


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reject_json_constant(raw: str) -> Any:
    raise ValueError(f"non-finite JSON constant {raw}")


def _parse_finite_float(raw: str) -> float:
    value = float(raw)
    if not math.isfinite(value):
        raise ValueError(f"non-finite JSON number {raw}")
    return value


def _object_without_duplicate_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _decode_json(raw: str, *, label: str) -> Any:
    try:
        return json.loads(
            raw,
            parse_constant=_reject_json_constant,
            parse_float=_parse_finite_float,
            object_pairs_hook=_object_without_duplicate_keys,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{label}: {exc}") from exc


def _is_regular_file_without_symlink(path: Path) -> bool:
    return path.is_file() and not path.is_symlink()


def read_events(path: Path) -> list[dict[str, Any]]:
    """Read strict object-only, finite-number JSONL from a regular file."""

    path = Path(path)
    if path.is_symlink():
        raise ValueError(f"event file is a symlink: {path}")
    if not path.is_file():
        raise ValueError(f"event file is not a regular file: {path}")
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"cannot read event file {path}: {exc}") from exc
    lines = text.splitlines()
    if not lines:
        raise ValueError(f"event file is empty: {path}")
    events: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            raise ValueError(f"{path}:{line_number}: blank JSONL line")
        value = _decode_json(line, label=f"{path}:{line_number}")
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: event must be a JSON object")
        events.append(value)
    return events


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    path = Path(path)
    if path.is_symlink():
        raise ValueError(f"{label} is a symlink: {path}")
    if not path.is_file():
        raise ValueError(f"{label} is not a regular file: {path}")
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"cannot read {label}: {exc}") from exc
    value = _decode_json(raw, label=label)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_finite_nonnegative_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and value >= 0
    )


def _number_reason(label: str, value: Any) -> str | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return f"{label} must be a finite nonnegative number"
    if not math.isfinite(float(value)):
        return f"{label} is non-finite"
    if value < 0:
        return f"{label} must be nonnegative"
    return None


def _sorted_unique(reasons: Sequence[str]) -> list[str]:
    return sorted(set(reasons))


def _semantic_measurement_reasons(value: Any, label: str) -> list[str]:
    """Validate numeric byte/timing values without judging unrelated metrics."""

    reasons: list[str] = []
    if isinstance(value, Mapping):
        for key in sorted(value):
            child = value[key]
            child_label = f"{label}.{key}"
            normalized_key = str(key).lower()
            exact_numeric_semantics = normalized_key.endswith(
                ("_bytes", "_ns")
            )
            broad_numeric_semantics = (
                "byte" in normalized_key or "timing" in normalized_key
            )
            if exact_numeric_semantics and child is not None:
                reason = _number_reason(child_label, child)
                if reason:
                    reasons.append(reason)
            elif broad_numeric_semantics and isinstance(child, (int, float)):
                if not isinstance(child, bool):
                    reason = _number_reason(child_label, child)
                    if reason:
                        reasons.append(reason)
            reasons.extend(_semantic_measurement_reasons(child, child_label))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            child_label = f"{label}[{index}]"
            reasons.extend(_semantic_measurement_reasons(child, child_label))
    return reasons


def _protocol_context(
    protocol: Mapping[str, Any], run_hint: str | None = None
) -> tuple[dict[str, Any], list[str]]:
    reasons: list[str] = []
    if not isinstance(protocol, Mapping):
        return {}, ["protocol manifest must be an object"]

    commit = protocol.get("confirmation_commit")
    source_sha = protocol.get("source_archive_sha256")
    dataset_sha = protocol.get("dataset_manifest_sha256")
    protocol_sha = protocol.get("protocol_manifest_sha256")
    if not isinstance(commit, str) or _COMMIT_RE.fullmatch(commit) is None:
        reasons.append("protocol confirmation_commit must be 40 hex characters")
    for field, value in (
        ("source_archive_sha256", source_sha),
        ("dataset_manifest_sha256", dataset_sha),
        ("protocol_manifest_sha256", protocol_sha),
    ):
        if not isinstance(value, str) or _HASH_RE.fullmatch(value) is None:
            reasons.append(f"protocol {field} must be 64 hex characters")

    schedule = protocol.get("schedule")
    if not isinstance(schedule, list) or any(
        not isinstance(row, Mapping) for row in schedule
    ):
        reasons.append("protocol schedule must be a list of objects")
        rows: list[Mapping[str, Any]] = []
    else:
        rows = list(schedule)

    if run_hint is None:
        candidate_ids = [row.get("run_id") for row in rows]
        run_hint = candidate_ids[0] if len(candidate_ids) == 1 else None
    matching = [row for row in rows if row.get("run_id") == run_hint]
    if run_hint is None or _RUN_RE.fullmatch(str(run_hint)) is None:
        reasons.append("current run_id is missing or outside the confirmation allowlist")
    if len(matching) != 1:
        reasons.append(
            f"protocol schedule must contain exactly one row for current run {run_hint!r}"
        )
        row: Mapping[str, Any] = {}
    else:
        row = matching[0]

    run_match = _RUN_RE.fullmatch(str(run_hint)) if run_hint is not None else None
    expected_group = run_match.group(1).upper() if run_match else None
    expected_seed = int(run_match.group(2)) if run_match else None
    if row:
        if row.get("group_id") != expected_group:
            reasons.append("protocol schedule group_id does not match run_id")
        if row.get("seed") != expected_seed:
            reasons.append("protocol schedule seed does not match run_id")
    algorithm_sha = row.get("algorithm_config_sha256")
    if not isinstance(algorithm_sha, str) or _HASH_RE.fullmatch(algorithm_sha) is None:
        reasons.append(
            "protocol schedule algorithm_config_sha256 must be 64 hex characters"
        )

    transport_values: list[tuple[str, Any]] = []
    if "transport_status" in protocol:
        transport_values.append(("protocol", protocol.get("transport_status")))
    if "transport_status" in row:
        transport_values.append(("schedule", row.get("transport_status")))
    if not transport_values:
        reasons.append("transport_status must be explicitly not_collected")
    else:
        distinct = {repr(value) for _, value in transport_values}
        if len(distinct) > 1:
            reasons.append("conflicting transport_status declarations")
        for location, value in transport_values:
            if value != "not_collected":
                reasons.append(
                    f"{location} transport_status must be explicitly not_collected"
                )

    active_clients = protocol.get("active_source_clients")
    if active_clients is not None and active_clients != ["C1", "C2"]:
        reasons.append("protocol active_source_clients must be exactly C1 and C2")

    context = {
        "run_id": run_hint,
        "group_id": expected_group,
        "training_seed": expected_seed,
        "confirmation_commit": commit,
        "source_archive_sha256": source_sha,
        "dataset_manifest_sha256": dataset_sha,
        "algorithm_config_sha256": algorithm_sha,
        "protocol_manifest_sha256": protocol_sha,
    }
    return context, _sorted_unique(reasons)


def _valid_wall_time(value: Any) -> bool:
    if not isinstance(value, str) or _WALL_TIME_RE.fullmatch(value) is None:
        return False
    try:
        datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError:
        return False
    return True


def validate_common_fields(
    events: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]
) -> list[str]:
    """Validate exact common schema, provenance, IDs and process sequences."""

    reasons: list[str] = []
    run_values = {
        event.get("run_id")
        for event in events
        if isinstance(event, Mapping) and isinstance(event.get("run_id"), str)
    }
    run_hint = next(iter(run_values)) if len(run_values) == 1 else None
    context, protocol_reasons = _protocol_context(protocol, run_hint)
    reasons.extend(protocol_reasons)
    if not events:
        reasons.append("attempt contains no readable events")
        return _sorted_unique(reasons)
    if len(run_values) != 1:
        reasons.append("events do not have one common run_id")

    seen_event_ids: set[str] = set()
    sequences: dict[tuple[Any, Any, Any], list[int]] = defaultdict(list)
    attempts: set[Any] = set()

    for index, event in enumerate(events, 1):
        label = f"event[{index}]"
        if not isinstance(event, Mapping):
            reasons.append(f"{label} is not an object")
            continue
        keys = set(event)
        if keys != COMMON_FIELDS:
            missing = sorted(COMMON_FIELDS - keys)
            unknown = sorted(keys - COMMON_FIELDS)
            reasons.append(
                f"{label} common fields are not exact; missing={missing}, unknown={unknown}"
            )
        if event.get("schema_version") != SCHEMA_VERSION:
            reasons.append(f"{label} schema_version mismatch")
        if not isinstance(event.get("payload"), Mapping):
            reasons.append(f"{label} payload must be an object")
        else:
            reasons.extend(
                _semantic_measurement_reasons(
                    event["payload"], f"{event.get('event_id')} payload"
                )
            )

        for field in (
            "run_id",
            "group_id",
            "training_seed",
            "confirmation_commit",
            "source_archive_sha256",
            "dataset_manifest_sha256",
            "algorithm_config_sha256",
        ):
            if event.get(field) != context.get(field):
                reasons.append(f"{label} {field} does not match protocol")

        attempt_id = event.get("attempt_id")
        attempts.add(attempt_id)
        attempt_match = (
            _ATTEMPT_RE.fullmatch(attempt_id)
            if isinstance(attempt_id, str)
            else None
        )
        if attempt_match is None or attempt_match.group(1) != event.get("run_id"):
            reasons.append(f"{label} attempt_id does not match run_id plus __aNNN")

        event_type = event.get("event_type")
        host_id = event.get("host_id")
        producer = event.get("producer")
        process_id = event.get("process_instance_id")
        if not isinstance(event_type, str) or not event_type:
            reasons.append(f"{label} event_type must be a non-empty string")
        for field, value in (
            ("host_id", host_id),
            ("producer", producer),
            ("process_instance_id", process_id),
        ):
            if not isinstance(value, str) or not value:
                reasons.append(f"{label} {field} must be a non-empty string")

        sequence = event.get("sequence")
        if not _is_int(sequence) or sequence < 1:
            reasons.append(f"{label} sequence must be a positive integer")
        else:
            sequences[(host_id, producer, process_id)].append(sequence)

        event_id = event.get("event_id")
        if isinstance(event_id, str):
            if event_id in seen_event_ids:
                reasons.append(f"duplicate event_id {event_id}")
            seen_event_ids.add(event_id)
        else:
            reasons.append(f"{label} event_id must be a string")
        if _is_int(sequence):
            expected_event_id = (
                f"{attempt_id}/{host_id}/{producer}/{process_id}/{sequence}"
            )
            if event_id != expected_event_id:
                reasons.append(f"{label} event_id does not match common identity")

        round_idx = event.get("round")
        if round_idx is not None and (
            not _is_int(round_idx) or round_idx not in EXPECTED_ROUNDS
        ):
            reasons.append(f"{label} round must be 1..25 or null")
        if not _valid_wall_time(event.get("wall_time_utc")):
            reasons.append(f"{label} wall_time_utc must be RFC 3339 UTC")
        monotonic_ns = event.get("monotonic_ns")
        if not _is_int(monotonic_ns) or monotonic_ns < 0:
            reasons.append(f"{label} monotonic_ns must be a nonnegative integer")
        if event.get("status") not in {"started", "succeeded", "failed", "aborted"}:
            reasons.append(f"{label} status is invalid")
        if event.get("status") in {"failed", "aborted"} or event_type == "producer_failure":
            reasons.append(f"{label} records producer failure or abort")

    if len(attempts) != 1:
        reasons.append("events do not have one common attempt_id")
    for key, actual in sequences.items():
        expected = list(range(1, len(actual) + 1))
        if actual != expected:
            reasons.append(
                f"sequence for host/producer/process {key!r} is not contiguous "
                "from 1 in file order"
            )
    return _sorted_unique(reasons)


def resolve_proxy_clients(
    events: Sequence[Mapping[str, Any]],
) -> dict[tuple[int, str], str]:
    """Resolve a server proxy only from a same-round C1/C2 FitRes event."""

    candidates: dict[tuple[int, str], set[str]] = defaultdict(set)
    for event in events:
        if event.get("event_type") != "flower_fitres_available":
            continue
        round_idx = event.get("round")
        client_id = event.get("client_id")
        payload = event.get("payload")
        proxy_id = payload.get("proxy_id") if isinstance(payload, Mapping) else None
        if (
            _is_int(round_idx)
            and isinstance(proxy_id, str)
            and proxy_id
            and client_id in EXPECTED_CLIENTS
        ):
            candidates[(round_idx, proxy_id)].add(client_id)
    return {
        key: next(iter(client_ids))
        for key, client_ids in candidates.items()
        if len(client_ids) == 1
    }


def _validate_application_audit(
    event: Mapping[str, Any], direction: str
) -> list[str]:
    reasons: list[str] = []
    event_id = str(event.get("event_id"))
    payload = event.get("payload")
    if not isinstance(payload, Mapping):
        return [f"{event_id} message payload must be an object"]
    audit_key = "downlink_audit" if direction == "downlink" else "uplink_audit"
    audit = payload.get(audit_key)
    if not isinstance(audit, Mapping):
        return [f"{event_id} {audit_key} must be an object"]
    required_audit_fields = {
        "logical",
        "application_message_bytes",
        "application_message_sha256",
    }
    if set(audit) != required_audit_fields:
        reasons.append(f"{event_id} {audit_key} fields are not exact")
    logical = audit.get("logical")
    expected_logical = (
        DOWNLINK_LOGICAL_FIELDS if direction == "downlink" else UPLINK_LOGICAL_FIELDS
    )
    if not isinstance(logical, Mapping):
        reasons.append(f"{event_id} {audit_key}.logical must be an object")
    else:
        if set(logical) != expected_logical:
            reasons.append(f"{event_id} {audit_key}.logical fields are not exact")
        for field in sorted(expected_logical):
            reason = _number_reason(
                f"{event_id} {field}", logical.get(field)
            )
            if reason is not None:
                reasons.append(reason)
        if all(_is_finite_nonnegative_number(logical.get(field)) for field in expected_logical):
            if direction == "downlink":
                expected_total = sum(
                    logical[field]
                    for field in (
                        "logical_downlink_parameter_blob_bytes",
                        "logical_downlink_semantic_proto_utf8_bytes",
                        "logical_downlink_other_config_value_bytes",
                    )
                )
                actual_total = logical["logical_downlink_total_bytes"]
            else:
                expected_total = sum(
                    logical[field]
                    for field in (
                        "logical_uplink_parameter_blob_bytes",
                        "logical_uplink_prototype_utf8_bytes",
                        "logical_uplink_prototype_var_utf8_bytes",
                        "logical_uplink_statistics_utf8_bytes",
                        "logical_uplink_diagnostic_value_bytes",
                    )
                )
                actual_total = logical["logical_uplink_total_bytes"]
            if actual_total != expected_total:
                reasons.append(f"{event_id} logical application byte total mismatch")
    application_bytes = audit.get("application_message_bytes")
    reason = _number_reason(
        f"{event_id} application_message_bytes", application_bytes
    )
    if reason is not None:
        reasons.append(reason)
    elif not _is_int(application_bytes):
        reasons.append(f"{event_id} application_message_bytes must be an integer")
    application_sha = audit.get("application_message_sha256")
    if not isinstance(application_sha, str) or _HASH_RE.fullmatch(application_sha) is None:
        reasons.append(f"{event_id} application_message_sha256 must be 64 hex characters")
    return reasons


def validate_message_matrix(
    events: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, int], list[str]]:
    """Require the exact 25-round by C1/C2 FitIns/FitRes matrix."""

    reasons: list[str] = []
    proxy_clients = resolve_proxy_clients(events)
    fitins_by_key: Counter[tuple[int, str]] = Counter()
    fitres_by_key: Counter[tuple[int, str]] = Counter()
    round_starts = [event for event in events if event.get("event_type") == "fit_round_start"]
    fitins = [event for event in events if event.get("event_type") == "flower_fitins_prepared"]
    fitres = [event for event in events if event.get("event_type") == "flower_fitres_available"]

    fitres_proxy_clients: dict[tuple[int, str], set[str]] = defaultdict(set)
    for event in fitres:
        round_idx = event.get("round")
        client_id = event.get("client_id")
        payload = event.get("payload")
        proxy_id = payload.get("proxy_id") if isinstance(payload, Mapping) else None
        if not _is_int(round_idx) or round_idx not in EXPECTED_ROUNDS:
            reasons.append(f"FitRes {event.get('event_id')} has invalid round")
        elif client_id not in EXPECTED_CLIENTS:
            reasons.append(
                f"round {round_idx} FitRes has invalid client_id {client_id!r}"
            )
        else:
            fitres_by_key[(round_idx, client_id)] += 1
        if not isinstance(proxy_id, str) or not proxy_id:
            reasons.append(f"round {round_idx} {client_id} FitRes proxy_id is missing")
        elif _is_int(round_idx) and client_id in EXPECTED_CLIENTS:
            fitres_proxy_clients[(round_idx, proxy_id)].add(client_id)
        reasons.extend(_validate_application_audit(event, "uplink"))

    for key, client_ids in fitres_proxy_clients.items():
        if len(client_ids) != 1:
            reasons.append(
                f"round {key[0]} proxy {key[1]!r} maps to multiple clients"
            )

    for event in fitins:
        round_idx = event.get("round")
        payload = event.get("payload")
        proxy_id = payload.get("proxy_id") if isinstance(payload, Mapping) else None
        if event.get("client_id") != proxy_id:
            reasons.append(
                f"round {round_idx} FitIns client_id does not equal proxy_id"
            )
        resolved = (
            proxy_clients.get((round_idx, proxy_id))
            if _is_int(round_idx) and isinstance(proxy_id, str)
            else None
        )
        if resolved is None:
            reasons.append(
                f"round {round_idx} FitIns proxy {proxy_id!r} does not resolve "
                "within the same round"
            )
        else:
            fitins_by_key[(round_idx, resolved)] += 1
        reasons.extend(_validate_application_audit(event, "downlink"))

    for round_idx in EXPECTED_ROUNDS:
        start_count = sum(1 for event in round_starts if event.get("round") == round_idx)
        if start_count != 1:
            reasons.append(
                f"round {round_idx} requires exactly one fit_round_start; got {start_count}"
            )
        for client_id in EXPECTED_CLIENTS:
            ins_count = fitins_by_key[(round_idx, client_id)]
            res_count = fitres_by_key[(round_idx, client_id)]
            if ins_count != 1:
                reasons.append(
                    f"round {round_idx} {client_id} requires exactly one FitIns; got {ins_count}"
                )
            if res_count != 1:
                reasons.append(
                    f"round {round_idx} {client_id} requires exactly one FitRes; got {res_count}"
                )

    counts = {
        "fitins": len(fitins),
        "fitres": len(fitres),
        "rounds": len({event.get("round") for event in round_starts}),
    }
    return dict(sorted(counts.items())), _sorted_unique(reasons)


def _events_for_key(
    events: Sequence[Mapping[str, Any]], event_type: str, round_idx: int, client_id: str | None
) -> list[Mapping[str, Any]]:
    return [
        event
        for event in events
        if event.get("event_type") == event_type
        and event.get("round") == round_idx
        and event.get("client_id") == client_id
    ]


def _one_phase_event(
    events: Sequence[Mapping[str, Any]],
    event_type: str,
    round_idx: int,
    client_id: str | None,
    reasons: list[str],
) -> Mapping[str, Any] | None:
    matches = _events_for_key(events, event_type, round_idx, client_id)
    label = f"round {round_idx} {client_id or 'server'} {event_type}"
    if len(matches) != 1:
        reasons.append(f"{label} requires exactly one event; got {len(matches)}")
        return None
    return matches[0]


def _monotonic_value(event: Mapping[str, Any] | None) -> int | None:
    if event is None:
        return None
    value = event.get("monotonic_ns")
    return value if _is_int(value) and value >= 0 else None


def validate_phase_times(events: Sequence[Mapping[str, Any]]) -> list[str]:
    """Validate phase cardinality and within-process monotonic timing values."""

    reasons: list[str] = []
    for event in events:
        if event.get("event_type") in {
            "observer_overhead",
            "resource_sample",
            "resource_sampler_end",
        }:
            continue
        payload = event.get("payload")
        if not isinstance(payload, Mapping):
            continue
        for field, value in payload.items():
            if isinstance(field, str) and field.endswith("_ns"):
                reason = _number_reason(
                    f"{event.get('event_id')} {field}", value
                )
                if reason:
                    reasons.append(reason)
    for round_idx in EXPECTED_ROUNDS:
        for client_id in EXPECTED_CLIENTS:
            phases = [
                _one_phase_event(events, event_type, round_idx, client_id, reasons)
                for event_type in (
                    "client_fit_start",
                    "client_train_start",
                    "client_train_end",
                    "client_fit_end",
                )
            ]
            times = [_monotonic_value(event) for event in phases]
            if all(value is not None for value in times) and times != sorted(times):
                reasons.append(
                    f"round {round_idx} {client_id} client phase monotonic order is invalid"
                )
            train_end = phases[2]
            fit_end = phases[3]
            if train_end is not None:
                payload = train_end.get("payload")
                value = payload.get("client_train_core_ns") if isinstance(payload, Mapping) else None
                reason = _number_reason(
                    f"round {round_idx} {client_id} client_train_core_ns", value
                )
                if reason:
                    reasons.append(reason)
            if fit_end is not None:
                payload = fit_end.get("payload")
                value = payload.get("client_fit_callback_ns") if isinstance(payload, Mapping) else None
                reason = _number_reason(
                    f"round {round_idx} {client_id} client_fit_callback_ns", value
                )
                if reason:
                    reasons.append(reason)

        server_events = [
            _one_phase_event(events, event_type, round_idx, None, reasons)
            for event_type in (
                "fit_round_start",
                "server_aggregate_start",
                "server_aggregate_end",
                "fit_round_end",
            )
        ]
        server_times = [_monotonic_value(event) for event in server_events]
        if all(value is not None for value in server_times) and server_times != sorted(server_times):
            reasons.append(f"round {round_idx} server phase monotonic order is invalid")

        aggregate_end = server_events[2]
        round_end = server_events[3]
        timing_payload: Mapping[str, Any] | None = None
        if aggregate_end is not None and isinstance(aggregate_end.get("payload"), Mapping):
            timing_payload = aggregate_end["payload"]
        required = (
            "server_aggregate_fit_total_ns",
            "server_da_total_ns",
            "server_aggregate_non_da_ns",
        )
        if timing_payload is not None:
            for field in required:
                reason = _number_reason(
                    f"round {round_idx} {field}", timing_payload.get(field)
                )
                if reason:
                    reasons.append(reason)
            if all(_is_finite_nonnegative_number(timing_payload.get(field)) for field in required):
                if timing_payload["server_aggregate_fit_total_ns"] < timing_payload["server_da_total_ns"]:
                    reasons.append(f"round {round_idx} server DA time exceeds aggregate time")
                if timing_payload["server_aggregate_non_da_ns"] != (
                    timing_payload["server_aggregate_fit_total_ns"]
                    - timing_payload["server_da_total_ns"]
                ):
                    reasons.append(f"round {round_idx} server non-DA timing mismatch")
            da_executed = timing_payload.get("da_executed")
            if not isinstance(da_executed, bool):
                reasons.append(f"round {round_idx} da_executed must be boolean")
            da_starts = _events_for_key(events, "server_da_start", round_idx, None)
            da_ends = _events_for_key(events, "server_da_end", round_idx, None)
            expected_da_events = 1 if da_executed is True else 0
            if len(da_starts) != expected_da_events or len(da_ends) != expected_da_events:
                reasons.append(f"round {round_idx} DA event cardinality mismatches da_executed")
            if da_executed is False and timing_payload.get("server_da_total_ns") != 0:
                reasons.append(f"round {round_idx} unexecuted DA time must be zero")

        if round_end is not None and isinstance(round_end.get("payload"), Mapping):
            round_payload = round_end["payload"]
            wall = round_payload.get("fit_round_wall_ns")
            reason = _number_reason(f"round {round_idx} fit_round_wall_ns", wall)
            if reason:
                reasons.append(reason)
            for field in (*required, "da_executed"):
                if round_payload.get(field) != (
                    timing_payload.get(field) if timing_payload is not None else None
                ):
                    reasons.append(
                        f"round {round_idx} fit_round_end timing does not match aggregate end"
                    )
                    break
            if (
                _is_finite_nonnegative_number(wall)
                and timing_payload is not None
                and _is_finite_nonnegative_number(
                    timing_payload.get("server_aggregate_fit_total_ns")
                )
                and wall < timing_payload["server_aggregate_fit_total_ns"]
            ):
                reasons.append(f"round {round_idx} fit round wall is below aggregate time")
    return _sorted_unique(reasons)


def _numeric_payload_reasons(
    payload: Mapping[str, Any], fields: set[str], label: str
) -> list[str]:
    reasons: list[str] = []
    for field in sorted(fields):
        reason = _number_reason(f"{label} {field}", payload.get(field))
        if reason:
            reasons.append(reason)
    return reasons


def _recursive_resource_number_reasons(value: Any, label: str) -> list[str]:
    reasons: list[str] = []
    if isinstance(value, Mapping):
        for key in sorted(value):
            reasons.extend(
                _recursive_resource_number_reasons(value[key], f"{label}.{key}")
            )
    elif isinstance(value, list):
        for index, item in enumerate(value):
            reasons.extend(
                _recursive_resource_number_reasons(item, f"{label}[{index}]")
            )
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        reason = _number_reason(label, value)
        if reason:
            reasons.append(reason)
    return reasons


def validate_resource_coverage(
    events_by_host: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[dict[str, Any], list[str]]:
    """Join client fits to sampler intervals only on their own host clock."""

    reasons: list[str] = []
    result: dict[str, Any] = {}

    for client_id in EXPECTED_CLIENTS:
        hosts = {
            host
            for host, events in events_by_host.items()
            if any(
                event.get("producer") == "client"
                and event.get("client_id") == client_id
                for event in events
            )
        }
        if len(hosts) != 1:
            reasons.append(f"{client_id} must have exactly one client/resource host")
            host_events: Sequence[Mapping[str, Any]] = []
        else:
            host_events = events_by_host[next(iter(hosts))]

        samples = [
            event
            for event in host_events
            if event.get("event_type") == "resource_sample"
            and event.get("client_id") == client_id
        ]
        valid_intervals: list[tuple[int, int, int]] = []
        for sample_index, sample in enumerate(samples):
            reason_count_before_sample = len(reasons)
            label = f"{client_id} resource sample {sample.get('event_id')}"
            if sample.get("round") is not None:
                reasons.append(f"{label} round must be null")
            if sample.get("status") != "succeeded":
                reasons.append(f"{label} status must be succeeded")
            payload = sample.get("payload")
            if not isinstance(payload, Mapping):
                reasons.append(f"{label} payload must be an object")
                continue
            reasons.extend(_recursive_resource_number_reasons(payload, label))
            reasons.extend(
                _numeric_payload_reasons(
                    payload, RESOURCE_NUMERIC_FIELDS, label
                )
            )
            for field in (
                "root_pid",
                "sampler_pid_excluded",
                "process_count_tree",
                "thread_count_tree",
                "logical_cpu_count",
                "sample_interval_start_monotonic_ns",
                "sample_interval_end_monotonic_ns",
                "sample_interval_wall_ns",
            ):
                if field in payload and not _is_int(payload.get(field)):
                    reasons.append(f"{label} {field} must be an integer")
            start = payload.get("sample_interval_start_monotonic_ns")
            end = payload.get("sample_interval_end_monotonic_ns")
            wall = payload.get("sample_interval_wall_ns")
            candidate_interval: tuple[int, int, int] | None = None
            if _is_int(start) and _is_int(end) and start >= 0 and end >= start:
                if wall != end - start:
                    reasons.append(f"{label} sample interval wall mismatch")
                candidate_interval = (sample_index, start, end)
            else:
                reasons.append(f"{label} sample interval is invalid")
            event_clock = sample.get("monotonic_ns")
            if _is_int(event_clock) and _is_int(end) and event_clock < end:
                reasons.append(f"{label} event clock precedes sample interval end")
            pids = payload.get("pids")
            if not isinstance(pids, list) or any(not _is_int(pid) or pid < 0 for pid in pids):
                reasons.append(f"{label} pids must be nonnegative integers")
            elif len(pids) != len(set(pids)):
                reasons.append(f"{label} pids must be unique")
            else:
                if payload.get("sampler_pid_excluded") in pids:
                    reasons.append(f"{label} sampler PID is included in target tree")
                if payload.get("process_count_tree") != len(pids):
                    reasons.append(f"{label} process_count_tree does not match pids")
            sample_errors = payload.get("sample_errors")
            if not isinstance(sample_errors, list):
                reasons.append(f"{label} sample_errors must be a list")
            elif sample_errors:
                reasons.append(f"{label} contains resource sampling errors")
            if (
                candidate_interval is not None
                and len(reasons) == reason_count_before_sample
            ):
                valid_intervals.append(candidate_interval)

        sampler_ends = [
            event
            for event in host_events
            if event.get("event_type") == "resource_sampler_end"
            and event.get("client_id") == client_id
        ]
        if len(sampler_ends) != 1:
            reasons.append(
                f"{client_id} requires exactly one resource_sampler_end; got {len(sampler_ends)}"
            )
        else:
            end_event = sampler_ends[0]
            if end_event.get("round") is not None:
                reasons.append(f"{client_id} resource_sampler_end round must be null")
            if end_event.get("status") != "succeeded":
                reasons.append(f"{client_id} resource sampler did not succeed")
            payload = end_event.get("payload")
            if not isinstance(payload, Mapping):
                reasons.append(f"{client_id} resource_sampler_end payload must be an object")
            else:
                reasons.extend(
                    _recursive_resource_number_reasons(
                        payload, f"{client_id} sampler end"
                    )
                )
                reasons.extend(
                    _numeric_payload_reasons(
                        payload, SAMPLER_END_NUMERIC_FIELDS, f"{client_id} sampler end"
                    )
                )
                peak = payload.get("sampler_rss_peak_bytes")
                if peak is not None:
                    reason = _number_reason(f"{client_id} sampler_rss_peak_bytes", peak)
                    if reason:
                        reasons.append(reason)
                if payload.get("sample_count") != len(samples):
                    reasons.append(
                        f"{client_id} sampler sample_count does not match resource events"
                    )

        covered_rounds: list[int] = []
        active_intervals: list[tuple[int, int, int]] = []
        expected_sample_points = 0
        for round_idx in EXPECTED_ROUNDS:
            starts = _events_for_key(
                host_events, "client_fit_start", round_idx, client_id
            )
            ends = _events_for_key(host_events, "client_fit_end", round_idx, client_id)
            covered = False
            valid_fit_interval = False
            if len(starts) == 1 and len(ends) == 1:
                fit_start = _monotonic_value(starts[0])
                fit_end = _monotonic_value(ends[0])
                if fit_start is not None and fit_end is not None and fit_start <= fit_end:
                    valid_fit_interval = True
                    duration_ns = fit_end - fit_start
                    expected_sample_points += max(
                        1, (duration_ns + 1_000_000_000 - 1) // 1_000_000_000
                    )
                    active_intervals.append((round_idx, fit_start, fit_end))
                    covered = any(
                        sample_start <= fit_end and sample_end >= fit_start
                        for _sample_index, sample_start, sample_end in valid_intervals
                    )
            if not valid_fit_interval:
                expected_sample_points += 1
            if covered:
                covered_rounds.append(round_idx)
            else:
                reasons.append(
                    f"{client_id} round {round_idx} has no overlapping resource sample"
                )
        covered_bins: set[tuple[int, int]] = set()
        for _sample_index, _sample_start, sample_end in valid_intervals:
            for round_idx, fit_start, fit_end in active_intervals:
                if not fit_start <= sample_end <= fit_end:
                    continue
                endpoint_offset_ns = sample_end - fit_start
                bin_index = (
                    0
                    if endpoint_offset_ns == 0
                    else (endpoint_offset_ns - 1) // 1_000_000_000
                )
                covered_bins.add((round_idx, bin_index))
                break
        covered_sample_points = len(covered_bins)
        coverage = (
            min(covered_sample_points / expected_sample_points, 1.0)
            if expected_sample_points > 0
            else 0.0
        )
        if coverage < RESOURCE_COVERAGE_MINIMUM:
            reasons.append(
                f"{client_id} resource coverage {coverage:.6f} is below 0.95"
            )
        result[client_id] = {
            "coverage": coverage,
            "covered_sample_points": covered_sample_points,
            "covered_rounds": len(covered_rounds),
            "expected_rounds": len(EXPECTED_ROUNDS),
            "expected_sample_points": expected_sample_points,
            "sample_count": len(samples),
        }
    return {key: result[key] for key in sorted(result)}, _sorted_unique(reasons)


def validate_observer_overhead(events: Sequence[Mapping[str, Any]]) -> list[str]:
    """Require one and only one valid overhead record per observed event."""

    reasons: list[str] = []
    domain_by_id = {
        event.get("event_id"): event
        for event in events
        if event.get("event_type") != "observer_overhead"
        and isinstance(event.get("event_id"), str)
    }
    reference_counts: Counter[str] = Counter()
    for overhead in events:
        if overhead.get("event_type") != "observer_overhead":
            continue
        overhead_id = str(overhead.get("event_id"))
        payload = overhead.get("payload")
        if not isinstance(payload, Mapping):
            reasons.append(f"observer overhead {overhead_id} payload must be an object")
            continue
        if set(payload) != OVERHEAD_FIELDS:
            reasons.append(f"observer overhead {overhead_id} fields are not exact")
        observed_id = payload.get("observed_event_id")
        if not isinstance(observed_id, str):
            reasons.append(f"observer overhead {overhead_id} observed_event_id is missing")
            continue
        reference_counts[observed_id] += 1
        observed = domain_by_id.get(observed_id)
        if observed is None:
            reasons.append(f"observer overhead {overhead_id} is orphaned")
        else:
            for field in ("host_id", "producer", "process_instance_id", "round", "client_id"):
                if overhead.get(field) != observed.get(field):
                    reasons.append(
                        f"observer overhead {overhead_id} identity differs from observed event"
                    )
                    break
            overhead_sequence = overhead.get("sequence")
            observed_sequence = observed.get("sequence")
            if (
                _is_int(overhead_sequence)
                and _is_int(observed_sequence)
                and overhead_sequence <= observed_sequence
            ):
                reasons.append(
                    f"observer overhead {overhead_id} does not follow observed event"
                )
        numeric_fields = OVERHEAD_FIELDS - {"observed_event_id"}
        for field in sorted(numeric_fields):
            reason = _number_reason(f"observer overhead {overhead_id} {field}", payload.get(field))
            if reason:
                reasons.append(reason)
            elif not _is_int(payload.get(field)):
                reasons.append(f"observer overhead {overhead_id} {field} must be an integer")
        total_inputs = (
            "observer_flower_serialize_ns",
            "observer_event_encode_ns",
            "observer_io_write_ns",
            "observer_fsync_ns",
        )
        if all(_is_finite_nonnegative_number(payload.get(field)) for field in total_inputs):
            if payload.get("observer_total_ns") != sum(payload[field] for field in total_inputs):
                reasons.append(f"observer overhead {overhead_id} total timing mismatch")
        event_count = payload.get("observer_event_count")
        if _is_int(event_count) and event_count < 1:
            reasons.append(
                f"observer overhead {overhead_id} event_count must be positive"
            )

    for event_id in sorted(domain_by_id):
        count = reference_counts[event_id]
        if count == 0:
            reasons.append(f"observed event {event_id} has unpaired observer overhead")
        elif count > 1:
            reasons.append(f"observed event {event_id} has duplicate observer overhead")
    return _sorted_unique(reasons)


def _within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except (OSError, ValueError):
        return False
    return True


def _inspect_evidence_tree(attempt_dir: Path) -> list[str]:
    reasons: list[str] = []
    raw_root = attempt_dir / "raw"
    if attempt_dir.is_symlink():
        reasons.append(f"attempt directory is a symlink: {attempt_dir}")
    if not attempt_dir.is_dir():
        reasons.append(f"attempt directory is not a directory: {attempt_dir}")
        return reasons
    if raw_root.is_symlink():
        reasons.append(f"raw evidence root is a symlink: {raw_root}")
    if not raw_root.is_dir():
        reasons.append("raw evidence root is missing")
        return reasons

    for root, directory_names, file_names in os.walk(raw_root, followlinks=False):
        root_path = Path(root)
        for name in [*directory_names, *file_names]:
            candidate = root_path / name
            if candidate.is_symlink():
                reasons.append(
                    f"evidence contains symlink and possible path escape: "
                    f"{candidate.relative_to(attempt_dir).as_posix()}"
                )
        if root_path == raw_root:
            for name in directory_names:
                if name not in {"ecs", "pi", "pc"}:
                    reasons.append(f"unknown host evidence directory raw/{name}")

    expected_jsonl = set(_EXPECTED_EVIDENCE)
    discovered_jsonl: set[str] = set()
    for candidate in raw_root.rglob("*.jsonl"):
        if not _within(candidate, attempt_dir):
            reasons.append(f"evidence path escapes attempt directory: {candidate}")
            continue
        relative = candidate.relative_to(attempt_dir).as_posix()
        discovered_jsonl.add(relative)
        if relative not in expected_jsonl:
            reasons.append(f"duplicate or unknown host evidence JSONL: {relative}")
    for relative in sorted(expected_jsonl - discovered_jsonl):
        reasons.append(f"missing expected host evidence: {relative}")
    return _sorted_unique(reasons)


def _validate_evidence_role(
    relative: str,
    events: Sequence[Mapping[str, Any]],
    attempt_id: str,
) -> list[str]:
    reasons: list[str] = []
    expected_host, expected_producer, expected_client = _EXPECTED_EVIDENCE[relative]
    allowed_types = {
        "server": SERVER_EVENT_TYPES,
        "client": CLIENT_EVENT_TYPES,
        "resource_sampler": SAMPLER_EVENT_TYPES,
    }[expected_producer]
    process_ids = {event.get("process_instance_id") for event in events}
    if len(process_ids) != 1:
        reasons.append(f"{relative} contains duplicate producer process evidence")
    for event in events:
        event_id = str(event.get("event_id"))
        if event.get("attempt_id") != attempt_id:
            reasons.append(f"{relative} event {event_id} attempt_id mismatches directory")
        if event.get("host_id") != expected_host:
            reasons.append(f"{relative} event {event_id} has unknown or duplicate host identity")
        if event.get("producer") != expected_producer:
            reasons.append(f"{relative} event {event_id} producer mismatch")
        if event.get("event_type") not in allowed_types:
            reasons.append(f"{relative} event {event_id} has unknown event_type")
        if expected_client is not None and event.get("client_id") != expected_client:
            reasons.append(f"{relative} event {event_id} client identity mismatch")
        if expected_producer == "server":
            event_type = event.get("event_type")
            client_id = event.get("client_id")
            if event_type == "flower_fitres_available" and client_id not in EXPECTED_CLIENTS:
                reasons.append(f"{relative} FitRes client identity is invalid")
            if event_type not in {
                "flower_fitins_prepared",
                "flower_fitres_available",
                "observer_overhead",
            } and client_id is not None:
                reasons.append(f"{relative} server event {event_id} client_id must be null")
    return reasons


def _validate_close_summary(
    path: Path, events: Sequence[Mapping[str, Any]]
) -> list[str]:
    reasons: list[str] = []
    relative_label = path.as_posix()
    try:
        summary = _load_json_object(path, label=f"close summary {relative_label}")
    except ValueError as exc:
        return [str(exc)]
    if set(summary) != CLOSE_SUMMARY_FIELDS:
        reasons.append(f"close summary {relative_label} fields are not exact")
    if summary.get("schema_version") != SCHEMA_VERSION:
        reasons.append(f"close summary {relative_label} schema mismatch")
    if events:
        first = events[0]
        for field in (
            "run_id",
            "attempt_id",
            "host_id",
            "producer",
            "process_instance_id",
        ):
            if summary.get(field) != first.get(field):
                reasons.append(f"close summary {relative_label} {field} mismatch")
    numeric_fields = CLOSE_SUMMARY_FIELDS - {
        "schema_version",
        "run_id",
        "attempt_id",
        "host_id",
        "producer",
        "process_instance_id",
    }
    for field in sorted(numeric_fields):
        reason = _number_reason(f"close summary {relative_label} {field}", summary.get(field))
        if reason:
            reasons.append(reason)
        elif not _is_int(summary.get(field)):
            reasons.append(f"close summary {relative_label} {field} must be an integer")
    if summary.get("observer_event_count") != len(events):
        reasons.append(f"close summary {relative_label} event count mismatch")
    total_inputs = (
        "observer_flower_serialize_ns",
        "observer_event_encode_ns",
        "observer_io_write_ns",
        "observer_fsync_ns",
    )
    if all(_is_finite_nonnegative_number(summary.get(field)) for field in total_inputs):
        if summary.get("observer_total_ns") != sum(summary[field] for field in total_inputs):
            reasons.append(f"close summary {relative_label} total timing mismatch")
    return reasons


def validate_attempt(
    attempt_dir: Path, protocol: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate all immutable evidence and return a deterministic audit object."""

    attempt_dir = Path(attempt_dir)
    reasons = _inspect_evidence_tree(attempt_dir)
    attempt_match = _ATTEMPT_RE.fullmatch(attempt_dir.name)
    if attempt_match is None:
        reasons.append("attempt directory name must be run_id plus __aNNN")
        attempt_id = attempt_dir.name
        run_hint = None
    else:
        attempt_id = attempt_dir.name
        run_hint = attempt_match.group(1)

    events_by_relative: dict[str, list[dict[str, Any]]] = {}
    input_hashes: dict[str, str] = {}
    for relative in sorted(_EXPECTED_EVIDENCE):
        path = attempt_dir / Path(relative)
        close_path = path.with_suffix(".close.json")
        for input_path in (path, close_path):
            input_relative = input_path.relative_to(attempt_dir).as_posix()
            if not _within(input_path, attempt_dir):
                reasons.append(f"input path escapes attempt directory: {input_relative}")
            elif input_path.is_symlink():
                reasons.append(f"input evidence is a symlink: {input_relative}")
            elif input_path.is_file():
                try:
                    input_hashes[input_relative] = _sha256_file(input_path)
                except OSError as exc:
                    reasons.append(f"cannot hash input {input_relative}: {exc}")
            else:
                reasons.append(f"missing input evidence: {input_relative}")
        try:
            rows = read_events(path)
        except ValueError as exc:
            reasons.append(f"{relative}: {exc}")
            rows = []
        events_by_relative[relative] = rows
        if rows:
            reasons.extend(_validate_evidence_role(relative, rows, attempt_id))
        if close_path.is_file() and not close_path.is_symlink():
            reasons.extend(_validate_close_summary(close_path, rows))

    all_events = [
        event
        for relative in sorted(events_by_relative)
        for event in events_by_relative[relative]
    ]
    reasons.extend(validate_common_fields(all_events, protocol))
    protocol_context, _ = _protocol_context(protocol, run_hint)
    if protocol_context.get("run_id") != run_hint:
        reasons.append("attempt directory run_id does not match protocol")

    server_events = events_by_relative["raw/ecs/events.jsonl"]
    counts, message_reasons = validate_message_matrix(server_events)
    reasons.extend(message_reasons)
    reasons.extend(validate_phase_times(all_events))

    host_events = {
        "pc-c2": [
            *events_by_relative["raw/pc/events.jsonl"],
            *events_by_relative["raw/pc/resource.jsonl"],
        ],
        "pi-c1": [
            *events_by_relative["raw/pi/events.jsonl"],
            *events_by_relative["raw/pi/resource.jsonl"],
        ],
    }
    resource, resource_reasons = validate_resource_coverage(host_events)
    reasons.extend(resource_reasons)
    reasons.extend(validate_observer_overhead(all_events))

    reasons = _sorted_unique(reasons)
    audit = {
        "attempt_id": attempt_id,
        "counts": dict(sorted(counts.items())),
        "inputs": {key: input_hashes[key] for key in sorted(input_hashes)},
        "protocol_manifest_sha256": protocol_context.get(
            "protocol_manifest_sha256"
        ),
        "reasons": reasons,
        "resource": {key: resource[key] for key in sorted(resource)},
        "run_id": run_hint,
        "schema_version": "iotj.confirmation.attempt_audit.v1",
        "status": "valid" if not reasons else "invalid",
    }
    return audit


def _write_exclusive_audit(path: Path, audit: Mapping[str, Any]) -> str:
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite audit output: {path}")
    parent = path.parent
    if parent.is_symlink() or not parent.is_dir():
        raise ValueError(f"audit output parent must be a regular directory: {parent}")
    payload = _canonical_json_bytes(audit) + b"\n"
    temporary = parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return hashlib.sha256(payload).hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt-dir", type=Path, required=True)
    parser.add_argument("--protocol-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"refusing to overwrite audit output: {args.output}")
    protocol = _load_json_object(
        args.protocol_manifest, label="protocol manifest"
    )
    audit = validate_attempt(args.attempt_dir, protocol)
    audit_sha256 = _write_exclusive_audit(args.output, audit)
    sys.stdout.buffer.write(
        _canonical_json_bytes({"audit_sha256": audit_sha256}) + b"\n"
    )
    return 0 if audit["status"] == "valid" else 2


if __name__ == "__main__":
    raise SystemExit(main())
