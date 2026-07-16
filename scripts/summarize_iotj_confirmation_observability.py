"""Build sealed, canonical-only IoT-J confirmation evidence summaries.

The C5 test evaluator is invoked only after the exact ten-attempt provenance,
audit, checkpoint, and protocol gate succeeds.  Training-side evidence is read
from immutable local attempt directories; the script never selects attempts by
classification metrics and never estimates transport-layer bytes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import shutil
import uuid
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from scripts.summarize_iotj_classification_ablation import (
    evaluate_checkpoint_stream,
    resolve_device,
)
from scripts.freeze_iotj_confirmation_protocol import build_dataset_manifest


SEEDS = (42, 43, 44, 45, 46)
GROUPS = ("B2", "B5")
EXPECTED_IDENTITIES = tuple((group, seed) for group in GROUPS for seed in SEEDS)
RUN_RE = re.compile(r"^c12_to_c5__(b2|b5)__s(42|43|44|45|46)$")
ATTEMPT_RE = re.compile(
    r"^(c12_to_c5__(?:b2|b5)__s(?:42|43|44|45|46))__a(\d{3})$"
)
AUDITED_EVIDENCE_PATHS = (
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
)
HASH_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
HISTORICAL_REVISION = "feaa75b"
CLAIM_STATUS = {
    "B2": "post_screen_exploratory",
    "B5": "predeclared_full_method",
}
METRICS = (
    "accuracy",
    "macro_f1",
    "nll",
    "ece",
    "recall_0",
    "recall_1",
    "recall_2",
    "recall_3",
    "worst_class_recall",
)

CLASSIFICATION_PER_RUN_FIELDS = (
    "run_id",
    "group_id",
    "seed",
    "claim_status",
    "N",
    "accuracy",
    "macro_f1",
    "nll",
    "ece",
    "recall_0",
    "recall_1",
    "recall_2",
    "recall_3",
    "worst_class_recall",
    "confirmation_commit",
    "source_archive_sha256",
    "dataset_manifest_sha256",
    "algorithm_config_sha256",
    "checkpoint_sha256",
)
CLASSIFICATION_MULTISEED_FIELDS = (
    "summary_type",
    "group_id",
    "claim_status",
    "metric",
    "seed_42",
    "seed_43",
    "seed_44",
    "seed_45",
    "seed_46",
    "mean",
    "sample_std_ddof1",
)

DOWNLINK_LOGICAL_FIELDS = (
    "logical_downlink_model_value_bytes",
    "logical_downlink_parameter_blob_bytes",
    "logical_downlink_semantic_proto_utf8_bytes",
    "logical_downlink_other_config_value_bytes",
    "logical_downlink_total_bytes",
)
UPLINK_LOGICAL_FIELDS = (
    "logical_uplink_model_value_bytes",
    "logical_uplink_parameter_blob_bytes",
    "logical_uplink_prototype_utf8_bytes",
    "logical_uplink_prototype_var_utf8_bytes",
    "logical_uplink_statistics_utf8_bytes",
    "logical_uplink_diagnostic_value_bytes",
    "logical_uplink_total_bytes",
)
COMMUNICATION_PER_ROUND_FIELDS = (
    "run_id",
    "attempt_id",
    "group_id",
    "seed",
    "round",
    "client_id",
    *DOWNLINK_LOGICAL_FIELDS,
    *UPLINK_LOGICAL_FIELDS,
    "application_downlink_message_bytes",
    "application_downlink_message_sha256",
    "application_uplink_message_bytes",
    "application_uplink_message_sha256",
    "application_round_total_bytes",
    "transport_status",
)
COMMUNICATION_SUMMARY_FIELDS = (
    "run_id",
    "attempt_id",
    "group_id",
    "seed",
    "logical_downlink_25round_total_bytes",
    "logical_uplink_25round_total_bytes",
    "application_downlink_25round_total_bytes",
    "application_uplink_25round_total_bytes",
    "application_25round_total_bytes",
    "application_round_mean_bytes",
    "transport_status",
)
ROUND_TIME_FIELDS = (
    "run_id",
    "attempt_id",
    "group_id",
    "seed",
    "statistic",
    "round",
    "c1_client_train_core_ns",
    "c2_client_train_core_ns",
    "client_train_critical_path_ns",
    "c1_client_fit_callback_ns",
    "c2_client_fit_callback_ns",
    "client_fit_critical_path_ns",
    "server_aggregate_fit_total_ns",
    "server_da_total_ns",
    "server_aggregate_non_da_ns",
    "fit_round_wall_ns",
    "parallel_client_times_are_not_serially_additive",
)
RESOURCE_FIELDS = (
    "run_id",
    "attempt_id",
    "group_id",
    "seed",
    "client_id",
    "host_role",
    "total_sample_count",
    "active_sample_count",
    "resource_coverage",
    "expected_sample_points",
    "covered_sample_points",
    "rss_all_sample_mean_bytes",
    "rss_active_mean_bytes",
    "rss_peak_bytes",
    "cpu_one_core_mean_percent",
    "cpu_one_core_peak_percent",
    "cpu_host_mean_percent",
    "cpu_host_peak_percent",
    "cpu_temperature_available",
    "cpu_temperature_mean_c",
    "cpu_temperature_peak_c",
    "throttling_available",
    "throttling_observed",
)
OVERHEAD_FIELDS = (
    "run_id",
    "attempt_id",
    "group_id",
    "seed",
    "host_id",
    "producer",
    "paired_domain_event_count",
    "paired_observer_flower_serialize_ns",
    "paired_observer_event_encode_ns",
    "paired_observer_io_write_ns",
    "paired_observer_fsync_ns",
    "paired_observer_total_ns",
    "close_observer_event_count",
    "close_observer_flower_serialize_ns",
    "close_observer_event_encode_ns",
    "close_observer_io_write_ns",
    "close_observer_fsync_ns",
    "close_observer_total_ns",
    "close_observer_event_bytes_written",
    "observer_reporting_tail_bytes",
    "fit_round_wall_ns_total",
    "observer_total_to_round_wall_ratio",
)
ATTEMPT_REGISTRY_FIELDS = (
    "run_id",
    "attempt_id",
    "group_id",
    "seed",
    "state",
    "claim_status",
    "confirmation_commit",
    "source_archive_sha256",
    "dataset_manifest_sha256",
    "algorithm_config_sha256",
    "protocol_manifest_sha256",
    "audit_sha256",
    "checkpoint_sha256",
    "classification_stream_sha256",
    "classification_stream_size_bytes",
    "classification_stream_device",
    "classification_stream_inode",
    "attempt_relative_path",
    "audit_relative_path",
    "checkpoint_relative_path",
    "classification_stream_relative_path",
    "transport_status",
)

TABLE_FIELDS = {
    "flower_communication_per_round.csv": COMMUNICATION_PER_ROUND_FIELDS,
    "flower_communication_summary.csv": COMMUNICATION_SUMMARY_FIELDS,
    "flower_round_time_breakdown.csv": ROUND_TIME_FIELDS,
    "training_resource_summary.csv": RESOURCE_FIELDS,
    "observer_overhead_summary.csv": OVERHEAD_FIELDS,
}

PublishedToken = tuple[int, int, int, str]
PublishedReceipt = tuple[Path, PublishedToken, Path, PublishedToken]
_ACTIVE_STREAM_RECEIPTS: ContextVar[list[PublishedReceipt] | None] = ContextVar(
    "iotj_confirmation_stream_receipts", default=None
)


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"value is not canonical finite JSON: {exc}") from exc


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reject_constant(raw: str) -> Any:
    raise ValueError(f"non-finite JSON constant {raw}")


def _parse_finite_float(raw: str) -> float:
    value = float(raw)
    if not math.isfinite(value):
        raise ValueError(f"non-finite JSON number {raw}")
    return value


def _object_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            Path(path).read_text(encoding="utf-8"),
            parse_constant=_reject_constant,
            parse_float=_parse_finite_float,
            object_pairs_hook=_object_no_duplicates,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError(f"invalid JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def _decode_json_text(raw: str, *, label: str) -> Any:
    try:
        return json.loads(
            raw,
            parse_constant=_reject_constant,
            parse_float=_parse_finite_float,
            object_pairs_hook=_object_no_duplicates,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError(f"invalid JSON {label}: {exc}") from exc


def _parse_json_object_bytes(data: bytes, *, label: str) -> dict[str, Any]:
    try:
        text = data.decode("utf-8")
    except UnicodeError as exc:
        raise RuntimeError(f"invalid UTF-8 {label}: {exc}") from exc
    value = _decode_json_text(text, label=label)
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {label}")
    return value


def _parse_jsonl_bytes(data: bytes, *, label: str) -> list[dict[str, Any]]:
    try:
        text = data.decode("utf-8")
    except UnicodeError as exc:
        raise RuntimeError(f"invalid UTF-8 {label}: {exc}") from exc
    lines = text.splitlines()
    if not lines:
        raise RuntimeError(f"audit input event file is empty: {label}")
    events: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            raise RuntimeError(f"audit input has blank JSONL line: {label}:{line_number}")
        value = _decode_json_text(line, label=f"{label}:{line_number}")
        if not isinstance(value, dict):
            raise RuntimeError(f"event must be a JSON object: {label}:{line_number}")
        events.append(value)
    return events


def _is_reparse(path: Path) -> bool:
    try:
        attrs = getattr(os.lstat(path), "st_file_attributes", 0)
    except OSError:
        return False
    return bool(attrs & getattr(__import__("stat"), "FILE_ATTRIBUTE_REPARSE_POINT", 0))


def _has_link_component(path: Path, stop: Path | None = None) -> bool:
    current = Path(path).absolute()
    stop_resolved = None if stop is None else Path(stop).resolve()
    while True:
        if current.exists() and (current.is_symlink() or _is_reparse(current)):
            return True
        if stop_resolved is not None:
            try:
                if current.resolve() == stop_resolved:
                    return False
            except OSError:
                pass
        if current.parent == current:
            return False
        current = current.parent


def _within(path: Path, root: Path) -> bool:
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except (OSError, ValueError):
        return False


def _finite_number(value: Any, label: str, *, integer: bool = False) -> float | int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{label} must be a finite nonnegative number")
    if not math.isfinite(float(value)):
        raise RuntimeError(f"{label} must be finite")
    if value < 0:
        raise RuntimeError(f"{label} must be nonnegative")
    if integer and not isinstance(value, int):
        raise RuntimeError(f"{label} must be an integer")
    return value


def _metric_trace(value: Any, path: str = "") -> str | None:
    tokens = ("accuracy", "macro_f1", "nll", "ece", "recall", "f1", "metric", "ranking")
    if isinstance(value, Mapping):
        for key, child in value.items():
            lowered = str(key).lower()
            if any(token in lowered for token in tokens):
                return f"{path}.{key}" if path else str(key)
            found = _metric_trace(child, f"{path}.{key}" if path else str(key))
            if found is not None:
                return found
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found = _metric_trace(child, f"{path}[{index}]")
            if found is not None:
                return found
    elif path.lower().endswith(("reason", "selection", "rerun")) and isinstance(value, str):
        lowered = value.lower()
        if any(token in lowered for token in tokens):
            return path
    return None


def _protocol_self_hash(protocol: Mapping[str, Any]) -> str:
    claimed = protocol.get("protocol_manifest_sha256")
    if not isinstance(claimed, str) or HASH_RE.fullmatch(claimed) is None:
        raise RuntimeError("protocol_manifest_sha256 is invalid")
    unhashed = {key: value for key, value in protocol.items() if key != "protocol_manifest_sha256"}
    actual = hashlib.sha256(_canonical_json_bytes(unhashed)).hexdigest()
    if actual != claimed:
        raise RuntimeError("protocol manifest SHA-256 self-hash mismatch")
    return claimed


def _read_audit(row: Mapping[str, Any]) -> dict[str, Any]:
    audit_path = Path(row["audit_path"])
    if not audit_path.is_file() or audit_path.is_symlink() or _is_reparse(audit_path):
        raise RuntimeError(f"audit must be a regular non-symlink file: {audit_path}")
    expected = row.get("audit_sha256")
    actual = _sha256_file(audit_path)
    if expected != actual:
        raise RuntimeError(f"audit SHA-256 mismatch for {row.get('attempt_id')}")
    audit = _load_json(audit_path)
    if (
        audit.get("status") != "valid"
        or audit.get("reasons") != []
        or audit.get("run_id") != row.get("run_id")
        or audit.get("attempt_id") != row.get("attempt_id")
        or audit.get("protocol_manifest_sha256") != row.get("protocol_manifest_sha256")
    ):
        raise RuntimeError(f"audit identity/status mismatch for {row.get('attempt_id')}")
    counts = audit.get("counts")
    if not isinstance(counts, Mapping) or {
        "fitins": counts.get("fitins"),
        "fitres": counts.get("fitres"),
        "rounds": counts.get("rounds"),
    } != {"fitins": 50, "fitres": 50, "rounds": 25}:
        raise RuntimeError(f"audit count mismatch for {row.get('attempt_id')}")
    return audit


def assert_test_gate(
    canonical_rows: Sequence[Mapping[str, Any]],
    *,
    protocol: Mapping[str, Any] | None = None,
    raw_root: Path | None = None,
) -> list[dict[str, Any]]:
    """Fail closed unless exactly the frozen ten canonical attempts are bound.

    Attempt selection is based only on immutable lifecycle state and structural
    audit evidence.  Classification metrics are forbidden in this input.
    """

    if len(canonical_rows) != 10:
        raise RuntimeError(f"test Gate requires exactly 10 canonical attempts; got {len(canonical_rows)}")
    rows = [dict(row) for row in canonical_rows]
    required = {
        "run_id",
        "attempt_id",
        "attempt_dir",
        "group_id",
        "seed",
        "state",
        "status_reason",
        "direction",
        "historical_seed42_included",
        "confirmation_commit",
        "source_archive_sha256",
        "dataset_manifest_sha256",
        "algorithm_config_sha256",
        "protocol_manifest_sha256",
        "audit_path",
        "audit_sha256",
        "checkpoint_path",
        "checkpoint_sha256",
        "transport_status",
    }
    identities: list[tuple[str, int]] = []
    run_ids: set[str] = set()
    attempt_ids: set[str] = set()
    common_fields = (
        "confirmation_commit",
        "source_archive_sha256",
        "dataset_manifest_sha256",
        "protocol_manifest_sha256",
    )
    common: dict[str, Any] = {}

    if protocol is not None:
        protocol_hash = _protocol_self_hash(protocol)
        if protocol.get("direction") != "C1/C2 -> C5":
            raise RuntimeError("protocol direction must be C1/C2 -> C5")
        if protocol.get("historical_seed42_included") is not False:
            raise RuntimeError("protocol historical evidence boundary is invalid")
        schedule = protocol.get("schedule")
        if not isinstance(schedule, list) or len(schedule) != 10:
            raise RuntimeError("protocol schedule must contain exactly 10 canonical runs")
        schedule_by_run = {
            item.get("run_id"): item for item in schedule if isinstance(item, Mapping)
        }
        if len(schedule_by_run) != 10:
            raise RuntimeError("protocol schedule contains duplicate or malformed runs")
    else:
        protocol_hash = None
        schedule_by_run = {}

    for index, row in enumerate(rows):
        missing = sorted(required - set(row))
        if missing:
            raise RuntimeError(f"canonical row {index} is missing fields: {missing}")
        trace = _metric_trace(row)
        if trace is not None:
            raise RuntimeError(f"metric-driven rerun/selection trace is forbidden: {trace}")
        if HISTORICAL_REVISION in str(row).lower() or row.get("historical_seed42_included") is not False:
            raise RuntimeError("historical feaa75b evidence cannot enter confirmation")
        if row.get("state") != "canonical" or row.get("status_reason") != "validator_accepted":
            raise RuntimeError("all ten attempts must have canonical validator-accepted state")
        if row.get("direction") != "C1/C2 -> C5":
            raise RuntimeError("cross-direction evidence is forbidden")
        if row.get("transport_status") != "not_collected":
            raise RuntimeError("transport_status must be explicitly not_collected")

        run_id = row.get("run_id")
        match = RUN_RE.fullmatch(run_id) if isinstance(run_id, str) else None
        if match is None:
            raise RuntimeError(f"run_id is outside main-direction allowlist: {run_id!r}")
        expected_group = match.group(1).upper()
        expected_seed = int(match.group(2))
        if row.get("group_id") != expected_group or row.get("seed") != expected_seed:
            raise RuntimeError(f"run/group/seed identity mismatch: {run_id}")
        identities.append((expected_group, expected_seed))
        if run_id in run_ids:
            raise RuntimeError(f"duplicate canonical run_id: {run_id}")
        run_ids.add(run_id)

        attempt_id = row.get("attempt_id")
        attempt_match = ATTEMPT_RE.fullmatch(attempt_id) if isinstance(attempt_id, str) else None
        if attempt_match is None or attempt_match.group(1) != run_id:
            raise RuntimeError(f"attempt_id does not bind run_id: {attempt_id!r}")
        if attempt_id in attempt_ids:
            raise RuntimeError(f"duplicate canonical attempt_id: {attempt_id}")
        attempt_ids.add(attempt_id)

        for field in common_fields:
            value = row.get(field)
            pattern = COMMIT_RE if field == "confirmation_commit" else HASH_RE
            if not isinstance(value, str) or pattern.fullmatch(value) is None:
                raise RuntimeError(f"{field} must be a full lowercase hash")
            if field not in common:
                common[field] = value
            elif common[field] != value:
                raise RuntimeError(f"all canonical attempts must share {field}")
        algorithm_sha = row.get("algorithm_config_sha256")
        if not isinstance(algorithm_sha, str) or HASH_RE.fullmatch(algorithm_sha) is None:
            raise RuntimeError("algorithm_config_sha256 must be a lowercase SHA-256")
        if protocol_hash is not None and row.get("protocol_manifest_sha256") != protocol_hash:
            raise RuntimeError("row protocol manifest SHA-256 mismatch")
        if protocol is not None:
            schedule_row = schedule_by_run.get(run_id)
            if not isinstance(schedule_row, Mapping):
                raise RuntimeError(f"run missing from protocol schedule: {run_id}")
            if (
                schedule_row.get("group_id") != expected_group
                or schedule_row.get("seed") != expected_seed
                or schedule_row.get("algorithm_config_sha256") != algorithm_sha
                or schedule_row.get("transport_status") != "not_collected"
            ):
                raise RuntimeError(f"protocol schedule binding mismatch: {run_id}")
            for field in ("confirmation_commit", "source_archive_sha256", "dataset_manifest_sha256"):
                if protocol.get(field) != row.get(field):
                    raise RuntimeError(f"protocol {field} mismatch")

        attempt_dir = Path(row["attempt_dir"])
        if attempt_dir.name != attempt_id or attempt_dir.parent.name != run_id:
            raise RuntimeError(f"attempt path identity mismatch: {attempt_dir}")
        if raw_root is not None and not _within(attempt_dir, Path(raw_root)):
            raise RuntimeError(f"attempt path escapes raw root: {attempt_dir}")
        if _has_link_component(attempt_dir, Path(raw_root) if raw_root is not None else None):
            raise RuntimeError(f"attempt path contains symlink/reparse point: {attempt_dir}")
        if not attempt_dir.is_dir():
            raise RuntimeError(f"attempt directory is missing: {attempt_dir}")
        for label, path_field, hash_field in (
            ("audit", "audit_path", "audit_sha256"),
            ("checkpoint", "checkpoint_path", "checkpoint_sha256"),
        ):
            path = Path(row[path_field])
            if not _within(path, attempt_dir):
                raise RuntimeError(f"{label} path escapes attempt: {path}")
            if _has_link_component(path, attempt_dir):
                raise RuntimeError(f"{label} path contains symlink/reparse point: {path}")
            if not path.is_file():
                raise RuntimeError(f"{label} is missing: {path}")
            actual_sha = _sha256_file(path)
            if row.get(hash_field) != actual_sha:
                raise RuntimeError(f"{label} SHA-256 mismatch for {attempt_id}")
        if Path(row["audit_path"]).resolve() != (attempt_dir / "attempt_audit.json").resolve():
            raise RuntimeError("audit path is not the canonical attempt audit")
        expected_checkpoint = (
            attempt_dir / "raw" / "ecs" / "training" / "server_latest_adapted.pth"
        )
        if Path(row["checkpoint_path"]).resolve() != expected_checkpoint.resolve():
            raise RuntimeError("checkpoint path is not the round-25 adapted checkpoint location")
        provenance = _load_json(attempt_dir / "attempt_provenance.json")
        status = _load_status(attempt_dir / "attempt_status.json")
        if (
            provenance.get("run_id") != run_id
            or provenance.get("attempt_id") != attempt_id
            or status.get("run_id") != run_id
            or status.get("attempt_id") != attempt_id
        ):
            raise RuntimeError("attempt provenance/status identity mismatch")
        for field in (
            "confirmation_commit",
            "source_archive_sha256",
            "dataset_manifest_sha256",
            "algorithm_config_sha256",
        ):
            if provenance.get(field) != row.get(field) or status.get(field) != row.get(field):
                raise RuntimeError(f"attempt provenance/status {field} mismatch")
        if (
            status.get("state") != "canonical"
            or status.get("reason") != "validator_accepted"
            or status.get("audit_sha256") != row.get("audit_sha256")
        ):
            raise RuntimeError("attempt status/audit identity mismatch")
        _read_audit(row)

    if sorted(identities) != sorted(EXPECTED_IDENTITIES):
        raise RuntimeError("canonical identity matrix must equal B2/B5 x seeds 42..46")
    return sorted(rows, key=lambda row: str(row["run_id"]))


def classification_row(row: Mapping[str, Any], metrics: Mapping[str, Any]) -> dict[str, Any]:
    recalls = metrics.get("per_class_recall")
    if not isinstance(recalls, Mapping):
        raise RuntimeError("classification per_class_recall must be an object")
    n_value = metrics.get("N")
    if isinstance(n_value, bool) or not isinstance(n_value, int) or n_value != 1360:
        raise RuntimeError("classification N must equal the frozen 1360-row C5 test set")
    result: dict[str, Any] = {
        "run_id": row["run_id"],
        "group_id": row["group_id"],
        "seed": row["seed"],
        "claim_status": CLAIM_STATUS[str(row["group_id"])],
        "N": n_value,
        "confirmation_commit": row["confirmation_commit"],
        "source_archive_sha256": row["source_archive_sha256"],
        "dataset_manifest_sha256": row["dataset_manifest_sha256"],
        "algorithm_config_sha256": row["algorithm_config_sha256"],
        "checkpoint_sha256": row["checkpoint_sha256"],
    }
    for field in ("accuracy", "macro_f1", "nll", "ece"):
        result[field] = float(_finite_number(metrics.get(field), f"classification {field}"))
    recall_values: list[float] = []
    for class_id in range(4):
        value = float(_finite_number(recalls.get(str(class_id)), f"recall_{class_id}"))
        result[f"recall_{class_id}"] = value
        recall_values.append(value)
    result["worst_class_recall"] = min(recall_values)
    if set(result) != set(CLASSIFICATION_PER_RUN_FIELDS):
        raise RuntimeError("classification_per_run schema construction error")
    return result


def _build_bound_dataset_manifest(
    data_root: Path,
    rows: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any] | None,
    *,
    changed_message: bool = False,
) -> dict[str, Any]:
    """Rebuild Task5's exact active-file manifest and bind it to the Gate."""

    try:
        manifest = build_dataset_manifest(Path(data_root))
    except (FileNotFoundError, OSError, ValueError) as exc:
        if changed_message:
            raise RuntimeError(
                f"dataset changed during sealed evaluation: {type(exc).__name__}: {exc}"
            ) from exc
        raise RuntimeError(f"dataset manifest validation failed: {exc}") from exc
    expected_sha = str(rows[0]["dataset_manifest_sha256"])
    actual_sha = manifest.get("dataset_manifest_sha256")
    if actual_sha != expected_sha:
        label = (
            "dataset changed during sealed evaluation"
            if changed_message
            else "dataset_manifest_sha256 does not match frozen confirmation"
        )
        raise RuntimeError(f"{label}: expected {expected_sha}, got {actual_sha}")
    if (
        manifest.get("direction") != "C1/C2 -> C5"
        or manifest.get("active_source_clients") != [1, 2]
        or manifest.get("active_target_clients") != [5]
        or manifest.get("sample_counts", {}).get("C5")
        != {"calibration": 320, "test": 1360}
    ):
        label = (
            "dataset changed during sealed evaluation"
            if changed_message
            else "dataset active-client/count contract mismatch"
        )
        raise RuntimeError(label)
    if protocol is not None:
        if (
            protocol.get("dataset_manifest_sha256") != actual_sha
            or protocol.get("direction") != "C1/C2 -> C5"
            or protocol.get("active_source_clients") != ["C1", "C2"]
            or protocol.get("active_target_clients") != ["C5"]
        ):
            label = (
                "dataset changed during sealed evaluation"
                if changed_message
                else "protocol dataset/direction/active-client binding mismatch"
            )
            raise RuntimeError(label)
    return manifest


def _write_exclusive_bytes(path: Path, payload: bytes) -> None:
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def csv_bytes(rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> bytes:
    materialized = [dict(row) for row in rows]
    expected = set(fields)
    for index, row in enumerate(materialized):
        if set(row) != expected:
            raise RuntimeError(
                f"CSV schema mismatch at row {index}: expected {sorted(expected)}, got {sorted(row)}"
            )
        for key, value in row.items():
            if isinstance(value, float) and not math.isfinite(value):
                raise RuntimeError(f"CSV field {key} must be finite")
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(fields), lineterminator="\n")
    writer.writeheader()
    writer.writerows(materialized)
    return buffer.getvalue().encode("utf-8")


def _stream_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    if not rows:
        raise RuntimeError("classification evaluator returned an empty target-test stream")
    fields = tuple(sorted(rows[0]))
    if not fields:
        raise RuntimeError("classification evaluator returned schema-less rows")
    return csv_bytes(rows, fields)


def _classification_stream_path(row: Mapping[str, Any]) -> Path:
    return (
        Path(row["attempt_dir"])
        / "raw"
        / "ecs"
        / "evaluation"
        / "classification_test_stream.csv"
    )


def _preflight_stream_destinations(
    rows: Sequence[Mapping[str, Any]],
) -> list[Path]:
    paths: list[Path] = []
    for row in rows:
        attempt_dir = Path(row["attempt_dir"])
        path = _classification_stream_path(row)
        if not _within(path, attempt_dir) or _has_link_component(path.parent, attempt_dir):
            raise RuntimeError(f"classification stream path is unsafe: {path}")
        if path.exists() or path.is_symlink() or _is_reparse(path):
            raise FileExistsError(f"refusing to overwrite output: {path}")
        paths.append(path)
    if len(paths) != len({path.resolve() for path in paths}):
        raise RuntimeError("classification stream destinations are not unique")
    return paths


def _publish_stream(path: Path, payload: bytes) -> None:
    """Publish one prepared stream exclusively; caller owns transaction rollback."""

    receipts = _ACTIVE_STREAM_RECEIPTS.get()
    if receipts is None:
        raise RuntimeError("classification stream publication lacks transaction ownership")
    path = Path(path)
    if path.exists() or path.is_symlink() or _is_reparse(path):
        raise FileExistsError(f"refusing to overwrite output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink() or _is_reparse(path.parent):
        raise RuntimeError(f"classification stream parent is unsafe: {path.parent}")
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    linked = False
    recorded = False
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary_token = _published_token(temporary)
        os.link(temporary, path)
        linked = True
        receipt = (path, temporary_token, temporary, temporary_token)
        try:
            _record_stream_receipt(receipts, receipt)
        except BaseException as exc:
            cleanup_errors: list[str] = []
            _unlink_owned_path(path, temporary_token, cleanup_errors)
            _unlink_owned_path(temporary, temporary_token, cleanup_errors)
            if cleanup_errors:
                raise RuntimeError(
                    "link-to-receipt failure cleanup was incomplete: "
                    + "; ".join(cleanup_errors)
                ) from exc
            raise
        recorded = True
        final_token = _published_token(path)
        if final_token != temporary_token:
            raise RuntimeError("published classification stream identity mismatch")
    finally:
        if not linked or recorded:
            temporary.unlink(missing_ok=True)


def _record_stream_receipt(
    receipts: list[PublishedReceipt], receipt: PublishedReceipt
) -> None:
    """Injectable seam; the publisher self-cleans if registration raises."""

    receipts.append(receipt)


def _published_token(path: Path) -> PublishedToken:
    stat_result = path.stat(follow_symlinks=False)
    return (
        int(stat_result.st_dev),
        int(stat_result.st_ino),
        int(stat_result.st_size),
        _sha256_file(path),
    )


def _unlink_owned_path(path: Path, token: PublishedToken, errors: list[str]) -> None:
    try:
        if not path.exists() and not path.is_symlink():
            return
        if (
            not path.is_file()
            or path.is_symlink()
            or _is_reparse(path)
            or _published_token(path) != token
        ):
            errors.append(f"ownership changed; preserved {path}")
            return
        path.unlink()
    except OSError as exc:
        errors.append(f"cannot rollback {path}: {type(exc).__name__}: {exc}")


def _rollback_published_streams(published: Sequence[PublishedReceipt]) -> list[str]:
    errors: list[str] = []
    for path, token, temporary, temporary_token in reversed(published):
        _unlink_owned_path(path, token, errors)
        _unlink_owned_path(temporary, temporary_token, errors)
    return errors


def _verify_checkpoint_stability(
    row: Mapping[str, Any], *, phase: str
) -> str:
    attempt_dir = Path(row["attempt_dir"])
    path = Path(row["checkpoint_path"])
    if (
        not _within(path, attempt_dir)
        or _has_link_component(path, attempt_dir)
        or not path.is_file()
    ):
        raise RuntimeError(f"checkpoint is missing or unsafe {phase}: {path}")
    actual = _sha256_file(path)
    if actual != row.get("checkpoint_sha256"):
        raise RuntimeError(
            f"checkpoint SHA-256 changed {phase}: {row['attempt_id']}; "
            f"expected {row.get('checkpoint_sha256')}, got {actual}"
        )
    return actual


def _verify_all_checkpoints(
    rows: Sequence[Mapping[str, Any]], *, phase: str
) -> None:
    for row in rows:
        _verify_checkpoint_stability(row, phase=phase)


class StreamTransaction:
    """Own final prediction streams until the summary directory is published."""

    def __init__(self, rows: Sequence[Mapping[str, Any]]) -> None:
        self.rows = [dict(row) for row in rows]
        self.paths = _preflight_stream_destinations(self.rows)
        self.receipts: list[PublishedReceipt] = []
        self.prepared: dict[Path, tuple[int, str]] = {}
        self._context_token: Any = None
        self.active = False
        self.committed = False

    def __enter__(self) -> "StreamTransaction":
        if self.active or self.committed:
            raise RuntimeError("stream transaction cannot be entered twice")
        # Repeat immediately before ownership begins to close construction races.
        _preflight_stream_destinations(self.rows)
        self._context_token = _ACTIVE_STREAM_RECEIPTS.set(self.receipts)
        self.active = True
        return self

    def register_prepared(self, path: Path, payload: bytes) -> None:
        if not self.active or self.committed:
            raise RuntimeError("stream transaction is not active")
        path = Path(path)
        if path not in self.paths or path in self.prepared:
            raise RuntimeError(f"unexpected or duplicate prepared stream: {path}")
        self.prepared[path] = (len(payload), hashlib.sha256(payload).hexdigest())

    def publish(self, prepared: Sequence[tuple[Path, bytes, dict[str, Any]]]) -> None:
        if not self.active or self.committed:
            raise RuntimeError("stream transaction is not active")
        if [Path(item[0]) for item in prepared] != self.paths:
            raise RuntimeError("prepared stream matrix does not match transaction paths")
        for path, payload, _classification in prepared:
            self.register_prepared(path, payload)
        for path, payload, _classification in prepared:
            _publish_stream(path, payload)
        self.verify_streams()

    def _receipt_by_path(self) -> dict[Path, PublishedReceipt]:
        receipts: dict[Path, PublishedReceipt] = {}
        for receipt in self.receipts:
            path = Path(receipt[0])
            if path in receipts:
                raise RuntimeError(f"duplicate stream transaction receipt: {path}")
            receipts[path] = receipt
        return receipts

    def stream_fingerprint(self, row: Mapping[str, Any]) -> dict[str, Any]:
        path = _classification_stream_path(row)
        receipt = self._receipt_by_path().get(path)
        prepared = self.prepared.get(path)
        if receipt is None or prepared is None:
            raise RuntimeError(f"missing prepared/receipt stream binding: {path}")
        expected_size, expected_sha = prepared
        try:
            actual_token = _published_token(path)
        except OSError as exc:
            raise RuntimeError(
                f"classification stream ownership changed or disappeared: {path}"
            ) from exc
        receipt_token = receipt[1]
        if actual_token != receipt_token:
            raise RuntimeError(f"classification stream ownership changed: {path}")
        if actual_token[2] != expected_size or actual_token[3] != expected_sha:
            raise RuntimeError(f"classification stream differs from prepared payload: {path}")
        return {
            "path": path,
            "size_bytes": expected_size,
            "sha256": expected_sha,
            "device": actual_token[0],
            "inode": actual_token[1],
        }

    def verify_streams(self) -> None:
        if not self.active or self.committed:
            raise RuntimeError("stream transaction is not active")
        if set(self.prepared) != set(self.paths):
            raise RuntimeError("stream transaction prepared matrix is incomplete")
        if set(self._receipt_by_path()) != set(self.paths):
            raise RuntimeError("stream transaction receipt matrix is incomplete")
        for row in self.rows:
            self.stream_fingerprint(row)

    def commit_after_summary_publish(self) -> None:
        if not self.active or self.committed:
            raise RuntimeError("stream transaction cannot be committed")
        self.committed = True

    def commit_standalone(self) -> None:
        self.verify_streams()
        _verify_all_checkpoints(
            self.rows, phase="before standalone stream transaction commit"
        )
        self.committed = True

    def __exit__(self, exc_type, exc, traceback) -> bool:
        rollback_errors: list[str] = []
        if not self.committed:
            rollback_errors = _rollback_published_streams(self.receipts)
        if self._context_token is not None:
            _ACTIVE_STREAM_RECEIPTS.reset(self._context_token)
            self._context_token = None
        self.active = False
        if rollback_errors:
            raise RuntimeError(
                "stream transaction rollback reported ownership changed/incomplete: "
                + "; ".join(rollback_errors)
            ) from exc
        return False


def evaluate_canonical_attempts(
    canonical_rows: Sequence[Mapping[str, Any]],
    *,
    data_root: Path,
    device: Any,
    batch_size: int,
    evaluator: Callable[..., tuple[list[dict[str, Any]], dict[str, Any]]] = evaluate_checkpoint_stream,
    protocol: Mapping[str, Any] | None = None,
    raw_root: Path | None = None,
    stream_transaction: StreamTransaction | None = None,
) -> list[dict[str, Any]]:
    """Open sealed C5 test only after the complete canonical Gate succeeds."""

    rows = assert_test_gate(canonical_rows, protocol=protocol, raw_root=raw_root)
    def run_evaluation(transaction: StreamTransaction) -> list[dict[str, Any]]:
        if not transaction.active or transaction.committed:
            raise RuntimeError("caller-owned stream transaction is not active")
        if [str(row["attempt_id"]) for row in transaction.rows] != [
            str(row["attempt_id"]) for row in rows
        ]:
            raise RuntimeError("stream transaction attempt matrix mismatch")
        initial_dataset = _build_bound_dataset_manifest(
            Path(data_root), rows, protocol
        )
        prepared: list[tuple[Path, bytes, dict[str, Any]]] = []
        for row, raw_path in zip(rows, transaction.paths):
            _verify_checkpoint_stability(row, phase="immediately before evaluator")
            stream, metrics = evaluator(
                Path(row["checkpoint_path"]),
                data_root=Path(data_root),
                target_client=5,
                split="test",
                device=device,
                batch_size=batch_size,
            )
            _verify_checkpoint_stability(row, phase="immediately after evaluator")
            if len(stream) != 1360:
                raise RuntimeError(
                    f"sealed C5 evaluator must return exactly 1360 rows; got {len(stream)}"
                )
            for sample_index, stream_row in enumerate(stream):
                if (
                    not isinstance(stream_row, Mapping)
                    or stream_row.get("client") != "C5"
                    or stream_row.get("split") != "test"
                    or stream_row.get("sample_index") != sample_index
                ):
                    raise RuntimeError(
                        "sealed C5 prediction stream identity/order mismatch"
                    )
            prepared.append(
                (raw_path, _stream_bytes(stream), classification_row(row, metrics))
            )
        final_dataset = _build_bound_dataset_manifest(
            Path(data_root), rows, protocol, changed_message=True
        )
        if final_dataset != initial_dataset:
            raise RuntimeError("dataset changed during sealed evaluation")
        _verify_all_checkpoints(rows, phase="after all evaluators before stream publish")
        transaction.publish(prepared)
        return [classification for _path, _payload, classification in prepared]

    if stream_transaction is not None:
        return run_evaluation(stream_transaction)
    owned_transaction = StreamTransaction(rows)
    with owned_transaction:
        result = run_evaluation(owned_transaction)
        owned_transaction.commit_standalone()
        return result


def build_classification_multiseed_summary(
    per_run_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if len(per_run_rows) != 10 or any(set(row) != set(CLASSIFICATION_PER_RUN_FIELDS) for row in per_run_rows):
        raise RuntimeError("classification multiseed input must be exact classification_per_run rows")
    by_identity = {(str(row["group_id"]), int(row["seed"])): row for row in per_run_rows}
    if set(by_identity) != set(EXPECTED_IDENTITIES):
        raise RuntimeError("classification multiseed identity matrix is incomplete or duplicate")
    output: list[dict[str, Any]] = []
    for metric in METRICS:
        for group in GROUPS:
            values = np.asarray(
                [float(by_identity[(group, seed)][metric]) for seed in SEEDS], dtype=np.float64
            )
            if not np.isfinite(values).all():
                raise RuntimeError(f"classification metric {metric} must be finite")
            result = {
                "summary_type": "group_five_seed",
                "group_id": group,
                "claim_status": CLAIM_STATUS[group],
                "metric": metric,
                **{f"seed_{seed}": float(values[index]) for index, seed in enumerate(SEEDS)},
                "mean": float(values.mean()),
                "sample_std_ddof1": float(values.std(ddof=1)),
            }
            output.append(result)
        differences = np.asarray(
            [
                float(by_identity[("B2", seed)][metric])
                - float(by_identity[("B5", seed)][metric])
                for seed in SEEDS
            ],
            dtype=np.float64,
        )
        output.append(
            {
                "summary_type": "paired_seed_difference",
                "group_id": "B2-B5",
                "claim_status": "paired_B2_minus_B5",
                "metric": metric,
                **{
                    f"seed_{seed}": float(differences[index])
                    for index, seed in enumerate(SEEDS)
                },
                "mean": float(differences.mean()),
                "sample_std_ddof1": float(differences.std(ddof=1)),
            }
        )
    return output


def _exact_one(
    events: Sequence[Mapping[str, Any]], event_type: str, round_idx: int, client_id: str | None
) -> Mapping[str, Any]:
    matches = [
        event
        for event in events
        if event.get("event_type") == event_type
        and event.get("round") == round_idx
        and event.get("client_id") == client_id
    ]
    if len(matches) != 1:
        label = f"{event_type} round={round_idx} client={client_id}"
        if len(matches) > 1:
            raise RuntimeError(f"duplicate communication/phase evidence: {label}")
        raise RuntimeError(f"missing communication/phase evidence: {label}")
    return matches[0]


def _audit_direction(event: Mapping[str, Any], direction: str) -> tuple[dict[str, int], int, str]:
    payload = event.get("payload")
    key = f"{direction}_audit"
    audit = payload.get(key) if isinstance(payload, Mapping) else None
    if not isinstance(audit, Mapping):
        raise RuntimeError(f"missing {key}")
    logical = audit.get("logical")
    fields = DOWNLINK_LOGICAL_FIELDS if direction == "downlink" else UPLINK_LOGICAL_FIELDS
    if not isinstance(logical, Mapping) or set(logical) != set(fields):
        raise RuntimeError(f"{key} logical schema mismatch")
    logical_result: dict[str, int] = {}
    for field in fields:
        logical_result[field] = int(_finite_number(logical.get(field), field, integer=True))
    if direction == "downlink":
        expected_total = sum(
            logical_result[field]
            for field in (
                "logical_downlink_parameter_blob_bytes",
                "logical_downlink_semantic_proto_utf8_bytes",
                "logical_downlink_other_config_value_bytes",
            )
        )
        total_field = "logical_downlink_total_bytes"
    else:
        expected_total = sum(
            logical_result[field]
            for field in (
                "logical_uplink_parameter_blob_bytes",
                "logical_uplink_prototype_utf8_bytes",
                "logical_uplink_prototype_var_utf8_bytes",
                "logical_uplink_statistics_utf8_bytes",
                "logical_uplink_diagnostic_value_bytes",
            )
        )
        total_field = "logical_uplink_total_bytes"
    if logical_result[total_field] != expected_total:
        raise RuntimeError(f"{key} logical total mismatch")
    application_bytes = int(
        _finite_number(
            audit.get("application_message_bytes"),
            f"{key} application_message_bytes",
            integer=True,
        )
    )
    application_sha = audit.get("application_message_sha256")
    if not isinstance(application_sha, str) or HASH_RE.fullmatch(application_sha) is None:
        raise RuntimeError(f"{key} application message SHA-256 is invalid")
    return logical_result, application_bytes, application_sha


def _communication_tables(
    rows: Sequence[Mapping[str, Any]],
    evidence_by_attempt: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    per_round: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for row in rows:
        evidence = evidence_by_attempt[str(row["attempt_id"])]
        server = evidence.get("server")
        if not isinstance(server, list):
            raise RuntimeError("server evidence must be a list")
        run_rows: list[dict[str, Any]] = []
        for round_idx in range(1, 26):
            for client_id in ("C1", "C2"):
                down_event = _exact_one(server, "flower_fitins_prepared", round_idx, client_id)
                up_event = _exact_one(server, "flower_fitres_available", round_idx, client_id)
                down_logical, down_bytes, down_sha = _audit_direction(down_event, "downlink")
                up_logical, up_bytes, up_sha = _audit_direction(up_event, "uplink")
                result = {
                    "run_id": row["run_id"],
                    "attempt_id": row["attempt_id"],
                    "group_id": row["group_id"],
                    "seed": row["seed"],
                    "round": round_idx,
                    "client_id": client_id,
                    **down_logical,
                    **up_logical,
                    "application_downlink_message_bytes": down_bytes,
                    "application_downlink_message_sha256": down_sha,
                    "application_uplink_message_bytes": up_bytes,
                    "application_uplink_message_sha256": up_sha,
                    "application_round_total_bytes": down_bytes + up_bytes,
                    "transport_status": "not_collected",
                }
                run_rows.append(result)
        per_round.extend(run_rows)
        app_down = sum(item["application_downlink_message_bytes"] for item in run_rows)
        app_up = sum(item["application_uplink_message_bytes"] for item in run_rows)
        summaries.append(
            {
                "run_id": row["run_id"],
                "attempt_id": row["attempt_id"],
                "group_id": row["group_id"],
                "seed": row["seed"],
                "logical_downlink_25round_total_bytes": sum(
                    item["logical_downlink_total_bytes"] for item in run_rows
                ),
                "logical_uplink_25round_total_bytes": sum(
                    item["logical_uplink_total_bytes"] for item in run_rows
                ),
                "application_downlink_25round_total_bytes": app_down,
                "application_uplink_25round_total_bytes": app_up,
                "application_25round_total_bytes": app_down + app_up,
                "application_round_mean_bytes": (app_down + app_up) / 25.0,
                "transport_status": "not_collected",
            }
        )
    return per_round, summaries


def _payload_number(event: Mapping[str, Any], field: str) -> int:
    payload = event.get("payload")
    value = payload.get(field) if isinstance(payload, Mapping) else None
    return int(_finite_number(value, field, integer=True))


def _round_time_rows(
    rows: Sequence[Mapping[str, Any]],
    evidence_by_attempt: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    numeric_fields = ROUND_TIME_FIELDS[6:-1]
    for row in rows:
        evidence = evidence_by_attempt[str(row["attempt_id"])]
        server = evidence["server"]
        c1 = evidence["C1"]
        c2 = evidence["C2"]
        raw_rows: list[dict[str, Any]] = []
        for round_idx in range(1, 26):
            c1_train = _payload_number(_exact_one(c1, "client_train_end", round_idx, "C1"), "client_train_core_ns")
            c2_train = _payload_number(_exact_one(c2, "client_train_end", round_idx, "C2"), "client_train_core_ns")
            c1_fit = _payload_number(_exact_one(c1, "client_fit_end", round_idx, "C1"), "client_fit_callback_ns")
            c2_fit = _payload_number(_exact_one(c2, "client_fit_end", round_idx, "C2"), "client_fit_callback_ns")
            aggregate = _exact_one(server, "server_aggregate_end", round_idx, None)
            aggregate_total = _payload_number(aggregate, "server_aggregate_fit_total_ns")
            da_total = _payload_number(aggregate, "server_da_total_ns")
            non_da = _payload_number(aggregate, "server_aggregate_non_da_ns")
            if aggregate_total != da_total + non_da:
                raise RuntimeError("server aggregate/DA/non-DA timing mismatch")
            wall = _payload_number(_exact_one(server, "fit_round_end", round_idx, None), "fit_round_wall_ns")
            raw_rows.append(
                {
                    "run_id": row["run_id"],
                    "attempt_id": row["attempt_id"],
                    "group_id": row["group_id"],
                    "seed": row["seed"],
                    "statistic": "raw",
                    "round": round_idx,
                    "c1_client_train_core_ns": c1_train,
                    "c2_client_train_core_ns": c2_train,
                    "client_train_critical_path_ns": max(c1_train, c2_train),
                    "c1_client_fit_callback_ns": c1_fit,
                    "c2_client_fit_callback_ns": c2_fit,
                    "client_fit_critical_path_ns": max(c1_fit, c2_fit),
                    "server_aggregate_fit_total_ns": aggregate_total,
                    "server_da_total_ns": da_total,
                    "server_aggregate_non_da_ns": non_da,
                    "fit_round_wall_ns": wall,
                    "parallel_client_times_are_not_serially_additive": True,
                }
            )
        output.extend(raw_rows)
        for statistic in ("mean", "p50", "p95", "total"):
            result: dict[str, Any] = {
                "run_id": row["run_id"],
                "attempt_id": row["attempt_id"],
                "group_id": row["group_id"],
                "seed": row["seed"],
                "statistic": statistic,
                "round": "",
                "parallel_client_times_are_not_serially_additive": True,
            }
            for field in numeric_fields:
                values = np.asarray([float(item[field]) for item in raw_rows], dtype=np.float64)
                if statistic == "mean":
                    result[field] = float(values.mean())
                elif statistic == "p50":
                    result[field] = float(np.percentile(values, 50))
                elif statistic == "p95":
                    result[field] = float(np.percentile(values, 95))
                else:
                    result[field] = float(values.sum())
            output.append(result)
    return output


def _resource_rows(
    rows: Sequence[Mapping[str, Any]],
    evidence_by_attempt: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        audit = _read_audit(row)
        resource_audit = audit.get("resource")
        if not isinstance(resource_audit, Mapping):
            raise RuntimeError("attempt audit lacks resource coverage")
        evidence = evidence_by_attempt[str(row["attempt_id"])]
        for client_id in ("C1", "C2"):
            samples = evidence.get(f"resource_{client_id}")
            if not isinstance(samples, list) or not samples:
                raise RuntimeError(f"{client_id} resource samples are missing")
            payloads: list[Mapping[str, Any]] = []
            for event in samples:
                payload = event.get("payload") if isinstance(event, Mapping) else None
                if event.get("event_type") != "resource_sample" or not isinstance(payload, Mapping):
                    raise RuntimeError(f"{client_id} resource sample schema mismatch")
                payloads.append(payload)
            client_events = evidence.get(client_id)
            if not isinstance(client_events, list):
                raise RuntimeError(f"{client_id} client events are missing")
            active_intervals: list[tuple[int, int]] = []
            for round_idx in range(1, 26):
                start_event = _exact_one(
                    client_events, "client_fit_start", round_idx, client_id
                )
                end_event = _exact_one(
                    client_events, "client_fit_end", round_idx, client_id
                )
                start = _finite_number(
                    start_event.get("monotonic_ns"), "client fit start", integer=True
                )
                end = _finite_number(
                    end_event.get("monotonic_ns"), "client fit end", integer=True
                )
                if end < start:
                    raise RuntimeError(f"{client_id} client fit interval is reversed")
                active_intervals.append((int(start), int(end)))
            active_payloads: list[Mapping[str, Any]] = []
            for payload in payloads:
                interval_start = payload.get("sample_interval_start_monotonic_ns")
                interval_end = payload.get("sample_interval_end_monotonic_ns")
                if interval_start is None and interval_end is None:
                    # Synthetic callers may omit intervals, but formal evidence
                    # produced by the sampler always supplies both.
                    active_payloads.append(payload)
                    continue
                start = int(
                    _finite_number(interval_start, "resource sample interval start", integer=True)
                )
                end = int(
                    _finite_number(interval_end, "resource sample interval end", integer=True)
                )
                if end < start:
                    raise RuntimeError("resource sample interval is reversed")
                if any(start <= fit_end and end >= fit_start for fit_start, fit_end in active_intervals):
                    active_payloads.append(payload)
            if not active_payloads:
                raise RuntimeError(f"{client_id} has no active-fit resource samples")

            rss_all = np.asarray(
                [float(_finite_number(payload.get("rss_tree_bytes"), "rss_tree_bytes")) for payload in payloads]
            )
            rss_active = np.asarray(
                [float(_finite_number(payload.get("rss_tree_bytes"), "rss_tree_bytes")) for payload in active_payloads]
            )
            rss_peaks = np.asarray(
                [float(_finite_number(payload.get("rss_tree_peak_bytes"), "rss_tree_peak_bytes")) for payload in payloads]
            )
            cpu_one = np.asarray(
                [float(_finite_number(payload.get("cpu_percent_tree_one_core_scale"), "cpu one-core CPU")) for payload in active_payloads]
            )
            cpu_host = np.asarray(
                [float(_finite_number(payload.get("cpu_percent_tree_host_scale"), "cpu host CPU")) for payload in active_payloads]
            )
            coverage_payload = resource_audit.get(client_id)
            if not isinstance(coverage_payload, Mapping):
                raise RuntimeError(f"{client_id} audit resource coverage is missing")
            coverage = float(_finite_number(coverage_payload.get("coverage"), "resource coverage"))
            if coverage < 0.95 or coverage > 1.0:
                raise RuntimeError(f"{client_id} resource coverage is outside valid canonical range")
            temp_values = [
                float(_finite_number(payload.get("cpu_temperature_c"), "CPU temperature"))
                for payload in active_payloads
                if payload.get("cpu_temperature_available") is True
            ]
            throttling_values = [
                int(_finite_number(payload.get("throttled_bits"), "throttled_bits", integer=True))
                for payload in active_payloads
                if payload.get("throttled_available") is True
            ]
            output.append(
                {
                    "run_id": row["run_id"],
                    "attempt_id": row["attempt_id"],
                    "group_id": row["group_id"],
                    "seed": row["seed"],
                    "client_id": client_id,
                    "host_role": "Raspberry Pi" if client_id == "C1" else "PC",
                    "total_sample_count": len(payloads),
                    "active_sample_count": len(active_payloads),
                    "resource_coverage": coverage,
                    "expected_sample_points": int(
                        _finite_number(coverage_payload.get("expected_sample_points"), "expected sample points", integer=True)
                    ),
                    "covered_sample_points": int(
                        _finite_number(coverage_payload.get("covered_sample_points"), "covered sample points", integer=True)
                    ),
                    "rss_all_sample_mean_bytes": float(rss_all.mean()),
                    "rss_active_mean_bytes": float(rss_active.mean()),
                    "rss_peak_bytes": int(rss_peaks.max()),
                    "cpu_one_core_mean_percent": float(cpu_one.mean()),
                    "cpu_one_core_peak_percent": float(cpu_one.max()),
                    "cpu_host_mean_percent": float(cpu_host.mean()),
                    "cpu_host_peak_percent": float(cpu_host.max()),
                    "cpu_temperature_available": bool(temp_values),
                    "cpu_temperature_mean_c": float(np.mean(temp_values)) if temp_values else "",
                    "cpu_temperature_peak_c": max(temp_values) if temp_values else "",
                    "throttling_available": bool(throttling_values),
                    "throttling_observed": any(value != 0 for value in throttling_values) if throttling_values else "",
                }
            )
    return output


def _all_evidence_events(evidence: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    result: list[Mapping[str, Any]] = []
    for key in ("server", "C1", "C2", "resource_C1", "resource_C2"):
        rows = evidence.get(key)
        if not isinstance(rows, list):
            raise RuntimeError(f"evidence {key} must be a list")
        result.extend(rows)
    return result


def _overhead_rows(
    rows: Sequence[Mapping[str, Any]],
    evidence_by_attempt: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    close_numeric = (
        "observer_flower_serialize_ns",
        "observer_event_encode_ns",
        "observer_io_write_ns",
        "observer_fsync_ns",
        "observer_total_ns",
        "observer_event_bytes_written",
        "observer_event_count",
        "observer_reporting_tail_bytes",
    )
    paired_numeric = (
        "observer_flower_serialize_ns",
        "observer_event_encode_ns",
        "observer_io_write_ns",
        "observer_fsync_ns",
        "observer_total_ns",
    )
    for row in rows:
        evidence = evidence_by_attempt[str(row["attempt_id"])]
        server = evidence["server"]
        wall_total = sum(
            _payload_number(_exact_one(server, "fit_round_end", round_idx, None), "fit_round_wall_ns")
            for round_idx in range(1, 26)
        )
        if wall_total <= 0:
            raise RuntimeError("fit-round wall total must be positive for overhead disclosure")
        closes = evidence.get("close_summaries")
        if not isinstance(closes, list) or not closes:
            raise RuntimeError("observer close summaries are missing")
        all_events = _all_evidence_events(evidence)
        producer_rows: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for close in sorted(closes, key=lambda item: (str(item.get("host_id")), str(item.get("producer")))):
            if not isinstance(close, Mapping):
                raise RuntimeError("observer close summary must be an object")
            key = (str(close.get("host_id")), str(close.get("producer")))
            if key in seen:
                raise RuntimeError("duplicate observer close summary")
            seen.add(key)
            if close.get("run_id") != row["run_id"] or close.get("attempt_id") != row["attempt_id"]:
                raise RuntimeError("observer close summary identity mismatch")
            close_values = {
                field: int(_finite_number(close.get(field), field, integer=True))
                for field in close_numeric
            }
            if close_values["observer_total_ns"] != sum(
                close_values[field]
                for field in (
                    "observer_flower_serialize_ns",
                    "observer_event_encode_ns",
                    "observer_io_write_ns",
                    "observer_fsync_ns",
                )
            ):
                raise RuntimeError("observer close summary total mismatch")
            overhead_events = [
                event
                for event in all_events
                if event.get("event_type") == "observer_overhead"
                and event.get("host_id") == close.get("host_id")
                and event.get("producer") == close.get("producer")
            ]
            paired_values = {field: 0 for field in paired_numeric}
            for event in overhead_events:
                payload = event.get("payload")
                if not isinstance(payload, Mapping):
                    raise RuntimeError("observer overhead payload is missing")
                for field in paired_numeric:
                    paired_values[field] += int(
                        _finite_number(payload.get(field), field, integer=True)
                    )
                if payload.get("observer_total_ns") != sum(
                    payload.get(field)
                    for field in (
                        "observer_flower_serialize_ns",
                        "observer_event_encode_ns",
                        "observer_io_write_ns",
                        "observer_fsync_ns",
                    )
                ):
                    raise RuntimeError("observer paired overhead total mismatch")
            producer_rows.append(
                {
                    "run_id": row["run_id"],
                    "attempt_id": row["attempt_id"],
                    "group_id": row["group_id"],
                    "seed": row["seed"],
                    "host_id": close["host_id"],
                    "producer": close["producer"],
                    "paired_domain_event_count": len(overhead_events),
                    "paired_observer_flower_serialize_ns": paired_values["observer_flower_serialize_ns"],
                    "paired_observer_event_encode_ns": paired_values["observer_event_encode_ns"],
                    "paired_observer_io_write_ns": paired_values["observer_io_write_ns"],
                    "paired_observer_fsync_ns": paired_values["observer_fsync_ns"],
                    "paired_observer_total_ns": paired_values["observer_total_ns"],
                    "close_observer_event_count": close_values["observer_event_count"],
                    "close_observer_flower_serialize_ns": close_values["observer_flower_serialize_ns"],
                    "close_observer_event_encode_ns": close_values["observer_event_encode_ns"],
                    "close_observer_io_write_ns": close_values["observer_io_write_ns"],
                    "close_observer_fsync_ns": close_values["observer_fsync_ns"],
                    "close_observer_total_ns": close_values["observer_total_ns"],
                    "close_observer_event_bytes_written": close_values["observer_event_bytes_written"],
                    "observer_reporting_tail_bytes": close_values["observer_reporting_tail_bytes"],
                    "fit_round_wall_ns_total": wall_total,
                    "observer_total_to_round_wall_ratio": close_values["observer_total_ns"] / wall_total,
                }
            )
        output.extend(producer_rows)
        total: dict[str, Any] = {
            "run_id": row["run_id"],
            "attempt_id": row["attempt_id"],
            "group_id": row["group_id"],
            "seed": row["seed"],
            "host_id": "ALL",
            "producer": "ALL",
            "fit_round_wall_ns_total": wall_total,
        }
        additive = OVERHEAD_FIELDS[6:20]
        for field in additive:
            total[field] = sum(float(item[field]) for item in producer_rows)
            if all(isinstance(item[field], int) for item in producer_rows):
                total[field] = int(total[field])
        total["observer_total_to_round_wall_ratio"] = total["close_observer_total_ns"] / wall_total
        output.append(total)
    return output


class SystemTables(dict[str, list[dict[str, Any]]]):
    """Summary tables plus the exact bytes consumed under Task7 audit approval."""

    def __init__(
        self,
        tables: Mapping[str, list[dict[str, Any]]],
        *,
        evidence_bindings: Mapping[str, Mapping[str, Mapping[str, str]]],
    ) -> None:
        super().__init__(tables)
        self.evidence_bindings = {
            attempt_id: {
                kind: dict(sorted(hashes.items()))
                for kind, hashes in binding.items()
            }
            for attempt_id, binding in sorted(evidence_bindings.items())
        }


def build_system_tables(
    canonical_rows: Sequence[Mapping[str, Any]],
    evidence_loader: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    *,
    protocol: Mapping[str, Any] | None = None,
    raw_root: Path | None = None,
) -> dict[str, list[dict[str, Any]]]:
    rows = assert_test_gate(canonical_rows, protocol=protocol, raw_root=raw_root)
    evidence_by_attempt: dict[str, Mapping[str, Any]] = {}
    evidence_bindings: dict[str, dict[str, dict[str, str]]] = {}
    for row in rows:
        attempt_id = str(row["attempt_id"])
        evidence = evidence_loader(row)
        evidence_by_attempt[attempt_id] = evidence
        binding = evidence.get("_evidence_binding")
        if binding is None:
            continue
        if not isinstance(binding, Mapping):
            raise RuntimeError("audit evidence binding must be an object")
        approved = binding.get("audit_approved")
        consumed = binding.get("consumed")
        if (
            not isinstance(approved, Mapping)
            or not isinstance(consumed, Mapping)
            or dict(approved) != dict(consumed)
            or set(approved) != set(AUDITED_EVIDENCE_PATHS)
        ):
            raise RuntimeError("audit-approved and consumed evidence SHA-256 bindings differ")
        evidence_bindings[attempt_id] = {
            "audit_approved": dict(approved),
            "consumed": dict(consumed),
        }
    communication, communication_summary = _communication_tables(rows, evidence_by_attempt)
    tables = {
        "flower_communication_per_round.csv": communication,
        "flower_communication_summary.csv": communication_summary,
        "flower_round_time_breakdown.csv": _round_time_rows(rows, evidence_by_attempt),
        "training_resource_summary.csv": _resource_rows(rows, evidence_by_attempt),
        "observer_overhead_summary.csv": _overhead_rows(rows, evidence_by_attempt),
    }
    for name, table_rows in tables.items():
        csv_bytes(table_rows, TABLE_FIELDS[name])
    return SystemTables(tables, evidence_bindings=evidence_bindings)


def _relative(path: Path, root: Path) -> str:
    try:
        return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"path escapes raw root: {path}") from exc


def _attempt_registry(
    rows: Sequence[Mapping[str, Any]],
    raw_root: Path,
    stream_transaction: StreamTransaction,
) -> list[dict[str, Any]]:
    registry: list[dict[str, Any]] = []
    for row in rows:
        fingerprint = stream_transaction.stream_fingerprint(row)
        stream_path = Path(fingerprint["path"])
        registry.append({
            "run_id": row["run_id"],
            "attempt_id": row["attempt_id"],
            "group_id": row["group_id"],
            "seed": row["seed"],
            "state": "canonical",
            "claim_status": CLAIM_STATUS[str(row["group_id"])],
            "confirmation_commit": row["confirmation_commit"],
            "source_archive_sha256": row["source_archive_sha256"],
            "dataset_manifest_sha256": row["dataset_manifest_sha256"],
            "algorithm_config_sha256": row["algorithm_config_sha256"],
            "protocol_manifest_sha256": row["protocol_manifest_sha256"],
            "audit_sha256": row["audit_sha256"],
            "checkpoint_sha256": row["checkpoint_sha256"],
            "classification_stream_sha256": fingerprint["sha256"],
            "classification_stream_size_bytes": fingerprint["size_bytes"],
            "classification_stream_device": fingerprint["device"],
            "classification_stream_inode": fingerprint["inode"],
            "attempt_relative_path": _relative(Path(row["attempt_dir"]), raw_root),
            "audit_relative_path": _relative(Path(row["audit_path"]), raw_root),
            "checkpoint_relative_path": _relative(Path(row["checkpoint_path"]), raw_root),
            "classification_stream_relative_path": _relative(stream_path, raw_root),
            "transport_status": "not_collected",
        })
    return registry


def _input_manifest(
    rows: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
    raw_root: Path,
    evidence_bindings: Mapping[str, Mapping[str, Mapping[str, str]]],
) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []
    for row in rows:
        input_files: dict[str, str] = {}
        attempt_dir = Path(row["attempt_dir"])
        for path in sorted(attempt_dir.rglob("*"), key=lambda item: item.as_posix()):
            if path.is_symlink() or _is_reparse(path):
                raise RuntimeError(f"input evidence tree contains symlink/reparse point: {path}")
            if path.is_file() and not path.is_symlink() and not _is_reparse(path):
                relative = _relative(path, raw_root)
                input_files[relative] = _sha256_file(path)
        binding = evidence_bindings.get(str(row["attempt_id"]))
        if not isinstance(binding, Mapping):
            raise RuntimeError(
                f"missing audit-approved consumed evidence binding: {row['attempt_id']}"
            )
        approved = binding.get("audit_approved")
        consumed = binding.get("consumed")
        if (
            not isinstance(approved, Mapping)
            or not isinstance(consumed, Mapping)
            or dict(approved) != dict(consumed)
            or set(approved) != set(AUDITED_EVIDENCE_PATHS)
        ):
            raise RuntimeError("audit-approved and consumed evidence binding mismatch")
        for relative, consumed_sha in consumed.items():
            current_sha = input_files.get(
                (attempt_dir / str(relative)).resolve().relative_to(raw_root.resolve()).as_posix()
            )
            if current_sha != consumed_sha:
                raise RuntimeError(
                    f"audit input changed after summary consumption: {row['attempt_id']} {relative}"
                )
        attempts.append(
            {
                "run_id": row["run_id"],
                "attempt_id": row["attempt_id"],
                "audit_sha256": row["audit_sha256"],
                "checkpoint_sha256": row["checkpoint_sha256"],
                "audit_approved_evidence_inputs": dict(sorted(approved.items())),
                "consumed_evidence_inputs": dict(sorted(consumed.items())),
                "input_files": input_files,
            }
        )
    return {
        "schema_version": "iotj.confirmation.summary_inputs.v1",
        "protocol_manifest_sha256": protocol["protocol_manifest_sha256"],
        "confirmation_commit": rows[0]["confirmation_commit"],
        "source_archive_sha256": rows[0]["source_archive_sha256"],
        "dataset_manifest_sha256": rows[0]["dataset_manifest_sha256"],
        "attempts": attempts,
    }


def _claim_boundary() -> str:
    return """# Confirmation claim boundary

- Scope: only `C1/C2 -> C5`, B2/B5, seeds 42–46 from one frozen confirmation revision.
- B2 is **post-screen exploratory**; it is not promoted to a preregistered confirmatory method.
- B5 is the **predeclared full method**.
- The historical `feaa75b` seed-42 screening evidence and all cross-direction runs are excluded from confirmation mean/std.
- Failed, invalid, aborted, incomplete, duplicate, or metric-driven rerun attempts are excluded.
- Communication claims are serialized Flower application-message bytes, with logical payload components reported separately.
- Transport bytes were **not collected** (`transport_status=not_collected`) and are neither reported as zero nor inferred.
- Client training is parallel; C1/C2 times are reported separately and by critical-path maximum, never serially added to server time.
"""


def _claim_map() -> str:
    return """# Claim-to-evidence map

| Claim | Evidence | Boundary |
|---|---|---|
| Five-seed C5 classification | `classification_per_run.csv`, `classification_multiseed_summary.csv` | Exact ten canonical checkpoints only |
| Flower application communication | `flower_communication_per_round.csv`, `flower_communication_summary.csv` | Logical and serialized application layers; no inferred transport |
| Training phase time | `flower_round_time_breakdown.csv` | Client parallel critical path is `max(C1,C2)` |
| Pi/PC training resource use | `training_resource_summary.csv` | Canonical attempts with audit coverage >=95% |
| Observer measurement overhead | `observer_overhead_summary.csv` | Raw phase times are not corrected by subtracting observer cost |
| Provenance and eligibility | `attempt_registry.csv`, `summary_input_manifest.json` | Audit/checkpoint/input SHA-256 bound |
"""


def _publish_summary_staging(staging: Path, destination: Path) -> None:
    """Injectable final atomic publish seam with an immediate no-overwrite check."""

    if destination.exists() or destination.is_symlink() or _is_reparse(destination):
        raise FileExistsError(f"refusing to overwrite summary output: {destination}")
    staging.rename(destination)


def write_summary_bundle(
    output_root: Path,
    canonical_rows: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
    classification_per_run: Sequence[Mapping[str, Any]],
    system_tables: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    raw_root: Path,
    stream_transaction: StreamTransaction,
) -> None:
    rows = assert_test_gate(canonical_rows, protocol=protocol, raw_root=raw_root)
    output_root = Path(output_root)
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"refusing to overwrite summary output: {output_root}")
    if not output_root.parent.is_dir() or output_root.parent.is_symlink():
        raise RuntimeError("summary output parent must be a regular directory")
    if _has_link_component(output_root.parent):
        raise RuntimeError("summary output parent contains a symlink/reparse point")
    if not stream_transaction.active or stream_transaction.committed:
        raise RuntimeError("summary publication requires an active stream transaction")
    if [str(row["attempt_id"]) for row in stream_transaction.rows] != [
        str(row["attempt_id"]) for row in rows
    ]:
        raise RuntimeError("summary/stream transaction attempt matrix mismatch")
    stream_transaction.verify_streams()
    per_run = [dict(row) for row in classification_per_run]
    multiseed = build_classification_multiseed_summary(per_run)
    if set(system_tables) != set(TABLE_FIELDS):
        raise RuntimeError("system table set is not exact")
    registry = _attempt_registry(
        rows, Path(raw_root), stream_transaction
    )
    bindings = getattr(system_tables, "evidence_bindings", None)
    if not isinstance(bindings, Mapping) or set(bindings) != {
        str(row["attempt_id"]) for row in rows
    }:
        raise RuntimeError("system tables lack exact audit-approved evidence bindings")
    input_manifest = _input_manifest(
        rows, protocol, Path(raw_root), bindings
    )
    staging = output_root.parent / f".{output_root.name}.{uuid.uuid4().hex}.staging"
    staging.mkdir()
    try:
        files: dict[str, bytes] = {
            "confirmation_protocol_manifest.json": _canonical_json_bytes(protocol) + b"\n",
            "classification_per_run.csv": csv_bytes(per_run, CLASSIFICATION_PER_RUN_FIELDS),
            "classification_multiseed_summary.csv": csv_bytes(
                multiseed, CLASSIFICATION_MULTISEED_FIELDS
            ),
            "attempt_registry.csv": csv_bytes(registry, ATTEMPT_REGISTRY_FIELDS),
            "summary_input_manifest.json": _canonical_json_bytes(input_manifest) + b"\n",
            "claim_boundary.md": _claim_boundary().encode("utf-8"),
            "claim_to_evidence_map.md": _claim_map().encode("utf-8"),
        }
        for name in sorted(TABLE_FIELDS):
            files[name] = csv_bytes(system_tables[name], TABLE_FIELDS[name])
        for name in sorted(files):
            path = staging / name
            with path.open("xb") as handle:
                handle.write(files[name])
                handle.flush()
                os.fsync(handle.fileno())
        # Recheck every mutable input immediately before the only summary
        # publish operation. The second manifest must be byte-for-byte stable.
        stream_transaction.verify_streams()
        _verify_all_checkpoints(
            rows, phase="immediately before final summary publish"
        )
        final_input_manifest = _input_manifest(
            rows, protocol, Path(raw_root), bindings
        )
        if final_input_manifest != input_manifest:
            raise RuntimeError("inputs changed before final summary publish")
        _publish_summary_staging(staging, output_root)
        stream_transaction.commit_after_summary_publish()
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _load_status(path: Path) -> dict[str, Any]:
    status = _load_json(path)
    trace = _metric_trace(status)
    if trace is not None:
        raise RuntimeError(f"metric-driven rerun/selection trace is forbidden: {trace}")
    return status


def discover_canonical_attempts(
    raw_root: Path, protocol: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Discover canonical attempts from immutable status, never from metrics."""

    raw_root = Path(raw_root)
    _protocol_self_hash(protocol)
    if not raw_root.is_dir() or raw_root.is_symlink() or _is_reparse(raw_root):
        raise RuntimeError("raw root must be a regular non-reparse directory")
    schedule = protocol.get("schedule")
    if not isinstance(schedule, list):
        raise RuntimeError("protocol schedule must be a list")
    expected_run_ids = {str(row.get("run_id")) for row in schedule if isinstance(row, Mapping)}
    actual_run_dirs = [path for path in raw_root.iterdir() if path.is_dir()]
    unexpected = sorted(path.name for path in actual_run_dirs if path.name not in expected_run_ids)
    if unexpected:
        raise RuntimeError(f"raw root contains cross-direction/extra run directories: {unexpected}")
    rows: list[dict[str, Any]] = []
    schedule_by_run = {str(row["run_id"]): row for row in schedule if isinstance(row, Mapping)}
    for run_id in sorted(expected_run_ids):
        run_root = raw_root / run_id
        if not run_root.is_dir() or _has_link_component(run_root, raw_root):
            continue
        canonical: list[tuple[int, Path, dict[str, Any]]] = []
        for attempt_dir in sorted(run_root.iterdir(), key=lambda item: item.name):
            match = ATTEMPT_RE.fullmatch(attempt_dir.name)
            if match is None or match.group(1) != run_id or not attempt_dir.is_dir():
                raise RuntimeError(f"malformed attempt directory: {attempt_dir}")
            if _has_link_component(attempt_dir, raw_root):
                raise RuntimeError(f"attempt path contains symlink/reparse point: {attempt_dir}")
            status_path = attempt_dir / "attempt_status.json"
            if not status_path.is_file() or status_path.is_symlink():
                raise RuntimeError(f"attempt status is missing: {attempt_dir}")
            status = _load_status(status_path)
            status_events = attempt_dir / "status_events"
            if status_events.exists():
                if not status_events.is_dir() or _has_link_component(status_events, attempt_dir):
                    raise RuntimeError(f"status event tree is unsafe: {status_events}")
                numbered = sorted(status_events.glob("status_*.json"))
                for status_event_path in numbered:
                    _load_status(status_event_path)
            if status.get("run_id") != run_id or status.get("attempt_id") != attempt_dir.name:
                raise RuntimeError("attempt status identity mismatch")
            state = status.get("state")
            if state == "canonical":
                if status.get("reason") != "validator_accepted":
                    raise RuntimeError("canonical status was not validator accepted")
                canonical.append((int(match.group(2)), attempt_dir, status))
            elif state not in {"failed", "invalid", "aborted"}:
                raise RuntimeError(f"nonterminal attempt cannot enter sealed summary: {attempt_dir}")
            elif (attempt_dir / "attempt_audit.json").is_file():
                earlier_audit = _load_json(attempt_dir / "attempt_audit.json")
                if earlier_audit.get("status") == "valid":
                    raise RuntimeError(
                        "a structurally valid earlier attempt was bypassed; possible metric-driven rerun"
                    )
        if len(canonical) != 1:
            raise RuntimeError(f"run {run_id} must have exactly one canonical attempt")
        _attempt_number, attempt_dir, status = canonical[0]
        provenance = _load_json(attempt_dir / "attempt_provenance.json")
        schedule_row = schedule_by_run[run_id]
        audit_path = attempt_dir / "attempt_audit.json"
        checkpoint_path = (
            attempt_dir / "raw" / "ecs" / "training" / "server_latest_adapted.pth"
        )
        audit_sha = _sha256_file(audit_path) if audit_path.is_file() else ""
        if status.get("audit_sha256") != audit_sha:
            raise RuntimeError(f"canonical status/audit SHA-256 mismatch: {attempt_dir.name}")
        rows.append(
            {
                "run_id": run_id,
                "attempt_id": attempt_dir.name,
                "attempt_dir": attempt_dir,
                "group_id": schedule_row.get("group_id"),
                "seed": schedule_row.get("seed"),
                "state": "canonical",
                "status_reason": "validator_accepted",
                "direction": protocol.get("direction"),
                "historical_seed42_included": protocol.get("historical_seed42_included"),
                "confirmation_commit": provenance.get("confirmation_commit"),
                "source_archive_sha256": provenance.get("source_archive_sha256"),
                "dataset_manifest_sha256": provenance.get("dataset_manifest_sha256"),
                "algorithm_config_sha256": provenance.get("algorithm_config_sha256"),
                "protocol_manifest_sha256": protocol.get("protocol_manifest_sha256"),
                "audit_path": audit_path,
                "audit_sha256": audit_sha,
                "checkpoint_path": checkpoint_path,
                "checkpoint_sha256": _sha256_file(checkpoint_path) if checkpoint_path.is_file() else "",
                "transport_status": schedule_row.get("transport_status"),
            }
        )
    return assert_test_gate(rows, protocol=protocol, raw_root=raw_root)


def load_attempt_evidence(row: Mapping[str, Any]) -> dict[str, Any]:
    attempt_dir = Path(row["attempt_dir"])
    evidence_keys = {
        "raw/ecs/events.jsonl": "server",
        "raw/pi/events.jsonl": "C1",
        "raw/pi/resource.jsonl": "resource_C1",
        "raw/pc/events.jsonl": "C2",
        "raw/pc/resource.jsonl": "resource_C2",
    }
    audit = _read_audit(row)
    approved = audit.get("inputs")
    if not isinstance(approved, Mapping) or set(approved) != set(
        AUDITED_EVIDENCE_PATHS
    ):
        raise RuntimeError(
            "audit input set must equal the exact five JSONL plus five close summaries"
        )
    approved_hashes: dict[str, str] = {}
    for relative in AUDITED_EVIDENCE_PATHS:
        value = approved.get(relative)
        if not isinstance(value, str) or HASH_RE.fullmatch(value) is None:
            raise RuntimeError(f"audit input SHA-256 is invalid: {relative}")
        approved_hashes[relative] = value

    raw_root = attempt_dir / "raw"
    actual_candidates: set[str] = set()
    if raw_root.is_dir():
        for path in raw_root.rglob("*"):
            if path.is_symlink() or _is_reparse(path):
                raise RuntimeError(f"audit input evidence tree contains symlink: {path}")
            if path.is_file() and (
                path.name.endswith(".jsonl") or path.name.endswith(".close.json")
            ):
                actual_candidates.add(path.relative_to(attempt_dir).as_posix())
    if actual_candidates != set(AUDITED_EVIDENCE_PATHS):
        missing = sorted(set(AUDITED_EVIDENCE_PATHS) - actual_candidates)
        extra = sorted(actual_candidates - set(AUDITED_EVIDENCE_PATHS))
        raise RuntimeError(
            f"audit input evidence file set mismatch; missing={missing}, extra={extra}"
        )

    result: dict[str, Any] = {}
    close_summaries: list[dict[str, Any]] = []
    consumed_hashes: dict[str, str] = {}
    for relative in AUDITED_EVIDENCE_PATHS:
        path = attempt_dir / relative
        if not _within(path, attempt_dir) or _has_link_component(path, attempt_dir):
            raise RuntimeError(f"audit input path escapes or is a symlink: {path}")
        try:
            data = path.read_bytes()
        except OSError as exc:
            raise RuntimeError(f"cannot read audit input {relative}: {exc}") from exc
        actual_sha = hashlib.sha256(data).hexdigest()
        if actual_sha != approved_hashes[relative]:
            raise RuntimeError(
                f"audit-approved evidence SHA-256 mismatch: {relative}; "
                f"expected {approved_hashes[relative]}, got {actual_sha}"
            )
        consumed_hashes[relative] = actual_sha
        if relative.endswith(".jsonl"):
            result[evidence_keys[relative]] = _parse_jsonl_bytes(
                data, label=relative
            )
        else:
            close_summaries.append(
                _parse_json_object_bytes(data, label=relative)
            )
    result["close_summaries"] = close_summaries
    result["_evidence_binding"] = {
        "audit_approved": dict(sorted(approved_hashes.items())),
        "consumed": dict(sorted(consumed_hashes.items())),
    }
    return result


def summarize_confirmation(
    *,
    raw_root: Path,
    protocol: Mapping[str, Any],
    data_root: Path,
    output_root: Path,
    device: Any,
    batch_size: int,
    evaluator: Callable[..., tuple[list[dict[str, Any]], dict[str, Any]]] = evaluate_checkpoint_stream,
) -> None:
    rows = discover_canonical_attempts(raw_root, protocol)
    # Complete training-side rendering first; no target-test access occurs here.
    system_tables = build_system_tables(
        rows, load_attempt_evidence, protocol=protocol, raw_root=raw_root
    )
    # Keep stream ownership inside one transaction until the summary directory
    # is durably published.  Any evaluator, evidence, staging, fsync, or rename
    # failure therefore removes every stream that this attempt still owns.
    with StreamTransaction(rows) as stream_transaction:
        per_run = evaluate_canonical_attempts(
            rows,
            data_root=data_root,
            device=device,
            batch_size=batch_size,
            evaluator=evaluator,
            protocol=protocol,
            raw_root=raw_root,
            stream_transaction=stream_transaction,
        )
        write_summary_bundle(
            output_root,
            rows,
            protocol,
            per_run,
            system_tables,
            raw_root=raw_root,
            stream_transaction=stream_transaction,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--protocol-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    if args.output_root.exists() or args.output_root.is_symlink():
        raise FileExistsError(f"refusing to overwrite summary output: {args.output_root}")
    protocol = _load_json(args.protocol_manifest)
    # Device resolution is deliberately after immutable protocol loading; C5 is
    # still not opened until discover/test Gate and system rendering succeed.
    device = resolve_device(args.device)
    summarize_confirmation(
        raw_root=args.raw_root,
        protocol=protocol,
        data_root=args.data_root,
        output_root=args.output_root,
        device=device,
        batch_size=args.batch_size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
