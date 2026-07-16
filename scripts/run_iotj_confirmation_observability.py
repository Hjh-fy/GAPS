"""Orchestrate immutable IoTJ confirmation attempts on the frozen topology.

The controller is intentionally transport- and process-injectable.  Its unit tests
never connect to ECS or Raspberry Pi and never start a training process.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import stat
import subprocess
import sys
import tarfile
import threading
import time
import uuid
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Sequence

from scripts.freeze_iotj_confirmation_protocol import (
    ALGORITHM_CONFIG_FIELDS,
    CONFIRMATION_SCHEDULE,
    canonical_sha256,
    confirmation_run_id,
    sha256_file,
)
from scripts.run_iotj_classification_cloud_edge import (
    _remote_python,
    _run,
    _ssh,
    _start_tunnels,
    _terminate_processes,
    _wait_for_pi,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUEUE = CONFIRMATION_SCHEDULE
DEFAULT_RAW_ROOT = (
    REPO_ROOT / "results" / "iotj_main_confirmation_observability_20260715" / "raw"
)
DEFAULT_PC_RUNTIME_ROOT = Path(
    "results/iotj_main_confirmation_observability_20260715/runtime"
)
ECS_REMOTE_RUNTIME_BASE = "/root/GAPS/confirmation_runtime"
PI_REMOTE_RUNTIME_BASE = "/home/gaps/GAPS/confirmation_runtime"
EXPECTED_DEPENDENCY_VERSIONS = {
    "flwr": "1.23.0",
    "protobuf": "4.25.8",
    "psutil": "7.0.0",
}
VALID_ATTEMPT_STATES = {"running", "failed", "aborted", "invalid", "canonical"}
TERMINAL_ATTEMPT_STATES = {"failed", "aborted", "invalid", "canonical"}
STATUS_REASON_CATEGORIES = {
    "attempt_allocated": "controller",
    "preflight_passed": "controller",
    "validator_accepted": "audit",
    "validator_rejected": "audit",
    "archive_integrity_failure": "archive_integrity",
    "dataset_integrity_failure": "dataset_integrity",
    "config_integrity_failure": "config_integrity",
    "dependency_mismatch": "dependency_mismatch",
    "transport_failure": "transport",
    "tunnel_failure": "tunnel",
    "process_failure": "process",
    "evidence_io_failure": "evidence_io",
    "audit_failure": "audit",
    "resource_coverage_failure": "resource_coverage",
    "observer_failure": "observer_failure",
    "operator_abort": "operator",
    "cleanup_failure": "process",
}
OBJECTIVE_RERUN_REASON_CATEGORIES = frozenset(
    {
        "archive_integrity",
        "dataset_integrity",
        "config_integrity",
        "dependency_mismatch",
        "transport",
        "tunnel",
        "process",
        "evidence_io",
        "audit",
        "resource_coverage",
        "observer_failure",
    }
)
EXPECTED_TOPOLOGY = {
    "server": "Alibaba Cloud ECS",
    "C1": "physical Raspberry Pi CPU",
    "C2": "physical Windows PC CPU",
    "C5": "server-side calibration only; no target test labels in training",
}
_RUN_ID_RE = re.compile(r"^c12_to_c5__(b2|b5)__s(42|43|44|45|46)$")
_ATTEMPT_ID_RE = re.compile(
    r"^(c12_to_c5__(?:b2|b5)__s(?:42|43|44|45|46))__a([0-9]{3})$"
)
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_INSTANCE_RE = re.compile(r"^[0-9a-f]{32}$")
_RFC3339_UTC_RE = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]{1,6})?Z$"
)
_FORBIDDEN_STATUS_KEY_PARTS = (
    "accuracy",
    "loss",
    "nll",
    "ece",
    "recall",
    "f1",
    "metric",
    "metrics",
)
_PROVENANCE_KEYS = {
    "run_id",
    "attempt_id",
    "confirmation_commit",
    "source_archive_sha256",
    "dataset_manifest_sha256",
    "algorithm_config_sha256",
    "controller_owner",
}
_STATUS_KEYS = {
    "sequence",
    "run_id",
    "attempt_id",
    "state",
    "event_type",
    "reason",
    "reason_category",
    "wall_time_utc",
    "confirmation_commit",
    "source_archive_sha256",
    "dataset_manifest_sha256",
    "algorithm_config_sha256",
    "audit_sha256",
}
_LAUNCH_REGISTRATION_KEYS = {
    "schema_version",
    "label",
    "launch_token",
    "registration_path",
    "child_pid",
    "owner_pid",
    "owner_pgid",
    "owner_start_ticks",
}
_ATTEMPT_LOCKS: dict[str, threading.RLock] = {}
_ATTEMPT_LOCKS_GUARD = threading.Lock()


class AttemptFailure(RuntimeError):
    """An objective controller failure with a stable rerun category."""

    reason_category = "process"


class ArchiveMismatch(AttemptFailure):
    reason_category = "archive_integrity"


class AttemptAborted(AttemptFailure):
    reason_category = "operator_abort"


@dataclass(frozen=True)
class Attempt:
    run_id: str
    attempt_id: str
    path: Path


@dataclass(frozen=True)
class Provenance:
    confirmation_commit: str | None
    source_archive_sha256: str | None
    dataset_manifest_sha256: str | None
    algorithm_config_sha256: str | None

    def require_complete(self) -> None:
        if not isinstance(self.confirmation_commit, str) or not _COMMIT_RE.fullmatch(
            self.confirmation_commit
        ):
            raise ValueError("confirmation_commit must be a full lowercase commit SHA")
        for name in (
            "source_archive_sha256",
            "dataset_manifest_sha256",
            "algorithm_config_sha256",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not _HASH_RE.fullmatch(value):
                raise ValueError(f"{name} must be a lowercase SHA-256")


@dataclass(frozen=True)
class HostDeployment:
    host_id: str
    archive_path: str | Path
    src_path: str | Path
    source_archive_sha256: str
    regular_members_sha256: str


@dataclass(frozen=True)
class FrozenRun:
    run_id: str
    group_id: str
    seed: int
    manifest_path: Path
    manifest: dict[str, Any]
    provenance: Provenance


@dataclass(frozen=True)
class FrozenInputs:
    protocol_path: Path
    protocol: dict[str, Any]
    source_manifest_path: Path
    source_manifest: dict[str, Any]
    dataset_manifest_path: Path
    dataset_manifest: dict[str, Any]
    command_root: Path
    archive_path: Path
    runs: tuple[FrozenRun, ...]


@dataclass(frozen=True)
class OwnedProcess:
    host_id: str
    label: str
    pid: int
    owner_pid: int | None = None
    owner_pgid: int | None = None
    owner_start_ticks: int | None = None
    registration_path: str | Path | None = None
    launch_token: str | None = None
    handle: Any | None = None
    host: str | None = None
    python_bin: str | None = None
    exit_path: str | Path | None = None
    stop_path: str | Path | None = None
    log_handles: tuple[Any, ...] = ()


@dataclass(frozen=True)
class ProductionRuntime:
    frozen: FrozenInputs
    frozen_run: FrozenRun
    deployments: Mapping[str, HostDeployment]
    ecs_host: str
    pi_host: str
    validator: Path
    poll_seconds: float
    timeout_seconds: float
    pc_runtime_root: Path


@dataclass(frozen=True)
class FormalSmokeConfig:
    """Explicit non-canonical two-round topology-smoke command overlay."""

    observer_enabled: bool
    trace_output: str
    initial_checkpoint: str
    mode: str = "off"
    namespace: str = "noncanonical_smoke"
    rounds: int = 2
    local_epochs: int = 1
    initial_checkpoint_source: Path | None = None
    initial_checkpoint_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.namespace != "noncanonical_smoke":
            raise ValueError("formal smoke namespace must be noncanonical_smoke")
        if self.rounds != 2:
            raise ValueError("formal smoke rounds must equal 2")
        if self.local_epochs != 1:
            raise ValueError("formal smoke local_epochs must equal 1")
        if self.mode not in {"off", "on"}:
            raise ValueError("formal smoke mode must be off or on")
        if self.observer_enabled != (self.mode == "on"):
            raise ValueError("formal smoke observer_enabled must match mode")
        if not self.trace_output or not self.initial_checkpoint:
            raise ValueError("formal smoke trace and initial checkpoint paths are required")
        if self.initial_checkpoint_source is not None:
            if not isinstance(self.initial_checkpoint_sha256, str) or not _HASH_RE.fullmatch(
                self.initial_checkpoint_sha256
            ):
                raise ValueError("formal smoke initial checkpoint SHA-256 is required")


@dataclass(frozen=True)
class ValidationOutcome:
    success: bool
    audit_sha256: str | None
    reason: str | None = None


@dataclass(frozen=True)
class LifecycleHooks:
    prepare: Callable[[Attempt], None]
    launch_server: Callable[[Attempt], object]
    start_tunnels: Callable[[Attempt], Sequence[object]]
    launch_pi_client: Callable[[Attempt], object]
    launch_pi_sampler: Callable[[Attempt, object], object]
    launch_pc_client: Callable[[Attempt], object]
    launch_pc_sampler: Callable[[Attempt, object], object]
    monitor_server: Callable[[Attempt, object], None]
    stop_sampler: Callable[[Attempt, object], None]
    wait_sampler: Callable[[Attempt, object], None]
    recover_evidence: Callable[[Attempt], None]
    validate_attempt: Callable[[Attempt], ValidationOutcome]
    cleanup_owned: Callable[[Sequence[object]], Sequence[str] | None]
    cleanup_tunnels: Callable[[Sequence[object]], None]


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def validate_requested_run(group: str, seed: int) -> None:
    if (group, seed) not in CONFIRMATION_SCHEDULE:
        raise ValueError(f"run is outside the exact confirmation allowlist: {group}/{seed}")


def validate_run_id(run_id: str) -> None:
    match = _RUN_ID_RE.fullmatch(run_id)
    if match is None:
        raise ValueError(f"run_id is outside the exact confirmation allowlist: {run_id!r}")
    validate_requested_run(match.group(1).upper(), int(match.group(2)))


def is_objective_rerun_allowed(state: str, reason_category: str | None) -> bool:
    return (
        state in {"failed", "aborted", "invalid"}
        and reason_category in OBJECTIVE_RERUN_REASON_CATEGORIES
    )


def _reject_symlink_components(path: Path, label: str) -> None:
    current = Path(path).absolute()
    while True:
        try:
            entry = os.lstat(current)
        except FileNotFoundError:
            pass
        else:
            attributes = getattr(entry, "st_file_attributes", 0)
            reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
            if stat.S_ISLNK(entry.st_mode) or attributes & reparse_flag:
                raise RuntimeError(
                    f"{label} contains symlink or reparse component: {current}"
                )
            if not os.path.isdir(current) and current == Path(path).absolute():
                raise RuntimeError(f"{label} must be a real directory: {current}")
        if current.parent == current:
            break
        current = current.parent


def _lstat_runtime_component(
    path: Path,
    label: str,
    *,
    kind: str,
    missing_ok: bool,
) -> os.stat_result | None:
    path = Path(path)
    _reject_symlink_components(path.parent, f"{label} parent")
    try:
        entry = os.lstat(path)
    except FileNotFoundError:
        if missing_ok:
            return None
        raise RuntimeError(f"{label} is missing: {path}")
    attributes = getattr(entry, "st_file_attributes", 0)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    if stat.S_ISLNK(entry.st_mode) or attributes & reparse_flag:
        raise RuntimeError(f"{label} is a symlink or reparse point: {path}")
    if kind == "directory" and not stat.S_ISDIR(entry.st_mode):
        raise RuntimeError(f"{label} is not a real directory: {path}")
    if kind == "file" and not stat.S_ISREG(entry.st_mode):
        raise RuntimeError(f"{label} is not a regular file: {path}")
    return entry


def _create_pinned_directory(path: Path, label: str) -> Path:
    path = Path(path)
    _reject_symlink_components(path, label)
    path.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(path, label)
    _lstat_runtime_component(path, label, kind="directory", missing_ok=False)
    return path


def _is_contained(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=True))
    except (FileNotFoundError, ValueError):
        return False
    return True


def _guard_real_directory(
    path: Path,
    label: str,
    *,
    create: bool = False,
    contained_by: Path | None = None,
) -> Path:
    path = Path(path)
    if create:
        path.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(path, label)
    try:
        path.lstat()
    except FileNotFoundError as exc:
        raise RuntimeError(f"{label} does not exist: {path}") from exc
    if not path.is_dir():
        raise RuntimeError(f"{label} must be a directory: {path}")
    if contained_by is not None and not _is_contained(path, Path(contained_by)):
        raise RuntimeError(f"{label} escapes its root: {path}")
    return path.resolve(strict=True)


def _guard_attempt_path(attempt_path: Path) -> tuple[Path, str, str]:
    attempt_path = Path(attempt_path)
    match = _ATTEMPT_ID_RE.fullmatch(attempt_path.name)
    if match is None:
        raise ValueError(f"invalid attempt directory name: {attempt_path.name}")
    run_id = match.group(1)
    if attempt_path.parent.name != run_id:
        raise RuntimeError("attempt directory is outside its run root")
    raw_root = _guard_real_directory(attempt_path.parent.parent, "raw root")
    run_root = _guard_real_directory(
        attempt_path.parent, "run root", contained_by=raw_root
    )
    resolved = _guard_real_directory(
        attempt_path, "attempt path", contained_by=run_root
    )
    return resolved, run_id, attempt_path.name


def _attempt_lock(attempt_path: Path) -> threading.RLock:
    key = str(Path(attempt_path).resolve(strict=False))
    with _ATTEMPT_LOCKS_GUARD:
        return _ATTEMPT_LOCKS.setdefault(key, threading.RLock())


def _load_json(path: Path) -> dict[str, Any]:
    path = Path(path)
    if path.is_symlink():
        raise RuntimeError(f"refusing symlink JSON path: {path}")
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _attempt_directories(run_root: Path, run_id: str) -> list[tuple[int, Path]]:
    rows: list[tuple[int, Path]] = []
    for path in run_root.iterdir():
        if not path.name.startswith(f"{run_id}__a"):
            continue
        match = _ATTEMPT_ID_RE.fullmatch(path.name)
        if match is None or match.group(1) != run_id:
            raise RuntimeError(f"malformed attempt directory name: {path.name}")
        if path.is_symlink():
            raise RuntimeError(f"attempt directory is a symlink: {path}")
        if not path.is_dir() or not _is_contained(path, run_root):
            raise RuntimeError(f"attempt directory escapes run root: {path}")
        rows.append((int(match.group(2)), path))
    return sorted(rows)


def allocate_attempt(raw_root: Path, run_id: str) -> Attempt:
    validate_run_id(run_id)
    raw_root_path = Path(raw_root)
    raw_root_path.mkdir(parents=True, exist_ok=True)
    raw_root_resolved = _guard_real_directory(raw_root_path, "raw root")
    run_root_path = raw_root_path / run_id
    if run_root_path.is_symlink():
        raise RuntimeError(f"run root is a symlink: {run_root_path}")
    run_root_path.mkdir(parents=False, exist_ok=True)
    run_root = _guard_real_directory(
        run_root_path, "run root", contained_by=raw_root_resolved
    )
    existing = _attempt_directories(run_root, run_id)
    statuses: list[tuple[int, dict[str, Any]]] = []
    for number, path in existing:
        chain = _read_status_chain(path, verify_current=False)
        if chain:
            status = chain[-1]
            statuses.append((number, status))
            if status["state"] == "canonical":
                raise RuntimeError(f"canonical attempt already exists for {run_id}")
            _read_status_chain(path, verify_current=True)
    if statuses:
        _, latest = max(statuses, key=lambda row: row[0])
        state = str(latest["state"])
        reason_category = latest.get("reason_category")
        if state == "running":
            raise RuntimeError(f"running attempt already exists for {run_id}")
        if not is_objective_rerun_allowed(
            state, reason_category if isinstance(reason_category, str) else None
        ):
            raise RuntimeError(
                "new attempt requires an objective infrastructure or audit failure category"
            )
    number = (existing[-1][0] + 1) if existing else 1
    if number > 999:
        raise RuntimeError(f"attempt sequence exhausted for {run_id}")
    attempt_id = f"{run_id}__a{number:03d}"
    path = run_root / attempt_id
    path.mkdir(parents=False, exist_ok=False)
    resolved = _guard_real_directory(path, "attempt path", contained_by=run_root)
    return Attempt(run_id, attempt_id, resolved)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _provenance_payload(provenance: Provenance) -> dict[str, str]:
    provenance.require_complete()
    return {
        "confirmation_commit": str(provenance.confirmation_commit),
        "source_archive_sha256": str(provenance.source_archive_sha256),
        "dataset_manifest_sha256": str(provenance.dataset_manifest_sha256),
        "algorithm_config_sha256": str(provenance.algorithm_config_sha256),
    }


def _reject_classification_metric_keys(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise RuntimeError("status JSON object keys must be strings")
            lowered = key.lower()
            if any(part in lowered for part in _FORBIDDEN_STATUS_KEY_PARTS):
                raise RuntimeError(f"classification metric key is forbidden: {key}")
            _reject_classification_metric_keys(child)
    elif isinstance(value, list):
        for child in value:
            _reject_classification_metric_keys(child)


def _validate_rfc3339_utc(value: Any) -> None:
    if not isinstance(value, str) or not _RFC3339_UTC_RE.fullmatch(value):
        raise RuntimeError("status wall_time_utc must be RFC3339 UTC")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise RuntimeError("status wall_time_utc must be RFC3339 UTC") from exc
    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise RuntimeError("status wall_time_utc must be RFC3339 UTC")


def _validate_bound_provenance_payload(
    payload: Mapping[str, Any], run_id: str, attempt_id: str
) -> None:
    _reject_classification_metric_keys(payload)
    if set(payload) != _PROVENANCE_KEYS:
        raise RuntimeError("attempt provenance has non-exact schema")
    if payload.get("run_id") != run_id or payload.get("attempt_id") != attempt_id:
        raise RuntimeError("bound attempt provenance identity mismatch")
    owner = payload.get("controller_owner")
    if not isinstance(owner, Mapping) or set(owner) != {"pid", "instance_id"}:
        raise RuntimeError("attempt provenance owner has non-exact schema")
    pid = owner.get("pid")
    instance_id = owner.get("instance_id")
    if (
        not isinstance(pid, int)
        or isinstance(pid, bool)
        or pid <= 0
        or not isinstance(instance_id, str)
        or not _INSTANCE_RE.fullmatch(instance_id)
    ):
        raise RuntimeError("attempt provenance owner has invalid type")


def _status_combination_is_legal(payload: Mapping[str, Any]) -> bool:
    state = payload["state"]
    event_type = payload["event_type"]
    reason = payload["reason"]
    audit = payload["audit_sha256"]
    if state == "running":
        return audit is None and (
            (reason == "attempt_allocated" and event_type == "attempt_start")
            or (reason == "preflight_passed" and event_type == "preflight_passed")
        )
    if state == "canonical":
        return (
            event_type == "attempt_end"
            and reason == "validator_accepted"
            and isinstance(audit, str)
            and bool(_HASH_RE.fullmatch(audit))
        )
    if state == "invalid":
        return (
            event_type == "attempt_failure"
            and reason == "validator_rejected"
            and audit is None
        )
    if state in {"failed", "aborted"}:
        return (
            event_type == "attempt_failure"
            and reason
            not in {
                "attempt_allocated",
                "preflight_passed",
                "validator_accepted",
                "validator_rejected",
            }
            and audit is None
        )
    return False


def _validate_status_payload(
    payload: Mapping[str, Any],
    *,
    sequence: int,
    run_id: str,
    attempt_id: str,
    provenance: Mapping[str, str],
) -> None:
    _reject_classification_metric_keys(payload)
    if set(payload) != _STATUS_KEYS:
        raise RuntimeError("status event has non-exact schema")
    if not isinstance(payload.get("sequence"), int) or isinstance(
        payload.get("sequence"), bool
    ):
        raise RuntimeError("status event sequence has invalid type")
    if payload["sequence"] != sequence:
        raise RuntimeError("status event sequence does not match immutable filename")
    for field in ("run_id", "attempt_id", "state", "event_type", "reason", "reason_category"):
        if not isinstance(payload.get(field), str):
            raise RuntimeError(f"status event {field} has invalid type")
    if payload["run_id"] != run_id or payload["attempt_id"] != attempt_id:
        raise RuntimeError("status event attempt identity mismatch")
    if payload["state"] not in VALID_ATTEMPT_STATES:
        raise RuntimeError("status event has invalid state")
    reason = payload["reason"]
    if reason not in STATUS_REASON_CATEGORIES:
        raise RuntimeError("status event has invalid reason code")
    if payload["reason_category"] != STATUS_REASON_CATEGORIES[reason]:
        raise RuntimeError("status reason category mismatch")
    _validate_rfc3339_utc(payload.get("wall_time_utc"))
    for field, value in provenance.items():
        if payload.get(field) != value:
            raise RuntimeError(f"status provenance mismatch: {field}")
    audit = payload.get("audit_sha256")
    if audit is not None and (
        not isinstance(audit, str) or not _HASH_RE.fullmatch(audit)
    ):
        raise RuntimeError("status audit SHA-256 has invalid type")
    if not _status_combination_is_legal(payload):
        raise RuntimeError("status event/state/reason/audit combination is invalid")


def _validate_lifecycle_chain(chain: Sequence[Mapping[str, Any]]) -> None:
    if not chain:
        return
    first = chain[0]
    if (
        first["state"],
        first["event_type"],
        first["reason"],
    ) != ("running", "attempt_start", "attempt_allocated"):
        raise RuntimeError(
            "status lifecycle sequence 1 must be running/attempt_start/attempt_allocated"
        )
    start_count = 0
    preflight_seen = False
    terminal_seen = False
    for payload in chain:
        event_type = payload["event_type"]
        state = payload["state"]
        if terminal_seen:
            raise RuntimeError("status lifecycle terminal event has a later event")
        if event_type == "attempt_start":
            start_count += 1
            if start_count != 1 or payload is not first:
                raise RuntimeError("status lifecycle attempt_start must occur exactly once")
        elif event_type == "preflight_passed":
            if preflight_seen:
                raise RuntimeError("status lifecycle preflight_passed may occur at most once")
            if start_count != 1:
                raise RuntimeError("status lifecycle preflight_passed requires attempt_start")
            preflight_seen = True
        if state in {"canonical", "invalid"} and not preflight_seen:
            raise RuntimeError(
                f"status lifecycle {state} terminal requires preflight_passed"
            )
        if state in TERMINAL_ATTEMPT_STATES:
            terminal_seen = True
    if start_count != 1:
        raise RuntimeError("status lifecycle attempt_start must occur exactly once")


def bind_attempt_provenance(attempt: Attempt, provenance: Provenance) -> Path:
    expected = _provenance_payload(provenance)
    attempt_path, run_id, attempt_id = _guard_attempt_path(attempt.path)
    if attempt.run_id != run_id or attempt.attempt_id != attempt_id:
        raise ValueError("attempt dataclass identity does not match its directory")
    path = attempt_path / "attempt_provenance.json"
    payload: dict[str, Any] = {
        "run_id": run_id,
        "attempt_id": attempt_id,
        **expected,
        "controller_owner": {
            "pid": os.getpid(),
            "instance_id": uuid.uuid4().hex,
        },
    }
    _write_json_exclusive(path, payload)
    return path


def _load_bound_provenance(attempt_path: Path) -> tuple[Provenance, dict[str, Any]]:
    path = attempt_path / "attempt_provenance.json"
    if not path.is_file() or path.is_symlink():
        raise RuntimeError("attempt provenance is not immutably bound")
    payload = _load_json(path)
    _validate_bound_provenance_payload(payload, attempt_path.parent.name, attempt_path.name)
    provenance = Provenance(
        confirmation_commit=payload.get("confirmation_commit"),
        source_archive_sha256=payload.get("source_archive_sha256"),
        dataset_manifest_sha256=payload.get("dataset_manifest_sha256"),
        algorithm_config_sha256=payload.get("algorithm_config_sha256"),
    )
    provenance.require_complete()
    return provenance, payload


def _read_status_chain(
    attempt_path: Path,
    *,
    verify_current: bool,
) -> list[dict[str, Any]]:
    attempt_path, run_id, attempt_id = _guard_attempt_path(attempt_path)
    events_root = attempt_path / "status_events"
    current_path = attempt_path / "attempt_status.json"
    if not events_root.exists():
        if current_path.exists() or current_path.is_symlink():
            raise RuntimeError("current status exists without immutable status events")
        return []
    events_root = _guard_real_directory(
        events_root, "status events", contained_by=attempt_path
    )
    numbered: list[tuple[int, Path]] = []
    for path in events_root.iterdir():
        match = re.fullmatch(r"status_([0-9]{3})\.json", path.name)
        if match is None:
            raise RuntimeError(f"malformed status event name: {path.name}")
        if path.is_symlink() or not path.is_file() or not _is_contained(path, events_root):
            raise RuntimeError(f"status event path is not a contained regular file: {path}")
        numbered.append((int(match.group(1)), path))
    numbered.sort(key=lambda row: row[0])
    expected_numbers = list(range(1, len(numbered) + 1))
    if [number for number, _ in numbered] != expected_numbers:
        raise RuntimeError("status event chain has a gap or duplicate")
    if not numbered:
        raise RuntimeError("status events directory is empty")
    provenance, bound = _load_bound_provenance(attempt_path)
    expected_provenance = _provenance_payload(provenance)
    chain: list[dict[str, Any]] = []
    terminal_seen = False
    for number, path in numbered:
        payload = _load_json(path)
        _validate_status_payload(
            payload,
            sequence=number,
            run_id=run_id,
            attempt_id=attempt_id,
            provenance=expected_provenance,
        )
        state = payload["state"]
        if terminal_seen:
            raise RuntimeError("terminal attempt status has a later event")
        if state in TERMINAL_ATTEMPT_STATES:
            terminal_seen = True
        chain.append(payload)
    _validate_lifecycle_chain(chain)
    if bound.get("run_id") != run_id or bound.get("attempt_id") != attempt_id:
        raise RuntimeError("bound attempt provenance identity mismatch")
    if verify_current:
        if not current_path.is_file() or current_path.is_symlink():
            raise RuntimeError("current status is missing or is a symlink")
        current = _load_json(current_path)
        _reject_classification_metric_keys(current)
        if current != chain[-1]:
            raise RuntimeError("current status does not match latest immutable event")
    return chain


def mark_attempt(
    attempt_path: Path,
    state: str,
    *,
    audit_sha256: str | None = None,
    event_type: str | None = None,
    reason: str,
) -> dict[str, Any]:
    if state not in VALID_ATTEMPT_STATES:
        raise ValueError(f"invalid attempt state: {state}")
    if reason not in STATUS_REASON_CATEGORIES:
        raise ValueError(f"reason must be a stable allowlisted code; got {reason!r}")
    attempt_path, run_id, attempt_id = _guard_attempt_path(Path(attempt_path))
    if state == "canonical" and (
        not isinstance(audit_sha256, str) or not _HASH_RE.fullmatch(audit_sha256)
    ):
        raise ValueError("canonical status requires validator audit_sha256")
    if audit_sha256 is not None and not _HASH_RE.fullmatch(audit_sha256):
        raise ValueError("audit_sha256 must be a lowercase SHA-256")
    with _attempt_lock(attempt_path):
        provenance, _bound = _load_bound_provenance(attempt_path)
        chain = _read_status_chain(attempt_path, verify_current=True)
        if chain and chain[-1]["state"] in TERMINAL_ATTEMPT_STATES:
            raise RuntimeError("terminal attempt status cannot be replaced")
        number = len(chain) + 1
        if number > 999:
            raise RuntimeError("status event sequence exhausted")
        if event_type is None:
            if state == "running":
                event_type = (
                    "attempt_start"
                    if reason == "attempt_allocated"
                    else "preflight_passed"
                )
            elif state == "canonical":
                event_type = "attempt_end"
            else:
                event_type = "attempt_failure"
        status = {
            "sequence": number,
            "run_id": run_id,
            "attempt_id": attempt_id,
            "state": state,
            "event_type": event_type,
            "reason": reason,
            "reason_category": STATUS_REASON_CATEGORIES[reason],
            "wall_time_utc": _utc_now(),
            **_provenance_payload(provenance),
            "audit_sha256": audit_sha256,
        }
        _validate_status_payload(
            status,
            sequence=number,
            run_id=run_id,
            attempt_id=attempt_id,
            provenance=_provenance_payload(provenance),
        )
        _validate_lifecycle_chain([*chain, status])
        events_root = attempt_path / "status_events"
        if not events_root.exists():
            events_root.mkdir(parents=False)
        _guard_real_directory(events_root, "status events", contained_by=attempt_path)
        _write_json_exclusive(events_root / f"status_{number:03d}.json", status)
        _atomic_write_json(attempt_path / "attempt_status.json", status)
        return status


def validate_archive_member_path(relative_path: str) -> tuple[str, ...]:
    if not isinstance(relative_path, str) or not relative_path:
        raise ArchiveMismatch("invalid archive member path")
    if (
        "\x00" in relative_path
        or "\\" in relative_path
        or ":" in relative_path
        or relative_path.startswith("/")
        or relative_path.startswith("//")
    ):
        raise ArchiveMismatch(f"unsafe archive member path: {relative_path!r}")
    parts = relative_path.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ArchiveMismatch(f"unsafe archive member path: {relative_path!r}")
    return tuple(parts)


def _expected_regular_members(
    source_manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows = source_manifest.get("regular_members")
    if not isinstance(rows, list) or any(not isinstance(row, Mapping) for row in rows):
        raise ArchiveMismatch("source archive manifest regular_members must be a list")
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in rows:
        relative = raw.get("relative_path")
        size = raw.get("byte_size")
        digest = raw.get("sha256")
        if (
            not isinstance(relative, str)
            or not relative
            or relative in seen
            or not isinstance(size, int)
            or size < 0
            or not isinstance(digest, str)
            or not _HASH_RE.fullmatch(digest)
        ):
            raise ArchiveMismatch("invalid tracked member entry")
        validate_archive_member_path(relative)
        seen.add(relative)
        normalized.append(
            {"relative_path": relative, "byte_size": size, "sha256": digest}
        )
    normalized.sort(key=lambda row: row["relative_path"])
    expected_members_hash = source_manifest.get("regular_members_sha256")
    actual_members_hash = canonical_sha256({"regular_members": normalized})
    if expected_members_hash != actual_members_hash:
        raise ArchiveMismatch("tracked member manifest SHA-256 mismatch")
    if source_manifest.get("tracked_files_manifest_sha256") != actual_members_hash:
        raise ArchiveMismatch("tracked-files manifest SHA-256 mismatch")
    return normalized


def _verify_archive(
    archive_path: Path, source_manifest: Mapping[str, Any]
) -> list[dict[str, Any]]:
    archive_path = Path(archive_path)
    if archive_path.is_symlink():
        raise ArchiveMismatch("source archive path cannot be a symlink")
    expected_archive_hash = source_manifest.get("source_archive_sha256")
    if not isinstance(expected_archive_hash, str) or not _HASH_RE.fullmatch(
        expected_archive_hash
    ):
        raise ArchiveMismatch("source archive manifest has invalid SHA-256")
    if not archive_path.is_file() or sha256_file(archive_path) != expected_archive_hash:
        raise ArchiveMismatch("source archive SHA-256 mismatch")
    expected = _expected_regular_members(source_manifest)
    actual: list[dict[str, Any]] = []
    with tarfile.open(archive_path, "r:") as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            extracted = archive.extractfile(member)
            if extracted is None:
                raise ArchiveMismatch(f"could not read archive member: {member.name}")
            with extracted:
                payload = extracted.read()
            actual.append(
                {
                    "relative_path": member.name,
                    "byte_size": len(payload),
                    "sha256": sha256_bytes(payload),
                }
            )
    actual.sort(key=lambda row: row["relative_path"])
    if actual != expected:
        raise ArchiveMismatch("archive tracked members do not match source manifest")
    return expected


def verify_and_extract_archive(
    archive_path: Path,
    destination: Path,
    source_manifest: Mapping[str, Any],
) -> dict[str, str]:
    expected = _verify_archive(Path(archive_path), source_manifest)
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite extracted source: {destination}")
    staging = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.staging")
    if staging.exists():
        raise FileExistsError(staging)
    staging.mkdir(parents=True)
    try:
        with tarfile.open(archive_path, "r:") as archive:
            by_name = {member.name: member for member in archive.getmembers() if member.isfile()}
            for row in expected:
                relative = str(row["relative_path"])
                member = by_name[relative]
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise ArchiveMismatch(f"could not extract tracked member: {relative}")
                parts = validate_archive_member_path(relative)
                target = staging.joinpath(*parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                if not _is_contained(target.parent, staging):
                    raise ArchiveMismatch(f"extracted member escapes fresh src: {relative}")
                with extracted, target.open("xb") as handle:
                    shutil.copyfileobj(extracted, handle)
                if target.stat().st_size != row["byte_size"] or sha256_file(target) != row["sha256"]:
                    raise ArchiveMismatch(f"extracted tracked member mismatch: {relative}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging.replace(destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        "source_archive_sha256": str(source_manifest["source_archive_sha256"]),
        "regular_members_sha256": str(source_manifest["regular_members_sha256"]),
    }


def _verify_extracted_source(
    destination: Path, source_manifest: Mapping[str, Any]
) -> dict[str, str]:
    destination = _guard_real_directory(Path(destination), "extracted source")
    expected = _expected_regular_members(source_manifest)
    actual: list[dict[str, Any]] = []
    for path in destination.rglob("*"):
        if path.is_symlink():
            raise ArchiveMismatch(f"extracted source contains symlink: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ArchiveMismatch(f"extracted source contains non-regular path: {path}")
        actual.append(
            {
                "relative_path": path.relative_to(destination).as_posix(),
                "byte_size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    actual.sort(key=lambda row: row["relative_path"])
    if actual != expected:
        raise ArchiveMismatch("extracted tracked members do not match source manifest")
    return {
        "source_archive_sha256": str(source_manifest["source_archive_sha256"]),
        "regular_members_sha256": str(source_manifest["regular_members_sha256"]),
    }


def _remote_runtime_path_helpers() -> str:
    return """
REPARSE_FLAG = getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0x400)
def checked_lstat(path, label, missing_ok=False):
    try:
        entry = os.lstat(path)
    except FileNotFoundError:
        if missing_ok:
            return None
        raise RuntimeError(f'{label} is missing: {path}')
    if stat.S_ISLNK(entry.st_mode) or getattr(entry, 'st_file_attributes', 0) & REPARSE_FLAG:
        raise RuntimeError(f'{label} is a symlink or reparse point: {path}')
    return entry
def pin_directory_chain(path, label, create=False):
    path = Path(path)
    missing = []
    current = path
    while True:
        entry = checked_lstat(current, label, missing_ok=True)
        if entry is not None:
            if not stat.S_ISDIR(entry.st_mode):
                raise RuntimeError(f'{label} component is not a directory: {current}')
            break
        missing.append(current)
        if current.parent == current:
            raise RuntimeError(f'{label} has no real ancestor: {path}')
        current = current.parent
    if missing and not create:
        raise RuntimeError(f'{label} is missing: {path}')
    for item in reversed(missing):
        os.mkdir(item)
        entry = checked_lstat(item, label)
        if not stat.S_ISDIR(entry.st_mode):
            raise RuntimeError(f'{label} component is not a directory: {item}')
    return checked_lstat(path, label)
def pin_leaf(path, label, kind, missing_ok=False):
    path = Path(path)
    pin_directory_chain(path.parent, f'{label} parent')
    entry = checked_lstat(path, label, missing_ok=missing_ok)
    if entry is None:
        return None
    expected = stat.S_ISREG if kind == 'file' else stat.S_ISDIR
    if not expected(entry.st_mode):
        raise RuntimeError(f'{label} is not a real {kind}: {path}')
    return entry
"""


def _remote_deploy_state_source(archive_path: str, src_path: str) -> str:
    helpers = _remote_runtime_path_helpers()
    return f"""# REMOTE_DEPLOY_STATE_V1
import json, os, stat
from pathlib import Path
{helpers}
archive_path = Path({archive_path!r})
src_path = Path({src_path!r})
if archive_path.parent != src_path.parent:
    raise RuntimeError('runtime components do not share one content root')
pin_directory_chain(archive_path.parent, 'content-addressed runtime root', create=True)
archive_entry = pin_leaf(archive_path, 'source archive', 'file', missing_ok=True)
src_entry = pin_leaf(src_path, 'extracted source', 'directory', missing_ok=True)
state = 'complete' if archive_entry is not None and src_entry is not None else ('absent' if archive_entry is None and src_entry is None else 'partial')
print(json.dumps({{'state': state}}, sort_keys=True))
"""


def _remote_reserve_archive_source(temp_path: str, runtime_root: str) -> str:
    helpers = _remote_runtime_path_helpers()
    return f"""# REMOTE_RESERVE_ARCHIVE_V1
import json, os, stat
from pathlib import Path
{helpers}
runtime_root = Path({runtime_root!r})
temp_path = Path({temp_path!r})
pin_directory_chain(runtime_root, 'content-addressed runtime root')
if temp_path.parent != runtime_root:
    raise RuntimeError('temporary archive escapes runtime root')
if checked_lstat(temp_path, 'temporary source archive', missing_ok=True) is not None:
    raise RuntimeError('temporary source archive already exists')
with temp_path.open('xb'):
    pass
pin_leaf(temp_path, 'temporary source archive', 'file')
print(json.dumps({{'state': 'reserved'}}, sort_keys=True))
"""


def _remote_install_archive_source(
    temp_path: str,
    archive_path: str,
    expected_sha256: str,
    runtime_root: str,
) -> str:
    helpers = _remote_runtime_path_helpers()
    return f"""# REMOTE_INSTALL_ARCHIVE_V1
import hashlib, json, os, stat
from pathlib import Path
{helpers}
temp_path = Path({temp_path!r})
archive_path = Path({archive_path!r})
runtime_root = Path({runtime_root!r})
pin_directory_chain(runtime_root, 'content-addressed runtime root')
if archive_path.parent != runtime_root or temp_path.parent != runtime_root:
    raise RuntimeError('temporary archive escapes runtime root')
pin_leaf(temp_path, 'temporary source archive', 'file')
if pin_leaf(archive_path, 'source archive', 'file', missing_ok=True) is not None:
    raise RuntimeError('source archive already exists')
def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            value.update(chunk)
    return value.hexdigest()
expected = {expected_sha256!r}
if digest(temp_path) != expected:
    raise RuntimeError('temporary source archive SHA-256 mismatch')
os.link(temp_path, archive_path, follow_symlinks=False)
temp_path.unlink()
pin_leaf(archive_path, 'source archive', 'file')
if digest(archive_path) != expected:
    raise RuntimeError('installed source archive SHA-256 mismatch')
print(json.dumps({{'source_archive_sha256': expected}}, sort_keys=True))
"""


def _remote_extract_source(
    archive_path: str,
    src_path: str,
    source_manifest: Mapping[str, Any],
    *,
    allow_fresh_extract: bool = False,
) -> str:
    """Return a fail-closed remote script; `_remote_python` transports it safely."""
    manifest_json = json.dumps(source_manifest, ensure_ascii=False, sort_keys=True)
    helpers = _remote_runtime_path_helpers()
    return f"""
# REMOTE_EXTRACT_SOURCE_V1
import hashlib, json, os, shutil, stat, tarfile
from pathlib import Path, PurePosixPath
{helpers}
manifest = json.loads({manifest_json!r})
archive_path = Path({archive_path!r})
src_path = Path({src_path!r})
allow_fresh_extract = {allow_fresh_extract!r}
if archive_path.parent != src_path.parent:
    raise RuntimeError('runtime components do not share one content root')
pin_directory_chain(archive_path.parent, 'content-addressed runtime root')
pin_leaf(archive_path, 'source archive', 'file')
src_entry = pin_leaf(src_path, 'extracted source', 'directory', missing_ok=True)
def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            value.update(chunk)
    return value.hexdigest()
def safe_parts(relative):
    if (not isinstance(relative, str) or not relative or '\\x00' in relative
            or '\\\\' in relative or ':' in relative or relative.startswith('/')):
        raise RuntimeError('unsafe archive member path')
    parts = relative.split('/')
    if any(part in ('', '.', '..') for part in parts):
        raise RuntimeError('unsafe archive member path')
    return parts
if digest(archive_path) != manifest['source_archive_sha256']:
    raise RuntimeError('source archive SHA-256 mismatch after transfer')
def verify_src():
    expected = sorted(manifest['regular_members'], key=lambda item: item['relative_path'])
    actual = []
    for path in src_path.rglob('*'):
        if path.is_symlink():
            raise RuntimeError('tracked member source contains symlink')
        if path.is_dir():
            continue
        if not path.is_file():
            raise RuntimeError('tracked member source contains non-regular path')
        actual.append({{
            'relative_path': path.relative_to(src_path).as_posix(),
            'byte_size': path.stat().st_size,
            'sha256': digest(path),
        }})
    actual.sort(key=lambda item: item['relative_path'])
    if actual != expected:
        raise RuntimeError('tracked member source mismatch')
if src_entry is not None:
    verify_src()
elif not allow_fresh_extract:
    raise RuntimeError('partial content-addressed runtime')
else:
    os.mkdir(src_path)
    pin_leaf(src_path, 'extracted source', 'directory')
    try:
        with tarfile.open(archive_path, 'r:') as archive:
            regular = {{item.name: item for item in archive.getmembers() if item.isfile()}}
            expected = manifest['regular_members']
            if sorted(regular) != sorted(item['relative_path'] for item in expected):
                raise RuntimeError('archive tracked member set mismatch')
            for item in expected:
                relative = item['relative_path']
                parts = safe_parts(relative)
                source = archive.extractfile(regular[relative])
                if source is None:
                    raise RuntimeError('missing tracked member')
                target = src_path.joinpath(*parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                if os.path.commonpath([str(src_path.resolve()), str(target.parent.resolve())]) != str(src_path.resolve()):
                    raise RuntimeError('tracked member escaped fresh src')
                with source, target.open('xb') as handle:
                    shutil.copyfileobj(source, handle)
                if target.stat().st_size != item['byte_size'] or digest(target) != item['sha256']:
                    raise RuntimeError('extracted tracked member mismatch')
        verify_src()
    except BaseException:
        shutil.rmtree(src_path, ignore_errors=True)
        raise
print(json.dumps({{
    'source_archive_sha256': digest(archive_path),
    'regular_members_sha256': manifest['regular_members_sha256'],
}}, sort_keys=True))
"""


def deploy_source_archive(
    archive_path: Path,
    source_manifest: Mapping[str, Any],
    *,
    ecs_host: str,
    pi_host: str,
    pc_runtime_root: Path = DEFAULT_PC_RUNTIME_ROOT,
    run: Callable[..., subprocess.CompletedProcess[str]] = _run,
    ssh: Callable[..., subprocess.CompletedProcess[str]] = _ssh,
    remote_python: Callable[..., str] = _remote_python,
) -> dict[str, HostDeployment]:
    archive_path = Path(archive_path)
    _verify_archive(archive_path, source_manifest)
    source_hash = str(source_manifest["source_archive_sha256"])
    members_hash = str(source_manifest["regular_members_sha256"])
    pc_runtime_parent = Path(pc_runtime_root)
    runtime_parent_entry = _lstat_runtime_component(
        pc_runtime_parent,
        "PC runtime parent",
        kind="directory",
        missing_ok=True,
    )
    if runtime_parent_entry is None:
        _create_pinned_directory(pc_runtime_parent, "PC runtime parent")
    pc_root = pc_runtime_parent / source_hash
    pc_archive = pc_root / "source.tar"
    pc_src = pc_root / "src"
    pc_root_entry = _lstat_runtime_component(
        pc_root,
        "PC content-addressed runtime",
        kind="directory",
        missing_ok=True,
    )
    pc_existed = pc_root_entry is not None
    if pc_root_entry is None:
        _create_pinned_directory(pc_root, "PC content-addressed runtime")
    archive_entry = _lstat_runtime_component(
        pc_archive, "PC source archive", kind="file", missing_ok=True
    )
    src_entry = _lstat_runtime_component(
        pc_src, "PC extracted source", kind="directory", missing_ok=True
    )
    if archive_entry is not None and src_entry is not None:
        _verify_archive(pc_archive, source_manifest)
        pc_report = _verify_extracted_source(pc_src, source_manifest)
    elif archive_entry is not None or src_entry is not None or pc_existed:
        raise RuntimeError(f"partial PC content-addressed runtime: {pc_root}")
    else:
        pc_temp = pc_root / f".source.tar.{uuid.uuid4().hex}.tmp"
        _lstat_runtime_component(
            pc_temp, "PC temporary source archive", kind="file", missing_ok=True
        )
        try:
            with archive_path.open("rb") as source, pc_temp.open("xb") as destination:
                shutil.copyfileobj(source, destination)
                destination.flush()
                os.fsync(destination.fileno())
            _lstat_runtime_component(
                pc_temp,
                "PC temporary source archive",
                kind="file",
                missing_ok=False,
            )
            _verify_archive(pc_temp, source_manifest)
            os.link(pc_temp, pc_archive, follow_symlinks=False)
            pc_temp.unlink()
            _lstat_runtime_component(
                pc_archive, "PC source archive", kind="file", missing_ok=False
            )
            _verify_archive(pc_archive, source_manifest)
        finally:
            pc_temp.unlink(missing_ok=True)
        pc_report = verify_and_extract_archive(pc_archive, pc_src, source_manifest)
    deployments: dict[str, HostDeployment] = {
        "pc": HostDeployment("pc", pc_archive, pc_src, **pc_report)
    }
    remote_rows = (
        (
            "ecs",
            ecs_host,
            f"{ECS_REMOTE_RUNTIME_BASE}/{source_hash}",
            "/root/gaps_env/bin/python",
        ),
        (
            "pi",
            pi_host,
            f"{PI_REMOTE_RUNTIME_BASE}/{source_hash}",
            "/home/gaps/GAPS/gaps_rpi_env/bin/python",
        ),
    )
    for host_id, host, root, python_bin in remote_rows:
        remote_archive = f"{root}/source.tar"
        remote_src = f"{root}/src"
        state_output = remote_python(
            host,
            python_bin,
            _remote_deploy_state_source(remote_archive, remote_src),
            timeout=30,
        )
        try:
            state_report = json.loads(state_output.splitlines()[-1])
        except (IndexError, json.JSONDecodeError) as exc:
            raise ArchiveMismatch(f"invalid {host_id} deployment state report") from exc
        if not isinstance(state_report, dict) or set(state_report) != {"state"}:
            raise ArchiveMismatch(f"invalid {host_id} deployment state report")
        state = state_report["state"]
        if state == "partial":
            raise ArchiveMismatch(f"partial {host_id} content-addressed runtime")
        if state not in {"absent", "complete"}:
            raise ArchiveMismatch(f"invalid {host_id} deployment state")
        if state == "absent":
            remote_temp = f"{root}/.source.tar.{uuid.uuid4().hex}.tmp"
            reserve_output = remote_python(
                host,
                python_bin,
                _remote_reserve_archive_source(remote_temp, root),
                timeout=30,
            )
            try:
                reserve_report = json.loads(reserve_output.splitlines()[-1])
            except (IndexError, json.JSONDecodeError) as exc:
                raise ArchiveMismatch(
                    f"invalid {host_id} archive reservation report"
                ) from exc
            if reserve_report != {"state": "reserved"}:
                raise ArchiveMismatch(f"invalid {host_id} archive reservation report")
            transfer = run(
                ["scp", "-p", str(archive_path), f"{host}:{remote_temp}"],
                timeout=300,
            )
            if transfer.returncode != 0:
                raise ArchiveMismatch(f"{host_id} source archive transfer failed")
            install_output = remote_python(
                host,
                python_bin,
                _remote_install_archive_source(
                    remote_temp, remote_archive, source_hash, root
                ),
                timeout=60,
            )
            try:
                install_report = json.loads(install_output.splitlines()[-1])
            except (IndexError, json.JSONDecodeError) as exc:
                raise ArchiveMismatch(
                    f"invalid {host_id} archive installation report"
                ) from exc
            if install_report != {"source_archive_sha256": source_hash}:
                raise ArchiveMismatch(f"invalid {host_id} archive installation report")
        output = remote_python(
            host,
            python_bin,
            _remote_extract_source(
                remote_archive,
                remote_src,
                source_manifest,
                allow_fresh_extract=state == "absent",
            ),
            timeout=300,
        )
        try:
            report = json.loads(output.splitlines()[-1])
        except (IndexError, json.JSONDecodeError) as exc:
            raise ArchiveMismatch(f"invalid {host_id} archive verification report") from exc
        if report != {
            "source_archive_sha256": source_hash,
            "regular_members_sha256": members_hash,
        }:
            raise ArchiveMismatch(f"{host_id} archive verification mismatch")
        deployments[host_id] = HostDeployment(
            host_id, remote_archive, remote_src, source_hash, members_hash
        )
    return deployments


def validate_host_preflight(
    report: Mapping[str, Any],
    *,
    host_id: str,
    provenance: Provenance,
    regular_members_sha256: str,
) -> None:
    provenance.require_complete()
    if report.get("host_id") != host_id:
        raise RuntimeError(f"{host_id} preflight host identity mismatch")
    if report.get("dependency_versions") != EXPECTED_DEPENDENCY_VERSIONS:
        raise RuntimeError(f"{host_id} dependency preflight mismatch")
    if report.get("confirmation_commit") != provenance.confirmation_commit:
        raise RuntimeError(f"{host_id} confirmation commit preflight mismatch")
    if report.get("source_archive_sha256") != provenance.source_archive_sha256:
        raise RuntimeError(f"{host_id} source archive preflight mismatch")
    if report.get("regular_members_sha256") != regular_members_sha256:
        raise RuntimeError(f"{host_id} tracked source preflight mismatch")
    if report.get("dataset_manifest_sha256") != provenance.dataset_manifest_sha256:
        raise RuntimeError(f"{host_id} dataset manifest preflight mismatch")
    if report.get("algorithm_config_sha256") != provenance.algorithm_config_sha256:
        raise RuntimeError(f"{host_id} algorithm config preflight mismatch")
    existing = report.get("existing_attempt_processes")
    if not isinstance(existing, list) or existing:
        raise RuntimeError(f"{host_id} existing attempt process preflight mismatch")


def validate_command_manifest(
    manifest: Mapping[str, Any],
    group_id: str,
    seed: int,
    *,
    expected_confirmation_commit: str | None = None,
    expected_protocol_manifest_sha256: str | None = None,
    expected_source_archive_sha256: str | None = None,
    expected_dataset_manifest_sha256: str | None = None,
    expected_algorithm_config_sha256: str | None = None,
) -> Provenance:
    validate_requested_run(group_id, seed)
    expected_run_id = confirmation_run_id(group_id, seed)
    if manifest.get("run_id") != expected_run_id or manifest.get("group_id") != group_id:
        raise ValueError("command manifest identity does not match allowlist")
    if manifest.get("transport_status") != "not_collected":
        raise ValueError(
            "command manifest transport_status must be explicitly not_collected"
        )
    protocol = manifest.get("protocol")
    if not isinstance(protocol, Mapping) or protocol.get("source_clients") != [1, 2] or protocol.get(
        "target_clients"
    ) != [5] or protocol.get("training_seed") != seed:
        raise ValueError("command manifest protocol does not match confirmation identity")
    if manifest.get("topology") != EXPECTED_TOPOLOGY:
        raise ValueError("command manifest topology is not the exact formal topology")
    commands = manifest.get("commands")
    if not isinstance(commands, Mapping) or set(commands) != {
        "server_ecs",
        "client_c1_pi",
        "client_c2_pc",
    }:
        raise ValueError("command manifest commands do not match the exact host topology")
    try:
        algorithm = {field: manifest[field] for field in ALGORITHM_CONFIG_FIELDS}
    except KeyError as exc:
        raise ValueError(f"command manifest missing algorithm field: {exc.args[0]}") from exc
    algorithm_hash = canonical_sha256(algorithm)
    if manifest.get("algorithm_config_sha256") != algorithm_hash:
        raise ValueError("algorithm config SHA-256 mismatch")
    provenance_payload = manifest.get("provenance")
    if not isinstance(provenance_payload, Mapping):
        raise ValueError("command manifest provenance must be an object")
    command_commit = provenance_payload.get("code_revision")
    provenance = Provenance(
        confirmation_commit=command_commit,
        source_archive_sha256=manifest.get("source_archive_sha256"),
        dataset_manifest_sha256=manifest.get("dataset_manifest_sha256"),
        algorithm_config_sha256=algorithm_hash,
    )
    provenance.require_complete()
    checks = (
        (
            "confirmation_commit",
            provenance.confirmation_commit,
            expected_confirmation_commit,
        ),
        (
            "protocol manifest SHA-256",
            manifest.get("protocol_manifest_sha256"),
            expected_protocol_manifest_sha256,
        ),
        (
            "source archive SHA-256",
            provenance.source_archive_sha256,
            expected_source_archive_sha256,
        ),
        (
            "dataset manifest SHA-256",
            provenance.dataset_manifest_sha256,
            expected_dataset_manifest_sha256,
        ),
        (
            "algorithm config SHA-256",
            algorithm_hash,
            expected_algorithm_config_sha256,
        ),
    )
    for label, actual, expected in checks:
        if expected is not None and actual != expected:
            raise ValueError(f"command {label} does not match frozen protocol")
    return provenance


def validate_protocol_schedule(protocol_manifest: Mapping[str, Any]) -> None:
    if (
        "transport_status" in protocol_manifest
        and protocol_manifest.get("transport_status") != "not_collected"
    ):
        raise ValueError(
            "protocol transport_status conflicts with explicit schedule declarations"
        )
    rows = protocol_manifest.get("schedule")
    expected = [
        {
            "run_id": confirmation_run_id(group, seed),
            "group_id": group,
            "seed": seed,
            "transport_status": "not_collected",
        }
        for group, seed in CONFIRMATION_SCHEDULE
    ]
    if not isinstance(rows, list) or len(rows) != len(expected):
        raise ValueError(
            "protocol schedule identity/transport does not equal confirmation allowlist"
        )
    actual = [
        {
            "run_id": row.get("run_id"),
            "group_id": row.get("group_id"),
            "seed": row.get("seed"),
            "transport_status": row.get("transport_status"),
        }
        if isinstance(row, Mapping)
        else None
        for row in rows
    ]
    if actual != expected:
        raise ValueError(
            "protocol schedule identity/transport does not equal confirmation allowlist"
        )


def _validate_self_hash(
    payload: Mapping[str, Any],
    hash_field: str,
    label: str,
) -> str:
    claimed = payload.get(hash_field)
    if not isinstance(claimed, str) or not _HASH_RE.fullmatch(claimed):
        raise ValueError(f"{label} has invalid {hash_field}")
    unhashed = {key: value for key, value in payload.items() if key != hash_field}
    if canonical_sha256(unhashed) != claimed:
        raise ValueError(f"{label} self SHA-256 mismatch")
    return claimed


def load_frozen_inputs(
    protocol_path: Path,
    source_manifest_path: Path,
    dataset_manifest_path: Path,
    command_root: Path,
    archive_path: Path,
) -> FrozenInputs:
    protocol_path = Path(protocol_path)
    source_manifest_path = Path(source_manifest_path)
    dataset_manifest_path = Path(dataset_manifest_path)
    command_root = _guard_real_directory(Path(command_root), "command root")
    archive_path = Path(archive_path)
    protocol = _load_json(protocol_path)
    source_manifest = _load_json(source_manifest_path)
    dataset_manifest = _load_json(dataset_manifest_path)
    protocol_hash = _validate_self_hash(
        protocol, "protocol_manifest_sha256", "protocol manifest"
    )
    dataset_hash = _validate_self_hash(
        dataset_manifest, "dataset_manifest_sha256", "dataset manifest"
    )
    validate_protocol_schedule(protocol)
    confirmation_commit = protocol.get("confirmation_commit")
    if not isinstance(confirmation_commit, str) or not _COMMIT_RE.fullmatch(
        confirmation_commit
    ):
        raise ValueError("protocol confirmation_commit is invalid")
    if source_manifest.get("confirmation_commit") != confirmation_commit:
        raise ValueError("source confirmation_commit does not match protocol")
    source_hash = source_manifest.get("source_archive_sha256")
    members_hash = source_manifest.get("regular_members_sha256")
    if protocol.get("source_archive_sha256") != source_hash:
        raise ValueError("protocol source archive SHA-256 mismatch")
    if protocol.get("regular_members_sha256") != members_hash:
        raise ValueError("protocol tracked source SHA-256 mismatch")
    if protocol.get("dataset_manifest_sha256") != dataset_hash:
        raise ValueError("protocol dataset manifest SHA-256 mismatch")
    if source_manifest.get("dependency_versions") != EXPECTED_DEPENDENCY_VERSIONS:
        raise ValueError("source dependency versions do not match confirmation freeze")
    _verify_archive(archive_path, source_manifest)
    schedule_rows = protocol["schedule"]
    runs: list[FrozenRun] = []
    for index, (group_id, seed) in enumerate(CONFIRMATION_SCHEDULE):
        row = schedule_rows[index]
        if not isinstance(row, Mapping):
            raise ValueError("protocol schedule row must be an object")
        expected_algorithm_hash = row.get("algorithm_config_sha256")
        if not isinstance(expected_algorithm_hash, str) or not _HASH_RE.fullmatch(
            expected_algorithm_hash
        ):
            raise ValueError("protocol schedule algorithm SHA-256 is invalid")
        run_id = confirmation_run_id(group_id, seed)
        manifest_path = command_root / run_id / "command_manifest.json"
        if not _is_contained(manifest_path.parent, command_root):
            raise ValueError("command manifest path escapes command root")
        manifest = _load_json(manifest_path)
        provenance = validate_command_manifest(
            manifest,
            group_id,
            seed,
            expected_confirmation_commit=confirmation_commit,
            expected_protocol_manifest_sha256=protocol_hash,
            expected_source_archive_sha256=str(source_hash),
            expected_dataset_manifest_sha256=dataset_hash,
            expected_algorithm_config_sha256=expected_algorithm_hash,
        )
        if manifest.get("regular_members_sha256") != members_hash:
            raise ValueError("command tracked source SHA-256 does not match protocol")
        runs.append(
            FrozenRun(
                run_id=run_id,
                group_id=group_id,
                seed=seed,
                manifest_path=manifest_path,
                manifest=manifest,
                provenance=provenance,
            )
        )
    return FrozenInputs(
        protocol_path=protocol_path,
        protocol=protocol,
        source_manifest_path=source_manifest_path,
        source_manifest=source_manifest,
        dataset_manifest_path=dataset_manifest_path,
        dataset_manifest=dataset_manifest,
        command_root=command_root,
        archive_path=archive_path,
        runs=tuple(runs),
    )


def write_host_contexts(
    attempt: Attempt,
    *,
    group_id: str,
    seed: int,
    provenance: Provenance,
) -> dict[str, Path]:
    validate_requested_run(group_id, seed)
    provenance.require_complete()
    if attempt.run_id != confirmation_run_id(group_id, seed):
        raise ValueError("attempt and requested run identity mismatch")
    contexts = (
        ("ecs_server", "ecs", "server", None),
        ("pi_client", "pi-c1", "client", "C1"),
        ("pi_sampler", "pi-c1", "resource_sampler", "C1"),
        ("pc_client", "pc-c2", "client", "C2"),
        ("pc_sampler", "pc-c2", "resource_sampler", "C2"),
    )
    result: dict[str, Path] = {}
    for name, host_id, producer, client_id in contexts:
        payload = {
            "run_id": attempt.run_id,
            "attempt_id": attempt.attempt_id,
            "group_id": group_id,
            "training_seed": seed,
            "client_id": client_id,
            "host_id": host_id,
            "producer": producer,
            "confirmation_commit": provenance.confirmation_commit,
            "source_archive_sha256": provenance.source_archive_sha256,
            "dataset_manifest_sha256": provenance.dataset_manifest_sha256,
            "algorithm_config_sha256": provenance.algorithm_config_sha256,
        }
        path = attempt.path / "contexts" / f"{name}.json"
        _write_json_exclusive(path, payload)
        result[name] = path
    return result


def copy_evidence_without_overwrite(source: Path, destination: Path) -> None:
    source = Path(source)
    destination = Path(destination)
    _reject_symlink_components(source, "evidence source")
    _reject_symlink_components(destination.parent, "evidence destination parent")
    if source.is_symlink():
        raise RuntimeError(f"evidence source is a symlink: {source}")
    if source.is_dir():
        for path in source.rglob("*"):
            if path.is_symlink():
                raise RuntimeError(f"evidence tree contains symlink: {path}")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite raw evidence: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(source, destination)
    elif source.is_file():
        shutil.copy2(source, destination)
    else:
        raise FileNotFoundError(source)


def invoke_validator(
    attempt: Attempt,
    *,
    validator: Path,
    protocol_manifest: Path,
    run: Callable[..., subprocess.CompletedProcess[str]] = _run,
) -> ValidationOutcome:
    attempt_path, _run_id, _attempt_id = _guard_attempt_path(attempt.path)
    output = attempt_path / "attempt_audit.json"
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite validator audit: {output}")
    argv = [
        sys.executable,
        str(Path(validator)),
        "--attempt-dir",
        str(attempt_path),
        "--protocol-manifest",
        str(Path(protocol_manifest)),
        "--output",
        str(output),
    ]
    result = run(argv, timeout=300, check=False)
    if result.returncode not in {0, 2}:
        raise RuntimeError(f"validator return code {result.returncode}")
    if not output.is_file() or output.is_symlink():
        raise RuntimeError("validator did not create a regular audit output")
    try:
        audit = _load_json(output)
    except (OSError, UnicodeError, ValueError, RuntimeError) as exc:
        raise RuntimeError("validator audit report is malformed") from exc
    status = audit.get("status")
    if status not in {"valid", "invalid"}:
        raise RuntimeError("validator audit status is missing or malformed")
    expected_status = "valid" if result.returncode == 0 else "invalid"
    if status != expected_status:
        raise RuntimeError(
            f"validator exit/status mismatch: return code {result.returncode}, "
            f"status {status}"
        )
    if status == "invalid":
        return ValidationOutcome(False, None, "validator rejected attempt")
    try:
        response = json.loads(result.stdout.strip())
    except json.JSONDecodeError as exc:
        raise RuntimeError("validator did not return JSON audit_sha256") from exc
    audit_sha256 = response.get("audit_sha256") if isinstance(response, Mapping) else None
    if not isinstance(audit_sha256, str) or not _HASH_RE.fullmatch(audit_sha256):
        raise RuntimeError("validator audit_sha256 is missing or malformed")
    if sha256_file(output) != audit_sha256:
        raise RuntimeError("validator audit_sha256 does not match output bytes")
    return ValidationOutcome(True, audit_sha256, None)


def _command_option(command: Sequence[str], option: str) -> str:
    matches = [index for index, value in enumerate(command) if value == option]
    if len(matches) != 1 or matches[0] + 1 >= len(command):
        raise ValueError(f"frozen command must contain exactly one {option}")
    return str(command[matches[0] + 1])


def _replace_command_option(
    command: Sequence[str], option: str, value: str
) -> list[str]:
    result = [str(item) for item in command]
    matches = [index for index, item in enumerate(result) if item == option]
    if len(matches) > 1:
        raise ValueError(f"command contains duplicate {option}")
    if matches:
        result[matches[0] + 1] = value
    else:
        result.extend([option, value])
    return result


def _remove_command_option(command: Sequence[str], option: str) -> list[str]:
    result = [str(item) for item in command]
    matches = [index for index, item in enumerate(result) if item == option]
    if len(matches) > 1:
        raise ValueError(f"command contains duplicate {option}")
    if not matches:
        return result
    index = matches[0]
    if index + 1 >= len(result):
        raise ValueError(f"command option {option} has no value")
    del result[index : index + 2]
    return result


def _wrap_smoke_server_command(
    command: Sequence[str], smoke: FormalSmokeConfig
) -> list[str]:
    original = [str(item) for item in command]
    if len(original) < 3 or original[1:3] != ["-m", "gaps_flower.server_app"]:
        raise ValueError("frozen smoke server command must invoke gaps_flower.server_app")
    return [
        original[0],
        "-m",
        "scripts.run_iotj_observer_equivalence_gate",
        "--trace-child-role",
        "server",
        "--trace-output",
        smoke.trace_output,
        "--trace-initial-checkpoint",
        smoke.initial_checkpoint,
        "--",
        *original[1:],
    ]


def derive_noncanonical_smoke_commands(
    command_manifest: Mapping[str, Any], smoke: FormalSmokeConfig
) -> dict[str, Any]:
    """Derive an ephemeral smoke overlay without mutating frozen commands."""

    commands = command_manifest.get("commands")
    if not isinstance(commands, Mapping) or set(commands) != {
        "server_ecs",
        "client_c1_pi",
        "client_c2_pc",
    }:
        raise ValueError("frozen smoke commands have non-exact topology schema")
    server = _replace_command_option(commands["server_ecs"], "--rounds", "2")
    c1 = _replace_command_option(commands["client_c1_pi"], "--local-epochs", "1")
    c2 = _replace_command_option(commands["client_c2_pc"], "--local-epochs", "1")
    for option in ("--observer-context", "--observer-events"):
        server = _remove_command_option(server, option)
        c1 = _remove_command_option(c1, option)
        c2 = _remove_command_option(c2, option)
    server = _wrap_smoke_server_command(server, smoke)
    identity_fields = (
        "run_id",
        "group_id",
        "protocol_manifest_sha256",
        "source_archive_sha256",
        "regular_members_sha256",
        "dataset_manifest_sha256",
        "algorithm_config_sha256",
    )
    identity = {field: command_manifest.get(field) for field in identity_fields}
    if any(not isinstance(value, str) for value in identity.values()):
        raise ValueError("frozen smoke command identity is incomplete")
    return {
        "server_ecs": server,
        "client_c1_pi": c1,
        "client_c2_pc": c2,
        "smoke_identity": {
            "namespace": smoke.namespace,
            "noncanonical_smoke": True,
            "mode": smoke.mode,
            "rounds": smoke.rounds,
            "local_epochs": smoke.local_epochs,
            **identity,
            "command_manifest_sha256": canonical_sha256(command_manifest),
        },
    }


def _rewrite_ecs_dataset_paths(command: Sequence[str]) -> list[str]:
    result = [str(item) for item in command]
    for option in ("--data-root", "--server-val-data", "--server-calib-data"):
        indexes = [index for index, item in enumerate(result) if item == option]
        if not indexes:
            continue
        if len(indexes) != 1 or indexes[0] + 1 >= len(result):
            raise ValueError(f"frozen server command must contain one value for {option}")
        index = indexes[0] + 1
        rewritten: list[str] = []
        for relative in result[index].split(","):
            parts = validate_archive_member_path(relative)
            if parts[0] != "dataset":
                raise ValueError(f"frozen {option} must be rooted at dataset/: {relative}")
            rewritten.append(str(PurePosixPath("/root/GAPS").joinpath(*parts)))
        result[index] = ",".join(rewritten)
    return result


def _host_attempt_root(deployment: HostDeployment, attempt_id: str) -> str:
    archive_parent = PurePosixPath(str(deployment.archive_path)).parent
    return str(archive_parent / "attempts" / attempt_id)


def _algorithm_payload(frozen_run: FrozenRun) -> dict[str, Any]:
    return {
        field: frozen_run.manifest[field]
        for field in ALGORITHM_CONFIG_FIELDS
    }


def _host_preflight_source(
    *,
    host_id: str,
    attempt_id: str,
    deployment: HostDeployment,
    dataset_root: str,
    dataset_manifest: Mapping[str, Any],
    frozen_run: FrozenRun,
) -> str:
    dataset_json = json.dumps(dataset_manifest, ensure_ascii=False, sort_keys=True)
    members_json = json.dumps(
        frozen_run.manifest.get("regular_members_sha256"), sort_keys=True
    )
    algorithm_json = json.dumps(_algorithm_payload(frozen_run), sort_keys=True)
    expected_json = json.dumps(
        {
            "confirmation_commit": frozen_run.provenance.confirmation_commit,
            "source_archive_sha256": frozen_run.provenance.source_archive_sha256,
            "regular_members_sha256": deployment.regular_members_sha256,
            "dataset_manifest_sha256": frozen_run.provenance.dataset_manifest_sha256,
            "algorithm_config_sha256": frozen_run.provenance.algorithm_config_sha256,
        },
        sort_keys=True,
    )
    source_members_json = json.dumps(
        frozen_run.manifest.get("source_regular_members", []), sort_keys=True
    )
    # The deployment already verified every tracked member.  This probe repeats
    # the archive digest and dataset files and binds the per-run config/process identity.
    return f"""# HOST_PREFLIGHT_V1:{host_id}
import hashlib, json, os
from importlib import metadata
from pathlib import Path
import psutil
host_id = {host_id!r}
attempt_id = {attempt_id!r}
archive_path = Path({str(deployment.archive_path)!r})
src_path = Path({str(deployment.src_path)!r})
dataset_root = Path({dataset_root!r})
dataset_manifest = json.loads({dataset_json!r})
algorithm = json.loads({algorithm_json!r})
expected = json.loads({expected_json!r})
def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            value.update(chunk)
    return value.hexdigest()
if digest(archive_path) != expected['source_archive_sha256']:
    raise RuntimeError('source archive mismatch')
for item in dataset_manifest.get('files', []):
    relative = item['relative_path']
    if ('\\x00' in relative or '\\\\' in relative or ':' in relative
            or relative.startswith('/') or any(part in ('', '.', '..') for part in relative.split('/'))):
        raise RuntimeError('unsafe dataset member path')
    path = dataset_root.joinpath(*relative.split('/'))
    if path.stat().st_size != item['byte_size'] or digest(path) != item['sha256']:
        raise RuntimeError('dataset file mismatch')
encoded = json.dumps(algorithm, ensure_ascii=False, separators=(',', ':'), sort_keys=True).encode('utf-8')
if hashlib.sha256(encoded).hexdigest() != expected['algorithm_config_sha256']:
    raise RuntimeError('algorithm config mismatch')
dependencies = {{name: metadata.version(name) for name in ('flwr', 'protobuf', 'psutil')}}
existing = []
for process in psutil.process_iter(['pid', 'cmdline']):
    if process.info['pid'] == os.getpid():
        continue
    command = ' '.join(process.info.get('cmdline') or [])
    if attempt_id in command and ('gaps_flower.server_app' in command or 'gaps_flower.client_app' in command):
        existing.append(process.info['pid'])
print(json.dumps({{
    'host_id': host_id,
    'dependency_versions': dependencies,
    'confirmation_commit': expected['confirmation_commit'],
    'source_archive_sha256': expected['source_archive_sha256'],
    'regular_members_sha256': expected['regular_members_sha256'],
    'dataset_manifest_sha256': expected['dataset_manifest_sha256'],
    'algorithm_config_sha256': expected['algorithm_config_sha256'],
    'existing_attempt_processes': sorted(existing),
}}, sort_keys=True))
"""


def preflight_frozen_run(runtime: ProductionRuntime, attempt_id: str) -> None:
    commands = runtime.frozen_run.manifest["commands"]
    data_root_name = str(runtime.frozen_run.manifest["protocol"]["data_root"])
    roots = {
        "ecs": f"/root/GAPS/dataset/{data_root_name}",
        "pi": _command_option(commands["client_c1_pi"], "--data-root"),
        "pc": _command_option(commands["client_c2_pc"], "--data-root"),
    }
    reports: dict[str, Mapping[str, Any]] = {}
    remote_rows = (
        ("ecs", runtime.ecs_host, "/root/gaps_env/bin/python"),
        ("pi", runtime.pi_host, "/home/gaps/GAPS/gaps_rpi_env/bin/python"),
    )
    for host_id, host, python_bin in remote_rows:
        source = _host_preflight_source(
            host_id=host_id,
            attempt_id=attempt_id,
            deployment=runtime.deployments[host_id],
            dataset_root=roots[host_id],
            dataset_manifest=runtime.frozen.dataset_manifest,
            frozen_run=runtime.frozen_run,
        )
        reports[host_id] = json.loads(
            _remote_python(host, python_bin, source, timeout=300).splitlines()[-1]
        )
    pc_source = _host_preflight_source(
        host_id="pc",
        attempt_id=attempt_id,
        deployment=runtime.deployments["pc"],
        dataset_root=roots["pc"],
        dataset_manifest=runtime.frozen.dataset_manifest,
        frozen_run=runtime.frozen_run,
    )
    pc_result = _run([sys.executable, "-c", pc_source], timeout=300, check=False)
    if pc_result.returncode != 0:
        raise RuntimeError("PC preflight process failed")
    reports["pc"] = json.loads(pc_result.stdout.splitlines()[-1])
    for host_id in ("ecs", "pi", "pc"):
        validate_host_preflight(
            reports[host_id],
            host_id=host_id,
            provenance=runtime.frozen_run.provenance,
            regular_members_sha256=runtime.deployments[host_id].regular_members_sha256,
        )


def _validate_launch_registration(
    payload: Any,
    *,
    label: str,
    launch_token: str,
    registration_path: str,
) -> dict[str, Any]:
    if not isinstance(payload, dict) or set(payload) != _LAUNCH_REGISTRATION_KEYS:
        raise RuntimeError("launch registration has non-exact schema")
    if payload.get("schema_version") != 1 or isinstance(
        payload.get("schema_version"), bool
    ):
        raise RuntimeError("launch registration schema version is invalid")
    if payload.get("label") != label:
        raise RuntimeError("launch registration label identity mismatch")
    if (
        payload.get("launch_token") != launch_token
        or not _INSTANCE_RE.fullmatch(launch_token)
    ):
        raise RuntimeError("launch registration token identity mismatch")
    if payload.get("registration_path") != registration_path:
        raise RuntimeError("launch registration path identity mismatch")
    for field in ("child_pid", "owner_pid", "owner_pgid", "owner_start_ticks"):
        value = payload.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise RuntimeError(f"launch registration {field} is invalid")
    if payload["owner_pid"] != payload["owner_pgid"]:
        raise RuntimeError("launch registration owner process group mismatch")
    if payload["child_pid"] == payload["owner_pid"]:
        raise RuntimeError("launch registration child and owner PID must differ")
    return dict(payload)


def _remote_read_registration_source(registration_path: str) -> str:
    return f"""# REMOTE_READ_REGISTRATION_V1
import os, stat, time
from pathlib import Path
registration_path = Path({registration_path!r})
deadline = time.monotonic() + 10
while True:
    try:
        entry = os.lstat(registration_path)
        break
    except FileNotFoundError:
        if time.monotonic() >= deadline:
            raise RuntimeError('launch registration is missing')
        time.sleep(0.05)
if stat.S_ISLNK(entry.st_mode) or getattr(entry, 'st_file_attributes', 0) & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0x400):
    raise RuntimeError('launch registration is a symlink or reparse point')
if not stat.S_ISREG(entry.st_mode):
    raise RuntimeError('launch registration is not a regular file')
print(registration_path.read_text(encoding='utf-8'), end='')
"""


def _remote_terminate_registration_source(
    registration: Mapping[str, Any],
) -> str:
    expected_json = json.dumps(dict(registration), ensure_ascii=False, sort_keys=True)
    return f"""# REMOTE_TERMINATE_REGISTRATION_V1
import json, os, signal, stat
from pathlib import Path
expected = json.loads({expected_json!r})
registration_path = Path(expected['registration_path'])
entry = os.lstat(registration_path)
if stat.S_ISLNK(entry.st_mode) or getattr(entry, 'st_file_attributes', 0) & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0x400):
    raise RuntimeError('launch registration is a symlink or reparse point')
if not stat.S_ISREG(entry.st_mode):
    raise RuntimeError('launch registration is not a regular file')
registration = json.loads(registration_path.read_text(encoding='utf-8'))
if registration != expected:
    raise RuntimeError('launch registration != expected identity')
owner_pid = expected['owner_pid']
owner_pgid = expected['owner_pgid']
owner_start_ticks = expected['owner_start_ticks']
def process_start_ticks(pid):
    fields = Path(f'/proc/{{pid}}/stat').read_text(encoding='ascii').rsplit(')', 1)[1].split()
    return int(fields[19])
try:
    actual_pgid = os.getpgid(owner_pid)
except ProcessLookupError:
    actual_pgid = None
if actual_pgid is not None:
    if actual_pgid != owner_pgid:
        raise RuntimeError('registered owner PGID changed')
    if process_start_ticks(owner_pid) != owner_start_ticks:
        raise RuntimeError('registered owner start time changed')
    os.killpg(owner_pgid, signal.SIGTERM)
archived = registration_path.with_name(registration_path.name + '.cleaned')
try:
    os.link(registration_path, archived, follow_symlinks=False)
except FileExistsError:
    archived_entry = os.lstat(archived)
    if stat.S_ISLNK(archived_entry.st_mode) or not stat.S_ISREG(archived_entry.st_mode):
        raise RuntimeError('archived launch registration is unsafe')
    if archived.read_bytes() != registration_path.read_bytes():
        raise RuntimeError('archived launch registration mismatch')
registration_path.unlink()
print(json.dumps({{'recovered': True}}, sort_keys=True))
"""


def _read_remote_launch_registration(
    *,
    host: str,
    python_bin: str,
    label: str,
    launch_token: str,
    registration_path: str,
    remote_python: Callable[..., str],
) -> dict[str, Any]:
    output = remote_python(
        host,
        python_bin,
        _remote_read_registration_source(registration_path),
        timeout=15,
    )
    try:
        payload = json.loads(output.splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise RuntimeError("launch registration is not valid JSON") from exc
    return _validate_launch_registration(
        payload,
        label=label,
        launch_token=launch_token,
        registration_path=registration_path,
    )


def _terminate_remote_launch_registration(
    *,
    host: str,
    python_bin: str,
    registration: Mapping[str, Any],
    remote_python: Callable[..., str],
) -> None:
    output = remote_python(
        host,
        python_bin,
        _remote_terminate_registration_source(registration),
        timeout=30,
    )
    try:
        report = json.loads(output.splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise RuntimeError("launch registration termination report is invalid") from exc
    if report != {"recovered": True}:
        raise RuntimeError("launch registration termination was not acknowledged")


def _remote_launch_process(
    *,
    host_id: str,
    label: str,
    host: str,
    python_bin: str,
    command: Sequence[str],
    cwd: str,
    log_path: str,
    exit_path: str,
    python_path: str,
    registration_path: str,
    remote_python: Callable[..., str] | None = None,
) -> OwnedProcess:
    remote_python = remote_python or _remote_python
    pid_path = f"{exit_path}.child.pid"
    launch_token = uuid.uuid4().hex
    supervisor = f"""
import os, subprocess
from pathlib import Path
command = {list(command)!r}
cwd = {cwd!r}
log_path = Path({log_path!r})
exit_path = Path({exit_path!r})
pid_path = Path({pid_path!r})
log_path.parent.mkdir(parents=True, exist_ok=True)
environment = os.environ.copy()
environment['PYTHONPATH'] = {python_path!r}
with log_path.open('ab', buffering=0) as log:
    child = subprocess.Popen(command, cwd=cwd, stdin=subprocess.DEVNULL, stdout=log, stderr=log, env=environment)
    try:
        with pid_path.open('x', encoding='ascii') as handle:
            handle.write(str(child.pid))
    except BaseException:
        child.terminate()
        child.wait()
        raise
    returncode = child.wait()
with exit_path.open('x', encoding='ascii') as handle:
    handle.write(str(returncode))
"""
    source = f"""# REMOTE_LAUNCH_V1:{label}
import json, os, signal, subprocess, time
from pathlib import Path
pid_path = Path({pid_path!r})
registration_path = Path({registration_path!r})
launch_token = {launch_token!r}
def process_start_ticks(pid):
    fields = Path(f'/proc/{{pid}}/stat').read_text(encoding='ascii').rsplit(')', 1)[1].split()
    return int(fields[19])
process = subprocess.Popen(
    [{python_bin!r}, '-c', {supervisor!r}],
    stdin=subprocess.DEVNULL,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
    start_new_session=True,
    close_fds=True,
)
try:
    owner_start_ticks = process_start_ticks(process.pid)
except BaseException:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    raise
deadline = time.monotonic() + 30
while not pid_path.is_file() and process.poll() is None and time.monotonic() < deadline:
    time.sleep(0.05)
if not pid_path.is_file():
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    raise RuntimeError('remote child PID was not published')
identity = {{
    'schema_version': 1,
    'label': {label!r},
    'launch_token': launch_token,
    'registration_path': str(registration_path),
    'child_pid': int(pid_path.read_text(encoding='ascii')),
    'owner_pid': process.pid,
    'owner_pgid': process.pid,
    'owner_start_ticks': owner_start_ticks,
}}
with registration_path.open('x', encoding='utf-8') as handle:
    json.dump(identity, handle, ensure_ascii=False, sort_keys=True)
    handle.write('\\n')
    handle.flush()
    os.fsync(handle.fileno())
print(json.dumps(identity, ensure_ascii=False, sort_keys=True))
"""

    def recover_and_raise(
        original: BaseException,
        message: str,
        registration: Mapping[str, Any] | None = None,
    ) -> None:
        try:
            verified = dict(registration) if registration is not None else (
                _read_remote_launch_registration(
                    host=host,
                    python_bin=python_bin,
                    label=label,
                    launch_token=launch_token,
                    registration_path=registration_path,
                    remote_python=remote_python,
                )
            )
            _terminate_remote_launch_registration(
                host=host,
                python_bin=python_bin,
                registration=verified,
                remote_python=remote_python,
            )
        except BaseException as recovery_exc:
            raise RuntimeError(
                f"{message}; launch registration recovery failed: "
                f"{type(recovery_exc).__name__}: {recovery_exc}"
            ) from original
        raise RuntimeError(
            f"{message}; registered process recovered after acknowledgement failure"
        ) from original

    try:
        output = remote_python(host, python_bin, source, timeout=40)
    except BaseException as exc:
        recover_and_raise(exc, f"remote launch acknowledgement failed: {label}")
    try:
        report = json.loads(output.splitlines()[-1])
        response_registration = _validate_launch_registration(
            report,
            label=label,
            launch_token=launch_token,
            registration_path=registration_path,
        )
    except BaseException as exc:
        recover_and_raise(exc, f"remote launch acknowledgement is invalid: {label}")
    try:
        stored_registration = _read_remote_launch_registration(
            host=host,
            python_bin=python_bin,
            label=label,
            launch_token=launch_token,
            registration_path=registration_path,
            remote_python=remote_python,
        )
    except BaseException as exc:
        recover_and_raise(exc, f"remote launch registration verification failed: {label}")
    if response_registration != stored_registration:
        recover_and_raise(
            RuntimeError("launch acknowledgement and registration identity mismatch"),
            f"remote launch identity mismatch: {label}",
            stored_registration,
        )
    return OwnedProcess(
        host_id=host_id,
        label=label,
        pid=report["child_pid"],
        owner_pid=report["owner_pid"],
        owner_pgid=report["owner_pgid"],
        owner_start_ticks=report["owner_start_ticks"],
        registration_path=registration_path,
        launch_token=launch_token,
        host=host,
        python_bin=python_bin,
        exit_path=exit_path,
    )


def _local_launch_process(
    *,
    label: str,
    command: Sequence[str],
    cwd: Path,
    log_root: Path,
    stop_path: Path | None = None,
) -> OwnedProcess:
    log_root.mkdir(parents=True, exist_ok=False)
    stdout = (log_root / "stdout.log").open("xb", buffering=0)
    stderr = (log_root / "stderr.log").open("xb", buffering=0)
    try:
        process = subprocess.Popen(
            [str(item) for item in command],
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
        )
    except BaseException:
        stdout.close()
        stderr.close()
        raise
    return OwnedProcess(
        host_id="pc",
        label=label,
        pid=int(process.pid),
        handle=process,
        stop_path=stop_path,
        log_handles=(stdout, stderr),
    )


def _remote_process_state(process: OwnedProcess) -> tuple[bool, int | None]:
    source = f"""# REMOTE_PROCESS_STATE_V1
import json, os
from pathlib import Path
pid = {process.owner_pid!r}
owner_pgid = {process.owner_pgid!r}
owner_start_ticks = {process.owner_start_ticks!r}
if not isinstance(pid, int) or pid <= 0:
    raise RuntimeError('missing owned supervisor PID')
exit_path = Path({str(process.exit_path)!r})
proc_path = Path(f'/proc/{{pid}}/stat')
running = proc_path.is_file()
if running:
    fields = proc_path.read_text(encoding='ascii').rsplit(')', 1)[1].split()
    actual_start_ticks = int(fields[19])
    if actual_start_ticks != owner_start_ticks or os.getpgid(pid) != owner_pgid:
        raise RuntimeError('owned supervisor identity changed')
returncode = int(exit_path.read_text(encoding='ascii')) if exit_path.is_file() else None
print(json.dumps({{'running': running, 'returncode': returncode}}))
"""
    report = json.loads(
        _remote_python(
            str(process.host), str(process.python_bin), source, timeout=30
        ).splitlines()[-1]
    )
    return bool(report["running"]), report["returncode"]


def _remote_wait(process: OwnedProcess) -> None:
    source = f"""# REMOTE_WAIT_V1:{process.label}
import json, time
from pathlib import Path
exit_path = Path({str(process.exit_path)!r})
deadline = time.monotonic() + 60
while not exit_path.is_file() and time.monotonic() < deadline:
    time.sleep(0.2)
returncode = int(exit_path.read_text(encoding='ascii')) if exit_path.is_file() else None
print(json.dumps({{'returncode': returncode}}))
"""
    report = json.loads(
        _remote_python(
            str(process.host), str(process.python_bin), source, timeout=70
        ).splitlines()[-1]
    )
    if report.get("returncode") != 0:
        raise RuntimeError(f"remote {process.label} exited unsuccessfully")


def _remote_cleanup(processes: Sequence[OwnedProcess]) -> list[str]:
    errors: list[str] = []
    for process in processes:
        if process.host_id == "pc":
            continue
        try:
            if any(
                not isinstance(value, int) or isinstance(value, bool) or value <= 0
                for value in (
                    process.owner_pid,
                    process.owner_pgid,
                    process.owner_start_ticks,
                )
            ):
                raise RuntimeError("remote process lacks owned identity")
            if not isinstance(process.registration_path, (str, Path)) or not isinstance(
                process.launch_token, str
            ):
                raise RuntimeError("remote process lacks owned launch registration")
            registration = _validate_launch_registration(
                {
                    "schema_version": 1,
                    "label": process.label,
                    "launch_token": process.launch_token,
                    "registration_path": str(process.registration_path),
                    "child_pid": process.pid,
                    "owner_pid": process.owner_pid,
                    "owner_pgid": process.owner_pgid,
                    "owner_start_ticks": process.owner_start_ticks,
                },
                label=process.label,
                launch_token=process.launch_token,
                registration_path=str(process.registration_path),
            )
            _terminate_remote_launch_registration(
                host=str(process.host),
                python_bin=str(process.python_bin),
                registration=registration,
                remote_python=_remote_python,
            )
        except BaseException as exc:
            errors.append(f"remote {process.label}: {type(exc).__name__}: {exc}")
    return errors


def build_production_hooks(
    runtime: ProductionRuntime, smoke: FormalSmokeConfig | None = None
) -> LifecycleHooks:
    state: dict[str, Any] = {}

    def active_runtime() -> ProductionRuntime:
        return state.get("runtime", runtime)

    def paths(attempt: Attempt) -> dict[str, str | Path]:
        deployments = active_runtime().deployments
        ecs_root = _host_attempt_root(deployments["ecs"], attempt.attempt_id)
        pi_root = _host_attempt_root(deployments["pi"], attempt.attempt_id)
        return {
            "ecs_root": ecs_root,
            "ecs_raw": f"{ecs_root}/raw/server",
            "pi_root": pi_root,
            "pi_raw": f"{pi_root}/raw/client_c1",
            "pc_raw": attempt.path / "raw" / "pc",
        }

    def prepare(attempt: Attempt) -> None:
        deployments = deploy_source_archive(
            runtime.frozen.archive_path,
            runtime.frozen.source_manifest,
            ecs_host=runtime.ecs_host,
            pi_host=runtime.pi_host,
            pc_runtime_root=runtime.pc_runtime_root,
            run=_run,
            ssh=_ssh,
            remote_python=_remote_python,
        )
        prepared_runtime = replace(runtime, deployments=deployments)
        state["runtime"] = prepared_runtime
        context_paths = write_host_contexts(
            attempt,
            group_id=runtime.frozen_run.group_id,
            seed=runtime.frozen_run.seed,
            provenance=runtime.frozen_run.provenance,
        )
        state["contexts"] = context_paths
        runtime_paths = paths(attempt)
        state["paths"] = runtime_paths
        if smoke is not None:
            smoke_bindings = {
                "ecs_root": str(runtime_paths["ecs_root"]),
                "ecs_raw": str(runtime_paths["ecs_raw"]),
            }
            resolved_trace_output = smoke.trace_output
            resolved_initial_checkpoint = smoke.initial_checkpoint
            for token, value in smoke_bindings.items():
                resolved_trace_output = resolved_trace_output.replace(
                    "{" + token + "}", value
                )
                resolved_initial_checkpoint = resolved_initial_checkpoint.replace(
                    "{" + token + "}", value
                )
            if "{" in resolved_trace_output or "{" in resolved_initial_checkpoint:
                raise ValueError("formal smoke path contains an unresolved binding")
            state["smoke_trace_output"] = resolved_trace_output
            state["smoke_initial_checkpoint"] = resolved_initial_checkpoint
            derived = derive_noncanonical_smoke_commands(
                runtime.frozen_run.manifest, smoke
            )
            state["commands"] = derived
            identity_path = attempt.path / "smoke_command_identity.json"
            if identity_path.exists() or identity_path.is_symlink():
                raise FileExistsError("smoke command identity already exists")
            identity_path.write_text(
                json.dumps(
                    derived["smoke_identity"],
                    ensure_ascii=False,
                    sort_keys=True,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
                newline="\n",
            )
        for host, root in (
            (runtime.ecs_host, runtime_paths["ecs_root"]),
            (runtime.pi_host, runtime_paths["pi_root"]),
        ):
            _ssh(host, f"test ! -e '{root}' && mkdir -p '{root}/contexts' '{root}/raw'")
        remote_contexts = (
            (context_paths["ecs_server"], runtime.ecs_host, f"{runtime_paths['ecs_root']}/contexts/server.json"),
            (context_paths["pi_client"], runtime.pi_host, f"{runtime_paths['pi_root']}/contexts/client.json"),
            (context_paths["pi_sampler"], runtime.pi_host, f"{runtime_paths['pi_root']}/contexts/sampler.json"),
        )
        for local, host, remote in remote_contexts:
            _run(["scp", "-p", str(local), f"{host}:{remote}"], timeout=120)
        if smoke is not None and smoke.initial_checkpoint_source is not None:
            source = Path(smoke.initial_checkpoint_source)
            if (
                not source.is_file()
                or source.is_symlink()
                or sha256_file(source) != smoke.initial_checkpoint_sha256
            ):
                raise ArchiveMismatch("formal smoke initial checkpoint hash mismatch")
            _run(
                [
                    "scp",
                    "-p",
                    str(source),
                    f"{runtime.ecs_host}:{state['smoke_initial_checkpoint']}",
                ],
                timeout=120,
            )
            verify_source = f"""
import hashlib, json
from pathlib import Path
path = Path({state['smoke_initial_checkpoint']!r})
digest = hashlib.sha256(path.read_bytes()).hexdigest()
expected = {smoke.initial_checkpoint_sha256!r}
if digest != expected:
    raise RuntimeError('formal smoke initial checkpoint hash mismatch')
print(json.dumps({{'initial_checkpoint_sha256': digest}}, sort_keys=True))
"""
            response = json.loads(
                _remote_python(
                    runtime.ecs_host,
                    "/root/gaps_env/bin/python",
                    verify_source,
                    timeout=120,
                ).splitlines()[-1]
            )
            if response != {
                "initial_checkpoint_sha256": smoke.initial_checkpoint_sha256
            }:
                raise ArchiveMismatch("formal smoke initial checkpoint verification failed")
        preflight_frozen_run(prepared_runtime, attempt.attempt_id)

    def launch_server(attempt: Attempt) -> OwnedProcess:
        runtime_paths = state["paths"]
        deployments = active_runtime().deployments
        source_commands = (
            state["commands"] if smoke is not None else runtime.frozen_run.manifest["commands"]
        )
        command = _rewrite_ecs_dataset_paths(source_commands["server_ecs"])
        command = _replace_command_option(
            command,
            "--output-dir",
            f"{runtime_paths['ecs_raw']}/training",
        )
        if smoke is None or smoke.observer_enabled:
            command = _replace_command_option(
                command, "--observer-context", f"{runtime_paths['ecs_root']}/contexts/server.json"
            )
            command = _replace_command_option(
                command, "--observer-events", f"{runtime_paths['ecs_raw']}/events.jsonl"
            )
        if smoke is not None:
            replacements = {
                smoke.trace_output: str(state["smoke_trace_output"]),
                smoke.initial_checkpoint: str(state["smoke_initial_checkpoint"]),
            }
            command = [replacements.get(item, item) for item in command]
        return _remote_launch_process(
            host_id="ecs",
            label="server",
            host=runtime.ecs_host,
            python_bin="/root/gaps_env/bin/python",
            command=command,
            cwd=str(deployments["ecs"].src_path),
            log_path=f"{runtime_paths['ecs_raw']}/server.log",
            exit_path=f"{runtime_paths['ecs_raw']}/server.exit",
            python_path=str(deployments["ecs"].src_path),
            registration_path=f"{runtime_paths['ecs_root']}/raw/server.registration.json",
        )

    def launch_pi_client(attempt: Attempt) -> OwnedProcess:
        runtime_paths = state["paths"]
        deployments = active_runtime().deployments
        source_commands = (
            state["commands"] if smoke is not None else runtime.frozen_run.manifest["commands"]
        )
        command = list(source_commands["client_c1_pi"])
        if smoke is None or smoke.observer_enabled:
            command = _replace_command_option(
                command,
                "--observer-context",
                f"{runtime_paths['pi_root']}/contexts/client.json",
            )
            command = _replace_command_option(
                command, "--observer-events", f"{runtime_paths['pi_raw']}/events.jsonl"
            )
        return _remote_launch_process(
            host_id="pi",
            label="pi-client",
            host=runtime.pi_host,
            python_bin="/home/gaps/GAPS/gaps_rpi_env/bin/python",
            command=command,
            cwd=str(deployments["pi"].src_path),
            log_path=f"{runtime_paths['pi_raw']}/client.log",
            exit_path=f"{runtime_paths['pi_raw']}/client.exit",
            python_path=str(deployments["pi"].src_path),
            registration_path=f"{runtime_paths['pi_root']}/raw/pi-client.registration.json",
        )

    def launch_pi_sampler(attempt: Attempt, client: object) -> OwnedProcess:
        assert isinstance(client, OwnedProcess)
        runtime_paths = state["paths"]
        deployments = active_runtime().deployments
        stop_path = f"{runtime_paths['pi_raw']}/sampler.stop"
        command = [
            "/home/gaps/GAPS/gaps_rpi_env/bin/python",
            "-m",
            "scripts.sample_iotj_process_resources",
            "--pid",
            str(client.pid),
            "--observer-context",
            f"{runtime_paths['pi_root']}/contexts/sampler.json",
            "--observer-events",
            f"{runtime_paths['pi_raw']}/resource.jsonl",
            "--stop-file",
            stop_path,
        ]
        process = _remote_launch_process(
            host_id="pi",
            label="pi-sampler",
            host=runtime.pi_host,
            python_bin="/home/gaps/GAPS/gaps_rpi_env/bin/python",
            command=command,
            cwd=str(deployments["pi"].src_path),
            log_path=f"{runtime_paths['pi_raw']}/sampler.log",
            exit_path=f"{runtime_paths['pi_raw']}/sampler.exit",
            python_path=str(deployments["pi"].src_path),
            registration_path=f"{runtime_paths['pi_root']}/raw/pi-sampler.registration.json",
        )
        return replace(process, stop_path=stop_path)

    def launch_pc_client(attempt: Attempt) -> OwnedProcess:
        runtime_paths = state["paths"]
        deployments = active_runtime().deployments
        source_commands = (
            state["commands"] if smoke is not None else runtime.frozen_run.manifest["commands"]
        )
        command = list(source_commands["client_c2_pc"])
        command[0] = sys.executable
        if smoke is None or smoke.observer_enabled:
            command = _replace_command_option(
                command,
                "--observer-context",
                str(state["contexts"]["pc_client"]),
            )
            command = _replace_command_option(
                command,
                "--observer-events",
                str(Path(runtime_paths["pc_raw"]) / "events.jsonl"),
            )
        return _local_launch_process(
            label="pc-client",
            command=command,
            cwd=Path(deployments["pc"].src_path),
            log_root=Path(runtime_paths["pc_raw"]) / "client_logs",
        )

    def launch_pc_sampler(attempt: Attempt, client: object) -> OwnedProcess:
        assert isinstance(client, OwnedProcess)
        runtime_paths = state["paths"]
        deployments = active_runtime().deployments
        stop_path = Path(runtime_paths["pc_raw"]) / "sampler.stop"
        command = [
            sys.executable,
            "-m",
            "scripts.sample_iotj_process_resources",
            "--pid",
            str(client.pid),
            "--observer-context",
            str(state["contexts"]["pc_sampler"]),
            "--observer-events",
            str(Path(runtime_paths["pc_raw"]) / "resource.jsonl"),
            "--stop-file",
            str(stop_path),
        ]
        return _local_launch_process(
            label="pc-sampler",
            command=command,
            cwd=Path(deployments["pc"].src_path),
            log_root=Path(runtime_paths["pc_raw"]) / "sampler_logs",
            stop_path=stop_path,
        )

    def monitor(_attempt: Attempt, server: object) -> None:
        assert isinstance(server, OwnedProcess)
        deadline = time.monotonic() + runtime.timeout_seconds
        while True:
            running, returncode = _remote_process_state(server)
            if not running:
                if returncode != 0:
                    raise RuntimeError("server process exited unsuccessfully")
                return
            if time.monotonic() >= deadline:
                raise TimeoutError("server process exceeded formal timeout")
            time.sleep(runtime.poll_seconds)

    def stop_sampler(_attempt: Attempt, process: object) -> None:
        assert isinstance(process, OwnedProcess)
        if process.host_id == "pc":
            stop_path = Path(str(process.stop_path))
            if stop_path.exists():
                raise FileExistsError(f"sampler stop file already exists: {stop_path}")
            stop_path.write_text("stop\n", encoding="utf-8")
        else:
            _ssh(str(process.host), f"test ! -e '{process.stop_path}' && touch '{process.stop_path}'")

    def wait_sampler(_attempt: Attempt, process: object) -> None:
        assert isinstance(process, OwnedProcess)
        if process.host_id == "pc":
            if process.handle.wait(timeout=60) != 0:
                raise RuntimeError("PC sampler exited unsuccessfully")
        else:
            _remote_wait(process)

    def recover(attempt: Attempt) -> None:
        runtime_paths = state["paths"]
        raw_root = attempt.path / "raw"
        for host_id, host, remote in (
            ("ecs", runtime.ecs_host, runtime_paths["ecs_raw"]),
            ("pi", runtime.pi_host, runtime_paths["pi_raw"]),
        ):
            destination = raw_root / host_id
            if destination.exists() or destination.is_symlink():
                raise FileExistsError(f"refusing to overwrite raw evidence: {destination}")
            _reject_symlink_components(destination.parent, "raw evidence root")
            _run(["scp", "-pr", f"{host}:{remote}", str(destination)], timeout=300)
            if not destination.is_dir() or destination.is_symlink():
                raise RuntimeError(f"raw evidence recovery failed: {host_id}")

    def validate(attempt: Attempt) -> ValidationOutcome:
        return invoke_validator(
            attempt,
            validator=runtime.validator,
            protocol_manifest=runtime.frozen.protocol_path,
            run=_run,
        )

    def cleanup_owned(processes: Sequence[object]) -> list[str]:
        typed = [process for process in processes if isinstance(process, OwnedProcess)]
        errors = _remote_cleanup(typed)
        for process in typed:
            if process.host_id != "pc" or process.handle is None:
                continue
            try:
                _terminate_processes((process.handle,))
            except BaseException as exc:
                errors.append(f"local {process.label}: {type(exc).__name__}: {exc}")
        for process in typed:
            for handle in process.log_handles:
                try:
                    handle.close()
                except BaseException as exc:
                    errors.append(f"log {process.label}: {type(exc).__name__}: {exc}")
        return errors

    return LifecycleHooks(
        prepare=prepare,
        launch_server=launch_server,
        start_tunnels=lambda _attempt: _start_tunnels(runtime.ecs_host, runtime.pi_host),
        launch_pi_client=launch_pi_client,
        launch_pi_sampler=launch_pi_sampler,
        launch_pc_client=launch_pc_client,
        launch_pc_sampler=launch_pc_sampler,
        monitor_server=monitor,
        stop_sampler=stop_sampler,
        wait_sampler=wait_sampler,
        recover_evidence=recover,
        validate_attempt=validate,
        cleanup_owned=cleanup_owned,
        cleanup_tunnels=lambda processes: _terminate_processes(processes),
    )


def _cleanup_lifecycle(
    attempt: Attempt,
    hooks: LifecycleHooks,
    samplers: Sequence[object],
    owned: Sequence[object],
    tunnels: Sequence[object],
) -> list[str]:
    errors: list[str] = []
    for sampler in samplers:
        try:
            hooks.stop_sampler(attempt, sampler)
        except BaseException as exc:  # cleanup must continue for every owned process
            errors.append(f"stop sampler: {type(exc).__name__}: {exc}")
    for sampler in samplers:
        try:
            hooks.wait_sampler(attempt, sampler)
        except BaseException as exc:
            errors.append(f"wait sampler: {type(exc).__name__}: {exc}")
    try:
        owned_errors = hooks.cleanup_owned(tuple(owned))
        if owned_errors is not None:
            if isinstance(owned_errors, (str, bytes)) or not isinstance(
                owned_errors, Sequence
            ):
                raise TypeError(
                    "cleanup_owned must return a sequence of errors or None"
                )
            for error in owned_errors:
                if not isinstance(error, str):
                    raise TypeError("cleanup_owned errors must be strings")
                errors.append(f"cleanup owned: {error}")
    except BaseException as exc:
        errors.append(f"cleanup owned: {type(exc).__name__}: {exc}")
    try:
        hooks.cleanup_tunnels(tuple(tunnels))
    except BaseException as exc:
        errors.append(f"cleanup tunnels: {type(exc).__name__}: {exc}")
    return errors


def _execute_hook_lifecycle(
    attempt: Attempt,
    hooks: LifecycleHooks,
    *,
    after_prepare: Callable[[], None] | None = None,
) -> None:
    """Run the one production lifecycle shared by canonical and smoke modes."""

    owned: list[object] = []
    samplers: list[object] = []
    tunnels: list[object] = []
    cleaned = False
    try:
        hooks.prepare(attempt)
        if after_prepare is not None:
            after_prepare()
        server = hooks.launch_server(attempt)
        owned.append(server)
        tunnels.extend(hooks.start_tunnels(attempt))
        pi_client = hooks.launch_pi_client(attempt)
        owned.append(pi_client)
        pi_sampler = hooks.launch_pi_sampler(attempt, pi_client)
        samplers.append(pi_sampler)
        owned.append(pi_sampler)
        pc_client = hooks.launch_pc_client(attempt)
        owned.append(pc_client)
        pc_sampler = hooks.launch_pc_sampler(attempt, pc_client)
        samplers.append(pc_sampler)
        owned.append(pc_sampler)
        hooks.monitor_server(attempt, server)
        cleanup_errors = _cleanup_lifecycle(attempt, hooks, samplers, owned, tunnels)
        cleaned = True
        if cleanup_errors:
            raise AttemptFailure("; ".join(cleanup_errors))
        hooks.recover_evidence(attempt)
    except BaseException as exc:
        cleanup_errors = [] if cleaned else _cleanup_lifecycle(
            attempt, hooks, samplers, owned, tunnels
        )
        if cleanup_errors:
            detail = f"{type(exc).__name__}: {exc}; cleanup: " + "; ".join(
                cleanup_errors
            )
            raise AttemptFailure(detail) from exc
        raise


def run_noncanonical_smoke_attempt(
    output_path: Path,
    *,
    run_id: str,
    mode: str,
    provenance: Provenance,
    hooks: LifecycleHooks,
) -> Attempt:
    """Execute formal smoke without touching the canonical attempt registry."""

    if mode not in {"off", "on"}:
        raise ValueError("formal smoke mode must be off or on")
    match = _RUN_ID_RE.fullmatch(run_id)
    if match is None:
        raise ValueError("formal smoke run_id is invalid")
    validate_requested_run(match.group(1).upper(), int(match.group(2)))
    provenance.require_complete()
    output = Path(output_path)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"noncanonical smoke output already exists: {output}")
    _reject_symlink_components(output.parent, "noncanonical smoke output parent")
    output.mkdir(parents=True)
    attempt_id = f"{run_id}__a998" if mode == "off" else f"{run_id}__a999"
    attempt = Attempt(run_id=run_id, attempt_id=attempt_id, path=output)
    marker = {
        "schema_version": 1,
        "namespace": "noncanonical_smoke",
        "noncanonical_smoke": True,
        "mode": mode,
        "run_id": run_id,
        "attempt_id": attempt_id,
        "confirmation_commit": provenance.confirmation_commit,
        "source_archive_sha256": provenance.source_archive_sha256,
        "dataset_manifest_sha256": provenance.dataset_manifest_sha256,
        "algorithm_config_sha256": provenance.algorithm_config_sha256,
    }
    marker["binding_sha256"] = canonical_sha256(marker)
    marker_path = output / "noncanonical_smoke.json"
    marker_path.write_text(
        json.dumps(marker, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    try:
        _execute_hook_lifecycle(attempt, hooks)
    except BaseException as exc:
        failure_path = output / "noncanonical_smoke_failure.json"
        failure = {
            **marker,
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        failure_path.write_text(
            json.dumps(failure, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        raise
    success = {**marker, "status": "evidence_recovered"}
    (output / "noncanonical_smoke_complete.json").write_text(
        json.dumps(success, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return attempt


def _failure_category(exc: BaseException) -> str:
    category = getattr(exc, "reason_category", None)
    return category if isinstance(category, str) else "process"


def _failure_reason(exc: BaseException) -> str:
    category = _failure_category(exc)
    return {
        "archive_integrity": "archive_integrity_failure",
        "dataset_integrity": "dataset_integrity_failure",
        "config_integrity": "config_integrity_failure",
        "dependency_mismatch": "dependency_mismatch",
        "transport": "transport_failure",
        "tunnel": "tunnel_failure",
        "evidence_io": "evidence_io_failure",
        "audit": "audit_failure",
        "resource_coverage": "resource_coverage_failure",
        "observer_failure": "observer_failure",
        "operator_abort": "operator_abort",
    }.get(category, "process_failure")


def _append_controller_log(attempt_path: Path, message: str) -> None:
    attempt_path, _run_id, _attempt_id = _guard_attempt_path(attempt_path)
    log_path = attempt_path / "controller.log"
    if log_path.is_symlink():
        raise RuntimeError("controller log cannot be a symlink")
    with log_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(f"{_utc_now()} {message}\n")
        handle.flush()
        os.fsync(handle.fileno())


def run_confirmation_attempt(
    raw_root: Path,
    run_id: str,
    *,
    provenance: Provenance,
    hooks: LifecycleHooks,
) -> Attempt:
    provenance.require_complete()
    attempt = allocate_attempt(Path(raw_root), run_id)
    bind_attempt_provenance(attempt, provenance)
    mark_attempt(
        attempt.path,
        "running",
        event_type="attempt_start",
        reason="attempt_allocated",
    )
    try:
        _execute_hook_lifecycle(
            attempt,
            hooks,
            after_prepare=lambda: mark_attempt(
                attempt.path,
                "running",
                event_type="preflight_passed",
                reason="preflight_passed",
            ),
        )
        outcome = hooks.validate_attempt(attempt)
        if not isinstance(outcome, ValidationOutcome):
            raise TypeError("validator must return ValidationOutcome")
        if not outcome.success:
            mark_attempt(
                attempt.path,
                "invalid",
                event_type="attempt_failure",
                reason="validator_rejected",
            )
            return attempt
        if not isinstance(outcome.audit_sha256, str) or not _HASH_RE.fullmatch(
            outcome.audit_sha256
        ):
            raise RuntimeError("successful validator result requires audit SHA-256")
        mark_attempt(
            attempt.path,
            "canonical",
            event_type="attempt_end",
            reason="validator_accepted",
            audit_sha256=outcome.audit_sha256,
        )
        return attempt
    except BaseException as exc:
        detail = f"{type(exc).__name__}: {exc}"
        state = "aborted" if isinstance(exc, (AttemptAborted, KeyboardInterrupt)) else "failed"
        try:
            _append_controller_log(attempt.path, detail)
            mark_attempt(
                attempt.path,
                state,
                event_type="attempt_failure",
                reason=_failure_reason(exc),
            )
        except BaseException as status_exc:
            raise RuntimeError(
                f"attempt failed and status append also failed: {status_exc}"
            ) from exc
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-manifest", type=Path, required=True)
    parser.add_argument("--source-archive-manifest", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--command-root", type=Path, required=True)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument(
        "--pc-runtime-root", type=Path, default=DEFAULT_PC_RUNTIME_ROOT
    )
    parser.add_argument("--ecs-host", default="root@121.40.139.213")
    parser.add_argument("--pi-hosts", default="gaps@192.168.31.184")
    parser.add_argument(
        "--runs",
        default=",".join(f"{group}:{seed}" for group, seed in DEFAULT_QUEUE),
        help="comma-separated exact confirmation run allowlist entries",
    )
    parser.add_argument("--poll-seconds", type=float, default=120.0)
    parser.add_argument("--run-timeout-seconds", type=float, default=18_000.0)
    parser.add_argument("--wait-for-pi-minutes", type=int, default=360)
    parser.add_argument("--pi-retry-seconds", type=int, default=60)
    parser.add_argument(
        "--validator",
        type=Path,
        default=REPO_ROOT / "scripts" / "validate_iotj_confirmation_attempt.py",
        help="Task 7 validator callable/CLI seam",
    )
    parser.add_argument(
        "--validate-inputs-only",
        action="store_true",
        help="validate frozen inputs without transport or process actions",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="deploy and verify every requested host without launching processes",
    )
    return parser


def _parse_requested_runs(specification: str) -> tuple[tuple[str, int], ...]:
    if not isinstance(specification, str) or not specification:
        raise ValueError("--runs must contain at least one confirmation run")
    requested: list[tuple[str, int]] = []
    for token in specification.split(","):
        match = re.fullmatch(r"(B2|B5):(42|43|44|45|46)", token)
        if match is None:
            raise ValueError(f"invalid confirmation run selector: {token!r}")
        run = (match.group(1), int(match.group(2)))
        validate_requested_run(*run)
        if run in requested:
            raise ValueError(f"duplicate confirmation run selector: {token}")
        requested.append(run)
    return tuple(requested)


def _validate_cli_inputs(args: argparse.Namespace) -> FrozenInputs:
    return load_frozen_inputs(
        args.protocol_manifest,
        args.source_archive_manifest,
        args.dataset_manifest,
        args.command_root,
        args.source_archive,
    )


def _select_frozen_runs(
    frozen: FrozenInputs, requested: Sequence[tuple[str, int]]
) -> tuple[FrozenRun, ...]:
    by_identity = {(run.group_id, run.seed): run for run in frozen.runs}
    try:
        return tuple(by_identity[identity] for identity in requested)
    except KeyError as exc:
        raise ValueError(f"requested run is absent from frozen inputs: {exc.args[0]}") from exc


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.poll_seconds <= 0 or args.run_timeout_seconds <= 0:
        raise ValueError("poll and run timeout seconds must be positive")
    if args.wait_for_pi_minutes < 0 or args.pi_retry_seconds <= 0:
        raise ValueError("Pi wait minutes must be nonnegative and retry seconds positive")
    if args.validate_inputs_only and args.preflight_only:
        raise ValueError("validation-only and preflight-only modes are mutually exclusive")
    frozen = _validate_cli_inputs(args)
    requested = _parse_requested_runs(args.runs)
    selected_runs = _select_frozen_runs(frozen, requested)
    if args.validate_inputs_only:
        print(
            json.dumps(
                {
                    "status": "validated",
                    "queue": [
                        {"group_id": run.group_id, "seed": run.seed}
                        for run in selected_runs
                    ],
                },
                sort_keys=True,
            )
        )
        return 0
    validator = Path(args.validator)
    if not args.preflight_only and (
        not validator.is_file() or validator.is_symlink()
    ):
        raise FileNotFoundError(f"validator must be a regular file: {validator}")
    pi_hosts = tuple(host for host in args.pi_hosts.split(",") if host)
    if not pi_hosts:
        raise ValueError("at least one Pi host is required")
    pi_host = _wait_for_pi(
        pi_hosts, args.wait_for_pi_minutes, args.pi_retry_seconds
    )
    deployments: Mapping[str, HostDeployment]
    if args.preflight_only:
        deployments = deploy_source_archive(
            frozen.archive_path,
            frozen.source_manifest,
            ecs_host=args.ecs_host,
            pi_host=pi_host,
            pc_runtime_root=args.pc_runtime_root,
            run=_run,
            ssh=_ssh,
            remote_python=_remote_python,
        )
    else:
        deployments = {}
    for frozen_run in selected_runs:
        runtime = ProductionRuntime(
            frozen=frozen,
            frozen_run=frozen_run,
            deployments=deployments,
            ecs_host=args.ecs_host,
            pi_host=pi_host,
            validator=validator,
            poll_seconds=args.poll_seconds,
            timeout_seconds=args.run_timeout_seconds,
            pc_runtime_root=args.pc_runtime_root,
        )
        if args.preflight_only:
            preflight_frozen_run(runtime, f"{frozen_run.run_id}__a000")
        else:
            run_confirmation_attempt(
                args.raw_root,
                frozen_run.run_id,
                provenance=frozen_run.provenance,
                hooks=build_production_hooks(runtime),
            )
    print(
        json.dumps(
            {
                "status": "preflighted" if args.preflight_only else "completed",
                "queue": [run.run_id for run in selected_runs],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
