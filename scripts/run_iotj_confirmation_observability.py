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
import shutil
import subprocess
import sys
import tarfile
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
EXPECTED_DEPENDENCY_VERSIONS = {
    "flwr": "1.23.0",
    "protobuf": "4.25.8",
    "psutil": "7.0.0",
}
VALID_ATTEMPT_STATES = {"running", "failed", "aborted", "invalid", "canonical"}
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
    cleanup_owned: Callable[[Sequence[object]], None]
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


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _attempt_directories(run_root: Path, run_id: str) -> list[tuple[int, Path]]:
    rows: list[tuple[int, Path]] = []
    for path in run_root.glob(f"{run_id}__a[0-9][0-9][0-9]"):
        match = _ATTEMPT_ID_RE.fullmatch(path.name)
        if match is None or match.group(1) != run_id or not path.is_dir():
            continue
        rows.append((int(match.group(2)), path))
    return sorted(rows)


def allocate_attempt(raw_root: Path, run_id: str) -> Attempt:
    validate_run_id(run_id)
    run_root = Path(raw_root) / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    existing = _attempt_directories(run_root, run_id)
    statuses: list[tuple[int, dict[str, Any]]] = []
    for number, path in existing:
        status_path = path / "attempt_status.json"
        if status_path.is_file():
            status = _load_json(status_path)
            if status.get("state") not in VALID_ATTEMPT_STATES:
                raise RuntimeError(f"invalid prior attempt state in {status_path}")
            statuses.append((number, status))
    if any(status["state"] == "canonical" for _, status in statuses):
        raise RuntimeError(f"canonical attempt already exists for {run_id}")
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
    return Attempt(run_id, attempt_id, path)


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


def mark_attempt(
    attempt_path: Path,
    state: str,
    *,
    audit_sha256: str | None = None,
    event_type: str | None = None,
    reason: str | None = None,
    reason_category: str | None = None,
    provenance: Provenance | None = None,
) -> dict[str, Any]:
    if state not in VALID_ATTEMPT_STATES:
        raise ValueError(f"invalid attempt state: {state}")
    attempt_path = Path(attempt_path)
    match = _ATTEMPT_ID_RE.fullmatch(attempt_path.name)
    if match is None:
        raise ValueError(f"invalid attempt directory name: {attempt_path.name}")
    if state == "canonical" and (
        not isinstance(audit_sha256, str) or not _HASH_RE.fullmatch(audit_sha256)
    ):
        raise ValueError("canonical status requires validator audit_sha256")
    if audit_sha256 is not None and not _HASH_RE.fullmatch(audit_sha256):
        raise ValueError("audit_sha256 must be a lowercase SHA-256")
    provenance = provenance or Provenance(None, None, None, None)
    defaults = {
        "running": ("controller state update", "controller"),
        "failed": ("attempt failed", "process"),
        "aborted": ("attempt aborted", "operator_abort"),
        "invalid": ("validator rejected attempt", "audit"),
        "canonical": ("validator accepted attempt", "audit"),
    }
    default_reason, default_category = defaults[state]
    status = {
        "state": state,
        "event_type": event_type or state,
        "reason": reason or default_reason,
        "reason_category": reason_category or default_category,
        "wall_time_utc": _utc_now(),
        "confirmation_commit": provenance.confirmation_commit,
        "source_archive_sha256": provenance.source_archive_sha256,
        "dataset_manifest_sha256": provenance.dataset_manifest_sha256,
        "algorithm_config_sha256": provenance.algorithm_config_sha256,
        "audit_sha256": audit_sha256,
    }
    events_root = attempt_path / "status_events"
    events_root.mkdir(parents=True, exist_ok=True)
    used: list[int] = []
    for path in events_root.glob("status_[0-9][0-9][0-9].json"):
        try:
            used.append(int(path.stem.rsplit("_", 1)[1]))
        except ValueError as exc:
            raise RuntimeError(f"invalid immutable status event name: {path.name}") from exc
    number = (max(used) + 1) if used else 1
    if number > 999:
        raise RuntimeError("status event sequence exhausted")
    _write_json_exclusive(events_root / f"status_{number:03d}.json", status)
    _atomic_write_json(attempt_path / "attempt_status.json", status)
    return status


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
            or PurePosixPath(relative).is_absolute()
            or ".." in PurePosixPath(relative).parts
            or not isinstance(size, int)
            or size < 0
            or not isinstance(digest, str)
            or not _HASH_RE.fullmatch(digest)
        ):
            raise ArchiveMismatch("invalid tracked member entry")
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
                target = staging.joinpath(*PurePosixPath(relative).parts)
                target.parent.mkdir(parents=True, exist_ok=True)
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


def _remote_extract_source(
    archive_path: str,
    src_path: str,
    source_manifest: Mapping[str, Any],
) -> str:
    """Return a fail-closed remote script; `_remote_python` transports it safely."""
    manifest_json = json.dumps(source_manifest, ensure_ascii=False, sort_keys=True)
    return f"""
import hashlib, json, os, shutil, tarfile
from pathlib import Path, PurePosixPath
manifest = json.loads({manifest_json!r})
archive_path = Path({archive_path!r})
src_path = Path({src_path!r})
def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            value.update(chunk)
    return value.hexdigest()
if digest(archive_path) != manifest['source_archive_sha256']:
    raise RuntimeError('source archive SHA-256 mismatch after transfer')
if src_path.exists():
    raise RuntimeError('refusing non-fresh src directory')
src_path.mkdir(parents=True)
try:
    with tarfile.open(archive_path, 'r:') as archive:
        regular = {{item.name: item for item in archive.getmembers() if item.isfile()}}
        expected = manifest['regular_members']
        if sorted(regular) != sorted(item['relative_path'] for item in expected):
            raise RuntimeError('archive tracked member set mismatch')
        for item in expected:
            relative = item['relative_path']
            pure = PurePosixPath(relative)
            if pure.is_absolute() or '..' in pure.parts:
                raise RuntimeError('unsafe tracked member path')
            source = archive.extractfile(regular[relative])
            if source is None:
                raise RuntimeError('missing tracked member')
            target = src_path.joinpath(*pure.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with source, target.open('xb') as handle:
                shutil.copyfileobj(source, handle)
            if target.stat().st_size != item['byte_size'] or digest(target) != item['sha256']:
                raise RuntimeError('extracted tracked member mismatch')
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
    pc_root = Path(pc_runtime_root) / source_hash
    pc_archive = pc_root / "source.tar"
    pc_src = pc_root / "src"
    pc_root.mkdir(parents=True, exist_ok=True)
    if pc_archive.exists():
        raise FileExistsError(f"refusing to overwrite immutable source archive: {pc_archive}")
    shutil.copyfile(archive_path, pc_archive)
    pc_report = verify_and_extract_archive(pc_archive, pc_src, source_manifest)
    deployments: dict[str, HostDeployment] = {
        "pc": HostDeployment("pc", pc_archive, pc_src, **pc_report)
    }
    remote_rows = (
        (
            "ecs",
            ecs_host,
            f"/root/GAPS/confirmation_runtime/{source_hash}",
            "/root/gaps_env/bin/python",
        ),
        (
            "pi",
            pi_host,
            f"/home/gaps/GAPS/confirmation_runtime/{source_hash}",
            "/home/gaps/GAPS/gaps_rpi_env/bin/python",
        ),
    )
    for host_id, host, root, python_bin in remote_rows:
        remote_archive = f"{root}/source.tar"
        remote_src = f"{root}/src"
        ssh(
            host,
            f"mkdir -p '{root}' && test ! -e '{remote_archive}' && test ! -e '{remote_src}'",
        )
        run(["scp", "-p", str(archive_path), f"{host}:{remote_archive}"], timeout=300)
        output = remote_python(
            host,
            python_bin,
            _remote_extract_source(remote_archive, remote_src, source_manifest),
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
) -> Provenance:
    validate_requested_run(group_id, seed)
    expected_run_id = confirmation_run_id(group_id, seed)
    if manifest.get("run_id") != expected_run_id or manifest.get("group_id") != group_id:
        raise ValueError("command manifest identity does not match allowlist")
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
    provenance = Provenance(
        confirmation_commit=(
            str(manifest["confirmation_commit"])
            if "confirmation_commit" in manifest
            else None
        ),
        source_archive_sha256=manifest.get("source_archive_sha256"),
        dataset_manifest_sha256=manifest.get("dataset_manifest_sha256"),
        algorithm_config_sha256=algorithm_hash,
    )
    return provenance


def validate_protocol_schedule(protocol_manifest: Mapping[str, Any]) -> None:
    rows = protocol_manifest.get("schedule")
    expected = [
        {
            "run_id": confirmation_run_id(group, seed),
            "group_id": group,
            "seed": seed,
        }
        for group, seed in CONFIRMATION_SCHEDULE
    ]
    if not isinstance(rows, list) or len(rows) != len(expected):
        raise ValueError("protocol schedule does not equal confirmation allowlist")
    actual = [
        {"run_id": row.get("run_id"), "group_id": row.get("group_id"), "seed": row.get("seed")}
        if isinstance(row, Mapping)
        else None
        for row in rows
    ]
    if actual != expected:
        raise ValueError("protocol schedule does not equal confirmation allowlist")


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
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite raw evidence: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(source, destination)
    elif source.is_file():
        shutil.copy2(source, destination)
    else:
        raise FileNotFoundError(source)


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
        hooks.cleanup_owned(tuple(owned))
    except BaseException as exc:
        errors.append(f"cleanup owned: {type(exc).__name__}: {exc}")
    try:
        hooks.cleanup_tunnels(tuple(tunnels))
    except BaseException as exc:
        errors.append(f"cleanup tunnels: {type(exc).__name__}: {exc}")
    return errors


def _failure_category(exc: BaseException) -> str:
    category = getattr(exc, "reason_category", None)
    return category if isinstance(category, str) else "process"


def run_confirmation_attempt(
    raw_root: Path,
    run_id: str,
    *,
    provenance: Provenance,
    hooks: LifecycleHooks,
) -> Attempt:
    provenance.require_complete()
    attempt = allocate_attempt(Path(raw_root), run_id)
    mark_attempt(
        attempt.path,
        "running",
        event_type="attempt_start",
        reason="controller allocated immutable attempt",
        reason_category="controller",
        provenance=provenance,
    )
    owned: list[object] = []
    samplers: list[object] = []
    tunnels: list[object] = []
    cleaned = False
    try:
        hooks.prepare(attempt)
        mark_attempt(
            attempt.path,
            "running",
            event_type="preflight_passed",
            reason="all host preflight gates passed",
            reason_category="controller",
            provenance=provenance,
        )
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
        outcome = hooks.validate_attempt(attempt)
        if not isinstance(outcome, ValidationOutcome):
            raise TypeError("validator must return ValidationOutcome")
        if not outcome.success:
            mark_attempt(
                attempt.path,
                "invalid",
                event_type="attempt_failure",
                reason=outcome.reason or "validator rejected attempt",
                reason_category="audit",
                provenance=provenance,
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
            reason="validator accepted attempt",
            reason_category="audit",
            provenance=provenance,
            audit_sha256=outcome.audit_sha256,
        )
        return attempt
    except BaseException as exc:
        cleanup_errors = [] if cleaned else _cleanup_lifecycle(
            attempt, hooks, samplers, owned, tunnels
        )
        detail = f"{type(exc).__name__}: {exc}"
        if cleanup_errors:
            detail += "; cleanup: " + "; ".join(cleanup_errors)
        state = "aborted" if isinstance(exc, (AttemptAborted, KeyboardInterrupt)) else "failed"
        try:
            mark_attempt(
                attempt.path,
                state,
                event_type="attempt_failure",
                reason=detail,
                reason_category=_failure_category(exc),
                provenance=provenance,
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
    parser.add_argument("--ecs-host", default="root@121.40.139.213")
    parser.add_argument("--pi-hosts", default="gaps@192.168.31.184")
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
    return parser


def _validate_cli_inputs(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = _load_json(args.protocol_manifest)
    source_manifest = _load_json(args.source_archive_manifest)
    dataset_manifest = _load_json(args.dataset_manifest)
    validate_protocol_schedule(protocol)
    _verify_archive(args.source_archive, source_manifest)
    if protocol.get("source_archive_sha256") != source_manifest.get("source_archive_sha256"):
        raise ValueError("protocol and source archive manifest SHA-256 mismatch")
    if protocol.get("dataset_manifest_sha256") != dataset_manifest.get(
        "dataset_manifest_sha256"
    ):
        raise ValueError("protocol and dataset manifest SHA-256 mismatch")
    for group, seed in CONFIRMATION_SCHEDULE:
        path = args.command_root / confirmation_run_id(group, seed) / "command_manifest.json"
        manifest = _load_json(path)
        provenance = validate_command_manifest(manifest, group, seed)
        if provenance.source_archive_sha256 != source_manifest.get("source_archive_sha256"):
            raise ValueError(f"source archive binding mismatch: {path}")
        if provenance.dataset_manifest_sha256 != dataset_manifest.get(
            "dataset_manifest_sha256"
        ):
            raise ValueError(f"dataset binding mismatch: {path}")
    return protocol, source_manifest


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_cli_inputs(args)
    if args.validate_inputs_only:
        print(
            json.dumps(
                {
                    "status": "validated",
                    "queue": [
                        {"group_id": group, "seed": seed}
                        for group, seed in CONFIRMATION_SCHEDULE
                    ],
                },
                sort_keys=True,
            )
        )
        return 0
    raise RuntimeError(
        "formal execution requires the Task 7 validator and explicit runtime hook binding"
    )


if __name__ == "__main__":
    raise SystemExit(main())
