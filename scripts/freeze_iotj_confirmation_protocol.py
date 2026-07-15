"""Freeze immutable manifests for the IoTJ C1/C2-to-C5 confirmation queue."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import subprocess
import tarfile
from importlib import metadata
from pathlib import Path
from typing import Any, BinaryIO, Mapping, Sequence

import numpy as np

from scripts.generate_iotj_classification_ablation_commands import build_run_manifest


CONFIRMATION_SCHEDULE = (
    ("B2", 42), ("B5", 42),
    ("B5", 43), ("B2", 43),
    ("B2", 44), ("B5", 44),
    ("B5", 45), ("B2", 45),
    ("B2", 46), ("B5", 46),
)
CONFIRMATION_GROUPS = frozenset({"B2", "B5"})
CONFIRMATION_SEEDS = frozenset({42, 43, 44, 45, 46})
ACTIVE_SOURCE_CLIENTS = (1, 2)
ACTIVE_TARGET_CLIENTS = (5,)
SOURCE_SPLITS = ("train", "calibration", "test")
TARGET_SPLITS = ("calibration", "test")
SPLIT_COMPONENTS = ("features", "classification_labels", "regression_labels")
EXPECTED_C5_COUNTS = {"calibration": 320, "test": 1360}
ALGORITHM_CONFIG_FIELDS = (
    "protocol",
    "training",
    "causal_factors",
    "server_adaptation",
)
DEPENDENCY_VERSIONS = {
    "flwr": "1.23.0",
    "protobuf": "4.25.8",
    "psutil": "7.0.0",
}
DEFAULT_RESULTS_ROOT = "results/iotj_main_confirmation_observability_20260715"
OBSERVER_CLI_SCOPE = (
    "Observer CLI is controller-local and excluded from algorithm config."
)
_HASH_CHUNK_SIZE = 1024 * 1024
_FULL_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one file without loading it all at once."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: Mapping[str, Any]) -> str:
    """Hash a mapping encoded as stable, compact canonical JSON."""
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_confirmation_identity(group_id: str, seed: int) -> None:
    if group_id not in CONFIRMATION_GROUPS:
        raise ValueError(f"confirmation group must be B2 or B5; got {group_id!r}")
    if seed not in CONFIRMATION_SEEDS:
        raise ValueError(f"confirmation seed must be one of 42-46; got {seed!r}")


def confirmation_run_id(group_id: str, seed: int) -> str:
    """Return the allowlisted logical run identifier."""
    _validate_confirmation_identity(group_id, seed)
    return f"c12_to_c5__{group_id.lower()}__s{seed}"


def _expected_dataset_files(data_root: Path) -> list[Path]:
    paths = [data_root / "split_info.json", data_root / "norm_stats.npz"]
    for client_id in ACTIVE_SOURCE_CLIENTS:
        for split in SOURCE_SPLITS:
            paths.extend(
                data_root / f"client_{client_id}" / f"{split}_{component}.npy"
                for component in SPLIT_COMPONENTS
            )
    for client_id in ACTIVE_TARGET_CLIENTS:
        for split in TARGET_SPLITS:
            paths.extend(
                data_root / f"client_{client_id}" / f"{split}_{component}.npy"
                for component in SPLIT_COMPONENTS
            )
    return sorted(paths, key=lambda path: path.relative_to(data_root).as_posix())


def _load_split_info(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid split_info.json: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("split_info.json must contain an object")
    if payload.get("source_clients") != [1, 2]:
        raise ValueError(
            f"split_info source_clients must equal [1, 2]; got {payload.get('source_clients')!r}"
        )
    if payload.get("target_clients") != [5]:
        raise ValueError(
            f"split_info target_clients must equal [5]; got {payload.get('target_clients')!r}"
        )
    if payload.get("seed") != 42:
        raise ValueError(f"split_info seed must equal 42; got {payload.get('seed')!r}")
    target_split = payload.get("target_split")
    if not isinstance(target_split, dict):
        raise ValueError("split_info target_split must be an object")
    expected_target_split = {"train_used": False, "calibration": 0.2, "test": 0.8}
    if any(target_split.get(key) != value for key, value in expected_target_split.items()):
        raise ValueError(
            "split_info target_split must keep train_used=false, calibration=0.2, test=0.8"
        )
    return payload


def _array_rows(path: Path) -> int:
    try:
        array = np.load(path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(f"failed to load dataset array {path}: {exc}") from exc
    if array.ndim == 0:
        raise ValueError(f"dataset array has no sample dimension: {path}")
    return int(array.shape[0])


def _split_sample_count(data_root: Path, client_id: int, split: str) -> int:
    paths = {
        component: data_root / f"client_{client_id}" / f"{split}_{component}.npy"
        for component in SPLIT_COMPONENTS
    }
    counts = {component: _array_rows(path) for component, path in paths.items()}
    if len(set(counts.values())) != 1:
        raise ValueError(f"C{client_id}/{split} array lengths differ: {counts}")
    return next(iter(counts.values()))


def build_dataset_manifest(data_root: Path) -> dict[str, Any]:
    """Validate and hash exactly the active C1/C2 source and C5 target inputs."""
    data_root = Path(data_root)
    expected_paths = _expected_dataset_files(data_root)
    missing = [path for path in expected_paths if not path.is_file()]
    if missing:
        relative = missing[0].relative_to(data_root).as_posix()
        raise FileNotFoundError(f"missing active dataset file: {relative}")

    split_info = _load_split_info(data_root / "split_info.json")
    sample_counts: dict[str, dict[str, int]] = {}
    for client_id in ACTIVE_SOURCE_CLIENTS:
        sample_counts[f"C{client_id}"] = {
            split: _split_sample_count(data_root, client_id, split)
            for split in SOURCE_SPLITS
        }
    for client_id in ACTIVE_TARGET_CLIENTS:
        sample_counts[f"C{client_id}"] = {
            split: _split_sample_count(data_root, client_id, split)
            for split in TARGET_SPLITS
        }
    for split, expected in EXPECTED_C5_COUNTS.items():
        actual = sample_counts["C5"][split]
        if actual != expected:
            raise ValueError(f"C5/{split} sample_count must equal {expected}; got {actual}")

    files = [
        {
            "relative_path": path.relative_to(data_root).as_posix(),
            "byte_size": int(path.stat().st_size),
            "sha256": sha256_file(path),
        }
        for path in expected_paths
    ]
    payload: dict[str, Any] = {
        "schema_version": 1,
        "direction": "C1/C2 -> C5",
        "active_source_clients": ["C1", "C2"],
        "active_target_clients": ["C5"],
        "inactive_shared_dataset_clients": ["C3", "C4"],
        "split_seed": int(split_info["seed"]),
        "sample_counts": sample_counts,
        "files": files,
    }
    payload["dataset_manifest_sha256"] = canonical_sha256(payload)
    return payload


def _algorithm_manifest_from_run_manifest(
    run_manifest: Mapping[str, Any],
    group_id: str,
    seed: int,
) -> dict[str, Any]:
    try:
        algorithm_config = {
            key: copy.deepcopy(run_manifest[key])
            for key in ALGORITHM_CONFIG_FIELDS
        }
    except KeyError as exc:
        raise ValueError(f"run manifest missing algorithm field: {exc.args[0]}") from exc
    protocol = algorithm_config["protocol"]
    if protocol.get("source_clients") != [1, 2] or protocol.get("target_clients") != [5]:
        raise ValueError("algorithm direction must be C1/C2 source to C5 target")
    if protocol.get("training_seed") != seed:
        raise ValueError(
            f"algorithm training seed must equal {seed}; got {protocol.get('training_seed')!r}"
        )
    return {
        "group_id": group_id,
        "seed": seed,
        "algorithm_config": algorithm_config,
        "algorithm_config_sha256": canonical_sha256(algorithm_config),
    }


def build_algorithm_manifest(repo_root: Path, group_id: str, seed: int) -> dict[str, Any]:
    """Freeze only the numerical protocol/training/causal/server-DA configuration."""
    _validate_confirmation_identity(group_id, seed)
    run_manifest = build_run_manifest(
        group_id,
        seed,
        repo_root=Path(repo_root),
        results_root=DEFAULT_RESULTS_ROOT,
    )
    return _algorithm_manifest_from_run_manifest(run_manifest, group_id, seed)


def _sha256_stream(handle: BinaryIO) -> str:
    digest = hashlib.sha256()
    while chunk := handle.read(_HASH_CHUNK_SIZE):
        digest.update(chunk)
    return digest.hexdigest()


def _exact_dependency_versions() -> dict[str, str]:
    installed = {name: metadata.version(name) for name in DEPENDENCY_VERSIONS}
    if installed != DEPENDENCY_VERSIONS:
        raise RuntimeError(
            f"confirmation dependency versions must equal {DEPENDENCY_VERSIONS}; got {installed}"
        )
    return installed


def create_source_archive(
    repo_root: Path,
    confirmation_commit: str,
    output: Path,
) -> dict[str, Any]:
    """Create one immutable git archive from the exact clean tracked HEAD."""
    repo_root = Path(repo_root)
    output = Path(output)
    if not _FULL_COMMIT_RE.fullmatch(confirmation_commit):
        raise ValueError("confirmation_commit must be an exact 40-character lowercase hex SHA")
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        text=True,
    ).strip()
    if head != confirmation_commit:
        raise ValueError(f"repository HEAD {head} does not equal confirmation commit {confirmation_commit}")
    tracked_status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=repo_root,
        text=True,
    )
    if tracked_status.strip():
        raise ValueError("repository tracked files must be clean before source archive creation")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite source archive: {output}")
    versions = _exact_dependency_versions()

    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "git",
            "archive",
            "--format=tar",
            "--output",
            str(output),
            confirmation_commit,
        ],
        cwd=repo_root,
        check=True,
    )

    regular_members: list[dict[str, Any]] = []
    with tarfile.open(output, "r:") as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            extracted = archive.extractfile(member)
            if extracted is None:
                raise ValueError(f"could not read regular archive member: {member.name}")
            with extracted:
                digest = _sha256_stream(extracted)
            regular_members.append(
                {
                    "relative_path": member.name,
                    "byte_size": int(member.size),
                    "sha256": digest,
                }
            )
    regular_members.sort(key=lambda item: item["relative_path"])
    members_hash = canonical_sha256({"regular_members": regular_members})
    return {
        "schema_version": 1,
        "confirmation_commit": confirmation_commit,
        "source_archive_sha256": sha256_file(output),
        "regular_members_sha256": members_hash,
        "tracked_files_manifest_sha256": members_hash,
        "regular_members": regular_members,
        "dependency_versions": versions,
    }


def _claim_fields(group_id: str) -> dict[str, str]:
    if group_id == "B2":
        return {"b2_claim_status": "post_screen_exploratory"}
    return {"b5_claim_status": "predeclared_full_method"}


def _protocol_contract(
    confirmation_commit: str,
    dataset_manifest: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
    run_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "protocol_id": "iotj_main_direction_confirmation",
        "direction": "C1/C2 -> C5",
        "active_source_clients": ["C1", "C2"],
        "active_target_clients": ["C5"],
        "groups": ["B2", "B5"],
        "seeds": [42, 43, 44, 45, 46],
        "historical_seed42_included": False,
        "confirmation_commit": confirmation_commit,
        "source_archive_sha256": source_manifest["source_archive_sha256"],
        "regular_members_sha256": source_manifest["regular_members_sha256"],
        "dataset_manifest_sha256": dataset_manifest["dataset_manifest_sha256"],
        "schedule": [dict(row) for row in run_rows],
    }


def build_protocol_manifest(
    repo_root: Path,
    data_root: Path,
    confirmation_commit: str,
    archive_path: Path,
) -> dict[str, Any]:
    """Build the complete, still-unwritten confirmation manifest bundle."""
    repo_root = Path(repo_root)
    dataset_manifest = build_dataset_manifest(Path(data_root))

    frozen_runs: list[tuple[dict[str, Any], dict[str, Any], str]] = []
    schedule_rows: list[dict[str, Any]] = []
    for group_id, seed in CONFIRMATION_SCHEDULE:
        _validate_confirmation_identity(group_id, seed)
        run_manifest = build_run_manifest(
            group_id,
            seed,
            repo_root=repo_root,
            results_root=DEFAULT_RESULTS_ROOT,
        )
        algorithm = _algorithm_manifest_from_run_manifest(run_manifest, group_id, seed)
        run_id = confirmation_run_id(group_id, seed)
        frozen_runs.append((run_manifest, algorithm, run_id))
        schedule_rows.append(
            {
                "run_id": run_id,
                "group_id": group_id,
                "seed": seed,
                "algorithm_config_sha256": algorithm["algorithm_config_sha256"],
                **_claim_fields(group_id),
            }
        )

    source_manifest = create_source_archive(repo_root, confirmation_commit, Path(archive_path))
    protocol = _protocol_contract(
        confirmation_commit,
        dataset_manifest,
        source_manifest,
        schedule_rows,
    )
    protocol["protocol_manifest_sha256"] = canonical_sha256(protocol)

    command_manifests: list[dict[str, Any]] = []
    for (group_id, seed), (run_manifest, algorithm, run_id) in zip(
        CONFIRMATION_SCHEDULE,
        frozen_runs,
    ):
        command_manifest = copy.deepcopy(run_manifest)
        command_manifest.update(
            {
                "run_id": run_id,
                "historical_seed42_included": False,
                "protocol_manifest_sha256": protocol["protocol_manifest_sha256"],
                "dataset_manifest_sha256": dataset_manifest["dataset_manifest_sha256"],
                "source_archive_sha256": source_manifest["source_archive_sha256"],
                "regular_members_sha256": source_manifest["regular_members_sha256"],
                "algorithm_config_sha256": algorithm["algorithm_config_sha256"],
                "transport_status": "not_collected",
                "observer_cli_scope": OBSERVER_CLI_SCOPE,
                **_claim_fields(group_id),
            }
        )
        command_manifests.append(command_manifest)

    protocol["source_archive_manifest"] = source_manifest
    protocol["dataset_manifest"] = dataset_manifest
    protocol["command_manifests"] = command_manifests
    return protocol


def _write_json(payload: Mapping[str, Any], output: Path) -> None:
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite immutable manifest: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_dataset_manifest(data_root: Path, output: Path) -> dict[str, Any]:
    """Validate fully, then write one standalone dataset manifest."""
    manifest = build_dataset_manifest(Path(data_root))
    _write_json(manifest, Path(output))
    return manifest


def _output_targets(
    summary_root: Path,
    command_root: Path,
    command_manifests: Sequence[Mapping[str, Any]],
) -> list[Path]:
    targets = [
        summary_root / "confirmation_protocol_manifest.json",
        summary_root / "source_archive_manifest.json",
        summary_root / "dataset_manifest.json",
    ]
    targets.extend(
        command_root / str(manifest["run_id"]) / "command_manifest.json"
        for manifest in command_manifests
    )
    return targets


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirmation-commit", required=True)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--archive-output", required=True, type=Path)
    parser.add_argument("--command-root", required=True, type=Path)
    parser.add_argument("--summary-root", required=True, type=Path)
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    bundle = build_protocol_manifest(
        repo_root,
        args.data_root,
        args.confirmation_commit,
        args.archive_output,
    )
    source_manifest = bundle["source_archive_manifest"]
    dataset_manifest = bundle["dataset_manifest"]
    command_manifests = bundle["command_manifests"]
    protocol_manifest = {
        key: value
        for key, value in bundle.items()
        if key not in {"source_archive_manifest", "dataset_manifest", "command_manifests"}
    }

    targets = _output_targets(args.summary_root, args.command_root, command_manifests)
    existing = [path for path in targets if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite immutable manifest: {existing[0]}")
    _write_json(protocol_manifest, args.summary_root / "confirmation_protocol_manifest.json")
    _write_json(source_manifest, args.summary_root / "source_archive_manifest.json")
    _write_json(dataset_manifest, args.summary_root / "dataset_manifest.json")
    for command_manifest in command_manifests:
        _write_json(
            command_manifest,
            args.command_root / command_manifest["run_id"] / "command_manifest.json",
        )

    counts = dataset_manifest["sample_counts"]["C5"]
    print(
        f"Wrote {len(command_manifests)} scheduled runs; "
        f"C5 calibration/test={counts['calibration']}/{counts['test']}; "
        f"source_archive_sha256={source_manifest['source_archive_sha256']}; "
        f"dataset_manifest_sha256={dataset_manifest['dataset_manifest_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
