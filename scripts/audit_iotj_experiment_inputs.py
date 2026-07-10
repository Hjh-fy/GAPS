"""Freeze and validate the corrected C1/C2-to-C5 IoT-J inputs."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


DEFAULT_DATA_ROOT = Path("dataset/client_data_c1234src_c5tgt_2080_timeaware_60_170_window_fullgrid")
DEFAULT_RUN_DIR = Path(
    "results/source_target_classification_matrix_20260630/"
    "F2_C12_to_C5_fixed_da_strong_r25"
)
DEFAULT_OUTPUT = Path("results/iotj_experiment_freeze_20260711/input_manifest.json")

ACTIVE_SOURCE_CLIENTS = (1, 2)
ACTIVE_TARGET_CLIENTS = (5,)
ACTIVE_CLIENTS = (*ACTIVE_SOURCE_CLIENTS, *ACTIVE_TARGET_CLIENTS)
SOURCE_SPLITS = ("train", "calibration", "test")
TARGET_SPLITS = ("calibration", "test")
SPLIT_COMPONENTS = ("features", "classification_labels", "regression_labels")
EXPECTED_C5_COUNTS = {"calibration": 320, "test": 1360}
HASH_CHUNK_SIZE = 1024 * 1024

F2_RUN_EXPECTED: dict[str, Any] = {
    "run_name": "F2_C12_to_C5_fixed_da_strong_r25",
    "rounds": 25,
    "strategy": "gaps",
    "profile": "strong_cls",
    "server_val_clients": [1, 2],
    "server_calib_clients": [5],
    "use_domain_adapt": True,
    "domain_adapt_steps": 100,
    "da_lambda_target_ce": 0.0,
    "use_adapted_as_global": True,
    "required_files": ["run_config.json", "history.json", "server_latest_adapted.pth"],
}


def _resolved(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(HASH_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def _generic_role(path: Path) -> str:
    text = str(path).replace("\\", "/").lower()
    if "h2_3" in text or "h2.3" in text:
        return "p4_h2_3_stream"
    if "h8" in text:
        return "p4_h8_stream"
    if "threshold_guard" in text:
        return "p4_threshold_guard_policy"
    return "input"


def _artifact_entry(path: Path, role: str | None = None) -> dict[str, Any]:
    resolved = _resolved(path)
    exists = resolved.is_file()
    return {
        "resolved_path": str(resolved),
        "role": role or _generic_role(resolved),
        "exists": exists,
        "byte_size": int(resolved.stat().st_size) if exists else 0,
        "sha256": _sha256(resolved) if exists else None,
        "status": "present" if exists else "missing",
    }


def _deduplicate_artifacts(entries: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    by_path: dict[str, dict[str, Any]] = {}
    for entry in entries:
        by_path.setdefault(entry["resolved_path"], entry)
    return [by_path[path] for path in sorted(by_path)]


def audit_inputs(paths: Sequence[Path]) -> dict[str, Any]:
    """Hash generic files/directories without guessing experiment semantics."""
    entries: list[dict[str, Any]] = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_dir():
            entries.extend(_artifact_entry(item) for item in sorted(path.rglob("*")) if item.is_file())
        else:
            entries.append(_artifact_entry(path))
    artifacts = _deduplicate_artifacts(entries)
    return {
        "artifacts": artifacts,
        "status": "complete" if all(item["exists"] for item in artifacts) else "incomplete",
    }


def _run_artifact_role(path: Path) -> str:
    return {
        "run_config.json": "matrix_run_config",
        "history.json": "matrix_history",
        "server_latest_adapted.pth": "matrix_checkpoint",
    }.get(path.name, "matrix_artifact")


def _client_ids(value: Any) -> list[int]:
    if not isinstance(value, str):
        return []
    ids: list[int] = []
    for item in value.split(","):
        match = re.search(r"(?:^|[\\/])client_(\d+)(?:$|[\\/])", item.strip())
        if match:
            ids.append(int(match.group(1)))
    return sorted(set(ids))


def _expected_text(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def _load_json_object(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"invalid JSON {path.name}: {exc}"
    if not isinstance(payload, dict):
        return None, f"invalid JSON {path.name}: expected an object"
    return payload, None


def _validate_run_config(config: dict[str, Any], expected: Mapping[str, Any]) -> tuple[list[str], dict[str, Any]]:
    args = config.get("args")
    if not isinstance(args, dict):
        return ["run_config.json args must be an object"], {}

    errors: list[str] = []
    validated: dict[str, Any] = {}
    scalar_keys = (
        "run_name",
        "rounds",
        "strategy",
        "profile",
        "use_domain_adapt",
        "domain_adapt_steps",
        "da_lambda_target_ce",
        "use_adapted_as_global",
    )
    for key in scalar_keys:
        actual = args.get(key)
        wanted = expected[key]
        validated[key] = actual
        if actual != wanted:
            errors.append(f"{key} must equal {_expected_text(wanted)}; got {actual!r}")

    for config_key, expected_key in (
        ("server_val_data", "server_val_clients"),
        ("server_calib_data", "server_calib_clients"),
    ):
        clients = _client_ids(args.get(config_key))
        wanted_clients = list(expected[expected_key])
        validated[f"{config_key}_clients"] = clients
        if clients != wanted_clients:
            errors.append(
                f"{config_key} clients must equal {wanted_clients}; got {clients}"
            )
    return errors, validated


def audit_matrix_run(run_dir: Path, expected: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one explicitly typed required run, independent of its path name."""
    root = _resolved(run_dir)
    required_files = [str(name) for name in expected.get("required_files", [])]
    entries = [
        _artifact_entry(root / filename, _run_artifact_role(root / filename))
        for filename in required_files
    ]
    if root.is_dir():
        entries.extend(
            _artifact_entry(path, _run_artifact_role(path))
            for path in sorted(root.rglob("*"))
            if path.is_file()
        )
    artifacts = _deduplicate_artifacts(entries)
    missing = [filename for filename in required_files if not (root / filename).is_file()]
    errors: list[str] = []
    warnings: list[str] = []
    validated_config: dict[str, Any] = {}

    config_path = root / "run_config.json"
    if config_path.is_file():
        config, error = _load_json_object(config_path)
        if error:
            errors.append(error)
        elif config is not None:
            config_errors, validated_config = _validate_run_config(config, expected)
            errors.extend(config_errors)

    existing = [entry for entry in artifacts if entry["exists"]]
    status = "complete" if not missing and not errors else "incomplete"
    return {
        "run_id": str(expected["run_name"]),
        "resolved_path": str(root),
        "role": "required_f2_run",
        "status": status,
        "missing_required": missing,
        "validation_errors": errors,
        "validation_warnings": warnings,
        "validated_config": validated_config,
        "file_count": len(existing),
        "byte_size": int(sum(entry["byte_size"] for entry in existing)),
        "artifacts": artifacts,
    }


def _class_counts(labels: np.ndarray) -> dict[str, int]:
    values, counts = np.unique(labels.astype(int).reshape(-1), return_counts=True)
    return {str(int(value)): int(count) for value, count in zip(values.tolist(), counts.tolist())}


def _concentration_counts(labels: np.ndarray, regression: np.ndarray) -> dict[str, dict[str, int]]:
    labels = labels.astype(int).reshape(-1)
    result: dict[str, dict[str, int]] = {}
    for class_id in sorted({int(value) for value in labels.tolist()}):
        mask = labels == class_id
        if regression.ndim == 2 and regression.shape[1] > class_id:
            selected = regression[mask, class_id]
        else:
            selected = regression[mask].reshape(-1)
        counts: dict[str, int] = {}
        for value in np.asarray(selected, dtype=float).reshape(-1):
            if np.isfinite(value):
                key = str(float(round(float(value), 6)))
                counts[key] = counts.get(key, 0) + 1
        result[str(class_id)] = dict(sorted(counts.items(), key=lambda item: float(item[0])))
    return result


def _split_role(client_id: int, split: str) -> str:
    domain = "source" if client_id in ACTIVE_SOURCE_CLIENTS else "target"
    return f"{domain}_{split}"


def _required_split_paths(data_root: Path) -> list[Path]:
    paths = [data_root / "split_info.json"]
    for client_id in ACTIVE_CLIENTS:
        splits = SOURCE_SPLITS if client_id in ACTIVE_SOURCE_CLIENTS else TARGET_SPLITS
        for split in splits:
            paths.extend(
                data_root / f"client_{client_id}" / f"{split}_{component}.npy"
                for component in SPLIT_COMPONENTS
            )
    return paths


def _validate_split_info(info: dict[str, Any]) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    if info.get("target_clients") != [5]:
        errors.append(f"target_clients must equal [5]; got {info.get('target_clients')!r}")

    source_clients = info.get("source_clients")
    if not isinstance(source_clients, list) or not set(ACTIVE_SOURCE_CLIENTS).issubset(source_clients):
        errors.append("source_clients must contain [1, 2]")
    if info.get("seed") != 42:
        errors.append(f"seed must equal 42; got {info.get('seed')!r}")

    target_split = info.get("target_split")
    if not isinstance(target_split, dict):
        errors.append("target_split must be an object")
    else:
        if target_split.get("train_used") is not False:
            errors.append("target_split.train_used must be false")
        if target_split.get("calibration") != 0.2:
            errors.append("target_split.calibration must equal 0.2")
        if target_split.get("test") != 0.8:
            errors.append("target_split.test must equal 0.8")

    stratify_by = info.get("stratify_by")
    required_stratification = {"client", "class", "concentration"}
    if not isinstance(stratify_by, list) or not required_stratification.issubset(stratify_by):
        errors.append("stratify_by must contain client, class, and concentration")

    protocol_text = str(info.get("protocol", ""))
    if protocol_text not in {"c12_source_c5_target_calib20_test80", "c12_to_c5"}:
        warnings.append(
            "split_info.protocol is stale; structured target_clients and target_split fields are authoritative"
        )
    return errors, warnings


def summarize_primary_dataset(data_root: Path) -> dict[str, Any]:
    """Validate metadata and summarize only active C1/C2/C5 splits."""
    root = _resolved(data_root)
    errors: list[str] = []
    warnings: list[str] = []
    metadata: dict[str, Any] = {}
    metadata_path = root / "split_info.json"
    if not metadata_path.is_file():
        errors.append("missing metadata: split_info.json")
    else:
        loaded, load_error = _load_json_object(metadata_path)
        if load_error:
            errors.append(load_error)
        elif loaded is not None:
            metadata = loaded
            metadata_errors, metadata_warnings = _validate_split_info(metadata)
            errors.extend(metadata_errors)
            warnings.extend(metadata_warnings)

    clients: dict[str, dict[str, Any]] = {}
    for client_id in ACTIVE_CLIENTS:
        client_dir = root / f"client_{client_id}"
        splits = SOURCE_SPLITS if client_id in ACTIVE_SOURCE_CLIENTS else TARGET_SPLITS
        split_summaries: dict[str, Any] = {}
        for split in splits:
            paths = {
                component: client_dir / f"{split}_{component}.npy"
                for component in SPLIT_COMPONENTS
            }
            missing_components = [component for component, path in paths.items() if not path.is_file()]
            for component in missing_components:
                errors.append(f"C{client_id}/{split} missing {split}_{component}.npy")
            if missing_components:
                split_summaries[split] = {
                    "role": _split_role(client_id, split),
                    "status": "incomplete",
                    "sample_count": 0,
                    "class_counts": {},
                    "concentration_counts": {},
                }
                continue

            try:
                features = np.load(paths["features"], mmap_mode="r", allow_pickle=True)
                labels = np.asarray(np.load(paths["classification_labels"], allow_pickle=True))
                regression = np.asarray(np.load(paths["regression_labels"], allow_pickle=True))
            except (OSError, ValueError) as exc:
                errors.append(f"C{client_id}/{split} failed to load arrays: {exc}")
                split_summaries[split] = {
                    "role": _split_role(client_id, split),
                    "status": "incomplete",
                    "sample_count": 0,
                    "class_counts": {},
                    "concentration_counts": {},
                }
                continue

            sample_count = int(len(features))
            if len(labels) != sample_count or len(regression) != sample_count:
                errors.append(
                    f"C{client_id}/{split} array lengths differ: "
                    f"features={sample_count}, labels={len(labels)}, regression={len(regression)}"
                )
            if client_id == 5 and sample_count != EXPECTED_C5_COUNTS[split]:
                errors.append(
                    f"C5/{split} sample_count must equal {EXPECTED_C5_COUNTS[split]}; got {sample_count}"
                )
            split_summaries[split] = {
                "role": _split_role(client_id, split),
                "status": "present",
                "sample_count": sample_count,
                "class_counts": _class_counts(labels),
                "concentration_counts": _concentration_counts(labels, regression),
            }
        clients[f"C{client_id}"] = split_summaries

    metadata_source_clients = metadata.get("source_clients", [])
    return {
        "resolved_path": str(root),
        "role": "primary_c12_to_c5_dataset",
        "status": "complete" if not errors else "incomplete",
        "metadata_path": str(metadata_path),
        "metadata_source_clients": [f"C{value}" for value in metadata_source_clients],
        "active_source_clients": ["C1", "C2"],
        "active_target_clients": ["C5"],
        "split_seed": metadata.get("seed"),
        "target_calibration_ratio": (
            metadata.get("target_split", {}).get("calibration")
            if isinstance(metadata.get("target_split"), dict)
            else None
        ),
        "target_test_ratio": (
            metadata.get("target_split", {}).get("test")
            if isinstance(metadata.get("target_split"), dict)
            else None
        ),
        "calibration_fit_ratio": 0.75,
        "calibration_validation_ratio": 0.25,
        "clients": clients,
        "validation_errors": errors,
        "validation_warnings": warnings,
    }


def _primary_dataset_artifacts(data_root: Path) -> list[dict[str, Any]]:
    root = _resolved(data_root)
    paths: list[Path] = []
    if root.is_dir():
        paths.extend(path for path in root.iterdir() if path.is_file())
        for client_id in ACTIVE_CLIENTS:
            client_dir = root / f"client_{client_id}"
            if client_dir.is_dir():
                paths.extend(path for path in client_dir.rglob("*") if path.is_file())
    paths.extend(_required_split_paths(root))
    entries = [
        _artifact_entry(
            path,
            "dataset_metadata" if path.name == "split_info.json" else "primary_dataset",
        )
        for path in paths
    ]
    return _deduplicate_artifacts(entries)


def _extra_run_inventory(run_dir: Path) -> list[dict[str, Any]]:
    required = _resolved(run_dir)
    parent = required.parent
    if not parent.is_dir():
        return []
    rows: list[dict[str, Any]] = []
    for candidate in sorted(parent.iterdir()):
        if not candidate.is_dir() or _resolved(candidate) == required:
            continue
        file_count = sum(1 for path in candidate.rglob("*") if path.is_file())
        rows.append(
            {
                "run_id": candidate.name,
                "resolved_path": str(_resolved(candidate)),
                "role": "inventory_only",
                "file_count": file_count,
                "status": "inventory_only",
            }
        )
    return rows


def build_manifest(data_root: Path, run_dir: Path) -> dict[str, Any]:
    """Build the deterministic manifest payload without provenance time."""
    dataset = summarize_primary_dataset(data_root)
    required_run = audit_matrix_run(run_dir, F2_RUN_EXPECTED)
    artifacts = _deduplicate_artifacts(
        [*_primary_dataset_artifacts(data_root), *required_run["artifacts"]]
    )
    errors = [f"dataset: {message}" for message in dataset["validation_errors"]]
    errors.extend(f"F2 run: {message}" for message in required_run["validation_errors"])
    warnings = [f"dataset: {message}" for message in dataset["validation_warnings"]]
    warnings.extend(f"F2 run: {message}" for message in required_run["validation_warnings"])
    required_run_row = {key: value for key, value in required_run.items() if key != "artifacts"}
    status = (
        "complete"
        if dataset["status"] == "complete"
        and required_run["status"] == "complete"
        and all(entry["status"] == "present" for entry in artifacts)
        else "incomplete"
    )
    return {
        "schema_version": 2,
        "protocol": {
            "name": "C12-to-C5",
            "active_source_clients": ["C1", "C2"],
            "active_target_clients": ["C5"],
            "active_clients": ["C1", "C2", "C5"],
            "inactive_shared_dataset_clients": ["C3", "C4"],
            "data_root": str(_resolved(data_root)),
            "required_run_dir": str(_resolved(run_dir)),
        },
        "primary_dataset": dataset,
        "required_runs": [required_run_row],
        "extra_run_inventory": _extra_run_inventory(run_dir),
        "artifacts": artifacts,
        "validation_errors": errors,
        "validation_warnings": warnings,
        "status": status,
    }


def with_provenance(payload: Mapping[str, Any], generated_at_utc: str | None = None) -> dict[str, Any]:
    result = copy.deepcopy(dict(payload))
    result["provenance"] = {
        "generated_at_utc": generated_at_utc or datetime.now(timezone.utc).isoformat()
    }
    return result


def validate_output_path(output: Path, protected_inputs: Sequence[Path]) -> None:
    resolved_output = _resolved(output)
    for protected in protected_inputs:
        resolved_protected = _resolved(Path(protected))
        if resolved_output == resolved_protected or resolved_protected in resolved_output.parents:
            raise ValueError(
                f"output path overlaps protected input: {resolved_output} vs {resolved_protected}"
            )


def write_manifest(
    payload: Mapping[str, Any],
    output: Path,
    protected_inputs: Sequence[Path],
) -> None:
    validate_output_path(output, protected_inputs)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    protected_inputs = [args.data_root, args.run_dir]
    validate_output_path(args.output, protected_inputs)
    stable_manifest = build_manifest(args.data_root, args.run_dir)
    manifest = with_provenance(stable_manifest)
    write_manifest(manifest, args.output, protected_inputs)
    print(
        f"Wrote {args.output}: {len(manifest['artifacts'])} artifacts, "
        f"{len(manifest['required_runs'])} required run, status={manifest['status']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
