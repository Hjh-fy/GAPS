"""Freeze read-only IoT-J experiment inputs in a hash-addressed manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np


DEFAULT_DATA_ROOT = Path("dataset/client_data_c12src_c345tgt_2080_timeaware_60_170_window_fullgrid")
DEFAULT_MATRIX_ROOT = Path("results/source_target_classification_matrix_20260708_clean")
DEFAULT_H23_STREAM = Path(
    "results/h2_3_plus_fusion_profile_20260630/"
    "r25_balanced_replay_gate/fusion_profile_predictions.csv"
)
DEFAULT_H8_C4_STREAM = Path(
    "results/f6_fixed_da_strong_r25_profile_replay_20260630/"
    "formal_c4_route_rescue_selector/formal_c4_route_rescue_predictions.csv"
)
DEFAULT_P4_POLICY = Path(
    "results/real_route_threshold_guard_deployment_candidate_20260707/threshold_guard_policy.json"
)
DEFAULT_OUTPUT = Path("results/iotj_experiment_freeze_20260711/input_manifest.json")

SOURCE_CLIENTS = (1, 2)
TARGET_CLIENTS = (3, 4, 5)
MATRIX_PRIORITY_RUN_DIRECTORIES = {
    "F4": "F4_C1234_to_C5_fixed_da_strong_r25",
    "F5": "F5_C1_to_C2345_fixed_da_strong_r25",
    "R1": "R1_C5_to_C1_fixed_da_strong_r25",
    "R2": "R2_C45_to_C1_fixed_da_strong_r25",
    "R3": "R3_C345_to_C1_fixed_da_strong_r25",
    "R4": "R4_C2345_to_C1_fixed_da_strong_r25",
}
MATRIX_REQUIRED_FILES = ("server_latest_adapted.pth", "run_config.json")
HASH_CHUNK_SIZE = 1024 * 1024
CALIBRATION_FIT_RATIO = 0.75
CALIBRATION_VALIDATION_RATIO = 0.25


def _resolved(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(HASH_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def _role_for(path: Path) -> str:
    text = str(path).replace("\\", "/").lower()
    if "source_target_classification_matrix" in text:
        if path.name == "server_latest_adapted.pth":
            return "matrix_checkpoint"
        if path.name == "run_config.json":
            return "matrix_run_config"
        return "matrix_artifact"
    if "h2_3" in text or "h2.3" in text:
        return "p4_h2_3_stream"
    if "h8" in text or "c4" in text:
        return "p4_h8_c4_stream"
    if "threshold_guard" in text:
        return "p4_threshold_guard_policy"
    if "client_data_c12src_c345tgt_2080_timeaware_60_170_window_fullgrid" in text:
        return "primary_dataset"
    return "input"


def _artifact_entry(path: Path, role: str | None = None) -> dict[str, Any]:
    resolved = _resolved(path)
    exists = resolved.is_file()
    return {
        "resolved_path": str(resolved),
        "role": role or _role_for(resolved),
        "exists": exists,
        "byte_size": resolved.stat().st_size if exists else 0,
        "sha256": _sha256(resolved) if exists else None,
        "status": "present" if exists else "missing",
    }


def _append_unique(entries: list[dict[str, Any]], entry: dict[str, Any]) -> None:
    if not any(item["resolved_path"] == entry["resolved_path"] for item in entries):
        entries.append(entry)


def _is_matrix_root(path: Path) -> bool:
    return "source_target_classification_matrix" in path.name.lower()


def _audit_matrix_root(root: Path, entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    resolved_root = _resolved(root)
    discovered = []
    if resolved_root.is_dir():
        discovered = [item.name for item in resolved_root.iterdir() if item.is_dir()]
    run_ids = sorted(set(discovered).union(MATRIX_PRIORITY_RUN_DIRECTORIES.values()))
    runs: list[dict[str, Any]] = []

    for run_id in run_ids:
        run_dir = resolved_root / run_id
        missing_required: list[str] = []
        for filename in MATRIX_REQUIRED_FILES:
            required_path = run_dir / filename
            entry = _artifact_entry(required_path)
            _append_unique(entries, entry)
            if not entry["exists"]:
                missing_required.append(filename)
        if run_dir.is_dir():
            for artifact in sorted(run_dir.rglob("*")):
                if artifact.is_file():
                    _append_unique(entries, _artifact_entry(artifact))
        runs.append(
            {
                "run_id": run_id,
                "resolved_path": str(_resolved(run_dir)),
                "status": "complete" if not missing_required else "missing",
                "missing_required": missing_required,
            }
        )
    return runs


def audit_inputs(paths: Sequence[Path]) -> dict[str, Any]:
    """Hash input files and inventory matrix runs without changing their contents."""
    artifacts: list[dict[str, Any]] = []
    matrix_runs: list[dict[str, Any]] = []

    for path in paths:
        candidate = Path(path)
        if _is_matrix_root(candidate):
            matrix_runs.extend(_audit_matrix_root(candidate, artifacts))
            continue
        if candidate.is_file() or not candidate.exists():
            _append_unique(artifacts, _artifact_entry(candidate))
            continue
        for artifact in sorted(candidate.rglob("*")):
            if artifact.is_file():
                _append_unique(artifacts, _artifact_entry(artifact))

    artifacts.sort(key=lambda item: item["resolved_path"])
    matrix_runs.sort(key=lambda item: item["run_id"])
    return {
        "artifacts": artifacts,
        "matrix_runs": matrix_runs,
        "status": "complete"
        if all(entry["status"] == "present" for entry in artifacts)
        and all(run["status"] == "complete" for run in matrix_runs)
        else "incomplete",
    }


def _concentration_counts(labels: np.ndarray, regression: np.ndarray | None) -> dict[str, dict[str, int]]:
    if regression is None:
        return {}
    values: dict[str, dict[str, int]] = {}
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
        values[str(class_id)] = dict(sorted(counts.items(), key=lambda item: float(item[0])))
    return values


def _split_role(client_id: int, split: str) -> str:
    return f"{'source' if client_id in SOURCE_CLIENTS else 'target'}_{split}"


def summarize_primary_dataset(data_root: Path) -> dict[str, Any]:
    """Record immutable per-client split counts from the primary saved arrays."""
    root = _resolved(data_root)
    protocol_path = root / "split_protocol_manifest.json"
    protocol: dict[str, Any] = {}
    if protocol_path.is_file():
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))

    clients: dict[str, dict[str, Any]] = {}
    required_splits = {**{client: ("train", "calibration", "test") for client in SOURCE_CLIENTS}, **{client: ("calibration", "test") for client in TARGET_CLIENTS}}
    missing_splits: list[str] = []
    for client_id in (*SOURCE_CLIENTS, *TARGET_CLIENTS):
        client_dir = root / f"client_{client_id}"
        client_summary: dict[str, Any] = {}
        for split in required_splits[client_id]:
            feature_path = client_dir / f"{split}_features.npy"
            label_path = client_dir / f"{split}_classification_labels.npy"
            regression_path = client_dir / f"{split}_regression_labels.npy"
            if not feature_path.is_file() or not label_path.is_file():
                missing_splits.append(f"C{client_id}/{split}")
                client_summary[split] = {"role": _split_role(client_id, split), "status": "missing"}
                continue
            features = np.load(feature_path, mmap_mode="r", allow_pickle=True)
            labels = np.asarray(np.load(label_path, allow_pickle=True)).astype(int).reshape(-1)
            regression = np.load(regression_path, allow_pickle=True) if regression_path.is_file() else None
            class_counts = {
                str(class_id): int(count)
                for class_id, count in zip(*np.unique(labels, return_counts=True))
            }
            client_summary[split] = {
                "role": _split_role(client_id, split),
                "status": "present",
                "sample_count": int(len(features)),
                "class_counts": class_counts,
                "concentration_counts": _concentration_counts(labels, regression),
            }
        clients[f"C{client_id}"] = client_summary

    return {
        "resolved_path": str(root),
        "role": "primary_dataset",
        "status": "complete" if not missing_splits else "incomplete",
        "split_seed": protocol.get("seed"),
        "source_clients": [f"C{client}" for client in SOURCE_CLIENTS],
        "target_clients": [f"C{client}" for client in TARGET_CLIENTS],
        "target_calibration_ratio": protocol.get("target_ratios", {}).get("calibration"),
        "target_test_ratio": protocol.get("target_ratios", {}).get("test"),
        "calibration_fit_ratio": CALIBRATION_FIT_RATIO,
        "calibration_validation_ratio": CALIBRATION_VALIDATION_RATIO,
        "clients": clients,
        "missing_splits": missing_splits,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--matrix-root", type=Path, default=DEFAULT_MATRIX_ROOT)
    parser.add_argument("--h23-stream", type=Path, default=DEFAULT_H23_STREAM)
    parser.add_argument("--h8-c4-stream", type=Path, default=DEFAULT_H8_C4_STREAM)
    parser.add_argument("--p4-policy", type=Path, default=DEFAULT_P4_POLICY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    manifest = audit_inputs(
        [args.data_root, args.h23_stream, args.h8_c4_stream, args.p4_policy, args.matrix_root]
    )
    primary_dataset = summarize_primary_dataset(args.data_root)
    manifest.update(
        {
            "schema_version": 1,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "primary_dataset": primary_dataset,
            "matrix_root": str(_resolved(args.matrix_root)),
        }
    )
    manifest["status"] = (
        "complete" if manifest["status"] == "complete" and primary_dataset["status"] == "complete" else "incomplete"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        f"Wrote {args.output}: {len(manifest['artifacts'])} artifacts, "
        f"{len(manifest['matrix_runs'])} matrix runs, status={manifest['status']}"
    )


if __name__ == "__main__":
    main()
