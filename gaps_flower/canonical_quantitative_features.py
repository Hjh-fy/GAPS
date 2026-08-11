"""Canonical-v1 5-Hz quantitative feature extraction and cache provenance.

The formulas are intentionally delegated to the frozen ``rich_feature_dict``
implementation.  This module adds the protocol boundary: only 50x8 canonical
windows are accepted, all 83D/H1 rows are built together, and a cache is usable
only when its content-addressed provenance is complete.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from run_regression_head_ablation import rich_feature_dict


STUDY_ID = "CAN-V1-CRRQ-20260811"
CANONICAL_WINDOW_SHAPE = (50, 8)
CANONICAL_SAMPLING_RATE_HZ = 5
DYNAMIC_DESCRIPTOR_INTERPRETATION = "fixed-5-Hz discrete per-sample descriptors"

METADATA_PHASE_FEATURE_NAMES = frozenset(
    {
        "window_start_s",
        "window_end_s",
        "window_center_s",
        "window_len_s",
        "t_onset",
        "t_min",
        "center_minus_onset",
        "center_minus_t_min",
        "interpolated_ratio",
        "max_gap_inside_window",
        "response_phase_main_response",
        "response_phase_recovery",
        "response_phase_unknown",
        "phase_label_early",
        "phase_label_middle",
        "phase_label_late",
        "phase_label_unknown",
        "phase_id_0",
        "phase_id_1",
        "phase_id_2",
        "phase_id_unknown",
    }
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_strings(values: Sequence[str]) -> str:
    payload = json.dumps(list(values), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _schema() -> tuple[tuple[str, ...], tuple[str, ...]]:
    template = rich_feature_dict(np.zeros(CANONICAL_WINDOW_SHAPE, dtype=np.float64))
    h1_names = tuple(sorted(template))
    sensor_names = tuple(name for name in h1_names if name not in METADATA_PHASE_FEATURE_NAMES)
    if len(sensor_names) != 83 or len(h1_names) != 104:
        raise RuntimeError(
            f"canonical quantitative schema mismatch: sensor={len(sensor_names)}, H1={len(h1_names)}"
        )
    if set(h1_names) - set(sensor_names) != set(METADATA_PHASE_FEATURE_NAMES):
        raise RuntimeError("canonical H1 metadata/phase schema mismatch")
    return sensor_names, h1_names


SENSOR_FEATURE_NAMES, H1_FEATURE_NAMES = _schema()


@dataclass(frozen=True)
class CanonicalFeatureRecord:
    sensor83: np.ndarray
    h1: np.ndarray
    sensor_feature_names: tuple[str, ...]
    h1_feature_names: tuple[str, ...]
    identity: dict[str, Any]
    provenance: dict[str, Any]


def _identity(
    metadata: Mapping[str, Any], *, client: str, split: str, sample_index: int
) -> dict[str, Any]:
    start = float(metadata.get("window_start_s", 0.0) or 0.0)
    end = float(metadata.get("window_end_s", 0.0) or 0.0)
    filename = str(metadata.get("filename") or metadata.get("raw_filename") or "")
    physical_identity = str(metadata.get("physical_identity") or "")
    if not physical_identity:
        physical_identity = f"{client}|{split}|{filename}|{start:.9g}|{end:.9g}|{sample_index}"
    return {
        "client": str(client).upper(),
        "split": str(split),
        "sample_index": int(sample_index),
        "physical_identity": physical_identity,
        "filename": filename,
        "window_start_s": start,
        "window_end_s": end,
    }


def extract_canonical_features(
    window: np.ndarray,
    *,
    phase: int,
    metadata: Mapping[str, Any],
    client: str,
    split: str,
    sample_index: int,
) -> CanonicalFeatureRecord:
    values = np.asarray(window)
    if values.shape != CANONICAL_WINDOW_SHAPE:
        raise ValueError(
            f"canonical quantitative input must be 50x8, observed {tuple(values.shape)}"
        )
    if not np.issubdtype(values.dtype, np.number) or not np.isfinite(values).all():
        raise ValueError("canonical quantitative input must be finite numeric values")
    features = rich_feature_dict(values.astype(np.float64, copy=False), int(phase), dict(metadata))
    if tuple(sorted(features)) != H1_FEATURE_NAMES:
        raise RuntimeError("canonical extractor returned a changed H1 schema")
    sensor = np.asarray([features[name] for name in SENSOR_FEATURE_NAMES], dtype=np.float64)
    h1 = np.asarray([features[name] for name in H1_FEATURE_NAMES], dtype=np.float64)
    if not np.isfinite(sensor).all() or not np.isfinite(h1).all():
        raise RuntimeError("canonical extractor returned non-finite values")
    return CanonicalFeatureRecord(
        sensor83=sensor,
        h1=h1,
        sensor_feature_names=SENSOR_FEATURE_NAMES,
        h1_feature_names=H1_FEATURE_NAMES,
        identity=_identity(metadata, client=client, split=split, sample_index=sample_index),
        provenance={
            "study_id": STUDY_ID,
            "window_shape": list(CANONICAL_WINDOW_SHAPE),
            "sampling_rate_hz": CANONICAL_SAMPLING_RATE_HZ,
            "dynamic_descriptor_interpretation": DYNAMIC_DESCRIPTOR_INTERPRETATION,
            "sampling_rate_invariant_claim": False,
            "legacy_10hz_5hz_numeric_equivalence_claim": False,
        },
    )


def validate_cache_manifest(
    manifest: Mapping[str, Any], *, expected_dataset_sha256: str
) -> None:
    required_hashes = (
        "dataset_aggregate_sha256",
        "source_array_sha256",
        "metadata_sha256",
        "extractor_file_sha256",
        "ordered_h1_feature_names_sha256",
        "ordered_sensor_feature_names_sha256",
    )
    valid = (
        manifest.get("study_id") == STUDY_ID
        and manifest.get("sampling_rate_hz") == CANONICAL_SAMPLING_RATE_HZ
        and manifest.get("window_shape") == list(CANONICAL_WINDOW_SHAPE)
        and manifest.get("dataset_aggregate_sha256") == expected_dataset_sha256
        and manifest.get("h1_dimensions") == 104
        and manifest.get("sensor_dimensions") == 83
        and manifest.get("created_from_canonical_arrays") is True
        and manifest.get("legacy_cache_reused") is False
        and all(
            isinstance(manifest.get(key), str) and len(str(manifest[key])) == 64
            for key in required_hashes
        )
    )
    if not valid:
        raise RuntimeError("canonical cache provenance mismatch")


def build_feature_cache(
    dataset_root: str | Path,
    cache_root: str | Path,
    *,
    client: str,
    split: str,
    dataset_aggregate_sha256: str,
    extractor_path: str | Path,
) -> dict[str, Any]:
    """Recompute one content-bound 83D/H1 cache; never load a prior cache."""
    dataset_root = Path(dataset_root)
    cache_dir = Path(cache_root) / str(client).upper() / str(split)
    if cache_dir.exists() and any(cache_dir.iterdir()):
        raise FileExistsError(f"refusing to reuse or overwrite feature cache: {cache_dir}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    client_number = int(str(client).upper().replace("C", ""))
    client_dir = dataset_root / f"client_{client_number}"
    feature_path = client_dir / f"{split}_features.npy"
    phase_path = client_dir / f"{split}_phase_labels.npy"
    metadata_path = client_dir / f"{split}_experiment_info.json"
    windows = np.load(feature_path, allow_pickle=False)
    phases = np.load(phase_path, allow_pickle=False).reshape(-1)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if windows.ndim != 3 or tuple(windows.shape[1:]) != CANONICAL_WINDOW_SHAPE:
        raise RuntimeError(f"canonical cache source must contain Nx50x8: {feature_path}")
    if len(windows) != len(phases) or len(windows) != len(metadata):
        raise RuntimeError(f"canonical cache row count mismatch: {client}/{split}")
    sensor_rows: list[np.ndarray] = []
    h1_rows: list[np.ndarray] = []
    identities: list[dict[str, Any]] = []
    for index, (window, phase, meta) in enumerate(zip(windows, phases, metadata)):
        record = extract_canonical_features(
            window,
            phase=int(phase),
            metadata=meta,
            client=str(client).upper(),
            split=split,
            sample_index=index,
        )
        sensor_rows.append(record.sensor83)
        h1_rows.append(record.h1)
        identities.append(record.identity)
    sensor_matrix = np.stack(sensor_rows).astype(np.float64, copy=False)
    h1_matrix = np.stack(h1_rows).astype(np.float64, copy=False)
    cache_path = cache_dir / "canonical_quantitative_features.npz"
    np.savez_compressed(cache_path, sensor83=sensor_matrix, h1=h1_matrix)
    identity_path = cache_dir / "row_identities.json"
    identity_path.write_text(
        json.dumps(identities, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "iotj.canonical_v1.quantitative_feature_cache.v1",
        "study_id": STUDY_ID,
        "client": str(client).upper(),
        "split": str(split),
        "row_count": int(len(windows)),
        "sampling_rate_hz": CANONICAL_SAMPLING_RATE_HZ,
        "window_shape": list(CANONICAL_WINDOW_SHAPE),
        "dataset_aggregate_sha256": dataset_aggregate_sha256,
        "source_array_sha256": sha256_file(feature_path),
        "phase_array_sha256": sha256_file(phase_path),
        "metadata_sha256": sha256_file(metadata_path),
        "extractor_file_sha256": sha256_file(extractor_path),
        "ordered_h1_feature_names_sha256": sha256_strings(H1_FEATURE_NAMES),
        "ordered_sensor_feature_names_sha256": sha256_strings(SENSOR_FEATURE_NAMES),
        "h1_dimensions": len(H1_FEATURE_NAMES),
        "sensor_dimensions": len(SENSOR_FEATURE_NAMES),
        "cache_sha256": sha256_file(cache_path),
        "row_identities_sha256": sha256_file(identity_path),
        "created_from_canonical_arrays": True,
        "legacy_cache_reused": False,
        "resized_or_interpolated_after_preprocessing": False,
        "dynamic_descriptor_interpretation": DYNAMIC_DESCRIPTOR_INTERPRETATION,
        "sampling_rate_invariant_claim": False,
        "legacy_10hz_5hz_numeric_equivalence_claim": False,
    }
    validate_cache_manifest(manifest, expected_dataset_sha256=dataset_aggregate_sha256)
    manifest_path = cache_dir / "cache_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def load_feature_cache(
    cache_dir: str | Path, *, expected_dataset_sha256: str
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    cache_dir = Path(cache_dir)
    manifest = json.loads((cache_dir / "cache_manifest.json").read_text(encoding="utf-8"))
    validate_cache_manifest(manifest, expected_dataset_sha256=expected_dataset_sha256)
    cache_path = cache_dir / "canonical_quantitative_features.npz"
    identity_path = cache_dir / "row_identities.json"
    if sha256_file(cache_path) != manifest["cache_sha256"]:
        raise RuntimeError("canonical cache content hash mismatch")
    if sha256_file(identity_path) != manifest["row_identities_sha256"]:
        raise RuntimeError("canonical cache identity hash mismatch")
    with np.load(cache_path, allow_pickle=False) as payload:
        sensor = np.asarray(payload["sensor83"], dtype=np.float64)
        h1 = np.asarray(payload["h1"], dtype=np.float64)
    identities = json.loads(identity_path.read_text(encoding="utf-8"))
    if sensor.shape != (manifest["row_count"], 83) or h1.shape != (manifest["row_count"], 104):
        raise RuntimeError("canonical cache matrix shape mismatch")
    if len(identities) != manifest["row_count"]:
        raise RuntimeError("canonical cache identity row count mismatch")
    return sensor, h1, identities, manifest
