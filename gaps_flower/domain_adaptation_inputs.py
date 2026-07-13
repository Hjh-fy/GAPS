"""Fail-closed input contracts for server-side domain adaptation."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np


def _parse_data_dirs(data_dirs_spec: str | Sequence[str | Path]) -> tuple[Path, ...]:
    if isinstance(data_dirs_spec, str):
        raw_items = data_dirs_spec.split(",")
    else:
        raw_items = list(data_dirs_spec)
    paths = tuple(Path(str(item).strip()).resolve() for item in raw_items if str(item).strip())
    if not paths:
        raise ValueError("domain adaptation data directory specification is empty")
    if len(set(paths)) != len(paths):
        raise ValueError("domain adaptation data directory specification contains duplicates")
    for path in paths:
        if not path.is_dir():
            raise ValueError(f"domain adaptation data directory does not exist: {path}")
    return paths


def validate_domain_adaptation_request(
    strategy: object,
    use_domain_adapt: bool,
    server_val_data: str | None,
    server_calib_data: str | None,
) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
    """Validate the server-level DA request before Flower starts."""
    if not use_domain_adapt:
        return (), ()
    strategy_name = (
        strategy.strip().lower()
        if isinstance(strategy, str)
        else type(strategy).__name__.lower()
    )
    if strategy_name not in {"gaps", "gapsstrategy"}:
        raise ValueError(
            "Domain adaptation requires the GAPS strategy; FedAvg cannot execute DA"
        )
    if not server_val_data:
        raise ValueError("server_val_data source-domain specification is required for DA")
    if not server_calib_data:
        raise ValueError("server_calib_data target-domain specification is required for DA")
    source_dirs = _parse_data_dirs(server_val_data)
    target_dirs = _parse_data_dirs(server_calib_data)
    overlap = sorted(set(source_dirs) & set(target_dirs), key=str)
    if overlap:
        raise ValueError(
            f"Source and target domain adaptation directories overlap: {overlap}"
        )
    return source_dirs, target_dirs


def _select_split_paths(directory: Path, strict: bool) -> tuple[Path, Path, Path]:
    prefixes = ("calibration_",) if strict else ("calibration_", "test_", "train_", "")
    selected_prefix: str | None = None
    for prefix in prefixes:
        if (directory / f"{prefix}features.npy").is_file():
            selected_prefix = prefix
            break
    if selected_prefix is None:
        expected = "calibration_features.npy" if strict else "features split"
        raise ValueError(f"Missing {expected} in {directory}")

    feature_path = directory / f"{selected_prefix}features.npy"
    class_path = directory / f"{selected_prefix}classification_labels.npy"
    phase_path = directory / f"{selected_prefix}phase_labels.npy"
    for path in (class_path, phase_path):
        if not path.is_file():
            raise ValueError(f"Missing {path.name} in {directory}")
    return feature_path, class_path, phase_path


def _load_npy(path: Path) -> np.ndarray:
    try:
        return np.load(path, allow_pickle=False)
    except Exception as exc:
        raise ValueError(f"Cannot load domain adaptation array {path}: {exc}") from exc


def _validate_label_array(
    values: np.ndarray,
    *,
    path: Path,
    rows: int,
    upper_bound: int,
    label: str,
) -> np.ndarray:
    if values.ndim != 1 or len(values) != rows:
        raise ValueError(
            f"{label} rows/shape mismatch in {path}: expected ({rows},), got {values.shape}"
        )
    if not np.issubdtype(values.dtype, np.integer):
        raise ValueError(f"{label} must use an integer dtype: {path}")
    normalized = values.astype(np.int64, copy=False)
    if np.any(normalized < 0) or np.any(normalized >= upper_bound):
        raise ValueError(
            f"{label} values are outside range [0, {upper_bound - 1}]: {path}"
        )
    return normalized


def load_domain_adaptation_arrays(
    data_dirs_spec: str | Sequence[str | Path],
    *,
    strict: bool,
    expected_window_shape: tuple[int, int] = (100, 8),
    num_classes: int = 4,
    num_phases: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load and validate one or more DA splits without label/phase fallback."""
    directories = _parse_data_dirs(data_dirs_spec)
    feature_parts: list[np.ndarray] = []
    class_parts: list[np.ndarray] = []
    phase_parts: list[np.ndarray] = []

    for directory in directories:
        feature_path, class_path, phase_path = _select_split_paths(directory, strict)
        features = _load_npy(feature_path)
        classes = _load_npy(class_path)
        phases = _load_npy(phase_path)
        if features.ndim != 3 or tuple(features.shape[1:]) != tuple(expected_window_shape):
            raise ValueError(
                f"Domain adaptation feature shape must be (N, {expected_window_shape[0]}, "
                f"{expected_window_shape[1]}), got {features.shape}: {feature_path}"
            )
        if len(features) == 0:
            raise ValueError(f"Domain adaptation split is empty: {feature_path}")
        if not np.issubdtype(features.dtype, np.number):
            raise ValueError(f"Domain adaptation features must be numeric: {feature_path}")
        if not np.all(np.isfinite(features)):
            raise ValueError(f"Domain adaptation features must be finite: {feature_path}")
        classes = _validate_label_array(
            classes,
            path=class_path,
            rows=len(features),
            upper_bound=num_classes,
            label="classification labels",
        )
        phases = _validate_label_array(
            phases,
            path=phase_path,
            rows=len(features),
            upper_bound=num_phases,
            label="phase labels",
        )
        feature_parts.append(features.astype(np.float32, copy=False))
        class_parts.append(classes)
        phase_parts.append(phases)

    return (
        np.concatenate(feature_parts, axis=0),
        np.concatenate(class_parts, axis=0),
        np.concatenate(phase_parts, axis=0),
    )
