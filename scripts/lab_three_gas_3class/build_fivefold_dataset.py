"""Build leakage-safe five-fold data for laboratory three-gas classification.

The raw experiment unit is one gas exposure. Windows derived from the same
exposure always remain in the same split.  The default schedule is provisional:

    1800 s air, then 6 * (1200 s target gas + 1800 s recovery)

Exposure 1 is kept in the boundary audit file but excluded from model data.
Exposures 2--6 from v1 and v2 form five paired fold groups.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


GAS_SCHEMA = {
    "乙醛": {"label": 0, "name_en": "acetaldehyde"},
    "甲烷": {"label": 1, "name_en": "methane"},
    "乙酸": {"label": 2, "name_en": "acetic_acid"},
}

CONCENTRATIONS = {
    ("乙醛", "v1"): ([399, 399, 548, 698, 847, 997], "ppb"),
    ("乙醛", "v2"): ([325, 325, 475, 624, 774, 921], "ppb"),
    ("甲烷", "v1"): ([4000, 4000, 5500, 7000, 8500, 10000], "ppm"),
    ("甲烷", "v2"): ([3260, 3260, 4760, 6260, 7760, 9242], "ppm"),
    ("乙酸", "v1"): ([28.52, 28.52, 39.215, 49.91, 60.605, 71.3], "ppm"),
    ("乙酸", "v2"): ([23.2, 23.2, 33.9, 44.6, 55.3, 66.0], "ppm"),
}

SELECTED_CHANNELS = (1, 2, 4, 6, 8, 9)
SESSION_RE = re.compile(
    r"^(?P<date>\d{8})_Unit(?P<unit>[123])_"
    r"(?P<gas>乙醛|甲烷|乙酸)_(?P<version>v[12])$"
)


@dataclass(frozen=True)
class BuildConfig:
    raw_root: str
    output_root: str
    raw_filename: str = "1.csv"
    sample_rate_hz: float = 1.0
    initial_air_s: float = 1800.0
    gas_s: float = 1200.0
    recovery_s: float = 1800.0
    baseline_s: float = 300.0
    baseline_gap_s: float = 0.0
    trim_start_s: float = 0.0
    trim_end_s: float = 0.0
    window_s: float = 100.0
    stride_s: float = 50.0
    smooth_s: float = 5.0
    transform: str = "relative"
    selected_channels: Tuple[int, ...] = SELECTED_CHANNELS
    normalization_clients: Tuple[int, ...] = (1, 2, 3)
    boundaries_csv: str | None = None


@dataclass(frozen=True)
class Session:
    session_id: str
    directory: Path
    csv_path: Path
    platform: int
    date: str
    gas_cn: str
    gas_en: str
    gas_label: int
    version: str


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve()
    project_root = here.parents[2]
    parser = argparse.ArgumentParser(
        description="Build five leakage-safe folds for lab three-gas classification."
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=project_root / "dataset" / "Dataset_self",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=project_root / "dataset" / "client_data_lab_3gas_5fold_nominal_v1",
    )
    parser.add_argument("--raw-filename", default="1.csv")
    parser.add_argument("--boundaries-csv", type=Path, default=None)
    parser.add_argument("--sample-rate-hz", type=float, default=1.0)
    parser.add_argument("--initial-air-s", type=float, default=1800.0)
    parser.add_argument("--gas-s", type=float, default=1200.0)
    parser.add_argument("--recovery-s", type=float, default=1800.0)
    parser.add_argument("--baseline-s", type=float, default=300.0)
    parser.add_argument("--baseline-gap-s", type=float, default=0.0)
    parser.add_argument("--trim-start-s", type=float, default=0.0)
    parser.add_argument("--trim-end-s", type=float, default=0.0)
    parser.add_argument("--window-s", type=float, default=100.0)
    parser.add_argument("--stride-s", type=float, default=50.0)
    parser.add_argument("--smooth-s", type=float, default=5.0)
    parser.add_argument(
        "--transform",
        choices=("relative", "relative_conductance", "delta", "none"),
        default="relative",
    )
    parser.add_argument(
        "--selected-channels",
        default=",".join(str(channel) for channel in SELECTED_CHANNELS),
        help="Comma-separated physical sensor channels from CH1..CH18.",
    )
    parser.add_argument(
        "--normalization-clients",
        default="1,2,3",
        help=(
            "Comma-separated client IDs whose training exposures fit the "
            "fold Z-score. Use '2' for P2->P3 and '1,2' for P1+P2->P3."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing files inside an existing output directory.",
    )
    return parser.parse_args()


def parse_normalization_clients(text: str) -> Tuple[int, ...]:
    try:
        clients = tuple(
            int(item.strip()) for item in text.split(",") if item.strip()
        )
    except ValueError as exc:
        raise ValueError(
            f"Invalid --normalization-clients value: {text!r}"
        ) from exc
    if not clients or len(set(clients)) != len(clients):
        raise ValueError("normalization clients must be non-empty and unique")
    if any(client not in (1, 2, 3) for client in clients):
        raise ValueError("normalization clients must be drawn from 1,2,3")
    return clients


def parse_selected_channels(text: str) -> Tuple[int, ...]:
    try:
        channels = tuple(
            int(item.strip()) for item in text.split(",") if item.strip()
        )
    except ValueError as exc:
        raise ValueError(f"Invalid --selected-channels value: {text!r}") from exc
    if not channels or len(set(channels)) != len(channels):
        raise ValueError("selected channels must be non-empty and unique")
    if any(channel < 1 or channel > 18 for channel in channels):
        raise ValueError("selected channels must be drawn from CH1..CH18")
    return channels


def discover_sessions(raw_root: Path, raw_filename: str) -> List[Session]:
    sessions: List[Session] = []
    for platform_dir in sorted(raw_root.glob("platform*")):
        if not platform_dir.is_dir():
            continue
        for session_dir in sorted(platform_dir.iterdir()):
            if not session_dir.is_dir():
                continue
            match = SESSION_RE.match(session_dir.name)
            if match is None:
                continue
            info = match.groupdict()
            platform = int(info["unit"])
            expected_platform = int(platform_dir.name.replace("platform", ""))
            if platform != expected_platform:
                raise ValueError(
                    f"Platform mismatch for {session_dir}: "
                    f"directory={expected_platform}, session Unit={platform}"
                )
            csv_path = session_dir / raw_filename
            if not csv_path.exists():
                raise FileNotFoundError(f"Missing {raw_filename}: {session_dir}")
            gas_info = GAS_SCHEMA[info["gas"]]
            sessions.append(
                Session(
                    session_id=session_dir.name,
                    directory=session_dir,
                    csv_path=csv_path,
                    platform=platform,
                    date=info["date"],
                    gas_cn=info["gas"],
                    gas_en=str(gas_info["name_en"]),
                    gas_label=int(gas_info["label"]),
                    version=info["version"],
                )
            )

    expected = {
        (platform, gas, version)
        for platform in (1, 2, 3)
        for gas in GAS_SCHEMA
        for version in ("v1", "v2")
    }
    actual = {(s.platform, s.gas_cn, s.version) for s in sessions}
    missing = sorted(expected - actual)
    duplicates = [
        key for key in sorted(actual)
        if sum((s.platform, s.gas_cn, s.version) == key for s in sessions) != 1
    ]
    if missing or duplicates or len(sessions) != 18:
        raise ValueError(
            "Expected exactly 18 formal sessions "
            f"(3 platforms x 3 gases x 2 versions); "
            f"found={len(sessions)}, missing={missing}, duplicates={duplicates}"
        )
    return sorted(
        sessions,
        key=lambda s: (s.platform, s.gas_label, s.version, s.session_id),
    )


def load_boundaries(
    path: Path | None,
) -> Dict[Tuple[str, int], Tuple[float, float, str]]:
    if path is None:
        return {}
    required = {"session_id", "exposure_index", "gas_start_s", "gas_end_s"}
    result: Dict[Tuple[str, int], Tuple[float, float, str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Boundary CSV missing columns: {sorted(missing)}")
        for row in reader:
            session_id = str(row["session_id"]).strip()
            exposure_index = int(row["exposure_index"])
            start = float(row["gas_start_s"])
            end = float(row["gas_end_s"])
            if not 1 <= exposure_index <= 6:
                raise ValueError(f"Invalid exposure_index in boundary CSV: {row}")
            if end <= start:
                raise ValueError(f"Invalid gas interval in boundary CSV: {row}")
            source = str(row.get("source") or "manual")
            key = (session_id, exposure_index)
            if key in result:
                raise ValueError(f"Duplicate boundary row: {key}")
            result[key] = (start, end, source)
    return result


def nominal_boundary(config: BuildConfig, exposure_index: int) -> Tuple[float, float]:
    cycle_s = config.gas_s + config.recovery_s
    start = config.initial_air_s + (exposure_index - 1) * cycle_s
    return start, start + config.gas_s


def concentration_metadata(
    gas_cn: str,
    version: str,
    exposure_index: int,
) -> Tuple[float, str, float]:
    values, unit = CONCENTRATIONS[(gas_cn, version)]
    original = float(values[exposure_index - 1])
    canonical_ppm = original / 1000.0 if unit == "ppb" else original
    return original, unit, canonical_ppm


def read_raw_csv(
    path: Path,
    selected_channels: Sequence[int] = SELECTED_CHANNELS,
) -> Tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path, delimiter=",", dtype=np.float64)
    if data.ndim != 2 or data.shape[1] != 19:
        raise ValueError(f"Expected 19 columns (Time + CH1..CH18), got {data.shape}: {path}")
    if not np.isfinite(data).all():
        raise ValueError(f"Non-finite raw values: {path}")
    time = data[:, 0]
    if np.any(np.diff(time) < 0):
        order = np.argsort(time, kind="stable")
        data = data[order]
        time = data[:, 0]
    unique_time, unique_index = np.unique(time, return_index=True)
    selected_columns = list(selected_channels)
    signals = data[unique_index][:, selected_columns]
    if len(unique_time) < 2 or np.any(np.diff(unique_time) <= 0):
        raise ValueError(f"Time axis is not strictly increasing after deduplication: {path}")
    return unique_time, signals


def signal_domain(
    resistance: np.ndarray,
    transform: str,
) -> np.ndarray:
    if transform != "relative_conductance":
        return resistance
    if np.any(resistance <= 0.0):
        minimum = float(np.min(resistance))
        raise ValueError(
            "relative_conductance requires strictly positive raw resistance; "
            f"minimum={minimum}"
        )
    conductance = np.reciprocal(resistance, dtype=np.float64)
    if not np.isfinite(conductance).all():
        raise ValueError("Non-finite conductance after reciprocal transform")
    return conductance


def moving_average_reflect(values: np.ndarray, width: int) -> np.ndarray:
    if width <= 1:
        return values
    if width % 2 == 0:
        width += 1
    pad = width // 2
    padded = np.pad(values, ((pad, pad), (0, 0)), mode="reflect")
    kernel = np.ones(width, dtype=np.float64) / width
    smoothed = np.empty_like(values, dtype=np.float64)
    for channel in range(values.shape[1]):
        smoothed[:, channel] = np.convolve(
            padded[:, channel], kernel, mode="valid"
        )
    return smoothed


def make_windows(
    values: np.ndarray,
    window_points: int,
    stride_points: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if len(values) < window_points:
        raise ValueError(
            f"Segment has {len(values)} points, shorter than window {window_points}"
        )
    starts = np.arange(
        0,
        len(values) - window_points + 1,
        stride_points,
        dtype=np.int64,
    )
    windows = np.stack(
        [values[start : start + window_points] for start in starts],
        axis=0,
    )
    return windows.astype(np.float32), starts


def build_exposure_records(
    sessions: Sequence[Session],
    config: BuildConfig,
    boundary_overrides: Mapping[Tuple[str, int], Tuple[float, float, str]],
) -> Tuple[List[dict], List[dict], List[dict]]:
    exposure_records: List[dict] = []
    exposure_manifest: List[dict] = []
    boundary_manifest: List[dict] = []

    step = 1.0 / config.sample_rate_hz
    window_points = int(round(config.window_s * config.sample_rate_hz))
    stride_points = int(round(config.stride_s * config.sample_rate_hz))
    smooth_points = int(round(config.smooth_s * config.sample_rate_hz))
    if window_points <= 0 or stride_points <= 0:
        raise ValueError("window_s and stride_s must map to positive point counts")

    for session in sessions:
        time, signals = read_raw_csv(
            session.csv_path,
            selected_channels=config.selected_channels,
        )
        domain_signals = signal_domain(signals, config.transform)
        dt = np.diff(time)
        sampling_median_s = float(np.median(dt))

        for exposure_index in range(1, 7):
            key = (session.session_id, exposure_index)
            if key in boundary_overrides:
                gas_start, gas_end, boundary_source = boundary_overrides[key]
            else:
                gas_start, gas_end = nominal_boundary(config, exposure_index)
                boundary_source = "nominal_schedule"

            boundary_manifest.append(
                {
                    "session_id": session.session_id,
                    "platform": session.platform,
                    "gas_cn": session.gas_cn,
                    "version": session.version,
                    "exposure_index": exposure_index,
                    "gas_start_s": gas_start,
                    "gas_end_s": gas_end,
                    "source": boundary_source,
                    "included_in_model": int(exposure_index >= 2),
                }
            )
            if exposure_index == 1:
                continue

            segment_start = gas_start + config.trim_start_s
            segment_end = gas_end - config.trim_end_s
            baseline_end = gas_start - config.baseline_gap_s
            baseline_start = baseline_end - config.baseline_s

            if baseline_start < time[0] or segment_end > time[-1]:
                raise ValueError(
                    f"Boundary outside raw time range for {session.session_id}, "
                    f"exposure {exposure_index}: requested "
                    f"baseline_start={baseline_start}, segment_end={segment_end}, "
                    f"available=({time[0]}, {time[-1]})"
                )
            baseline_mask = (time >= baseline_start) & (time < baseline_end)
            if baseline_mask.sum() < max(10, int(config.baseline_s / (3 * step))):
                raise ValueError(
                    f"Too few baseline samples for {session.session_id}, "
                    f"exposure {exposure_index}: {baseline_mask.sum()}"
                )
            baseline = np.median(domain_signals[baseline_mask], axis=0)

            grid = np.arange(segment_start, segment_end, step, dtype=np.float64)
            resampled = np.column_stack(
                [
                    np.interp(grid, time, domain_signals[:, c])
                    for c in range(domain_signals.shape[1])
                ]
            )
            if config.transform in ("relative", "relative_conductance"):
                denom = np.maximum(np.abs(baseline), 1e-12)
                transformed = (resampled - baseline) / denom
            elif config.transform == "delta":
                transformed = resampled - baseline
            else:
                transformed = resampled

            transformed = moving_average_reflect(transformed, smooth_points)
            windows, starts = make_windows(
                transformed,
                window_points=window_points,
                stride_points=stride_points,
            )

            retained_rank = exposure_index - 1
            exposure_id = (
                f"P{session.platform}_{session.gas_en}_{session.version}"
                f"_E{exposure_index}"
            )
            original_conc, original_unit, concentration_ppm = concentration_metadata(
                session.gas_cn,
                session.version,
                exposure_index,
            )
            common = {
                "exposure_id": exposure_id,
                "session_id": session.session_id,
                "platform": session.platform,
                "date": session.date,
                "gas_cn": session.gas_cn,
                "gas_en": session.gas_en,
                "gas_label": session.gas_label,
                "version": session.version,
                "exposure_index": exposure_index,
                "retained_rank": retained_rank,
                "fold_group": retained_rank,
                "concentration_original": original_conc,
                "concentration_unit": original_unit,
                "concentration_ppm": concentration_ppm,
                "gas_start_s": gas_start,
                "gas_end_s": gas_end,
                "segment_start_s": segment_start,
                "segment_end_s": segment_end,
                "baseline_start_s": baseline_start,
                "baseline_end_s": baseline_end,
                "boundary_source": boundary_source,
            }
            exposure_manifest.append(
                {
                    **common,
                    "raw_csv": str(session.csv_path),
                    "raw_time_end_s": float(time[-1]),
                    "raw_median_dt_s": sampling_median_s,
                    "resampled_points": len(transformed),
                    "n_windows": len(windows),
                }
            )
            window_rows = []
            for window_index, start_point in enumerate(starts):
                start_s = segment_start + start_point * step
                window_rows.append(
                    {
                        **common,
                        "window_index": int(window_index),
                        "window_start_s": float(start_s),
                        "window_end_s": float(start_s + config.window_s),
                    }
                )
            exposure_records.append(
                {
                    **common,
                    "features": windows,
                    "window_rows": window_rows,
                }
            )

    return exposure_records, exposure_manifest, boundary_manifest


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def select_records(
    records: Iterable[dict],
    *,
    platform: int | None = None,
    fold_groups: Iterable[int],
) -> List[dict]:
    groups = set(fold_groups)
    return [
        record
        for record in records
        if record["fold_group"] in groups
        and (platform is None or record["platform"] == platform)
    ]


def concatenate_records(records: Sequence[dict]) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
    if not records:
        raise ValueError("Cannot concatenate an empty record list")
    features = np.concatenate([record["features"] for record in records], axis=0)
    labels = np.concatenate(
        [
            np.full(len(record["features"]), record["gas_label"], dtype=np.int64)
            for record in records
        ],
        axis=0,
    )
    rows: List[dict] = []
    for record in records:
        rows.extend(record["window_rows"])
    if len(features) != len(labels) or len(features) != len(rows):
        raise AssertionError("Feature, label, and manifest lengths differ")
    return features, labels, rows


def save_split(
    client_dir: Path,
    prefix: str,
    records: Sequence[dict],
    mean: np.ndarray,
    std: np.ndarray,
) -> dict:
    features, labels, rows = concatenate_records(records)
    normalized = ((features - mean) / std).astype(np.float32)
    phase_labels = np.zeros(len(labels), dtype=np.int64)

    np.save(client_dir / f"{prefix}_features.npy", normalized)
    np.save(client_dir / f"{prefix}_classification_labels.npy", labels)
    # One compatibility phase means "whole gas exposure", not early/middle/late.
    np.save(client_dir / f"{prefix}_phase_labels.npy", phase_labels)
    write_csv(client_dir / f"{prefix}_window_manifest.csv", rows)

    exposure_ids = sorted({str(row["exposure_id"]) for row in rows})
    class_counts = {
        str(label): int(np.sum(labels == label))
        for label in sorted(np.unique(labels).tolist())
    }
    return {
        "n_windows": len(normalized),
        "n_exposures": len(exposure_ids),
        "shape": list(normalized.shape),
        "class_window_counts": class_counts,
        "exposure_ids": exposure_ids,
    }


def build_folds(
    records: Sequence[dict],
    config: BuildConfig,
    output_root: Path,
) -> dict:
    summaries: Dict[str, dict] = {}
    all_groups = {1, 2, 3, 4, 5}

    for test_group in range(1, 6):
        validation_group = test_group % 5 + 1
        train_groups = sorted(all_groups - {test_group, validation_group})
        fold_dir = output_root / f"fold_{test_group}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        global_train = [
            record
            for record in select_records(records, fold_groups=train_groups)
            if record["platform"] in config.normalization_clients
        ]
        train_features, _, _ = concatenate_records(global_train)
        mean = train_features.mean(
            axis=(0, 1), keepdims=True, dtype=np.float64
        ).astype(np.float32)
        std = train_features.std(
            axis=(0, 1), keepdims=True, dtype=np.float64
        ).astype(np.float32)
        std = np.maximum(std, np.float32(1e-8))
        np.savez(
            fold_dir / "norm_stats.npz",
            mean=mean,
            std=std,
            selected_channels=np.asarray(config.selected_channels, dtype=np.int64),
        )

        fold_summary = {
            "fold": test_group,
            "test_group": test_group,
            "validation_group": validation_group,
            "train_groups": train_groups,
            "features_are_zscored": True,
            "normalization_fit_scope": "source_clients_train_only",
            "normalization_fit_clients": list(config.normalization_clients),
            "clients": {},
        }
        for platform in (1, 2, 3):
            client_dir = fold_dir / f"client_{platform}"
            client_dir.mkdir(parents=True, exist_ok=True)
            train_records = select_records(
                records,
                platform=platform,
                fold_groups=train_groups,
            )
            validation_records = select_records(
                records,
                platform=platform,
                fold_groups={validation_group},
            )
            test_records = select_records(
                records,
                platform=platform,
                fold_groups={test_group},
            )
            client_summary = {
                "train": save_split(
                    client_dir, "train", train_records, mean, std
                ),
                # Existing GAPS code calls the validation split "calibration".
                "validation": save_split(
                    client_dir,
                    "calibration",
                    validation_records,
                    mean,
                    std,
                ),
                "test": save_split(
                    client_dir, "test", test_records, mean, std
                ),
            }
            fold_summary["clients"][str(platform)] = client_summary
            (client_dir / "stats.json").write_text(
                json.dumps(client_summary, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

        (fold_dir / "fold_config.json").write_text(
            json.dumps(fold_summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        summaries[f"fold_{test_group}"] = fold_summary
    return summaries


def build_dataset(config: BuildConfig, overwrite: bool = False) -> dict:
    raw_root = Path(config.raw_root).resolve()
    output_root = Path(config.output_root).resolve()
    if output_root.exists() and any(output_root.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output_root}. "
            "Use --overwrite to replace generated files."
        )
    output_root.mkdir(parents=True, exist_ok=True)

    sessions = discover_sessions(raw_root, config.raw_filename)
    boundaries_path = Path(config.boundaries_csv) if config.boundaries_csv else None
    boundary_overrides = load_boundaries(boundaries_path)
    valid_session_ids = {session.session_id for session in sessions}
    unknown_boundaries = sorted(
        key for key in boundary_overrides if key[0] not in valid_session_ids
    )
    if unknown_boundaries:
        raise ValueError(f"Boundary CSV contains unknown sessions: {unknown_boundaries}")

    records, exposure_manifest, boundary_manifest = build_exposure_records(
        sessions,
        config,
        boundary_overrides,
    )
    if len(records) != 90:
        raise AssertionError(f"Expected 90 retained exposures, got {len(records)}")

    fold_summaries = build_folds(records, config, output_root)
    write_csv(output_root / "exposure_manifest.csv", exposure_manifest)
    write_csv(output_root / "boundary_manifest.csv", boundary_manifest)
    write_csv(
        output_root / "fold_assignments.csv",
        [
            {
                "exposure_id": row["exposure_id"],
                "session_id": row["session_id"],
                "platform": row["platform"],
                "gas_label": row["gas_label"],
                "gas_cn": row["gas_cn"],
                "version": row["version"],
                "exposure_index": row["exposure_index"],
                "fold_group": row["fold_group"],
            }
            for row in exposure_manifest
        ],
    )

    schema = {
        "task": "three_gas_classification",
        "num_classes": 3,
        "num_clients": 3,
        "input_shape": [
            int(round(config.window_s * config.sample_rate_hz)),
            len(config.selected_channels),
        ],
        "selected_channels": list(config.selected_channels),
        "classes": {
            str(info["label"]): {
                "name_cn": gas_cn,
                "name_en": info["name_en"],
            }
            for gas_cn, info in GAS_SCHEMA.items()
        },
        "regression_task": False,
        "response_phase_partition": False,
        "compatibility_phase": {
            "enabled": True,
            "value": 0,
            "meaning": "whole_target_gas_exposure",
        },
        "double_pentene_included": False,
    }
    (output_root / "class_schema.json").write_text(
        json.dumps(schema, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_root / "build_config.json").write_text(
        json.dumps(asdict(config), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    summary = {
        "output_root": str(output_root),
        "n_sessions": len(sessions),
        "n_exposures": len(records),
        "n_folds": len(fold_summaries),
        "boundary_mode": (
            "manual_or_mixed" if boundary_overrides else "nominal_schedule"
        ),
        "anomalies": [
            {
                "session_id": session.session_id,
                "issue": "date starts with 2024; confirm whether this should be 2026",
            }
            for session in sessions
            if session.date.startswith("2024")
        ],
        "folds": fold_summaries,
    }
    (output_root / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    args = parse_args()
    config = BuildConfig(
        raw_root=str(args.raw_root),
        output_root=str(args.output_root),
        raw_filename=args.raw_filename,
        sample_rate_hz=args.sample_rate_hz,
        initial_air_s=args.initial_air_s,
        gas_s=args.gas_s,
        recovery_s=args.recovery_s,
        baseline_s=args.baseline_s,
        baseline_gap_s=args.baseline_gap_s,
        trim_start_s=args.trim_start_s,
        trim_end_s=args.trim_end_s,
        window_s=args.window_s,
        stride_s=args.stride_s,
        smooth_s=args.smooth_s,
        transform=args.transform,
        selected_channels=parse_selected_channels(args.selected_channels),
        normalization_clients=parse_normalization_clients(
            args.normalization_clients
        ),
        boundaries_csv=str(args.boundaries_csv) if args.boundaries_csv else None,
    )
    summary = build_dataset(config, overwrite=args.overwrite)
    print(
        "Built laboratory three-gas dataset: "
        f"sessions={summary['n_sessions']}, "
        f"exposures={summary['n_exposures']}, "
        f"folds={summary['n_folds']}, "
        f"output={summary['output_root']}"
    )
    if summary["anomalies"]:
        print("Warnings:")
        for anomaly in summary["anomalies"]:
            print(f"  - {anomaly['session_id']}: {anomaly['issue']}")


if __name__ == "__main__":
    main()
