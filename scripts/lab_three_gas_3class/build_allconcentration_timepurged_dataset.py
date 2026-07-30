"""Build an auxiliary all-concentration, raw-time-purged P2-to-P3 dataset.

Every retained exposure contributes windows to both active roles:

* P2: train and source calibration
* P3: target calibration and target test

Calibration windows are spread across the exposure. Their immediate neighbors
are purged because 100-second windows at 50-second stride overlap in raw time.
The compatibility-only split for each client aliases calibration and is never
used for training, selection, adaptation, or target testing.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

if __package__:
    from .build_fivefold_dataset import (
        GAS_SCHEMA,
        BuildConfig,
        build_exposure_records,
        concatenate_records,
        discover_sessions,
        load_boundaries,
        parse_normalization_clients,
        parse_selected_channels,
        save_split,
        write_csv,
    )
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from build_fivefold_dataset import (  # type: ignore[no-redef]
        GAS_SCHEMA,
        BuildConfig,
        build_exposure_records,
        concatenate_records,
        discover_sessions,
        load_boundaries,
        parse_normalization_clients,
        parse_selected_channels,
        save_split,
        write_csv,
    )

CALIBRATION_INDICES = (3, 11, 19)
PURGED_INDICES = (2, 4, 10, 12, 18, 20)
MAIN_INDICES = tuple(
    index
    for index in range(23)
    if index not in set(CALIBRATION_INDICES) | set(PURGED_INDICES)
)


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description=(
            "Build the auxiliary all-concentration time-purged P2-to-P3 "
            "laboratory dataset."
        )
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=project_root / "dataset" / "Dataset_self",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=(
            project_root
            / "dataset"
            / "client_data_lab_3gas_allconc_timepurged_p2src_v1"
        ),
    )
    parser.add_argument("--raw-filename", default="1.csv")
    parser.add_argument("--boundaries-csv", type=Path, default=None)
    parser.add_argument("--normalization-clients", default="2")
    parser.add_argument(
        "--transform",
        choices=("relative", "relative_conductance", "delta", "none"),
        default="relative",
    )
    parser.add_argument(
        "--selected-channels",
        default="1,2,4,6,8,9",
        help="Comma-separated physical sensor channels from CH1..CH18.",
    )
    parser.add_argument(
        "--main-min-offset-s",
        type=float,
        default=0.0,
        help=(
            "Minimum gas-relative start time for source-train and target-test "
            "windows. Calibration indices remain fixed and time-purged."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def subset_record(record: dict, indices: Sequence[int], role: str) -> dict:
    if len(record["features"]) != 23:
        raise ValueError(
            f"{record['exposure_id']}: expected 23 base windows, "
            f"got {len(record['features'])}"
        )
    rows = []
    for index in indices:
        row = dict(record["window_rows"][index])
        row["timepurged_role"] = role
        row["base_window_index"] = int(index)
        rows.append(row)
    return {
        **{
            key: value
            for key, value in record.items()
            if key not in {"features", "window_rows"}
        },
        "features": record["features"][list(indices)].copy(),
        "window_rows": rows,
    }


def subset_records(
    records: Iterable[dict],
    *,
    platform: int,
    indices: Sequence[int],
    role: str,
) -> list[dict]:
    return [
        subset_record(record, indices, role)
        for record in records
        if int(record["platform"]) == platform
    ]


def resolve_main_indices(config: BuildConfig, minimum_offset_s: float) -> tuple[int, ...]:
    if minimum_offset_s < 0.0:
        raise ValueError("main minimum offset must be non-negative")
    main_indices = tuple(
        index
        for index in MAIN_INDICES
        if index * config.stride_s >= minimum_offset_s
    )
    if not main_indices:
        raise ValueError(
            f"No main windows remain at minimum offset {minimum_offset_s}s"
        )
    return main_indices


def build_dataset(
    config: BuildConfig,
    overwrite: bool = False,
    main_min_offset_s: float = 0.0,
) -> dict:
    raw_root = Path(config.raw_root).resolve()
    output_root = Path(config.output_root).resolve()
    if output_root.exists() and any(output_root.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output_root}. "
            "Use --overwrite to replace generated files."
        )
    output_root.mkdir(parents=True, exist_ok=True)

    if tuple(config.normalization_clients) != (2,):
        raise ValueError(
            "This direction-specific protocol requires normalization clients=(2,)"
        )
    if config.window_s != 100.0 or config.stride_s != 50.0:
        raise ValueError("Time-purged indices require window_s=100 and stride_s=50")
    main_indices = resolve_main_indices(config, main_min_offset_s)

    sessions = discover_sessions(raw_root, config.raw_filename)
    boundaries_path = Path(config.boundaries_csv) if config.boundaries_csv else None
    boundary_overrides = load_boundaries(boundaries_path)
    records, exposure_manifest, boundary_manifest = build_exposure_records(
        sessions,
        config,
        boundary_overrides,
    )
    if len(records) != 90:
        raise AssertionError(f"Expected 90 retained exposures, got {len(records)}")

    p2_train = subset_records(
        records,
        platform=2,
        indices=main_indices,
        role="source_train",
    )
    p2_calibration = subset_records(
        records,
        platform=2,
        indices=CALIBRATION_INDICES,
        role="source_calibration",
    )
    p3_calibration = subset_records(
        records,
        platform=3,
        indices=CALIBRATION_INDICES,
        role="target_calibration",
    )
    p3_test = subset_records(
        records,
        platform=3,
        indices=main_indices,
        role="target_test",
    )

    train_features, _, _ = concatenate_records(p2_train)
    mean = train_features.mean(
        axis=(0, 1), keepdims=True, dtype=np.float64
    ).astype(np.float32)
    std = train_features.std(
        axis=(0, 1), keepdims=True, dtype=np.float64
    ).astype(np.float32)
    std = np.maximum(std, np.float32(1e-8))

    fold_dir = output_root / "fold_1"
    fold_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        fold_dir / "norm_stats.npz",
        mean=mean,
        std=std,
        selected_channels=np.asarray(config.selected_channels, dtype=np.int64),
    )

    client2_dir = fold_dir / "client_2"
    client3_dir = fold_dir / "client_3"
    client2_dir.mkdir(parents=True, exist_ok=True)
    client3_dir.mkdir(parents=True, exist_ok=True)

    p2_summary = {
        "train": save_split(client2_dir, "train", p2_train, mean, std),
        "validation": save_split(
            client2_dir,
            "calibration",
            p2_calibration,
            mean,
            std,
        ),
        "test": save_split(
            client2_dir,
            "test",
            p2_calibration,
            mean,
            std,
        ),
        "compatibility_alias": {
            "test": "source_calibration",
            "used_by_protocol": False,
        },
    }
    p3_summary = {
        "train": save_split(
            client3_dir,
            "train",
            p3_calibration,
            mean,
            std,
        ),
        "validation": save_split(
            client3_dir,
            "calibration",
            p3_calibration,
            mean,
            std,
        ),
        "test": save_split(client3_dir, "test", p3_test, mean, std),
        "compatibility_alias": {
            "train": "target_calibration",
            "used_by_protocol": False,
        },
    }
    for client_dir, summary in (
        (client2_dir, p2_summary),
        (client3_dir, p3_summary),
    ):
        (client_dir / "stats.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    split_config = {
        "fold": 1,
        "split_protocol": "all_concentration_timepurged_p2_to_p3_v1",
        "source_clients": [2],
        "target_client": 3,
        "normalization_fit_scope": "source_clients_train_only",
        "normalization_fit_clients": [2],
        "base_window_count_per_exposure": 23,
        "calibration_window_indices_zero_based": list(CALIBRATION_INDICES),
        "purged_window_indices_zero_based": list(PURGED_INDICES),
        "main_window_indices_zero_based": list(main_indices),
        "main_min_offset_s": float(main_min_offset_s),
        "raw_time_overlap_allowed": False,
        "target_test_open_after_source_round_selection": True,
        "clients": {
            "2": p2_summary,
            "3": p3_summary,
        },
    }
    (fold_dir / "fold_config.json").write_text(
        json.dumps(split_config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    write_csv(output_root / "exposure_manifest.csv", exposure_manifest)
    write_csv(output_root / "boundary_manifest.csv", boundary_manifest)
    schema = {
        "task": "three_gas_classification",
        "num_classes": 3,
        "num_clients": 3,
        "input_shape": [100, len(config.selected_channels)],
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
    }
    (output_root / "class_schema.json").write_text(
        json.dumps(schema, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    build_config = {
        **asdict(config),
        "split_protocol": split_config["split_protocol"],
        "calibration_window_indices_zero_based": list(CALIBRATION_INDICES),
        "purged_window_indices_zero_based": list(PURGED_INDICES),
        "main_window_indices_zero_based": list(MAIN_INDICES),
        "effective_main_window_indices_zero_based": list(main_indices),
        "main_min_offset_s": float(main_min_offset_s),
    }
    (output_root / "build_config.json").write_text(
        json.dumps(build_config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    summary = {
        "output_root": str(output_root),
        "n_sessions": len(sessions),
        "n_exposures": len(records),
        "n_folds": 1,
        "boundary_mode": (
            "manual_or_mixed" if boundary_overrides else "nominal_schedule"
        ),
        "split_protocol": split_config["split_protocol"],
        "folds": {"fold_1": split_config},
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
        transform=args.transform,
        selected_channels=parse_selected_channels(args.selected_channels),
        normalization_clients=parse_normalization_clients(
            args.normalization_clients
        ),
        boundaries_csv=str(args.boundaries_csv) if args.boundaries_csv else None,
    )
    summary = build_dataset(
        config,
        overwrite=args.overwrite,
        main_min_offset_s=args.main_min_offset_s,
    )
    print(
        "Built all-concentration time-purged dataset: "
        f"sessions={summary['n_sessions']}, "
        f"exposures={summary['n_exposures']}, "
        f"output={summary['output_root']}"
    )


if __name__ == "__main__":
    main()
