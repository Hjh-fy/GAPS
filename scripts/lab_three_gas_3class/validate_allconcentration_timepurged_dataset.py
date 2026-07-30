"""Validate the auxiliary all-concentration time-purged P2-to-P3 dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ACTIVE_SPLITS = {
    2: ("train", "calibration"),
    3: ("calibration", "test"),
}
def read_manifest(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def concentration_keys(rows: list[dict]) -> set[tuple[str, str, str, str]]:
    return {
        (
            row["gas_label"],
            row["version"],
            row["exposure_index"],
            row["concentration_ppm"],
        )
        for row in rows
    }


def interval_overlap(left: dict, right: dict) -> bool:
    return max(float(left["window_start_s"]), float(right["window_start_s"])) < min(
        float(left["window_end_s"]),
        float(right["window_end_s"]),
    )


def audit_dataset(root: Path) -> dict:
    root = root.resolve()
    errors: list[str] = []
    warnings: list[str] = []
    result: dict = {
        "root": str(root),
        "split_protocol": "all_concentration_timepurged_p2_to_p3_v1",
        "clients": {},
    }
    schema = json.loads((root / "class_schema.json").read_text(encoding="utf-8"))
    expected_shape = tuple(schema["input_shape"])
    fold_dir = root / "fold_1"
    fold_config = json.loads(
        (fold_dir / "fold_config.json").read_text(encoding="utf-8")
    )
    if fold_config.get("split_protocol") != result["split_protocol"]:
        errors.append("fold_config split_protocol mismatch")
    if fold_config.get("normalization_fit_clients") != [2]:
        errors.append("normalization_fit_clients must be [2]")

    manifests: dict[tuple[int, str], list[dict]] = {}
    for client_id, splits in ACTIVE_SPLITS.items():
        client_result = {}
        client_dir = fold_dir / f"client_{client_id}"
        for split in splits:
            prefix = "calibration" if split == "calibration" else split
            feature_path = client_dir / f"{prefix}_features.npy"
            label_path = client_dir / f"{prefix}_classification_labels.npy"
            phase_path = client_dir / f"{prefix}_phase_labels.npy"
            manifest_path = client_dir / f"{prefix}_window_manifest.csv"
            for path in (feature_path, label_path, phase_path, manifest_path):
                if not path.is_file():
                    errors.append(f"missing {path}")
            if errors and any(not p.is_file() for p in (
                feature_path, label_path, phase_path, manifest_path
            )):
                continue
            features = np.load(feature_path)
            labels = np.load(label_path)
            phases = np.load(phase_path)
            rows = read_manifest(manifest_path)
            manifests[(client_id, split)] = rows
            summary_key = "validation" if split == "calibration" else split
            expected_n = int(
                fold_config["clients"][str(client_id)][summary_key]["n_windows"]
            )
            if features.shape != (expected_n, *expected_shape):
                errors.append(
                    f"{feature_path}: expected {(expected_n, *expected_shape)}, "
                    f"got {features.shape}"
                )
            if len(labels) != expected_n or len(phases) != expected_n:
                errors.append(f"{client_dir}/{prefix}: array lengths mismatch")
            if len(rows) != expected_n:
                errors.append(f"{manifest_path}: expected {expected_n} rows")
            if not np.isfinite(features).all():
                errors.append(f"{feature_path}: non-finite values")
            if set(np.unique(labels)) != {0, 1, 2}:
                errors.append(f"{label_path}: labels are not 0,1,2")
            if set(np.unique(phases)) != {0}:
                errors.append(f"{phase_path}: phases are not all zero")
            class_counts = {
                str(label): int(np.sum(labels == label)) for label in (0, 1, 2)
            }
            expected_per_class = expected_n // 3
            if set(class_counts.values()) != {expected_per_class}:
                errors.append(
                    f"{client_dir}/{prefix}: class imbalance {class_counts}"
                )
            exposure_ids = {row["exposure_id"] for row in rows}
            if len(exposure_ids) != 30:
                errors.append(
                    f"{client_dir}/{prefix}: expected 30 exposures, "
                    f"got {len(exposure_ids)}"
                )
            client_result[split] = {
                "n_windows": expected_n,
                "n_exposures": len(exposure_ids),
                "shape": list(features.shape),
                "class_counts": class_counts,
                "n_concentration_keys": len(concentration_keys(rows)),
            }
        result["clients"][str(client_id)] = client_result

    for client_id, (left_name, right_name) in ACTIVE_SPLITS.items():
        left_rows = manifests.get((client_id, left_name), [])
        right_rows = manifests.get((client_id, right_name), [])
        left_keys = concentration_keys(left_rows)
        right_keys = concentration_keys(right_rows)
        if left_keys != right_keys or len(left_keys) != 30:
            errors.append(
                f"client {client_id}: concentration coverage differs between "
                f"{left_name} and {right_name}"
            )
        overlap_count = 0
        right_by_exposure: dict[str, list[dict]] = {}
        for row in right_rows:
            right_by_exposure.setdefault(row["exposure_id"], []).append(row)
        for left in left_rows:
            for right in right_by_exposure.get(left["exposure_id"], []):
                overlap_count += int(interval_overlap(left, right))
        result["clients"][str(client_id)]["raw_time_overlap_count"] = overlap_count
        if overlap_count:
            errors.append(
                f"client {client_id}: {overlap_count} raw-time interval overlaps "
                f"between {left_name} and {right_name}"
            )

    client2_dir = fold_dir / "client_2"
    client3_dir = fold_dir / "client_3"
    aliases = (
        (
            client2_dir / "calibration_features.npy",
            client2_dir / "test_features.npy",
            "P2 compatibility test",
        ),
        (
            client3_dir / "calibration_features.npy",
            client3_dir / "train_features.npy",
            "P3 compatibility train",
        ),
    )
    for source, alias, label in aliases:
        if sha256(source) != sha256(alias):
            errors.append(f"{label} is not an exact calibration alias")

    p2_train = np.load(client2_dir / "train_features.npy")
    mean = p2_train.mean(axis=(0, 1), dtype=np.float64)
    std = p2_train.std(axis=(0, 1), dtype=np.float64)
    result["normalization_train_mean"] = mean.tolist()
    result["normalization_train_std"] = std.tolist()
    if not np.allclose(mean, 0.0, atol=2e-5):
        errors.append(f"P2 train normalization mean is not zero: {mean}")
    if not np.allclose(std, 1.0, atol=2e-4):
        errors.append(f"P2 train normalization std is not one: {std}")

    boundary_manifest = read_manifest(root / "boundary_manifest.csv")
    if boundary_manifest and {
        row["source"] for row in boundary_manifest
    } == {"nominal_schedule"}:
        warnings.append(
            "All gas boundaries use the nominal schedule; this remains "
            "preliminary screening evidence."
        )
    result["errors"] = errors
    result["warnings"] = warnings
    result["ok"] = not errors
    return result


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=(
            project_root
            / "dataset"
            / "client_data_lab_3gas_allconc_timepurged_p2src_v1"
        ),
    )
    args = parser.parse_args()
    report = audit_dataset(args.data_root)
    output = args.data_root / "validation_report.json"
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Validation ok={report['ok']}; report={output}")
    for warning in report["warnings"]:
        print(f"WARNING: {warning}")
    if not report["ok"]:
        for error in report["errors"]:
            print(f"ERROR: {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
