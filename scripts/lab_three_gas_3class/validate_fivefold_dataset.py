"""Audit the generated laboratory three-gas five-fold dataset."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Set

import numpy as np


PREFIX_BY_SPLIT = {
    "train": "train",
    "validation": "calibration",
    "test": "test",
}


def read_manifest(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def audit_dataset(root: Path) -> dict:
    errors: List[str] = []
    warnings: List[str] = []
    fold_results: Dict[str, dict] = {}

    schema_path = root / "class_schema.json"
    if not schema_path.exists():
        errors.append(f"Missing {schema_path}")
        return {"ok": False, "errors": errors, "warnings": warnings}
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    expected_shape = tuple(schema["input_shape"])
    if schema.get("num_classes") != 3:
        errors.append("class_schema num_classes is not 3")

    for fold_index in range(1, 6):
        fold_dir = root / f"fold_{fold_index}"
        fold_config_path = fold_dir / "fold_config.json"
        if not fold_config_path.is_file():
            errors.append(f"Missing {fold_config_path}")
            continue
        fold_config = json.loads(fold_config_path.read_text(encoding="utf-8"))
        normalization_clients = fold_config.get(
            "normalization_fit_clients", [1, 2, 3]
        )
        if (
            not normalization_clients
            or len(set(normalization_clients)) != len(normalization_clients)
            or any(client not in (1, 2, 3) for client in normalization_clients)
        ):
            errors.append(
                f"{fold_config_path}: invalid normalization_fit_clients "
                f"{normalization_clients}"
            )
        fold_result = {
            "clients": {},
            "normalization_fit_scope": fold_config.get(
                "normalization_fit_scope", "unknown"
            ),
            "normalization_fit_clients": normalization_clients,
        }
        global_train_arrays = []
        for platform in (1, 2, 3):
            client_dir = fold_dir / f"client_{platform}"
            split_exposures: Dict[str, Set[str]] = {}
            client_result = {}
            for split_name, prefix in PREFIX_BY_SPLIT.items():
                feature_path = client_dir / f"{prefix}_features.npy"
                label_path = client_dir / f"{prefix}_classification_labels.npy"
                phase_path = client_dir / f"{prefix}_phase_labels.npy"
                manifest_path = client_dir / f"{prefix}_window_manifest.csv"
                missing = [
                    str(path)
                    for path in (feature_path, label_path, phase_path, manifest_path)
                    if not path.exists()
                ]
                if missing:
                    errors.extend(f"Missing {path}" for path in missing)
                    continue

                features = np.load(feature_path)
                labels = np.load(label_path)
                phases = np.load(phase_path)
                manifest = read_manifest(manifest_path)
                if split_name == "train" and platform in normalization_clients:
                    global_train_arrays.append(features)

                if features.ndim != 3 or tuple(features.shape[1:]) != expected_shape:
                    errors.append(
                        f"{feature_path}: expected (*,{expected_shape}), got {features.shape}"
                    )
                if len(features) != len(labels) or len(features) != len(manifest):
                    errors.append(
                        f"{client_dir}/{prefix}: feature/label/manifest lengths differ"
                    )
                if len(phases) != len(features) or set(np.unique(phases)) != {0}:
                    errors.append(
                        f"{phase_path}: expected one compatibility phase with value 0"
                    )
                if not np.isfinite(features).all():
                    errors.append(f"{feature_path}: non-finite values")
                if set(np.unique(labels)) != {0, 1, 2}:
                    errors.append(f"{label_path}: labels are not exactly 0,1,2")

                exposure_ids = {row["exposure_id"] for row in manifest}
                expected_exposures = 18 if split_name == "train" else 6
                if len(exposure_ids) != expected_exposures:
                    errors.append(
                        f"{client_dir}/{prefix}: expected {expected_exposures} "
                        f"exposures, got {len(exposure_ids)}"
                    )
                manifest_labels = np.asarray(
                    [int(row["gas_label"]) for row in manifest],
                    dtype=np.int64,
                )
                if not np.array_equal(labels, manifest_labels):
                    errors.append(f"{client_dir}/{prefix}: labels disagree with manifest")

                exposure_class_counts = {
                    gas_label: len(
                        {
                            row["exposure_id"]
                            for row in manifest
                            if int(row["gas_label"]) == gas_label
                        }
                    )
                    for gas_label in (0, 1, 2)
                }
                expected_per_class = 6 if split_name == "train" else 2
                if set(exposure_class_counts.values()) != {expected_per_class}:
                    errors.append(
                        f"{client_dir}/{prefix}: exposure class imbalance "
                        f"{exposure_class_counts}"
                    )
                split_exposures[split_name] = exposure_ids
                client_result[split_name] = {
                    "n_windows": len(features),
                    "n_exposures": len(exposure_ids),
                    "shape": list(features.shape),
                    "window_class_counts": {
                        str(label): int(np.sum(labels == label))
                        for label in (0, 1, 2)
                    },
                    "exposure_class_counts": exposure_class_counts,
                }

            if set(split_exposures) == set(PREFIX_BY_SPLIT):
                pairs = (
                    ("train", "validation"),
                    ("train", "test"),
                    ("validation", "test"),
                )
                for left, right in pairs:
                    overlap = split_exposures[left] & split_exposures[right]
                    if overlap:
                        errors.append(
                            f"{client_dir}: exposure leakage {left}/{right}: "
                            f"{sorted(overlap)}"
                        )
            fold_result["clients"][str(platform)] = client_result

        if global_train_arrays:
            train = np.concatenate(global_train_arrays, axis=0)
            mean = train.mean(axis=(0, 1), dtype=np.float64)
            std = train.std(axis=(0, 1), dtype=np.float64)
            if not np.allclose(mean, 0.0, atol=2e-5):
                errors.append(
                    f"{fold_dir}: normalization-client train mean is not zero: {mean}"
                )
            if not np.allclose(std, 1.0, atol=2e-4):
                errors.append(
                    f"{fold_dir}: normalization-client train std is not one: {std}"
                )
            fold_result["normalization_train_mean"] = mean.tolist()
            fold_result["normalization_train_std"] = std.tolist()
        fold_results[f"fold_{fold_index}"] = fold_result

    boundary_path = root / "boundary_manifest.csv"
    if boundary_path.exists():
        boundary_rows = read_manifest(boundary_path)
        sources = sorted({row["source"] for row in boundary_rows})
        if sources == ["nominal_schedule"]:
            warnings.append(
                "All gas boundaries use the nominal schedule. Rebuild with an "
                "edited --boundaries-csv before treating results as final."
            )
    else:
        errors.append(f"Missing {boundary_path}")

    return {
        "ok": not errors,
        "root": str(root),
        "errors": errors,
        "warnings": warnings,
        "folds": fold_results,
    }


def main() -> None:
    default_root = (
        Path(__file__).resolve().parents[2]
        / "dataset"
        / "client_data_lab_3gas_5fold_nominal_v1"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=default_root)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    report = audit_dataset(args.data_root.resolve())
    report_path = args.report or args.data_root / "validation_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Validation ok={report['ok']}; report={report_path}")
    for warning in report["warnings"]:
        print(f"WARNING: {warning}")
    for error in report["errors"]:
        print(f"ERROR: {error}")
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
