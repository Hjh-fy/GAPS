"""Fail-closed preflight and SHA256 verification for canonical v1."""
from __future__ import annotations

import argparse, csv, hashlib, json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def assert_no_target_overlap(rows: Iterable[Mapping[str, Any]]) -> None:
    roles: dict[tuple[int, str], set[str]] = defaultdict(set)
    for row in rows:
        client = int(row["client_id"])
        if client in (3, 4, 5):
            roles[(client, str(row["physical_identity"]))].add(str(row["role"]))
    overlap = [key for key, value in roles.items() if {"calibration", "test"} <= value]
    if overlap:
        raise RuntimeError(f"FAIL_CLOSED target calibration/test overlap: {overlap[:3]}")


def assert_finite_arrays(paths: Iterable[Path]) -> None:
    for path in paths:
        array = np.load(path, mmap_mode="r")
        if not np.isfinite(array).all():
            raise RuntimeError(f"FAIL_CLOSED NaN/Inf array: {path}")


def verify_dataset_hashes(root: Path, manifest: Mapping[str, Any]) -> str:
    aggregate = hashlib.sha256()
    for name, expected in sorted(dict(manifest["files"]).items()):
        path = root / name
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"FAIL_CLOSED SHA256 mismatch: {name}")
        aggregate.update(name.encode()); aggregate.update(b"\0"); aggregate.update(expected.encode()); aggregate.update(b"\n")
    actual = aggregate.hexdigest()
    if actual != manifest["aggregate_sha256"]:
        raise RuntimeError("FAIL_CLOSED aggregate SHA256 mismatch")
    return actual


def run_preflight(dataset_root: Path) -> dict[str, Any]:
    required = ["canonical_preprocessing_manifest.json", "raw_file_manifest.csv", "raw_sha256.json", "processing_manifest.csv", "window_identity_manifest.csv", "split_manifest.csv", "dataset_sha256.json"]
    missing = [name for name in required if not (dataset_root / name).is_file()]
    if missing:
        raise RuntimeError(f"FAIL_CLOSED missing canonical assets: {missing}")
    manifest = json.loads((dataset_root / "canonical_preprocessing_manifest.json").read_text(encoding="utf-8"))
    expected = {"candidate_id": "HZ5_MEAN_W10S", "sampling_rate_hz": 5, "points_per_window": 50, "reuse_historical_checkpoint": False}
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise RuntimeError(f"FAIL_CLOSED preprocessing freeze differs: {key}")
    dataset_hash = json.loads((dataset_root / "dataset_sha256.json").read_text(encoding="utf-8"))
    aggregate = verify_dataset_hashes(dataset_root, dataset_hash)
    split_rows = read_csv(dataset_root / "split_manifest.csv")
    assert_no_target_overlap(split_rows)
    identities = [row["physical_identity"] for row in split_rows]
    if len(identities) != len(set(identities)):
        raise RuntimeError("FAIL_CLOSED physical-window identity not unique")
    feature_paths = sorted(dataset_root.glob("client_*/*_features.npy"))
    assert_finite_arrays(feature_paths)
    counts: dict[str, dict[str, int]] = {}
    coverage: dict[tuple[int, str, str, str], int] = defaultdict(int)
    for client in range(1, 6):
        directory = dataset_root / f"client_{client}"
        splits = ("train", "calibration", "test") if client in (1, 2) else ("calibration", "test")
        client_counts: dict[str, int] = {}
        for split in splits:
            features = np.load(directory / f"{split}_features.npy", mmap_mode="r")
            cls = np.load(directory / f"{split}_classification_labels.npy", mmap_mode="r")
            reg = np.load(directory / f"{split}_regression_labels.npy", mmap_mode="r")
            phase = np.load(directory / f"{split}_phase_labels.npy", mmap_mode="r")
            meta = json.loads((directory / f"{split}_experiment_info.json").read_text(encoding="utf-8"))
            n = len(features)
            if not (n == len(cls) == len(reg) == len(phase) == len(meta)):
                raise RuntimeError(f"FAIL_CLOSED row alignment C{client}/{split}")
            if features.shape[1:] != (50, 8):
                raise RuntimeError(f"FAIL_CLOSED temporal shape C{client}/{split}: {features.shape}")
            for i, item in enumerate(meta):
                label = int(cls[i]); ppm = float(reg[i, label])
                if label != int(item["classification_label"]) or not np.isclose(ppm, float(item["concentration"])):
                    raise RuntimeError(f"FAIL_CLOSED label mismatch C{client}/{split}/{i}")
                coverage[(client, split, str(label), str(float(ppm)))] += 1
            client_counts[split] = n
        if client in (3, 4, 5) and "train" in client_counts:
            raise RuntimeError(f"FAIL_CLOSED target train role C{client}")
        counts[f"C{client}"] = client_counts
    missing_coverage = []
    for client in range(1, 6):
        splits = ("train", "calibration", "test") if client in (1, 2) else ("calibration", "test")
        for split in splits:
            for cls in range(4):
                levels = [key for key in coverage if key[0] == client and key[1] == split and key[2] == str(cls)]
                if len(levels) != 10:
                    missing_coverage.append((client, split, cls, len(levels)))
    if missing_coverage:
        raise RuntimeError(f"FAIL_CLOSED class x concentration coverage: {missing_coverage[:5]}")
    processing = read_csv(dataset_root / "processing_manifest.csv")
    repeat1 = [row for row in processing if row["client_id"] == "5" and row["gas"].lower() == "methane" and float(row["concentration"]) == 225.0 and row["repeat_id"] == "1"]
    if not repeat1:
        raise RuntimeError("FAIL_CLOSED C5 Methane 225 repeat1 missing")
    return {"schema_version": "iotj.canonical_v1.preflight", "status": "PASS", "dataset_root": str(dataset_root), "aggregate_sha256": aggregate, "counts": counts, "split_identity_count": len(split_rows), "c5_methane_225_repeat1_windows": len(repeat1), "checkpoint_reuse": False}


def render_report(payload: Mapping[str, Any]) -> str:
    lines = ["# Canonical v1 Dataset Preflight", "", f"**Status: {payload['status']}**", "", f"Dataset SHA256: `{payload['aggregate_sha256']}`", "", "| Client | Split counts |", "|---|---|"]
    for client, counts in payload["counts"].items():
        lines.append(f"| {client} | " + ", ".join(f"{key}={value}" for key, value in counts.items()) + " |")
    lines += ["", f"Unique included identities: {payload['split_identity_count']}", f"C5 Methane 225 repeat1 raw windows recorded: {payload['c5_methane_225_repeat1_windows']}", "", "No training is authorized unless this report remains PASS and dataset SHA256 is unchanged."]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    payload = run_preflight(args.dataset.resolve())
    if not args.verify_only:
        if args.output is None:
            raise ValueError("--output required unless --verify-only")
        output = args.output.resolve()
        output.mkdir(parents=True, exist_ok=True)
        path = output / "preflight.json"
        if path.exists():
            observed = json.loads(path.read_text(encoding="utf-8"))
            if observed != payload:
                raise RuntimeError("FAIL_CLOSED existing preflight differs")
        else:
            path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            (output / "DATASET_PREFLIGHT.md").write_text(render_report(payload), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
