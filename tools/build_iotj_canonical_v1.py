"""Build the immutable HZ5_MEAN_W10S canonical IoT-J dataset from raw files."""
from __future__ import annotations

import argparse, csv, hashlib, json, subprocess, sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WS = ROOT.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.preprocessor_canonical_candidate import TA, process_file

RAW_ROOT = WS / "dataset/data1"
FROZEN_ROLE_ROOT = WS / "dataset/client_data_c12src_c345tgt_2080_timeaware_60_170_window_fullgrid"
DEFAULT_OUTPUT = ROOT / "dataset/iotj_canonical_v1"
PHASE = {"early": 0, "middle": 1, "late": 2}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, text=True, stdout=subprocess.PIPE).stdout.strip()


def canonical_preprocessing_manifest(code_commit: str) -> dict[str, Any]:
    return {
        "schema_version": "iotj.canonical_preprocessing.v1",
        "candidate_id": "HZ5_MEAN_W10S",
        "sampling_rate_hz": 5,
        "bin_width_s": 0.2,
        "aggregation": "mean",
        "baseline": "raw_observation_mean_G0_20_50s",
        "timestamp_sort": "stable",
        "duplicate_timestamp_merge": "mean",
        "gap_policy": "short_gap_only_max_1_bin",
        "long_gap_continuous_interpolation": False,
        "physical_crop_s": [60.0, 170.0],
        "window_duration_s": 10.0,
        "stride_s": 5.0,
        "points_per_window": 50,
        "source_clients": [1, 2],
        "target_clients": [3, 4, 5],
        "target_train_ratio": 0.0,
        "target_calibration_ratio": 0.2,
        "target_test_ratio": 0.8,
        "split_source": "frozen_physical_identity_reuse",
        "seed": 42,
        "reuse_historical_checkpoint": False,
        "code_commit": code_commit,
    }


def physical_window_key(meta: Mapping[str, Any]) -> tuple[int, str, float]:
    return (int(meta["client_id"]), str(meta["filename"]), round(float(meta["window_start_s"]), 6))


def merge_role_maps(per_client: Mapping[int, Mapping[tuple[int, str, float], str]], client_order: Sequence[int]) -> dict[tuple[int, str, float], str]:
    merged: dict[tuple[int, str, float], str] = {}
    for client in sorted(set(int(value) for value in client_order)):
        for key, role in per_client[client].items():
            if key in merged and merged[key] != role:
                raise RuntimeError(f"FAIL_CLOSED conflicting role for {key}")
            merged[key] = role
    return merged


def load_frozen_roles(root: Path = FROZEN_ROLE_ROOT, client_order: Sequence[int] = (1, 2, 3, 4, 5)) -> dict[tuple[int, str, float], str]:
    per_client: dict[int, dict[tuple[int, str, float], str]] = {}
    for client in client_order:
        roles: dict[tuple[int, str, float], str] = {}
        splits = ("train", "calibration", "test") if client in (1, 2) else ("calibration", "test")
        for split in splits:
            path = root / f"client_{client}/{split}_experiment_info.json"
            for meta in json.loads(path.read_text(encoding="utf-8")):
                meta = {**meta, "client_id": client}
                key = physical_window_key(meta)
                if key in roles:
                    raise RuntimeError(f"FAIL_CLOSED split identity overlap: {key}")
                roles[key] = split
        per_client[client] = roles
    return merge_role_maps(per_client, client_order)


def save_split(directory: Path, split: str, rows: list[dict[str, Any]]) -> None:
    features = np.asarray([row["feature"] for row in rows], dtype=np.float32)
    if not rows:
        features = np.empty((0, 50, 8), dtype=np.float32)
    cls = np.asarray([row["classification_label"] for row in rows], dtype=np.int64)
    reg = np.asarray([row["regression_label"] for row in rows], dtype=np.float32).reshape(-1, 4)
    phase = np.asarray([row["phase_label"] for row in rows], dtype=np.int64)
    np.save(directory / f"{split}_features.npy", features)
    np.save(directory / f"{split}_classification_labels.npy", cls)
    np.save(directory / f"{split}_regression_labels.npy", reg)
    np.save(directory / f"{split}_phase_labels.npy", phase)
    metadata = [{k: v for k, v in row.items() if k != "feature"} for row in rows]
    (directory / f"{split}_experiment_info.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_canonical_dataset(raw_root: Path, output: Path, seed: int = 42) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"FAIL_CLOSED canonical output already exists: {output}")
    output.mkdir(parents=True)
    commit = git_head()
    manifest = canonical_preprocessing_manifest(commit)
    if seed != int(manifest["seed"]):
        raise ValueError("canonical seed must be 42")
    roles = load_frozen_roles()
    by_client_split: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    raw_rows: list[dict[str, Any]] = []
    processing_rows: list[dict[str, Any]] = []
    identity_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    raw_hashes: dict[str, str] = {}
    files = TA.iter_raw_txt_files(raw_root)
    for path in files:
        info = TA.parse_filename(path.name)
        client = int(info["client_id"])
        relative = str(path.relative_to(raw_root)).replace("\\", "/")
        digest = sha256(path)
        raw_hashes[relative] = digest
        raw_rows.append({"relative_path": relative, "filename": path.name, "client_id": client, "gas": info["gas"], "class_id": info["classification_label"], "concentration": info["concentration"], "repeat_id": info["repeat_id"], "bytes": path.stat().st_size, "sha256": digest})
        result = process_file(path, 5, "mean", 10, "mean", True)
        for window, quality in zip(result["windows"], result["metadata"]):
            key = (client, path.name, round(float(quality["physical_window_start_s"]), 6))
            role = roles.get(key)
            if role is None:
                raise RuntimeError(f"FAIL_CLOSED missing frozen role: {key}")
            identity = f"C{client}|{path.name}|{info['repeat_id']}|{info['classification_label']}|{float(info['concentration']):.6f}|{quality['physical_window_start_s']:.6f}|{quality['physical_window_end_s']:.6f}"
            base = {"physical_identity": identity, "client_id": client, "filename": path.name, "repeat_id": int(info["repeat_id"]), "gas": info["gas"], "gas_code": info["gas_code"], "class_id": int(info["classification_label"]), "concentration": float(info["concentration"]), "window_start_s": float(quality["physical_window_start_s"]), "window_end_s": float(quality["physical_window_end_s"]), "role": role, **quality, "baseline_n_raw_samples": int(result["baseline_n_raw_samples"]), "duplicate_timestamps": int(result["duplicate_timestamps"]), "sampling_completeness": float(result["sampling_completeness"])}
            included = bool(quality["valid"] and np.isfinite(window).all())
            processing_rows.append({**base, "included": included, "inactive_reason": "" if included else "invalid_long_gap"})
            identity_rows.append({**base, "included": included})
            if not included:
                continue
            row = {**base, "feature": window, "classification_label": int(info["classification_label"]), "regression_label": info["regression_label"], "phase_label": PHASE.get(str(info["phase_label"]), -1)}
            by_client_split[(client, role)].append(row)
            split_rows.append({k: base[k] for k in ("physical_identity", "client_id", "filename", "repeat_id", "gas", "class_id", "concentration", "window_start_s", "window_end_s", "role")})
    for client in range(1, 6):
        directory = output / f"client_{client}"
        directory.mkdir()
        splits = ("train", "calibration", "test") if client in (1, 2) else ("calibration", "test")
        counts: dict[str, int] = {}
        for split in splits:
            rows = sorted(by_client_split[(client, split)], key=lambda row: row["physical_identity"])
            save_split(directory, split, rows)
            counts[split] = len(rows)
        stats = {"schema_version": "iotj.canonical_v1.stats", "client_id": f"C{client}", "role": "source" if client in (1, 2) else "target", "counts": counts, "n_total_included": sum(counts.values())}
        (directory / "stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    (output / "canonical_preprocessing_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    write_csv(output / "raw_file_manifest.csv", raw_rows)
    (output / "raw_sha256.json").write_text(json.dumps(raw_hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(output / "processing_manifest.csv", processing_rows)
    write_csv(output / "window_identity_manifest.csv", identity_rows)
    write_csv(output / "split_manifest.csv", split_rows)
    file_hashes: dict[str, str] = {}
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "dataset_sha256.json":
            file_hashes[str(path.relative_to(output)).replace("\\", "/")] = sha256(path)
    aggregate = hashlib.sha256()
    for name, digest in sorted(file_hashes.items()):
        aggregate.update(name.encode()); aggregate.update(b"\0"); aggregate.update(digest.encode()); aggregate.update(b"\n")
    dataset_hash = {"schema_version": "iotj.canonical_v1.sha256", "aggregate_sha256": aggregate.hexdigest(), "files": file_hashes}
    (output / "dataset_sha256.json").write_text(json.dumps(dataset_hash, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"output": str(output), "raw_files": len(files), "included_windows": len(split_rows), "aggregate_sha256": dataset_hash["aggregate_sha256"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(build_canonical_dataset(args.raw_root.resolve(), args.output.resolve(), args.seed), indent=2))


if __name__ == "__main__":
    main()
