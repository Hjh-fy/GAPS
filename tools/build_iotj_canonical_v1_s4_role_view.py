"""Build a fail-closed S4 source-role view without modifying canonical-v1."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "dataset/iotj_canonical_v1"
DEFAULT_OUTPUT = ROOT / "dataset/iotj_canonical_v1_s4_role_view"
SPLITS = ("train", "calibration", "test")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_split(directory: Path, split: str) -> list[dict[str, Any]]:
    metadata = json.loads((directory / f"{split}_experiment_info.json").read_text(encoding="utf-8"))
    features = np.load(directory / f"{split}_features.npy")
    classes = np.load(directory / f"{split}_classification_labels.npy")
    regression = np.load(directory / f"{split}_regression_labels.npy")
    phases = np.load(directory / f"{split}_phase_labels.npy")
    if not (len(metadata) == len(features) == len(classes) == len(regression) == len(phases)):
        raise RuntimeError(f"FAIL_CLOSED split arrays differ in length: {directory}/{split}")
    return [
        {
            "metadata": dict(metadata[index]),
            "feature": features[index],
            "classification": int(classes[index]),
            "regression": regression[index],
            "phase": int(phases[index]),
        }
        for index in range(len(metadata))
    ]


def _concentration(row: Mapping[str, Any]) -> float:
    metadata = row["metadata"]
    if "concentration" in metadata:
        return float(metadata["concentration"])
    class_id = int(row["classification"])
    return float(np.asarray(row["regression"])[class_id])


def partition_added_source(
    rows: Sequence[Mapping[str, Any]], *, seed: int, client: int
) -> dict[str, list[dict[str, Any]]]:
    """Partition one added source independently by class and concentration."""
    groups: dict[tuple[int, float], list[dict[str, Any]]] = {}
    for value in rows:
        row = dict(value)
        key = (int(row["classification"]), _concentration(row))
        groups.setdefault(key, []).append(row)
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(client)]))
    output: dict[str, list[dict[str, Any]]] = {name: [] for name in SPLITS}
    for key in sorted(groups):
        bucket = sorted(groups[key], key=lambda row: str(row["metadata"]["physical_identity"]))
        order = rng.permutation(len(bucket))
        bucket = [bucket[int(index)] for index in order]
        n = len(bucket)
        if n < 3:
            raise RuntimeError(f"FAIL_CLOSED source stratum too small: C{client}/{key}/{n}")
        n_test = max(1, int(round(n * 0.20)))
        n_calibration = max(1, int(round(n * 0.10)))
        if n_test + n_calibration >= n:
            n_calibration = max(1, n - n_test - 1)
        output["test"].extend(bucket[:n_test])
        output["calibration"].extend(bucket[n_test : n_test + n_calibration])
        output["train"].extend(bucket[n_test + n_calibration :])
    for split in SPLITS:
        values = output[split]
        order = rng.permutation(len(values))
        output[split] = [values[int(index)] for index in order]
    return output


def _save_split(directory: Path, split: str, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"FAIL_CLOSED empty derived split: {directory}/{split}")
    np.save(directory / f"{split}_features.npy", np.asarray([row["feature"] for row in rows]))
    np.save(
        directory / f"{split}_classification_labels.npy",
        np.asarray([row["classification"] for row in rows], dtype=np.int64),
    )
    np.save(
        directory / f"{split}_regression_labels.npy",
        np.asarray([row["regression"] for row in rows], dtype=np.float32),
    )
    np.save(
        directory / f"{split}_phase_labels.npy",
        np.asarray([row["phase"] for row in rows], dtype=np.int64),
    )
    metadata = []
    for row in rows:
        item = dict(row["metadata"])
        item["role"] = split
        metadata.append(item)
    (directory / f"{split}_experiment_info.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _copy_client(source: Path, output: Path, client: int) -> dict[str, str]:
    src = source / f"client_{client}"
    dst = output / f"client_{client}"
    shutil.copytree(src, dst)
    return {
        str(path.relative_to(source)).replace("\\", "/"): sha256_file(path)
        for path in sorted(src.iterdir())
        if path.is_file()
    }


def build_role_view(source: Path, output: Path, *, seed: int = 42) -> dict[str, Any]:
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"FAIL_CLOSED derived role view already exists: {output}")
    if int(seed) != 42:
        raise ValueError("Gate A role-view seed must be 42")
    manifest_path = source / "canonical_preprocessing_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("candidate_id") != "HZ5_MEAN_W10S":
        raise RuntimeError("FAIL_CLOSED source is not canonical HZ5_MEAN_W10S")
    output.mkdir(parents=True)
    clients: dict[str, Any] = {}
    copied_hashes: dict[str, str] = {}
    for client in (1, 2, 5):
        copied_hashes.update(_copy_client(source, output, client))
        clients[f"C{client}"] = {"role": "source" if client in (1, 2) else "target", "copy_mode": "byte_identical"}
    identity_records: list[str] = []
    for client in (3, 4):
        pool = _load_split(source / f"client_{client}", "calibration") + _load_split(
            source / f"client_{client}", "test"
        )
        identities = [str(row["metadata"]["physical_identity"]) for row in pool]
        if len(set(identities)) != len(identities):
            raise RuntimeError(f"FAIL_CLOSED duplicate canonical identity in C{client}")
        partitions = partition_added_source(pool, seed=seed, client=client)
        directory = output / f"client_{client}"
        directory.mkdir()
        counts = {}
        for split, rows in partitions.items():
            _save_split(directory, split, rows)
            counts[split] = len(rows)
            identity_records.extend(
                f"C{client}\0{split}\0{row['metadata']['physical_identity']}" for row in rows
            )
        stats = {
            "schema_version": "iotj.canonical_v1.s4_role_view.stats.v1",
            "client_id": f"C{client}",
            "role": "source",
            "counts": counts,
            "n_total_included": len(pool),
        }
        (directory / "stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
        clients[f"C{client}"] = {
            "role": "source",
            "copy_mode": "derived_stratified_70_10_20",
            "source_pool_count": len(pool),
            "counts": counts,
            "rng": f"SeedSequence([{seed},{client}])",
        }
    partition_digest = hashlib.sha256("\n".join(sorted(identity_records)).encode("utf-8")).hexdigest()
    payload = {
        "schema_version": "iotj.canonical_v1.s4_role_view.v1",
        "status": "FROZEN",
        "source": str(source),
        "seed": int(seed),
        "preprocessing_candidate": "HZ5_MEAN_W10S",
        "source_clients": ["C1", "C2", "C3", "C4"],
        "target_clients": ["C5"],
        "c5_rng_access": False,
        "clients": clients,
        "partition_identity_sha256": partition_digest,
        "byte_identical_source_hashes": copied_hashes,
    }
    shutil.copy2(manifest_path, output / manifest_path.name)
    (output / "s4_role_view_manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    file_hashes = {
        str(path.relative_to(output)).replace("\\", "/"): sha256_file(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "dataset_sha256.json"
    }
    aggregate = hashlib.sha256()
    for name, digest in sorted(file_hashes.items()):
        aggregate.update(name.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(digest.encode("ascii"))
        aggregate.update(b"\n")
    (output / "dataset_sha256.json").write_text(
        json.dumps(
            {"schema_version": "iotj.canonical_v1.s4_role_view.sha256.v1", "aggregate_sha256": aggregate.hexdigest(), "files": file_hashes},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(build_role_view(args.source, args.output, seed=args.seed), indent=2))


if __name__ == "__main__":
    main()

