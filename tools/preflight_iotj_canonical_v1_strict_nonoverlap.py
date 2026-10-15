"""Fail-closed preflight for the strict raw-file-disjoint robustness dataset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.build_iotj_canonical_v1_strict_nonoverlap import assert_strict_nonoverlap, read_csv, sha256
from tools.preflight_iotj_canonical_v1 import verify_dataset_hashes


def _rows(directory: Path, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    arrays = (
        np.load(directory / f"{split}_features.npy", allow_pickle=False),
        np.load(directory / f"{split}_classification_labels.npy", allow_pickle=False),
        np.load(directory / f"{split}_regression_labels.npy", allow_pickle=False),
        np.load(directory / f"{split}_phase_labels.npy", allow_pickle=False),
    )
    metadata = json.loads((directory / f"{split}_experiment_info.json").read_text(encoding="utf-8"))
    if len({len(value) for value in (*arrays, metadata)}) != 1:
        raise RuntimeError(f"FAIL_CLOSED row alignment: {directory.name}/{split}")
    if arrays[0].shape[1:] != (50, 8) or not np.isfinite(arrays[0]).all():
        raise RuntimeError(f"FAIL_CLOSED feature contract: {directory.name}/{split}")
    return *arrays, metadata


def run_preflight(dataset: Path, parent: Path) -> dict[str, Any]:
    required = (
        "dataset_sha256.json", "strict_non_overlap_protocol.json",
        "strict_non_overlap_split_manifest.csv", "strict_non_overlap_assignment_manifest.csv",
    )
    missing = [name for name in required if not (dataset / name).is_file()]
    if missing:
        raise RuntimeError(f"FAIL_CLOSED missing strict assets: {missing}")
    hashes = json.loads((dataset / "dataset_sha256.json").read_text(encoding="utf-8"))
    aggregate = verify_dataset_hashes(dataset, hashes)
    protocol = json.loads((dataset / "strict_non_overlap_protocol.json").read_text(encoding="utf-8"))
    parent_hash = json.loads((parent / "dataset_sha256.json").read_text(encoding="utf-8"))["aggregate_sha256"]
    if protocol["parent_dataset_sha256"] != parent_hash:
        raise RuntimeError("FAIL_CLOSED parent dataset hash differs")

    split_rows = read_csv(dataset / "strict_non_overlap_split_manifest.csv")
    audit = assert_strict_nonoverlap(split_rows)
    if audit != protocol["audit"]:
        raise RuntimeError("FAIL_CLOSED strict overlap audit differs from freeze")

    # Source arrays/provenance are byte-identical to canonical-v1.
    source_checked = 0
    for client in (1, 2):
        for path in sorted((parent / f"client_{client}").glob("*")):
            counterpart = dataset / f"client_{client}" / path.name
            if not counterpart.is_file() or sha256(path) != sha256(counterpart):
                raise RuntimeError(f"FAIL_CLOSED source asset differs: C{client}/{path.name}")
            source_checked += 1

    counts: dict[str, dict[str, int]] = {}
    c5_anomaly_in_test = False
    for client in (3, 4, 5):
        counts[f"C{client}"] = {}
        available_phases = protocol["coverage"][f"C{client}"]["available_phases"]
        for split in ("calibration", "test"):
            features, cls, reg, phase, metadata = _rows(dataset / f"client_{client}", split)
            counts[f"C{client}"][split] = len(features)
            cells = {(int(cls[index]), float(reg[index, int(cls[index])])) for index in range(len(cls))}
            if len(cells) != 40 or sorted(set(map(int, cls))) != [0, 1, 2, 3]:
                raise RuntimeError(f"FAIL_CLOSED class/concentration coverage: C{client}/{split}")
            if sorted(set(map(int, phase))) != available_phases:
                raise RuntimeError(f"FAIL_CLOSED phase coverage: C{client}/{split}")
            for index, item in enumerate(metadata):
                if int(item["classification_label"]) != int(cls[index]):
                    raise RuntimeError(f"FAIL_CLOSED metadata label mismatch: C{client}/{split}/{index}")
                if client == 5 and split == "test" and int(item["repeat_id"]) == 1 and float(item["concentration"]) == 225.0 and str(item["gas"]).lower() == "methane":
                    c5_anomaly_in_test = True
    if not c5_anomaly_in_test:
        raise RuntimeError("FAIL_CLOSED C5 methane 225 ppm repeat1 not retained in strict test")
    expected = {"C3": {"calibration": 678, "test": 2515}, "C4": {"calibration": 320, "test": 840}, "C5": {"calibration": 320, "test": 840}}
    if counts != expected:
        raise RuntimeError(f"FAIL_CLOSED strict counts differ: {counts}")
    return {
        "schema_version": "iotj.canonical_v1.strict_nonoverlap.preflight.v1",
        "status": "PASS",
        "dataset_aggregate_sha256": aggregate,
        "parent_dataset_aggregate_sha256": parent_hash,
        "counts": counts,
        "overlap": audit,
        "source_files_byte_identical": source_checked,
        "label_and_phase_coverage": "PASS",
        "c5_methane_225_repeat1_in_test": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = run_preflight(args.dataset.resolve(), args.parent.resolve())
    if args.output:
        args.output.resolve().write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
