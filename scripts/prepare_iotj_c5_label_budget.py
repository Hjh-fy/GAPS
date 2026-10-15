"""Build immutable nested C5 calibration budgets by indexing canonical-v1 arrays."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "dataset/iotj_canonical_v1/client_5"
DEFAULT_OUTPUT = ROOT / "results/iotj_canonical_v1_c5_budget_20260810"
BUDGET_QUOTA = {20: 8, 15: 6, 10: 4, 5: 2}
ARRAY_NAMES = (
    "features",
    "classification_labels",
    "phase_labels",
    "regression_labels",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stable_key(value: str) -> str:
    return hashlib.sha256(f"42|{value}".encode("utf-8")).hexdigest()


def _stratum_key(row: dict[str, Any]) -> tuple[int, float]:
    return int(row["class_id"]), float(row["concentration"])


def _diverse_order(indices: list[int], info: list[dict[str, Any]]) -> list[int]:
    groups: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index in indices:
        row = info[index]
        groups[(str(row["filename"]), int(row["repeat_id"]))].append(index)
    keys = sorted(groups, key=lambda key: _stable_key(f"{key[0]}|{key[1]}"))
    for key in keys:
        groups[key].sort(key=lambda index: _stable_key(str(info[index]["physical_identity"])))
    ordered: list[int] = []
    for position in range(max(len(values) for values in groups.values())):
        for key in keys:
            if position < len(groups[key]):
                ordered.append(groups[key][position])
    return ordered


def build_nested_indices(info: list[dict[str, Any]]) -> dict[int, list[int]]:
    if len(info) != 320:
        raise ValueError(f"expected 320 canonical C5 calibration windows, got {len(info)}")
    groups: dict[tuple[int, float], list[int]] = defaultdict(list)
    identities: set[str] = set()
    for index, row in enumerate(info):
        identity = str(row["physical_identity"])
        if identity in identities:
            raise ValueError(f"duplicate calibration identity: {identity}")
        identities.add(identity)
        groups[_stratum_key(row)].append(index)
    if len(groups) != 40:
        raise ValueError(f"expected 40 strata, got {len(groups)}")
    if {len(indices) for indices in groups.values()} != {8}:
        raise ValueError("each canonical C5 stratum must contain exactly 8 windows")
    ranked = {key: _diverse_order(indices, info) for key, indices in groups.items()}
    nested: dict[int, list[int]] = {}
    for budget, quota in BUDGET_QUOTA.items():
        nested[budget] = [
            index
            for key in sorted(ranked)
            for index in ranked[key][:quota]
        ]
    if not set(nested[5]) < set(nested[10]) < set(nested[15]) < set(nested[20]):
        raise RuntimeError("nested calibration identity contract failed")
    return nested


def _membership(index: int, membership_sets: dict[int, set[int]]) -> str:
    return ";".join(f"{budget:02d}" for budget in (5, 10, 15, 20) if index in membership_sets[budget])


def _manifest_row(row: dict[str, Any], membership: str) -> dict[str, Any]:
    return {
        "client_id": int(row["client_id"]),
        "raw_filename": str(row["filename"]),
        "repeat_id": int(row["repeat_id"]),
        "gas": str(row["gas"]),
        "class_id": int(row["class_id"]),
        "concentration": float(row["concentration"]),
        "physical_window_start_s": float(row["physical_window_start_s"]),
        "physical_window_end_s": float(row["physical_window_end_s"]),
        "canonical_identity": str(row["physical_identity"]),
        "budget_membership": membership,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prepare(source: Path, output: Path) -> dict[str, Any]:
    source = source.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    info_path = source / "calibration_experiment_info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    nested = build_nested_indices(info)
    arrays = {
        name: np.load(source / f"calibration_{name}.npy", allow_pickle=False)
        for name in ARRAY_NAMES
    }
    if any(len(values) != len(info) for values in arrays.values()):
        raise ValueError("canonical calibration array lengths differ from metadata")
    test_info = json.loads((source / "test_experiment_info.json").read_text(encoding="utf-8"))
    test_identities = {str(row["physical_identity"]) for row in test_info}
    calibration_identities = {str(row["physical_identity"]) for row in info}
    overlap = calibration_identities & test_identities
    if overlap:
        raise ValueError(f"calibration/test exact identity overlap: {len(overlap)}")

    output.mkdir(parents=True)
    membership_sets = {budget: set(indices) for budget, indices in nested.items()}
    memberships = {index: _membership(index, membership_sets) for index in nested[20]}
    coverage_rows: list[dict[str, Any]] = []
    for budget in (20, 15, 10, 5):
        indices = nested[budget]
        suffix = f"{budget:02d}"
        directory = output / f"budget_data/client_5_budget_{suffix}"
        directory.mkdir(parents=True)
        for name, values in arrays.items():
            np.save(directory / f"calibration_{name}.npy", values[indices])
        selected_info = [info[index] for index in indices]
        (directory / "calibration_experiment_info.json").write_text(
            json.dumps(selected_info, indent=2) + "\n", encoding="utf-8"
        )
        manifest_rows = [_manifest_row(info[index], memberships[index]) for index in indices]
        _write_csv(output / f"c5_calibration_budget_{suffix}pct.csv", manifest_rows)
        strata = {_stratum_key(row) for row in selected_info}
        coverage_rows.append({
            "budget_pct": budget,
            "total_calibration_n": len(indices),
            "covered_strata": len(strata),
            "total_strata": 40,
            "coverage_ratio": len(strata) / 40.0,
            "levels_per_class": json.dumps({
                str(class_id): len({float(row["concentration"]) for row in selected_info if int(row["class_id"]) == class_id})
                for class_id in range(4)
            }, sort_keys=True),
        })
    _write_csv(output / "c5_budget_strata_coverage.csv", coverage_rows)

    strata_counts: dict[tuple[int, float], int] = defaultdict(int)
    for row in info:
        strata_counts[_stratum_key(row)] += 1
    class_counts = {str(class_id): sum(int(row["class_id"]) == class_id for row in info) for class_id in range(4)}
    repeats = sorted({int(row["repeat_id"]) for row in info})
    repeat_counts = {str(repeat): sum(int(row["repeat_id"]) == repeat for row in info) for repeat in repeats}
    audit = {
        "schema_version": "gaps.iotj.canonical_v1.c5_label_budget.audit.v1",
        "status": "PASS",
        "source": str(source),
        "source_file_sha256": {
            path.name: sha256(path)
            for path in sorted(source.glob("calibration_*"))
            if path.is_file()
        },
        "counts": {str(budget): len(nested[budget]) for budget in (20, 15, 10, 5)},
        "strata": len(strata_counts),
        "stratum_min_n_20pct": min(strata_counts.values()),
        "stratum_max_n_20pct": max(strata_counts.values()),
        "class_counts": class_counts,
        "raw_file_count": len({str(row["filename"]) for row in info}),
        "repeat_counts": repeat_counts,
        "nested": True,
        "calibration_test_exact_identity_overlap": 0,
        "test_arrays_copied_to_budget_directories": False,
    }
    (output / "C5_CALIBRATION_POOL_AUDIT.md").write_text(
        "# C5 calibration pool audit\n\n"
        "Status: **PASS**.\n\n"
        "The frozen pool contains 320 windows, 80 raw files, four classes, and all 40 class × concentration strata. "
        "Every stratum contains exactly eight windows. The nominal 15/10/5% nested budgets contain 240/160/80 windows.\n",
        encoding="utf-8",
    )
    (output / "C5_CALIBRATION_BUDGET_MANIFEST_AUDIT.md").write_text(
        "# C5 calibration budget manifest audit\n\n"
        "Status: **PASS**.\n\n"
        "The deterministic family satisfies 5% ⊂ 10% ⊂ 15% ⊂ 20%, covers 40/40 strata at every budget, "
        "contains no duplicates, copies no target-test arrays, and has zero exact calibration/test identity overlap.\n",
        encoding="utf-8",
    )
    (output / "calibration_budget_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    files = {
        path.relative_to(output).as_posix(): sha256(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "c5_calibration_budget_manifest_sha256.json"
    }
    (output / "c5_calibration_budget_manifest_sha256.json").write_text(
        json.dumps({"schema_version": "gaps.iotj.c5_budget.sha256.v1", "files": files}, indent=2) + "\n",
        encoding="utf-8",
    )
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.output), indent=2))


if __name__ == "__main__":
    main()
