"""Build the preregistered raw-file-disjoint canonical-v1 robustness split.

The parent arrays are reused byte-for-value. Only target calibration/test
membership changes. The highest repeat in each class x concentration cell is
the calibration file; all other repeats are test files. Calibration files are
deterministically subsampled to preserve the frozen canonical calibration N.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARENT = ROOT / "dataset/iotj_canonical_v1"
DEFAULT_OUTPUT = ROOT / "dataset/iotj_canonical_v1_strict_nonoverlap"
TARGET_CALIBRATION_N = {3: 678, 4: 320, 5: 320}
TARGETS = (3, 4, 5)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def allocate_balanced_quotas(cells: Iterable[tuple[str, str]], total: int) -> dict[tuple[str, str], int]:
    ordered = sorted(set(cells), key=lambda item: (int(item[0]), float(item[1])))
    if not ordered:
        raise ValueError("no class x concentration cells")
    base, remainder = divmod(int(total), len(ordered))
    return {cell: base + int(index < remainder) for index, cell in enumerate(ordered)}


def evenly_spaced_indices(n: int, k: int) -> list[int]:
    if k < 0 or k > n:
        raise ValueError(f"cannot select {k} from {n}")
    if k == 0:
        return []
    # Midpoint quantiles give a deterministic temporal spread without a seed.
    indices = [min(n - 1, int((index + 0.5) * n / k)) for index in range(k)]
    if len(set(indices)) != k:
        raise RuntimeError("deterministic calibration indices are not unique")
    return indices


def choose_calibration_file(client: int, ordered_files: Sequence[str], cell_index: int) -> str:
    """Pre-result repeat assignment that retains C5 repeat1 in the test role."""
    if not ordered_files:
        raise ValueError("no raw files in cell")
    if client == 5:
        return ordered_files[-1]
    return ordered_files[cell_index % len(ordered_files)]


def assert_strict_nonoverlap(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = [dict(row) for row in rows if int(row["client_id"]) in TARGETS]
    by_role_identity: dict[tuple[int, str], set[str]] = defaultdict(set)
    by_role_file: dict[tuple[int, str], set[str]] = defaultdict(set)
    for row in rows:
        client = int(row["client_id"])
        role = str(row["role"])
        by_role_identity[(client, role)].add(str(row["physical_identity"]))
        by_role_file[(client, role)].add(str(row["filename"]))
    identity_overlap: list[tuple[int, str]] = []
    file_overlap: list[tuple[int, str]] = []
    for client in TARGETS:
        identity_overlap.extend(
            (client, value)
            for value in by_role_identity[(client, "calibration")] & by_role_identity[(client, "test")]
        )
        file_overlap.extend(
            (client, value)
            for value in by_role_file[(client, "calibration")] & by_role_file[(client, "test")]
        )
    if identity_overlap:
        raise RuntimeError(f"FAIL_CLOSED exact-window overlap: {identity_overlap[:3]}")
    if file_overlap:
        raise RuntimeError(f"FAIL_CLOSED raw-file overlap: {file_overlap[:3]}")
    # Physical time is scoped to a client/raw file; raw-file disjointness makes
    # the calibration/test intersection empty by construction.
    return {
        "exact_window_overlap_count": 0,
        "raw_file_overlap_count": 0,
        "raw_time_overlap_seconds": 0.0,
        "audit_basis": "client_id + raw filename + [window_start_s, window_end_s)",
    }


def load_split_rows(directory: Path, split: str) -> list[dict[str, Any]]:
    features = np.load(directory / f"{split}_features.npy", allow_pickle=False)
    cls = np.load(directory / f"{split}_classification_labels.npy", allow_pickle=False)
    reg = np.load(directory / f"{split}_regression_labels.npy", allow_pickle=False)
    phase = np.load(directory / f"{split}_phase_labels.npy", allow_pickle=False)
    metadata = json.loads((directory / f"{split}_experiment_info.json").read_text(encoding="utf-8"))
    if not (len(features) == len(cls) == len(reg) == len(phase) == len(metadata)):
        raise RuntimeError(f"FAIL_CLOSED parent row alignment: {directory.name}/{split}")
    return [
        {
            **dict(metadata[index]),
            "feature": features[index],
            "classification_label": int(cls[index]),
            "regression_label": reg[index],
            "phase_label": int(phase[index]),
            "parent_role": split,
        }
        for index in range(len(features))
    ]


def save_split(directory: Path, split: str, rows: Sequence[Mapping[str, Any]]) -> None:
    ordered = sorted(rows, key=lambda row: str(row["physical_identity"]))
    np.save(directory / f"{split}_features.npy", np.asarray([row["feature"] for row in ordered], dtype=np.float32))
    np.save(directory / f"{split}_classification_labels.npy", np.asarray([row["classification_label"] for row in ordered], dtype=np.int64))
    np.save(directory / f"{split}_regression_labels.npy", np.asarray([row["regression_label"] for row in ordered], dtype=np.float32).reshape(-1, 4))
    np.save(directory / f"{split}_phase_labels.npy", np.asarray([row["phase_label"] for row in ordered], dtype=np.int64))
    metadata = []
    for row in ordered:
        item = {key: value for key, value in row.items() if key not in {"feature", "regression_label", "phase_label"}}
        item["role"] = split
        metadata.append(item)
    (directory / f"{split}_experiment_info.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _coverage(rows: Sequence[Mapping[str, Any]], role: str) -> dict[str, Any]:
    selected = [row for row in rows if row["strict_role"] == role]
    classes = sorted({int(row["classification_label"]) for row in selected})
    cells = sorted({(int(row["classification_label"]), float(row["concentration"])) for row in selected})
    phases = sorted({int(row["phase_label"]) for row in selected})
    return {"N": len(selected), "classes": classes, "class_concentration_cell_count": len(cells), "phases": phases}


def build_strict_dataset(parent: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"FAIL_CLOSED strict output already exists: {output}")
    output.mkdir(parents=True)

    # Source clients and immutable preprocessing/raw provenance are unchanged.
    for client in (1, 2):
        shutil.copytree(parent / f"client_{client}", output / f"client_{client}")
    for name in (
        "canonical_preprocessing_manifest.json",
        "raw_file_manifest.csv",
        "raw_sha256.json",
        "processing_manifest.csv",
        "window_identity_manifest.csv",
    ):
        shutil.copy2(parent / name, output / name)

    parent_split_rows = read_csv(parent / "split_manifest.csv")
    selected_manifest = [row for row in parent_split_rows if int(row["client_id"]) in (1, 2)]
    assignment_rows: list[dict[str, Any]] = []
    coverage: dict[str, Any] = {}

    for client in TARGETS:
        source_dir = parent / f"client_{client}"
        rows = load_split_rows(source_dir, "calibration") + load_split_rows(source_dir, "test")
        unique = {str(row["physical_identity"]): row for row in rows}
        if len(unique) != len(rows):
            raise RuntimeError(f"FAIL_CLOSED duplicate parent identity C{client}")
        rows = list(unique.values())

        cells: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            cell = (str(int(row["classification_label"])), str(float(row["concentration"])))
            cells[cell].append(row)
        if len(cells) != 40:
            raise RuntimeError(f"FAIL_CLOSED expected 40 class x concentration cells C{client}")
        quotas = allocate_balanced_quotas(cells, TARGET_CALIBRATION_N[client])
        selected: list[dict[str, Any]] = []

        ordered_cells = sorted(cells, key=lambda item: (int(item[0]), float(item[1])))
        for cell_index, cell in enumerate(ordered_cells):
            by_file: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in cells[cell]:
                by_file[str(row["filename"])].append(row)
            ordered_files = sorted(
                by_file,
                key=lambda filename: (int(by_file[filename][0]["repeat_id"]), filename),
            )
            if len(ordered_files) < 2:
                raise RuntimeError(f"FAIL_CLOSED fewer than two repeats C{client}/{cell}")
            calibration_file = choose_calibration_file(client, ordered_files, cell_index)
            calibration_candidates = sorted(by_file[calibration_file], key=lambda row: float(row["window_start_s"]))
            chosen = set(evenly_spaced_indices(len(calibration_candidates), quotas[cell]))
            for index, row in enumerate(calibration_candidates):
                row["strict_role"] = "calibration" if index in chosen else "excluded"
                row["inactive_reason"] = "" if index in chosen else "calibration_file_budget_subsample"
                if index in chosen:
                    selected.append(row)
            for filename in ordered_files:
                if filename == calibration_file:
                    continue
                for row in by_file[filename]:
                    row["strict_role"] = "test"
                    row["inactive_reason"] = ""
                    selected.append(row)

        client_dir = output / f"client_{client}"
        client_dir.mkdir()
        save_split(client_dir, "calibration", [row for row in selected if row["strict_role"] == "calibration"])
        save_split(client_dir, "test", [row for row in selected if row["strict_role"] == "test"])
        counts = {role: sum(row["strict_role"] == role for row in selected) for role in ("calibration", "test")}
        (client_dir / "stats.json").write_text(
            json.dumps({"schema_version": "iotj.strict_nonoverlap.stats.v1", "client_id": f"C{client}", "role": "target", "counts": counts, "n_total_included": sum(counts.values())}, indent=2) + "\n",
            encoding="utf-8",
        )

        for row in rows:
            assignment_rows.append({
                "physical_identity": row["physical_identity"], "client_id": client,
                "filename": row["filename"], "repeat_id": row["repeat_id"],
                "class_id": row["classification_label"], "concentration": row["concentration"],
                "window_start_s": row["window_start_s"], "window_end_s": row["window_end_s"],
                "parent_role": row["parent_role"], "strict_role": row["strict_role"],
                "inactive_reason": row["inactive_reason"],
            })
        for row in selected:
            selected_manifest.append({
                "physical_identity": row["physical_identity"], "client_id": client,
                "filename": row["filename"], "repeat_id": row["repeat_id"],
                "gas": row["gas"], "class_id": row["classification_label"],
                "concentration": row["concentration"], "window_start_s": row["window_start_s"],
                "window_end_s": row["window_end_s"], "role": row["strict_role"],
            })
        coverage[f"C{client}"] = {
            "calibration": _coverage(rows, "calibration"),
            "test": _coverage(rows, "test"),
            "excluded_N": sum(row["strict_role"] == "excluded" for row in rows),
            "available_phases": sorted({int(row["phase_label"]) for row in rows}),
        }

    audit = assert_strict_nonoverlap(selected_manifest)
    for client in TARGETS:
        for role in ("calibration", "test"):
            item = coverage[f"C{client}"][role]
            if item["classes"] != [0, 1, 2, 3] or item["class_concentration_cell_count"] != 40:
                raise RuntimeError(f"FAIL_CLOSED label coverage C{client}/{role}: {item}")
            if item["phases"] != coverage[f"C{client}"]["available_phases"]:
                raise RuntimeError(f"FAIL_CLOSED phase coverage C{client}/{role}: {item}")

    write_csv(output / "split_manifest.csv", selected_manifest)
    write_csv(output / "strict_non_overlap_split_manifest.csv", selected_manifest)
    write_csv(output / "strict_non_overlap_assignment_manifest.csv", assignment_rows)
    parent_hash = json.loads((parent / "dataset_sha256.json").read_text(encoding="utf-8"))["aggregate_sha256"]
    protocol = {
        "schema_version": "iotj.canonical_v1.strict_nonoverlap.protocol.v1",
        "status": "FROZEN",
        "parent_dataset_sha256": parent_hash,
        "preprocessing": "HZ5_MEAN_W10S_UNCHANGED",
        "assignment": "C3/C4 cycle calibration repeat by ordered class-concentration cell; C5 uses highest repeat; remaining repeats are test files",
        "calibration_subsampling": "deterministic midpoint-quantile windows within calibration-only raw files",
        "target_calibration_N": {f"C{k}": v for k, v in TARGET_CALIBRATION_N.items()},
        "split_seed": None,
        "target_test_used_for_split_selection": False,
        "c5_methane_225_repeat1": "retained_in_test_by_highest-repeat calibration rule",
        "audit": audit,
        "coverage": coverage,
        "limitations": ["C4/C5 test N is reduced because two-repeat raw-file grouping cannot realize an 80% raw-file-disjoint test split"],
    }
    (output / "strict_non_overlap_protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    files: dict[str, str] = {}
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "dataset_sha256.json":
            files[str(path.relative_to(output)).replace("\\", "/")] = sha256(path)
    aggregate = hashlib.sha256()
    for name, digest in sorted(files.items()):
        aggregate.update(name.encode()); aggregate.update(b"\0"); aggregate.update(digest.encode()); aggregate.update(b"\n")
    dataset_hash = {"schema_version": "iotj.canonical_v1.strict_nonoverlap.sha256.v1", "aggregate_sha256": aggregate.hexdigest(), "files": files}
    (output / "dataset_sha256.json").write_text(json.dumps(dataset_hash, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"output": str(output), "aggregate_sha256": dataset_hash["aggregate_sha256"], "audit": audit, "coverage": coverage}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(build_strict_dataset(args.parent.resolve(), args.output.resolve()), indent=2))


if __name__ == "__main__":
    main()
