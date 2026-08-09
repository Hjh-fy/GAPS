import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.prepare_iotj_c5_label_budget import build_nested_indices, prepare


def synthetic_info() -> list[dict]:
    rows = []
    index = 0
    for class_id in range(4):
        for level in range(10):
            concentration = float((level + 1) * 10)
            for repeat_id in (1, 2):
                filename = f"C{class_id}_L{level}_R{repeat_id}.txt"
                for window in range(4):
                    rows.append({
                        "client_id": 5,
                        "filename": filename,
                        "repeat_id": repeat_id,
                        "gas": f"gas_{class_id}",
                        "gas_code": f"G{class_id}",
                        "class_id": class_id,
                        "classification_label": class_id,
                        "concentration": concentration,
                        "physical_window_start_s": float(60 + 5 * window),
                        "physical_window_end_s": float(70 + 5 * window),
                        "phase_label": 2,
                        "physical_identity": f"id-{index}",
                    })
                    index += 1
    return rows


def test_nested_indices_keep_exact_balanced_counts_and_repeat_diversity() -> None:
    info = synthetic_info()
    nested = build_nested_indices(info)

    assert {budget: len(indices) for budget, indices in nested.items()} == {
        20: 320,
        15: 240,
        10: 160,
        5: 80,
    }
    assert set(nested[5]) < set(nested[10]) < set(nested[15]) < set(nested[20])
    for budget, per_stratum in ((20, 8), (15, 6), (10, 4), (5, 2)):
        selected = [info[index] for index in nested[budget]]
        counts: dict[tuple[int, float], int] = {}
        for row in selected:
            key = (row["class_id"], row["concentration"])
            counts[key] = counts.get(key, 0) + 1
        assert len(counts) == 40
        assert set(counts.values()) == {per_stratum}
    for class_id in range(4):
        for level in range(10):
            chosen = [info[index] for index in nested[5] if info[index]["class_id"] == class_id and info[index]["concentration"] == float((level + 1) * 10)]
            assert {row["repeat_id"] for row in chosen} == {1, 2}


def test_nested_indices_fail_closed_if_canonical_stratum_is_missing() -> None:
    rows = synthetic_info()
    replacement = []
    for offset, row in enumerate(rows[:8]):
        replacement.append({**row, "physical_identity": f"replacement-{offset}"})
    with pytest.raises(ValueError, match="40 strata"):
        build_nested_indices(rows[:-8] + replacement)


def test_prepare_indexes_existing_features_without_copying_test_arrays(tmp_path: Path) -> None:
    source = tmp_path / "client_5"
    source.mkdir()
    info = synthetic_info()
    features = np.arange(320 * 50 * 8, dtype=np.float32).reshape(320, 50, 8)
    classes = np.asarray([row["class_id"] for row in info], dtype=np.int64)
    phases = np.asarray([row["phase_label"] for row in info], dtype=np.int64)
    regression = np.zeros((320, 4), dtype=np.float32)
    np.save(source / "calibration_features.npy", features)
    np.save(source / "calibration_classification_labels.npy", classes)
    np.save(source / "calibration_phase_labels.npy", phases)
    np.save(source / "calibration_regression_labels.npy", regression)
    (source / "calibration_experiment_info.json").write_text(json.dumps(info), encoding="utf-8")
    (source / "test_features.npy").write_bytes(b"sealed-test-sentinel")
    test_info = [{"physical_identity": "sealed-test-id"}]
    (source / "test_experiment_info.json").write_text(json.dumps(test_info), encoding="utf-8")

    output = tmp_path / "study"
    audit = prepare(source, output)

    assert audit["status"] == "PASS"
    assert audit["counts"] == {"20": 320, "15": 240, "10": 160, "5": 80}
    budget5 = output / "budget_data/client_5_budget_05"
    observed = np.load(budget5 / "calibration_features.npy", allow_pickle=False)
    assert observed.shape == (80, 50, 8)
    assert not (budget5 / "test_features.npy").exists()
    assert not (budget5 / "test_experiment_info.json").exists()
    manifest_rows = list(csv.DictReader((output / "c5_calibration_budget_05pct.csv").open(encoding="utf-8")))
    assert len(manifest_rows) == 80
    assert {row["budget_membership"] for row in manifest_rows} == {"05;10;15;20"}
    index = json.loads((output / "c5_calibration_budget_manifest_sha256.json").read_text(encoding="utf-8"))
    manifest_sha = hashlib.sha256((output / "c5_calibration_budget_05pct.csv").read_bytes()).hexdigest()
    assert index["files"]["c5_calibration_budget_05pct.csv"] == manifest_sha


def test_prepare_refuses_existing_destination(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    output = tmp_path / "study"
    output.mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        prepare(source, output)
