from __future__ import annotations

from collections import Counter

import pytest

from scripts.evaluate_iotj_low_calibration_sensitivity import (
    assign_group_folds,
    classify_sensitivity,
    select_nested_group_subsets,
)


def _metadata() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    sizes = [1, 2, 3, 4, 5, 6, 7, 4, 4, 4, 4, 4]
    for group_id, size in enumerate(sizes):
        for _ in range(size):
            rows.append(
                {
                    "filename": f"file_{group_id:02d}.txt",
                    "classification_label": group_id % 4,
                    "concentration": float(25 * (1 + group_id % 5)),
                }
            )
    return rows


def test_nested_subsets_are_deterministic_and_group_complete() -> None:
    metadata = _metadata()
    first = select_nested_group_subsets(metadata, (40, 24, 12), seed=42)
    second = select_nested_group_subsets(metadata, (40, 24, 12), seed=42)
    assert first == second
    assert set(first[12]) <= set(first[24]) <= set(first[40])
    by_file = {}
    for index, row in enumerate(metadata):
        by_file.setdefault(row["filename"], set()).add(index)
    for indexes in first.values():
        chosen = set(indexes)
        for group in by_file.values():
            assert not (chosen & group) or group <= chosen


def test_group_folds_do_not_split_filename_and_are_deterministic() -> None:
    metadata = _metadata()
    indexes = select_nested_group_subsets(metadata, (40,), seed=7)[40]
    folds = assign_group_folds(metadata, indexes, n_splits=5, seed=7)
    assert folds == assign_group_folds(metadata, indexes, n_splits=5, seed=7)
    assert set(folds) == set(indexes)
    assert set(folds.values()) == set(range(5))
    observed: dict[str, set[int]] = {}
    for index, fold in folds.items():
        observed.setdefault(str(metadata[index]["filename"]), set()).add(fold)
    assert all(len(values) == 1 for values in observed.values())


@pytest.mark.parametrize(
    ("deltas", "expected"),
    [
        ({160: 0.01, 80: 0.04, 40: 0.05}, "ROBUST_TO_REDUCED_CALIBRATION"),
        ({160: 0.03, 80: 0.08, 40: 0.30}, "MODERATE_CALIBRATION_SENSITIVITY"),
        ({160: 0.11, 80: 0.05, 40: 0.02}, "HIGH_CALIBRATION_SENSITIVITY"),
    ],
)
def test_sensitivity_rule_is_frozen(
    deltas: dict[int, float], expected: str
) -> None:
    assert classify_sensitivity(deltas) == expected


def test_fold_assignment_balances_group_count() -> None:
    metadata = _metadata()
    folds = assign_group_folds(metadata, range(len(metadata)), n_splits=5, seed=9)
    counts = Counter(folds.values())
    assert max(counts.values()) - min(counts.values()) <= 8
