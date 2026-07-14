import importlib
import subprocess
import sys
from pathlib import Path

import pytest


def _summary():
    return importlib.import_module("scripts.summarize_iotj_cross_direction_classification")


def _row(
    sample_index: int,
    true_class: int,
    pred_class: int,
    probabilities: tuple[float, float, float, float],
) -> dict[str, object]:
    row: dict[str, object] = {
        "client": "C1",
        "split": "test",
        "sample_index": sample_index,
        "true_class": true_class,
        "pred_class": pred_class,
    }
    for class_id, probability in enumerate(probabilities):
        row[f"prob_{class_id}"] = probability
    return row


B2_ROWS = [
    _row(0, 0, 0, (0.9, 0.05, 0.03, 0.02)),
    _row(1, 1, 1, (0.05, 0.9, 0.03, 0.02)),
    _row(2, 2, 2, (0.02, 0.03, 0.9, 0.05)),
    _row(3, 3, 0, (0.6, 0.1, 0.1, 0.2)),
]
B5_ROWS = [
    _row(0, 0, 0, (0.8, 0.1, 0.05, 0.05)),
    _row(1, 1, 1, (0.1, 0.8, 0.05, 0.05)),
    _row(2, 2, 0, (0.6, 0.1, 0.2, 0.1)),
    _row(3, 3, 0, (0.5, 0.1, 0.1, 0.3)),
]


def test_paired_comparison_uses_identical_row_keys() -> None:
    summary = _summary()

    result = summary.compare_streams(
        B2_ROWS, B5_ROWS, bootstrap_seed=20260713, bootstrap_reps=200
    )

    assert result["N"] == 4
    assert result["accuracy_delta_pp"] == pytest.approx(25.0)
    assert result["b2_only_correct"] == 1
    assert result["b5_only_correct"] == 0
    assert 0.0 <= result["mcnemar_exact_p"] <= 1.0
    assert result["accuracy_delta_pp_ci_low"] <= result["accuracy_delta_pp"]
    assert result["accuracy_delta_pp_ci_high"] >= result["accuracy_delta_pp"]


def test_paired_comparison_rejects_misaligned_rows() -> None:
    summary = _summary()

    with pytest.raises(ValueError, match="row keys"):
        summary.compare_streams(B2_ROWS, list(reversed(B5_ROWS)))


def test_noninferiority_rule_requires_accuracy_macro_f1_and_worst_recall() -> None:
    summary = _summary()

    assert summary.classify_direction(
        accuracy_delta_pp=-0.4,
        macro_f1_delta_pp=-0.3,
        worst_recall_delta_pp=-0.2,
        margin_pp=0.5,
    ) == "B2_noninferior"
    assert summary.classify_direction(
        accuracy_delta_pp=-0.6,
        macro_f1_delta_pp=-0.2,
        worst_recall_delta_pp=-0.2,
        margin_pp=0.5,
    ) == "B5_favored"


def test_evaluation_count_contract_rejects_wrong_target_split() -> None:
    summary = _summary()
    manifest = {
        "run_name": "B2_r1",
        "protocol": {
            "expected_target_counts": {"calibration": 680, "test": 2680}
        },
    }
    payload = {
        "metrics": {
            "calibration": {"N": 320},
            "test": {"N": 680},
        }
    }

    with pytest.raises(ValueError, match="B2_r1: test N=680, expected 2680"):
        summary.validate_evaluation_counts(payload, manifest)


def test_evaluation_count_contract_accepts_manifest_counts() -> None:
    summary = _summary()
    manifest = {
        "run_name": "B2_r1",
        "protocol": {
            "expected_target_counts": {"calibration": 680, "test": 2680}
        },
    }
    payload = {
        "metrics": {
            "calibration": {"N": 680},
            "test": {"N": 2680},
        }
    }

    summary.validate_evaluation_counts(payload, manifest)


def test_summary_can_filter_the_frozen_queue_by_direction() -> None:
    summary = _summary()
    manifests = [
        (Path("f1_b2.json"), {"direction_id": "F1_C1_TO_C5", "group_id": "B2"}),
        (Path("r1_b2.json"), {"direction_id": "R1_C5_TO_C1", "group_id": "B2"}),
        (Path("r1_b5.json"), {"direction_id": "R1_C5_TO_C1", "group_id": "B5"}),
    ]

    selected = summary.filter_manifests(
        manifests, directions={"R1_C5_TO_C1"}, groups={"B2", "B5"}
    )

    assert [path.name for path, _manifest in selected] == ["r1_b2.json", "r1_b5.json"]


def test_summary_cli_help_runs_from_repo_root() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        [
            sys.executable,
            "scripts/summarize_iotj_cross_direction_classification.py",
            "--help",
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--command-root" in result.stdout
