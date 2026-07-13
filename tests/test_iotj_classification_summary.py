import numpy as np
import pytest
from pathlib import Path

from scripts.summarize_iotj_classification_ablation import (
    _run_identity,
    aggregate_groups,
    classification_metrics,
    validate_confirmation_seeds,
    validate_expected_groups,
)


@pytest.mark.parametrize(
    ("run_name", "expected"),
    [
        ("A0_ce_only_no_da_c12_to_c5_s42_r25", ("A0", 42)),
        ("A0T_ce_only_server_da_c12_to_c5_s43_r25", ("A0T", 43)),
        ("A4S_align_replay_no_da_c12_to_c5_s44_r25", ("A4S", 44)),
        ("A7_proto_replay_full_da_c12_to_c5_s46_r25", ("A7", 46)),
        ("B1_proto_replay_corrected_server_da_c12_to_c5_s42_r25", ("B1", 42)),
        ("B5_proto_replay_corrected_full_da_c12_to_c5_s45_r25", ("B5", 45)),
    ],
)
def test_run_identity_accepts_all_scheduled_group_id_shapes(
    run_name: str, expected: tuple[str, int]
) -> None:
    assert _run_identity(Path(run_name)) == expected


def test_classification_metrics_reports_macro_f1_nll_ece_and_recall() -> None:
    true = [0, 0, 1, 1]
    probs = np.asarray(
        [
            [0.9, 0.05, 0.03, 0.02],
            [0.4, 0.5, 0.05, 0.05],
            [0.1, 0.8, 0.05, 0.05],
            [0.1, 0.7, 0.1, 0.1],
        ],
        dtype=np.float64,
    )

    metrics = classification_metrics(true, probs, ece_bins=5)

    assert metrics["N"] == 4
    assert metrics["accuracy"] == 0.75
    assert 0.0 < metrics["macro_f1"] < 1.0
    assert metrics["nll"] > 0.0
    assert metrics["ece"] >= 0.0
    assert metrics["per_class_recall"]["0"] == 0.5
    assert metrics["per_class_recall"]["1"] == 1.0
    assert metrics["confusion_matrix"] == [[1, 1, 0, 0], [0, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]


def test_group_aggregation_uses_seed_mean_and_sample_std() -> None:
    rows = [
        {
            "group_id": "A7",
            "seed": 42,
            "accuracy": 0.9,
            "macro_f1": 0.8,
            "nll": 0.2,
            "ece": 0.1,
            "mean_confidence": 0.85,
        },
        {
            "group_id": "A7",
            "seed": 43,
            "accuracy": 1.0,
            "macro_f1": 1.0,
            "nll": 0.1,
            "ece": 0.05,
            "mean_confidence": 0.95,
        },
    ]

    summary = aggregate_groups(rows)[0]

    assert summary["num_seeds"] == 2
    assert summary["seeds"] == "42,43"
    assert summary["accuracy_mean"] == pytest.approx(0.95)
    assert summary["accuracy_std"] > 0.0


def test_confirmation_seed_validation_rejects_missing_seed() -> None:
    rows = [
        {"group_id": group, "seed": seed}
        for group in ("A0", "A0T", "A4", "A4S", "A5", "A7")
        for seed in (42, 43, 44, 45, 46)
        if not (group == "A7" and seed == 46)
    ]

    with pytest.raises(ValueError, match="A7 confirmation seeds"):
        validate_confirmation_seeds(rows)


def test_expected_group_validation_rejects_silently_missing_b_group() -> None:
    rows = [{"group_id": group, "seed": 42} for group in ("B1", "B2", "B3", "B5")]

    with pytest.raises(ValueError, match="missing expected groups: B4"):
        validate_expected_groups(rows, ("B1", "B2", "B3", "B4", "B5"))
