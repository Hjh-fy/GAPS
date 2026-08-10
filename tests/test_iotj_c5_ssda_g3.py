from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest
import torch


def _rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for class_id in range(4):
        for concentration_index in range(10):
            concentration = float(25 * (concentration_index + 1))
            for window_index in range(8):
                rows.append(
                    {
                        "physical_identity": (
                            f"C5|c{class_id}|q{concentration_index}|w{window_index}"
                        ),
                        "client_id": 5,
                        "role": "calibration",
                        "class_id": class_id,
                        "classification_label": class_id,
                        "concentration": concentration,
                    }
                )
    return rows


def test_unlabeled_dataset_exposes_only_x_and_identity() -> None:
    from gaps_flower.ssda import UnlabeledTargetDataset

    dataset = UnlabeledTargetDataset(
        np.zeros((3, 50, 8), dtype=np.float32),
        ("u0", "u1", "u2"),
    )
    sample = dataset[0]
    assert set(sample) == {"x", "identity"}
    assert sample["identity"] == "u0"
    assert not hasattr(dataset, "labels")
    assert not hasattr(dataset, "y_true")
    assert not hasattr(dataset, "phases")


def test_g3_request_rejects_target_test_from_training_api(tmp_path: Path) -> None:
    from gaps_flower.ssda import G3Request

    with pytest.raises(ValueError, match="test manifest"):
        G3Request(
            source_checkpoint=tmp_path / "source.pth",
            calibration_manifest=tmp_path / "calibration.json",
            labeled_manifest=tmp_path / "labeled.json",
            target_test_manifest=tmp_path / "test.json",
        ).validate_static_boundary()

    fields = {field.name for field in dataclasses.fields(G3Request)}
    assert "target_test_labels" not in fields
    assert "unlabeled_labels" not in fields


def test_partition_is_80_labeled_240_unlabeled_and_stratified(tmp_path: Path) -> None:
    from gaps_flower.ssda import build_g3_partition

    calibration = _rows()
    labeled = [row for index, row in enumerate(calibration) if index % 8 < 2]
    calibration_path = tmp_path / "calibration.json"
    labeled_path = tmp_path / "labeled.json"
    calibration_path.write_text(json.dumps(calibration), encoding="utf-8")
    labeled_path.write_text(json.dumps(labeled), encoding="utf-8")
    partition = build_g3_partition(calibration_path, labeled_path)

    assert len(partition.labeled_indices) == 80
    assert len(partition.unlabeled_indices) == 240
    assert set(partition.labeled_indices).isdisjoint(partition.unlabeled_indices)
    assert set(partition.labeled_indices) | set(partition.unlabeled_indices) == set(
        range(320)
    )
    assert set(partition.labeled_stratum_counts.values()) == {2}
    assert set(partition.unlabeled_stratum_counts.values()) == {6}


def test_deterministic_two_fold_uses_one_labeled_window_per_stratum(tmp_path: Path) -> None:
    from gaps_flower.ssda import build_g3_partition, deterministic_two_fold

    calibration = _rows()
    labeled = [row for index, row in enumerate(calibration) if index % 8 < 2]
    calibration_path = tmp_path / "calibration.json"
    labeled_path = tmp_path / "labeled.json"
    calibration_path.write_text(json.dumps(calibration), encoding="utf-8")
    labeled_path.write_text(json.dumps(labeled), encoding="utf-8")
    partition = build_g3_partition(calibration_path, labeled_path)
    folds = deterministic_two_fold(partition)

    assert len(folds) == 2
    assert all(len(fold.train_indices) == 40 for fold in folds)
    assert all(len(fold.validation_indices) == 40 for fold in folds)
    assert all(
        set(fold.train_indices).isdisjoint(fold.validation_indices) for fold in folds
    )
    assert set(folds[0].train_indices) == set(folds[1].validation_indices)
    assert set(folds[1].train_indices) == set(folds[0].validation_indices)


def test_g3_hyperparameters_and_search_space_are_bounded() -> None:
    from gaps_flower.ssda import G3Config

    config = G3Config()
    assert config.steps == 100
    assert config.optimizer == "Adam"
    assert config.lr == 5e-4
    assert config.seed == 42
    assert config.ema_alpha == 0.99
    assert config.lambda_proto == 0.05
    assert config.mme_lambda == 0.1
    assert config.taus == (0.90, 0.95)
    assert config.lambda_us == (0.25, 0.5, 1.0)
    assert len(config.grid()) == 6


def test_gradient_reversal_flips_feature_gradient_only() -> None:
    from gaps_flower.ssda import gradient_reverse

    feature = torch.tensor([[1.0, 2.0]], requires_grad=True)
    weight = torch.tensor([[3.0, 4.0]], requires_grad=True)
    loss = (gradient_reverse(feature) @ weight.T).sum()
    loss.backward()
    assert torch.equal(feature.grad, -weight.detach())
    assert torch.equal(weight.grad, feature.detach())


def test_ema_teacher_update_is_frozen_weighted_average() -> None:
    from gaps_flower.ssda import update_ema_teacher

    student = torch.nn.Linear(2, 1, bias=False)
    teacher = torch.nn.Linear(2, 1, bias=False)
    student.weight.data.fill_(2.0)
    teacher.weight.data.fill_(1.0)
    update_ema_teacher(teacher, student, alpha=0.99)
    assert torch.allclose(teacher.weight, torch.full_like(teacher.weight, 1.01))
    assert all(not parameter.requires_grad for parameter in teacher.parameters())


def test_gate3_decision_thresholds_are_pre_registered() -> None:
    from scripts.run_iotj_c5_ssda_g3 import decide_gate3

    assert decide_gate3(a0t_f1=0.90, mme_f1=0.93, gaps_f1=0.94)["decision"] == (
        "SSDA_COMPONENT_SUPPORTED"
    )
    assert decide_gate3(a0t_f1=0.90, mme_f1=0.94, gaps_f1=0.92)["decision"] == (
        "MME_DOMINATES"
    )
    assert decide_gate3(a0t_f1=0.930, mme_f1=0.932, gaps_f1=0.931)[
        "decision"
    ] == "NO_SSDA_SPACE"
