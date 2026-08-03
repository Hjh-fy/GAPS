from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

import scripts.run_iotj_p0_zero_label_commissioning as p0u


class TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.feature = torch.nn.Linear(8, 64)
        self.classifier = torch.nn.Linear(64, 4)

    def forward(self, x: torch.Tensor):
        feat = torch.nn.functional.normalize(self.feature(x.mean(dim=1)), dim=1)
        return self.classifier(feat), feat, feat


def source_loader() -> DataLoader:
    x = torch.randn(8, 100, 8)
    y = torch.arange(8) % 4
    return DataLoader(TensorDataset(x, y, torch.zeros(8, 4), torch.zeros(8, dtype=torch.long)), batch_size=4)


def target_loader() -> DataLoader:
    return DataLoader(torch.randn(8, 100, 8), batch_size=4)


def test_feature_only_dataset_loads_no_target_labels(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(np, "load", lambda path, **_kwargs: calls.append(Path(path).name) or np.zeros((320, 100, 8), np.float32))
    dataset = p0u.FeatureOnlyCalibrationDataset(tmp_path)
    assert calls == ["calibration_features.npy"]
    assert isinstance(dataset[0], torch.Tensor)


def test_runtime_guard_rejects_label_bearing_target_batch() -> None:
    with pytest.raises(RuntimeError, match="non-feature object"):
        p0u.require_x_only((torch.zeros(2, 100, 8), torch.zeros(2)), method="test")


def test_static_label_access_audit_passes() -> None:
    result = p0u.static_label_access_audit()
    assert result["status"] == "PASS"
    assert all(result["checks"].values())


def test_u1_disables_all_target_conditioned_losses(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(p0u, "STEPS", 1)
    model, diagnostics, _seconds = p0u.run_unsupervised_global_alignment(TinyModel(), source_loader(), target_loader(), torch.device("cpu"))
    assert isinstance(model, TinyModel)
    row = diagnostics[0]
    assert row["target_label_object_present"] is False
    assert row["target_ce_status"] == "UNAVAILABLE"
    assert row["class_conditional_coral_status"] == "DISABLED"
    assert row["class_mmd_status"] == "DISABLED"
    assert row["stage_mmd_status"] == "DISABLED"
    assert row["target_proto_anchor_status"] == "UNAVAILABLE"


def test_u2_pseudo_labels_come_from_frozen_teacher_at_fixed_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(p0u, "STEPS", 1)
    model = TinyModel()
    with torch.no_grad():
        model.classifier.weight.zero_(); model.classifier.bias.copy_(torch.tensor([8.0, 0.0, 0.0, 0.0]))
    _student, teacher, diagnostics, _seconds = p0u.run_pseudo_label_self_training(model, target_loader(), torch.device("cpu"))
    assert all(not parameter.requires_grad for parameter in teacher.parameters())
    row = diagnostics[0]
    assert row["threshold"] == 0.90
    assert row["coverage"] == 1.0
    assert row["pseudo_class_0"] == row["selected_count"]
    assert row["pseudo_label_origin"] == "teacher_argmax_only"
    assert row["target_label_object_present"] is False


def test_both_branches_leave_original_source_model_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(p0u, "STEPS", 1)
    source = TinyModel(); before = {key: value.detach().clone() for key, value in source.state_dict().items()}
    p0u.run_unsupervised_global_alignment(source, source_loader(), target_loader(), torch.device("cpu"))
    p0u.run_pseudo_label_self_training(source, target_loader(), torch.device("cpu"))
    assert all(torch.equal(before[key], value) for key, value in source.state_dict().items())


def test_test_split_opens_after_posthoc_and_both_training() -> None:
    source = inspect.getsource(p0u.main)
    assert source.index('"u2_training_completed"') < source.index("posthoc_pseudo_precision")
    assert source.index("posthoc_pseudo_precision") < source.index('make_loader(data_root, 5, "test"')


def test_no_hyperparameter_search_contract() -> None:
    assert p0u.SEED == 42
    assert p0u.STEPS == 100
    assert p0u.MODEL_LR == 5e-4
    assert p0u.PSEUDO_THRESHOLD == 0.90
    assert p0u.U1_WEIGHTS == {"source_ce": 1.0, "coral": 0.5, "global_mmd2": 0.5, "adversarial": 0.5}
