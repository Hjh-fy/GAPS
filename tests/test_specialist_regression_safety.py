from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from gaps_flower.specialist_calibration_fit import (
    InsufficientValidationDataError,
    _build_auto_v2_routing,
    _collect_deployable_predictions,
    _gate_accept,
    _score,
    _selection_provenance,
    _split_loader,
)


class _CalibrationDataset(Dataset):
    def __init__(self, classes: list[int], concentrations: list[float]) -> None:
        self.classification_labels = np.asarray(classes, dtype=np.int64)
        self.regression_labels = np.zeros((len(classes), 4), dtype=np.float32)
        for index, (class_id, concentration) in enumerate(zip(classes, concentrations)):
            self.regression_labels[index, class_id] = concentration
        self.phase_labels = np.zeros(len(classes), dtype=np.int64)
        self.features = np.zeros((len(classes), 2, 2), dtype=np.float32)

    def __len__(self) -> int:
        return len(self.classification_labels)

    def __getitem__(self, index: int):
        return (
            torch.from_numpy(self.features[index]),
            torch.tensor(self.classification_labels[index]),
            torch.from_numpy(self.regression_labels[index]),
            torch.tensor(self.phase_labels[index]),
        )


def _subset_indices(loader: DataLoader) -> set[int]:
    assert isinstance(loader.dataset, Subset)
    return {int(index) for index in loader.dataset.indices}


def test_class_concentration_split_is_disjoint_and_complete() -> None:
    loader = DataLoader(
        _CalibrationDataset([0, 0, 1, 1], [10.0, 20.0, 30.0, 40.0]),
        batch_size=2,
    )

    train, val = _split_loader(loader, 2, 0.25, 42, "class_concentration")
    train_ids = _subset_indices(train)
    val_ids = _subset_indices(val)

    assert train_ids.isdisjoint(val_ids)
    assert train_ids | val_ids == set(range(4))


def test_split_rejects_dataset_without_independent_validation_capacity() -> None:
    loader = DataLoader(_CalibrationDataset([0], [10.0]), batch_size=1)

    with pytest.raises(InsufficientValidationDataError):
        _split_loader(loader, 1, 0.25, 42, "class_concentration")


def _candidate_metrics(value) -> dict:
    return {
        mode: {
            "per_class": {
                str(class_id): {"R2": value}
                for class_id in range(4)
            }
        }
        for mode in ["none", "bias_only", "affine_only", "phase_affine_only", "full"]
    }


@pytest.mark.parametrize("value", [None, 0.5])
def test_missing_or_tied_metrics_keep_none(value) -> None:
    routing, diagnostics = _build_auto_v2_routing(
        _candidate_metrics(value), {}, {}, {}, "R2", 0.0, 4
    )

    assert set(routing["selected_modes"].values()) == {"none"}
    assert diagnostics["selection_available"] == {
        str(class_id): False for class_id in range(4)
    }


@pytest.mark.parametrize("value", [None, np.nan, np.inf, -np.inf])
def test_score_returns_none_for_missing_or_nonfinite_metrics(value) -> None:
    metrics = {"per_class": {"0": {"R2": value}}}

    assert _score(metrics, 0, "R2") is None


def test_gate_fails_closed_for_missing_scores_and_guard_metrics() -> None:
    args = SimpleNamespace(
        min_delta=0.0,
        gate_mode="guarded",
        use_p90_guard=True,
        p90_max_worsen=0.0,
        use_bias_guard=True,
        bias_max_worsen=0.0,
    )

    accepted, details = _gate_accept({}, {}, None, None, args)

    assert accepted is False
    assert details["primary_ok"] is False
    assert details["p90_ok"] is False
    assert details["bias_ok"] is False


class _AlwaysClassOne(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        logits = torch.zeros((len(x), 4), device=x.device)
        logits[:, 1] = 10.0
        features = x.mean(dim=1)
        return logits, features, features


class _RouteRecordingRegressor(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.route_classes: list[int] = []

    def forward(self, x: torch.Tensor):
        features = x.mean(dim=1)
        return torch.empty(0), torch.empty(0), features

    def forward_reg(self, features, y_cls, y_phase):
        self.route_classes.extend(y_cls.detach().cpu().tolist())
        return torch.full((len(features),), 0.5, device=features.device)


def test_deployable_predictions_use_predicted_route_but_true_concentration() -> None:
    dataset = _CalibrationDataset([0], [20.0])
    loader = DataLoader(dataset, batch_size=1)
    regressor = _RouteRecordingRegressor()

    true_ppm, pred_ppm, true_class, route_class, phase = (
        _collect_deployable_predictions(
            _AlwaysClassOne(), regressor, loader, torch.device("cpu")
        )
    )

    assert true_ppm.tolist() == [20.0]
    assert pred_ppm.tolist() == pytest.approx([137.5])
    assert true_class.tolist() == [0]
    assert route_class.tolist() == [1]
    assert phase.tolist() == [0]
    assert regressor.route_classes == [1]


def test_pre_refit_selection_metrics_are_frozen_from_deployment_refit() -> None:
    pre_refit = {"per_class": {"1": {"R2": 0.75}}}

    audit = _selection_provenance(
        pre_refit,
        deployment_models_refit=True,
    )
    pre_refit["per_class"]["1"]["R2"] = -100.0

    assert audit["selection_metrics_source"] == "pre_refit_independent_validation"
    assert audit["deployment_models_refit_on_full_calibration"] is True
    assert audit["selected_pre_refit_val_metrics"]["per_class"]["1"]["R2"] == 0.75
