from __future__ import annotations

import numpy as np
import pytest
import torch

from gaps_deploy.calibration import RegressionCalibrator
from gaps_deploy.deploy_config import DeployConfig
from gaps_deploy.inference import DeployPredictor, normalize_phase_ids
from gaps_deploy.qc_policy import RiskScoreComputer, TwoThresholdDecider


class _Classifier(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        logits = torch.zeros((len(x), 4), dtype=x.dtype, device=x.device)
        logits[:, 0] = 5.0
        features = x.mean(dim=1)
        return logits, features, features


class _PhaseRecordingRegressor(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model_phases: list[int] = []

    def forward(self, x: torch.Tensor):
        features = x.mean(dim=1)
        return torch.empty(0), torch.empty(0), features

    def forward_reg(
        self,
        features: torch.Tensor,
        y_cls: torch.Tensor,
        y_phase: torch.Tensor,
    ) -> torch.Tensor:
        phases = y_phase.detach().cpu().reshape(-1).tolist()
        self.model_phases.extend(int(value) for value in phases)
        return 0.25 + 0.10 * y_phase.to(dtype=features.dtype)


def _predictor() -> tuple[DeployPredictor, _PhaseRecordingRegressor]:
    regressor = _PhaseRecordingRegressor()
    predictor = DeployPredictor(
        model_A=_Classifier(),
        model_B=regressor,
        calibrator=RegressionCalibrator(num_classes=4, num_phases=3),
        risk_computer=RiskScoreComputer(),
        qc_decider=TwoThresholdDecider(),
        config=DeployConfig(num_phases=3),
    )
    return predictor, regressor


def _loader(phase: int) -> torch.utils.data.DataLoader:
    dataset = torch.utils.data.TensorDataset(
        torch.zeros((1, 2, 2), dtype=torch.float32),
        torch.zeros(1, dtype=torch.long),
        torch.zeros((1, 4), dtype=torch.float32),
        torch.full((1,), phase, dtype=torch.long),
    )
    return torch.utils.data.DataLoader(dataset, batch_size=1)


def test_unknown_phase_uses_zero_for_batch_and_generator_but_preserves_raw_value() -> None:
    predictor, recorder = _predictor()
    features = np.zeros((1, 2, 2), dtype=np.float32)

    batch = predictor.predict_batch(features, phase=-1)[0]
    streamed = next(predictor.predict_generator(_loader(-1)))[0]

    assert batch.phase == -1
    assert streamed.phase == -1
    assert batch.final_ppm == pytest.approx(streamed.final_ppm)
    assert recorder.model_phases == [0, 0]


@pytest.mark.parametrize("phase", [-2, 3, 1.5, True])
def test_invalid_scalar_phase_is_rejected(phase) -> None:
    predictor, _ = _predictor()

    with pytest.raises(ValueError, match="phase"):
        predictor.predict_batch(np.zeros((1, 2, 2), dtype=np.float32), phase=phase)


def test_invalid_generator_phase_is_rejected() -> None:
    predictor, _ = _predictor()

    with pytest.raises(ValueError, match="phase"):
        next(predictor.predict_generator(_loader(3)))


def test_normalize_phase_ids_preserves_raw_and_returns_model_safe_values() -> None:
    raw, model = normalize_phase_ids(
        np.asarray([-1, 0, 2], dtype=np.int64),
        n_samples=3,
        num_phases=3,
    )

    assert raw.tolist() == [-1, 0, 2]
    assert model.tolist() == [0, 0, 2]
