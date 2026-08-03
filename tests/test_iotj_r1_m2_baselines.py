from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from client import Client
from gaps_flower.task import make_config
from scripts.run_iotj_r1_m2_local_baselines import apply_ds, fit_ds_mapping


class TinyClassifier(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(3, 2)

    def forward(self, x: torch.Tensor):
        features = x.mean(dim=1)
        return self.linear(features), features, features


def test_fedprox_penalty_is_active_after_first_local_update() -> None:
    config = make_config(
        device="cpu",
        local_epochs=2,
        batch_size=2,
        profile="ce_only",
        seed=42,
        proximal_mu=0.01,
    )
    config.NUM_CLASSES = 2
    model = TinyClassifier()
    x = torch.arange(24, dtype=torch.float32).view(4, 2, 3) / 24.0
    y = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    reg = torch.zeros((4, 4), dtype=torch.float32)
    phase = torch.zeros(4, dtype=torch.long)
    client = Client(client_id=1, config=config)
    client.set_model(model)
    client.update_dataloader(DataLoader(TensorDataset(x, y, reg, phase), batch_size=2))

    client.train_one_round(current_round=1)

    assert client.last_fedprox_penalty > 0.0


def test_direct_standardization_recovers_affine_target_to_source_map() -> None:
    rng = np.random.default_rng(42)
    true_mapping = np.vstack(
        [np.diag([1.2, 0.8]), np.asarray([[0.3, -0.2]])]
    )
    keys = [(idx % 2, f"{idx:03d}", "early") for idx in range(8)]
    target = {key: rng.normal(size=(5, 2)) for key in keys}
    source = {key: apply_ds(value[None, ...], true_mapping)[0] for key, value in target.items()}

    fitted = fit_ds_mapping(target, source, keys, alpha=1e-10)

    np.testing.assert_allclose(fitted, true_mapping, rtol=0.0, atol=1e-6)


def test_direct_standardization_requires_mapping_with_intercept() -> None:
    with pytest.raises(ValueError):
        apply_ds(np.zeros((1, 3, 2), dtype=np.float32), np.zeros((2, 2)))
