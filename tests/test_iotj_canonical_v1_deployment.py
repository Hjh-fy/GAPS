from __future__ import annotations

import numpy as np
import pytest
import torch

from gaps_deploy.canonical_v1_runtime import preprocess_canonical_window
from scripts.build_iotj_canonical_v1_deployment import model_size_stats


def test_canonical_runtime_accepts_only_50_by_8_finite_windows() -> None:
    values = preprocess_canonical_window(np.zeros((50, 8), dtype=np.float64))
    assert values.shape == (50, 8)
    assert values.dtype == np.float32
    with pytest.raises(ValueError, match="50x8"):
        preprocess_canonical_window(np.zeros((100, 8), dtype=np.float32))
    invalid = np.zeros((50, 8), dtype=np.float32)
    invalid[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        preprocess_canonical_window(invalid)


def test_model_size_stats_distinguish_tensor_count_and_parameter_count() -> None:
    model = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.Linear(2, 1))
    stats = model_size_stats(model)
    assert stats["state_tensor_count"] == 4
    assert stats["total_parameter_count"] == 11
    assert stats["trainable_parameter_count"] == 11
    assert stats["fp32_model_bytes"] == 44

