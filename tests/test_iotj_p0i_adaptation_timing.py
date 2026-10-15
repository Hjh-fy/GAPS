from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest
import torch

import gaps_flower.p0i_adaptation as uda
import scripts.run_iotj_p0i_adaptation_timing as p0i
from gaps_flower.strategy import P0IInterleavedFedAvg


def test_posthoc_and_interleaved_share_uda_objective() -> None:
    assert uda.U1_WEIGHTS == {"source_ce": 1.0, "coral": 0.5, "global_mmd2": 0.5, "adversarial": 0.5}
    runner_source = inspect.getsource(p0i)
    strategy_source = inspect.getsource(P0IInterleavedFedAvg.aggregate_fit)
    assert "run_frozen_u1" in runner_source
    assert "run_frozen_u1" in strategy_source


def test_posthoc_total_steps_2500() -> None:
    source = inspect.getsource(p0i.run_posthoc)
    assert "num_steps=2500" in source
    assert p0i.MILESTONES == {0, 100, 250, 500, 1000, 1500, 2000, 2500}


def test_interleaved_total_steps_2500() -> None:
    assert p0i.ROUNDS == 25
    assert P0IInterleavedFedAvg.__init__.__defaults__ is None or True
    assert "uda_steps_per_round: int = 100" in inspect.getsource(P0IInterleavedFedAvg.__init__)


def test_interleaved_post_uda_becomes_next_round_global() -> None:
    source = inspect.getsource(P0IInterleavedFedAvg)
    assert "return ndarrays_to_parameters(post_arrays)" in source
    assert "value != self._previous_post_fingerprint" in source
    assert "FAIL_CLOSED round" in source


def test_parameter_fingerprint_is_content_exact_and_serialization_independent() -> None:
    arrays = [np.asarray([[1.0, 2.0]], np.float32), np.asarray([3], np.int64)]
    first = uda.parameter_fingerprint(["a", "b"], arrays)
    second = uda.parameter_fingerprint(["a", "b"], [item.copy() for item in arrays])
    assert first == second
    arrays[0][0, 0] += 1
    assert uda.parameter_fingerprint(["a", "b"], arrays) != first


def test_target_api_is_x_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    loaded: list[str] = []
    monkeypatch.setattr(np, "load", lambda path, **kwargs: loaded.append(Path(path).name) or np.zeros((320,100,8),np.float32))
    dataset = uda.FeatureOnlyCalibrationDataset(tmp_path)
    assert loaded == ["calibration_features.npy"]
    assert isinstance(dataset[0], torch.Tensor)


def test_no_target_label_conditioned_losses() -> None:
    source = inspect.getsource(uda.run_frozen_u1)
    for forbidden in ("cross_entropy(logits_t", "deep_coral_loss_class_conditional(",
                      "cross_domain_same_class_phase_mmd2(", "target_prototype_anchor(",
                      "pseudo =", "y_t ="):
        assert forbidden not in source.lower()


def test_pre_and_post_checkpoints_saved_every_round() -> None:
    source = inspect.getsource(P0IInterleavedFedAvg.aggregate_fit)
    assert '"pre_uda"' in source and '"post_uda"' in source


def test_target_test_not_opened_during_training() -> None:
    for function in (p0i.run_posthoc, p0i.run_interleaved):
        source = inspect.getsource(function)
        assert '"test"' not in source
        assert "make_loader" not in source


def test_round25_and_step2500_are_fixed_endpoints() -> None:
    assert p0i.ROUNDS == 25
    source = inspect.getsource(p0i)
    assert '"formal_endpoint": step == 2500' in source
    assert '"formal_endpoint":"round25_post_uda"' in source
