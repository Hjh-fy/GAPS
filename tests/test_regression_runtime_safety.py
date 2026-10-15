from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

import gaps_flower.evaluate_regression_pipeline as evaluator
import gaps_flower.regression_server as regression_server
import gaps_flower.regression_task as regression_task
from gaps_flower.regression_server import (
    _checkpoint_n_samples,
    aggregate_regression_checkpoints,
)


@pytest.mark.parametrize("value", [None, True, False, 0, -1, 1.5, "3"])
def test_checkpoint_n_samples_rejects_invalid_metadata(tmp_path, value) -> None:
    checkpoint = {} if value is None else {"n_samples": value}

    with pytest.raises(ValueError, match="n_samples"):
        _checkpoint_n_samples(checkpoint, 3, tmp_path / "client3.pth")


def test_checkpoint_n_samples_accepts_positive_integer(tmp_path) -> None:
    assert _checkpoint_n_samples(
        {"n_samples": 7}, 2, tmp_path / "client2.pth"
    ) == 7


def test_legacy_regression_producer_persists_n_samples(tmp_path, monkeypatch) -> None:
    model = torch.nn.Linear(2, 1)
    loaders = {
        1: DataLoader(TensorDataset(torch.zeros((3, 2))), batch_size=2),
        2: DataLoader(TensorDataset(torch.zeros((7, 2))), batch_size=2),
    }
    sample_counts = {1: 3, 2: 7}
    monkeypatch.setattr(regression_task, "train_regression_local", lambda *args, **kwargs: 0.0)
    monkeypatch.setattr(
        regression_task,
        "get_regression_state_keys",
        lambda current_model: list(current_model.state_dict()),
    )

    regression_task.train_federated_source_regression(
        model,
        loaders,
        sample_counts,
        torch.device("cpu"),
        total_steps_per_client=1,
        source_rounds=1,
        save_dir=str(tmp_path),
    )

    for client_id, expected in sample_counts.items():
        checkpoint = torch.load(
            tmp_path / f"regression_source_client{client_id}_local.pth",
            map_location="cpu",
            weights_only=False,
        )
        assert checkpoint["n_samples"] == expected


def _server_config() -> SimpleNamespace:
    return SimpleNamespace(
        REG_HEAD_DEPTH=3,
        REG_OUTPUT_MODE="sigmoid",
        REG_WINDOW_STATS=False,
        REG_WINDOW_STATS_MODE="global",
        REG_WINDOW_STATS_DIM=8,
        REG_RESPONSE_BRANCH="none",
        REG_DCT_K=16,
        REG_DCT_GAMMA_INIT=0.0,
        REG_DCT_DROPOUT=0.1,
        REG_MSCONV_CHANNELS=16,
        REG_MSCONV_KERNELS="3,7,15,31",
        REG_MSCONV_GAMMA_INIT=0.0,
        REG_MSCONV_DROPOUT=0.1,
        REG_TCN_ADAPTER=False,
        REG_TCN_ADAPTER_KERNEL=3,
        REG_TCN_ADAPTER_GAMMA_INIT=0.0,
        REG_TCN_ADAPTER_DROPOUT=0.05,
        REG_USE_SHARED_TRUNK=False,
        REG_SHARED_TRUNK_DIM=128,
        REG_GAS_EMB_DIM=16,
        REG_RESIDUAL_HEAD_DEPTH=2,
        USE_REG_RATIO_BRANCH=False,
        REG_RATIO_GAMMA_INIT=0.0,
        REG_RATIO_DROPOUT=0.05,
    )


def _write_client_checkpoints(directory: Path) -> None:
    directory.mkdir(parents=True)
    model = torch.nn.Linear(2, 1)
    for client_id, count in {1: 3, 2: 7}.items():
        torch.save(
            {
                "model_state": model.state_dict(),
                "n_samples": count,
                "model_config": {},
            },
            directory / f"regression_source_client{client_id}_local.pth",
        )


def _patch_server_model(monkeypatch, captured_counts: list[dict[int, int]]) -> None:
    monkeypatch.setattr(regression_server, "make_regression_config", lambda **kwargs: _server_config())
    monkeypatch.setattr(regression_server, "create_regression_model", lambda config: torch.nn.Linear(2, 1))
    monkeypatch.setattr(regression_server, "load_classifier_weights", lambda *args, **kwargs: None)
    monkeypatch.setattr(regression_server, "init_regression_branch_from_classifier", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        regression_server,
        "get_regression_state_keys",
        lambda model: list(model.state_dict()),
    )

    def capture(states, counts, keys, device):
        captured_counts.append(dict(counts))
        return regression_task.fedavg_regression_states(states, counts, keys, device)

    monkeypatch.setattr(regression_server, "fedavg_regression_states", capture)


def test_aggregation_uses_checkpoint_counts_without_live_reconstruction(
    tmp_path, monkeypatch
) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    _write_client_checkpoints(checkpoint_dir)
    captured: list[dict[int, int]] = []
    _patch_server_model(monkeypatch, captured)

    def forbidden_live_counts(**kwargs):
        raise AssertionError("live data counts must not be reconstructed by default")

    monkeypatch.setattr(
        regression_server,
        "build_source_regression_loaders",
        forbidden_live_counts,
    )

    aggregate_regression_checkpoints(
        classifier_ckpt="unused.pth",
        client_ckpt_dir=str(checkpoint_dir),
        data_root="unused",
        client_ids=[1, 2],
        device=torch.device("cpu"),
        output_dir=str(tmp_path / "output"),
    )

    assert captured == [{1: 3, 2: 7}]


def test_optional_live_count_verification_rejects_mismatch_before_fedavg(
    tmp_path, monkeypatch
) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    _write_client_checkpoints(checkpoint_dir)
    captured: list[dict[int, int]] = []
    _patch_server_model(monkeypatch, captured)
    monkeypatch.setattr(
        regression_server,
        "build_source_regression_loaders",
        lambda **kwargs: ({}, {1: 30, 2: 70}),
    )

    with pytest.raises(ValueError, match="sample count|n_samples"):
        aggregate_regression_checkpoints(
            classifier_ckpt="unused.pth",
            client_ckpt_dir=str(checkpoint_dir),
            data_root="unused",
            client_ids=[1, 2],
            device=torch.device("cpu"),
            verify_live_sample_counts=True,
            output_dir=str(tmp_path / "output"),
        )

    assert captured == []


@pytest.fixture
def tiny_regression_factory(monkeypatch):
    monkeypatch.setattr(
        evaluator,
        "create_regression_model",
        lambda config: torch.nn.Linear(2, 1),
    )


def test_selected_full_and_specialist_assets_are_required(
    tmp_path, tiny_regression_factory
) -> None:
    config = SimpleNamespace()

    with pytest.raises(ValueError, match="full"):
        evaluator.load_full_model(None, torch.device("cpu"), config, required=True)
    with pytest.raises(ValueError, match="specialist"):
        evaluator.load_specialist_models(
            None,
            {0: "specialist"},
            torch.device("cpu"),
            config,
        )
    with pytest.raises(ValueError, match="class_0.pth"):
        evaluator.load_specialist_models(
            str(tmp_path),
            {0: "specialist_full"},
            torch.device("cpu"),
            config,
        )


@pytest.mark.parametrize("damage", ["missing", "unexpected", "shape"])
def test_evaluation_assets_require_exact_state_keys(
    tmp_path, tiny_regression_factory, damage: str
) -> None:
    model = torch.nn.Linear(2, 1)
    state = {key: value.clone() for key, value in model.state_dict().items()}
    if damage == "missing":
        state.pop("bias")
    elif damage == "unexpected":
        state["extra"] = torch.zeros(1)
    else:
        state["weight"] = torch.zeros((1, 3))
    path = tmp_path / "full_model.pth"
    torch.save({"model_state": state}, path)

    with pytest.raises(ValueError, match="full_model.pth"):
        evaluator.load_full_model(
            str(path), torch.device("cpu"), SimpleNamespace(), required=True
        )


def test_valid_evaluation_checkpoint_loads_strictly(
    tmp_path, tiny_regression_factory
) -> None:
    model = torch.nn.Linear(2, 1)
    path = tmp_path / "full_model.pth"
    torch.save({"model_state": model.state_dict()}, path)

    loaded = evaluator.load_full_model(
        str(path), torch.device("cpu"), SimpleNamespace(), required=True
    )

    assert isinstance(loaded, torch.nn.Linear)


def test_base_evaluation_checkpoint_requires_exact_state_keys(
    tmp_path, tiny_regression_factory, monkeypatch
) -> None:
    path = tmp_path / "base_regression.pth"
    torch.save({"model_state": {"weight": torch.zeros((1, 2))}}, path)
    monkeypatch.setattr(
        evaluator,
        "make_regression_config_from_checkpoint",
        lambda checkpoint, device, batch_size, args: SimpleNamespace(),
    )

    with pytest.raises(ValueError, match="base_regression.pth"):
        evaluator.load_regression_model(
            str(path),
            torch.device("cpu"),
            4,
            SimpleNamespace(),
        )


def test_specialist_checkpoint_requires_exact_state_keys(
    tmp_path, tiny_regression_factory
) -> None:
    torch.save(
        {"model_state": {"weight": torch.zeros((1, 2))}},
        tmp_path / "class_0.pth",
    )

    with pytest.raises(ValueError, match="class_0.pth"):
        evaluator.load_specialist_models(
            str(tmp_path),
            {0: "specialist"},
            torch.device("cpu"),
            SimpleNamespace(),
        )
