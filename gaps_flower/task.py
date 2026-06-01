"""Shared helpers for the minimal GAPS Flower deployment.

This module intentionally keeps the first deployment path small: it reuses the
existing model, dataset, and Client training code, while Flower handles only the
cloud-edge parameter exchange.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import torch

from client import Client
from config import FLConfig
from federated_dataset import create_client_full_test_loader, create_train_loader
from utils import create_model_by_config, evaluate_model, set_random_seed


NDArrays = List[np.ndarray]


def make_config(device: str = "cpu", local_epochs: int = 1, batch_size: int = 32) -> FLConfig:
    """Create a conservative runtime config for cloud-edge smoke tests."""
    config = FLConfig()
    config.DEVICE = device
    config.LOCAL_EPOCHS = local_epochs
    config.BATCH_SIZE = batch_size
    config.USE_REG_LOSS = False
    config.USE_ALIGN = False
    config.USE_REPLAY_DISTILL = False
    config.USE_SERVER_OPT = False
    config.USE_LEARNABLE_AGG = False
    config.USE_SELECTIVE_AGG = False
    config.USE_PROTO_DECOUPLING = False
    config.USE_SOFT_AGGREGATION = False
    config.USE_SENSOR_AUG = False
    config.USE_MMD_REG = False
    config.USE_DEEP_CORAL = False
    config.USE_ADVERSARIAL_DOMAIN = False
    config.USE_MMD_ALIGNMENT = False
    set_random_seed(config.SEED)
    return config


def create_model(config: FLConfig) -> torch.nn.Module:
    """Create the lightweight classification model used for first-link tests."""
    model = create_model_by_config(config, with_reg_head=False)
    return model.to(config.DEVICE)


def get_parameters(model: torch.nn.Module) -> Tuple[NDArrays, List[str]]:
    """Return model parameters and the matching state_dict key order."""
    state = model.state_dict()
    keys = list(state.keys())
    arrays = [state[key].detach().cpu().numpy() for key in keys]
    return arrays, keys


def set_parameters(model: torch.nn.Module, parameters: Iterable[np.ndarray], keys: List[str]) -> None:
    """Load Flower ndarray parameters into a PyTorch model."""
    state = model.state_dict()
    new_state = OrderedDict()
    for key, value in zip(keys, parameters):
        tensor = torch.tensor(value)
        if key in state:
            tensor = tensor.to(dtype=state[key].dtype)
        new_state[key] = tensor
    model.load_state_dict(new_state, strict=True)


def client_dir(data_root: str | Path, client_id: int) -> Path:
    return Path(data_root) / f"client_{client_id}"


def load_client_loaders(data_root: str | Path, client_id: int, config: FLConfig):
    """Load local train/test data for one edge client."""
    cdir = client_dir(data_root, client_id)
    train_loader = create_train_loader(cdir, batch_size=config.BATCH_SIZE, shuffle=True)
    test_loader = create_client_full_test_loader(cdir, batch_size=config.BATCH_SIZE)
    return train_loader, test_loader


def make_client(client_id: int, model: torch.nn.Module, train_loader, config: FLConfig) -> Client:
    gaps_client = Client(client_id=client_id, config=config)
    gaps_client.set_model(model)
    gaps_client.update_dataloader(train_loader)
    return gaps_client


def train_one_round(gaps_client: Client, round_idx: int) -> Tuple[NDArrays, int, dict]:
    params, _mus, _counts, _features, _residual, _vars = gaps_client.train_one_round(
        current_round=max(1, round_idx),
        global_protos=None,
        semantic_protos=None,
    )
    ordered = OrderedDict(params)
    arrays = [value.detach().cpu().numpy() for value in ordered.values()]
    num_examples = len(gaps_client.train_loader.dataset) if gaps_client.train_loader is not None else 0
    return arrays, num_examples, {"num_examples": num_examples}


def evaluate(model: torch.nn.Module, test_loader, config: FLConfig) -> Tuple[float, int, dict]:
    acc = evaluate_model(model, test_loader, torch.device(config.DEVICE))
    num_examples = len(test_loader.dataset)
    loss = float(1.0 - acc)
    return loss, num_examples, {"accuracy": float(acc)}
