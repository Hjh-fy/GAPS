"""Shared helpers for the minimal GAPS Flower deployment.

This module intentionally keeps the first deployment path small: it reuses the
existing model, dataset, and Client training code, while Flower handles only the
cloud-edge parameter exchange.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

from client import Client
from config import FLConfig
from federated_dataset import create_client_test_only_loader, create_train_loader
from utils import create_model_by_config, evaluate_model, set_random_seed


NDArrays = List[np.ndarray]

CLASSIFICATION_PROFILE_FLAGS = {
    "ce_only": {"align": False, "replay": False, "decouple": False},
    "proto_only": {"align": True, "replay": False, "decouple": True},
    "replay_only": {"align": False, "replay": True, "decouple": False},
    "proto_replay": {"align": True, "replay": True, "decouple": True},
}

CLASSIFICATION_PROFILE_ALIASES = {
    "smoke": "ce_only",
    "gaps": "proto_replay",
    "gaps_cls": "proto_replay",
    "gaps_classification": "proto_replay",
    "classification": "proto_replay",
    "strong_cls": "proto_replay",
}


def canonical_profile(profile: str) -> str:
    profile_key = str(profile or "ce_only").lower()
    profile_key = CLASSIFICATION_PROFILE_ALIASES.get(profile_key, profile_key)
    if profile_key not in CLASSIFICATION_PROFILE_FLAGS:
        raise ValueError(f"Unsupported Flower runtime profile: {profile}")
    return profile_key


def serialize_counts(count_dict: dict) -> str:
    """Serialize class-phase sample counts for Flower metrics."""
    serializable = {}
    for key, value in sorted(count_dict.items(), key=lambda item: tuple(item[0])):
        cls, phase = key
        serializable[f"{int(cls)},{int(phase)}"] = int(value)
    return json.dumps(serializable, ensure_ascii=False, sort_keys=True)


def serialize_tensor_dict(tensor_dict: dict) -> str:
    """Serialize class-phase tensors as JSON lists for Flower metrics."""
    serializable = {}
    for key, value in sorted(tensor_dict.items(), key=lambda item: tuple(item[0])):
        cls, phase = key
        tensor = value.detach().cpu().float().view(-1)
        serializable[f"{int(cls)},{int(phase)}"] = tensor.tolist()
    return json.dumps(serializable, ensure_ascii=False, sort_keys=True)


def summarize_feature_vector(feature: torch.Tensor) -> dict:
    """Return scalar summary and JSON payload for a client feature mean."""
    vec = feature.detach().cpu().float().view(-1)
    return {
        "feature_norm": float(torch.norm(vec, p=2).item()),
        "feature_mean": float(vec.mean().item()),
        "feature_std": float(vec.std(unbiased=False).item()),
        "global_feature_json": json.dumps(vec.tolist(), ensure_ascii=False),
    }


def make_config(
    device: str = "cpu",
    local_epochs: int = 1,
    batch_size: int = 32,
    profile: str = "smoke",
    seed: int = 42,
) -> FLConfig:
    """Create the runtime config used by Flower clients and the DA server.

    Profiles
    --------
    smoke:
        Minimal CE-only training for communication tests.
    gaps_cls:
        Classification deployment training.  It keeps regression disabled,
        but enables global prototype alignment and device-residual statistics
        once the server starts broadcasting semantic prototypes.
    """
    config = FLConfig()
    config.DEVICE = device
    config.LOCAL_EPOCHS = local_epochs
    config.BATCH_SIZE = batch_size
    config.SEED = int(seed)

    # Always keep the Flower classification path independent from regression.
    # Regression has a separate offline/FedAvg-style script path in this package.
    config.USE_REG_LOSS = False

    # Conservative defaults: exact smoke-test behavior.
    config.USE_ALIGN = False
    config.USE_CONTRASTIVE_ALIGN = False
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

    profile_key = canonical_profile(profile)
    flags = CLASSIFICATION_PROFILE_FLAGS[profile_key]
    config.USE_ALIGN = flags["align"]
    config.USE_CONTRASTIVE_ALIGN = flags["align"]
    config.USE_REPLAY_DISTILL = flags["replay"]
    config.USE_PROTO_DECOUPLING = flags["decouple"]

    set_random_seed(config.SEED)
    return config




def _parse_proto_key(key: str) -> Optional[Tuple[int, int]]:
    """Parse Flower/legacy prototype keys into ``(class_id, phase_id)``."""
    text = str(key).strip()
    if not text:
        return None
    if text.startswith("(") and text.endswith(")"):
        text = text[1:-1]
    text = text.replace("_", ",")
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if len(parts) < 2:
        return None
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        return None


def deserialize_tensor_dict(
    payload: str | None,
    *,
    device: str = "cpu",
    key_style: str = "tuple",
) -> Dict:
    """Deserialize prototypes broadcast by the Flower server.

    Args:
        payload: JSON object whose keys are class-phase ids and values are
            flattened feature vectors.
        device: Destination device for tensors.
        key_style: ``tuple`` for Client._compute_align_loss, or ``paren`` for
            Client residual/statistics code expecting keys like ``"(0,1)"``.
    """
    if not payload:
        return {}
    try:
        raw = json.loads(payload)
    except (TypeError, json.JSONDecodeError):
        return {}
    out: Dict = {}
    for key, value in raw.items():
        parsed = _parse_proto_key(key)
        if parsed is None:
            continue
        tensor = torch.tensor(value, dtype=torch.float32, device=device).view(-1)
        if key_style == "paren":
            out[f"({parsed[0]},{parsed[1]})"] = tensor
        else:
            out[parsed] = tensor
    return out

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


def parameters_to_state_dict(
    parameters: Iterable[np.ndarray],
    keys: List[str],
    reference_state: Dict[str, torch.Tensor],
) -> OrderedDict:
    """Convert Flower ndarray parameters into a typed PyTorch state_dict.

    This is used on the client to cache the previous server model for replay
    distillation without changing the Flower wire protocol.
    """
    state = OrderedDict()
    for key, value in zip(keys, parameters):
        tensor = torch.tensor(value)
        if key in reference_state:
            tensor = tensor.to(dtype=reference_state[key].dtype)
        state[key] = tensor.detach().cpu().clone()
    return state


def set_prev_model_from_state(gaps_client: Client, state_dict: Dict[str, torch.Tensor]) -> None:
    """Install a previous-round model for feature replay distillation.

    The original single-machine Client.set_prev_parameters creates a generic
    model class.  In Flower we create the model through create_model_by_config so
    that encoder options such as TCN normalization remain identical to the
    current deployment model.
    """
    prev_model = create_model_by_config(gaps_client.config, with_reg_head=False).to(gaps_client.device)
    prev_model.load_state_dict(state_dict, strict=False)
    prev_model.eval()
    for param in prev_model.parameters():
        param.requires_grad = False
    gaps_client.prev_model = prev_model


def client_dir(data_root: str | Path, client_id: int) -> Path:
    return Path(data_root) / f"client_{client_id}"


def load_client_loaders(data_root: str | Path, client_id: int, config: FLConfig):
    """Load local train/test data for one edge client."""
    cdir = client_dir(data_root, client_id)
    train_loader = create_train_loader(
        cdir,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        normalize=False,
    )
    test_loader = create_client_test_only_loader(cdir, batch_size=config.BATCH_SIZE)
    return train_loader, test_loader


def make_client(client_id: int, model: torch.nn.Module, train_loader, config: FLConfig) -> Client:
    gaps_client = Client(client_id=client_id, config=config)
    gaps_client.set_model(model)
    gaps_client.update_dataloader(train_loader)
    return gaps_client


def train_one_round(
    gaps_client: Client,
    round_idx: int,
    fit_config: Optional[dict] = None,
) -> Tuple[NDArrays, int, dict]:
    fit_config = fit_config or {}
    proto_payload = fit_config.get("semantic_protos_json") or fit_config.get("global_protos_json")
    global_protos = deserialize_tensor_dict(
        proto_payload, device=gaps_client.config.DEVICE, key_style="tuple"
    )
    semantic_protos = deserialize_tensor_dict(
        proto_payload, device=gaps_client.config.DEVICE, key_style="paren"
    )
    params, prototypes, count_dict, global_feature, device_residual, proto_vars = gaps_client.train_one_round(
        current_round=max(1, round_idx),
        global_protos=global_protos or None,
        semantic_protos=semantic_protos or None,
    )
    ordered = OrderedDict(params)
    arrays = [value.detach().cpu().numpy() for value in ordered.values()]
    num_examples = len(gaps_client.train_loader.dataset) if gaps_client.train_loader is not None else 0
    proto_examples = int(sum(count_dict.values())) if count_dict else 0
    metrics = {
        "client_id": int(gaps_client.client_id),
        "num_examples": int(num_examples),
        "proto_examples": int(proto_examples),
        "local_epochs": int(gaps_client.config.LOCAL_EPOCHS),
        "class_phase_counts_json": serialize_counts(count_dict or {}),
        "prototype_json": serialize_tensor_dict(prototypes or {}),
        "prototype_count": int(len(prototypes or {})),
        "prototype_var_json": serialize_tensor_dict(proto_vars or {}),
        "prototype_var_count": int(len(proto_vars or {})),
        "received_global_prototypes": int(len(global_protos)),
        "use_align": int(bool(gaps_client.config.USE_ALIGN and global_protos)),
        "use_replay_distill": int(bool(gaps_client.config.USE_REPLAY_DISTILL and gaps_client.prev_model is not None)),
        "has_prev_model": int(gaps_client.prev_model is not None),
        "use_proto_decoupling": int(bool(gaps_client.config.USE_PROTO_DECOUPLING and semantic_protos)),
    }
    if device_residual is not None:
        vec = device_residual.detach().cpu().float().view(-1)
        metrics["device_residual_json"] = json.dumps(vec.tolist(), ensure_ascii=False)
        metrics["device_residual_norm"] = float(torch.norm(vec, p=2).item())
    metrics.update(summarize_feature_vector(global_feature))
    return arrays, num_examples, metrics


def evaluate(model: torch.nn.Module, test_loader, config: FLConfig, client_id: int | None = None) -> Tuple[float, int, dict]:
    acc = evaluate_model(model, test_loader, torch.device(config.DEVICE))
    num_examples = len(test_loader.dataset)
    loss = float(1.0 - acc)
    metrics = {
        "accuracy": float(acc),
        "num_examples": int(num_examples),
    }
    if client_id is not None:
        metrics["client_id"] = int(client_id)
    return loss, num_examples, metrics
