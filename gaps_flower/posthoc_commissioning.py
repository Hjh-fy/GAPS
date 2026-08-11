"""Fail-closed one-time commissioning utilities with no Flower lifecycle API.

The functions in this module operate on an already completed source checkpoint.
They intentionally have no server address, round count, Flower strategy, or
client-process interface.
"""

from __future__ import annotations

import copy
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from gaps_flower.domain_adaptation import ServerDomainAdaptation
from utils import set_random_seed


METHODS = ("a0t_full", "a4", "target_head", "classifier_only", "low_rank_adapter")
STEPS = 100
LR = 5e-4
BATCH_SIZE = 32
SEED = 42


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ordered_state_fingerprint(state: dict[str, torch.Tensor]) -> str:
    """Hash state tensors in their declared order, including dtype and shape."""
    digest = hashlib.sha256()
    for name, tensor in state.items():
        value = tensor.detach().cpu().contiguous()
        digest.update(str(name).encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def load_calibration_identity_manifest(
    path: Path,
    *,
    expected_client: int,
    expected_count: int,
) -> tuple[str, ...]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list) or len(payload) != int(expected_count):
        raise ValueError(
            f"calibration manifest must contain exactly {expected_count} rows"
        )
    identities: list[str] = []
    for row in payload:
        if not isinstance(row, dict) or row.get("role") != "calibration":
            raise ValueError("calibration manifest contains a non-calibration role")
        if int(row.get("client_id", -1)) != int(expected_client):
            raise ValueError("calibration manifest contains the wrong client")
        identity = str(row.get("physical_identity", "")).strip()
        if not identity:
            raise ValueError("calibration manifest contains an empty identity")
        identities.append(identity)
    if len(set(identities)) != len(identities):
        raise ValueError("calibration manifest contains duplicate identities")
    return tuple(identities)


@dataclass(frozen=True)
class PosthocRequest:
    source_checkpoint: Path
    calibration_manifest: Path
    method: str
    target_test_manifest: Path | None = None

    def validate_static_boundary(self) -> None:
        if self.method not in METHODS:
            raise ValueError(f"unsupported post-hoc method: {self.method}")
        if self.target_test_manifest is not None:
            raise ValueError("target test manifest must never enter adaptation API")


def configure_trainable_parameters(model: torch.nn.Module, method: str) -> list[str]:
    if method not in METHODS:
        raise ValueError(f"unsupported post-hoc method: {method}")
    for parameter in model.parameters():
        parameter.requires_grad_(method in {"a0t_full", "a4"})
    if method == "target_head":
        if getattr(model, "feat_proj", None) is None:
            raise ValueError("target-head protocol requires a concrete feat_proj")
        for parameter in model.feat_proj.parameters():
            parameter.requires_grad_(True)
        for parameter in model.classifier.parameters():
            parameter.requires_grad_(True)
    elif method in {"classifier_only", "low_rank_adapter"}:
        for parameter in model.classifier.parameters():
            parameter.requires_grad_(True)
    return [name for name, parameter in model.named_parameters() if parameter.requires_grad]


class LowRankPosthocAdapter(torch.nn.Module):
    """Rank-r residual adapter on the normalized 64-D classification feature."""

    def __init__(self, source_model: torch.nn.Module, *, rank: int = 4) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("adapter rank must be positive")
        self.base_model = copy.deepcopy(source_model)
        configure_trainable_parameters(self.base_model, "low_rank_adapter")
        feature_dim = int(self.base_model.classifier.in_features)
        self.rank = int(rank)
        self.down = torch.nn.Linear(feature_dim, self.rank, bias=False)
        self.up = torch.nn.Linear(self.rank, feature_dim, bias=False)
        torch.nn.init.kaiming_uniform_(self.down.weight, a=5**0.5)
        torch.nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor):
        _base_logits, cls_feat, reg_feat = self.base_model(x)
        adapted_feat = cls_feat + self.up(self.down(cls_feat))
        logits = self.base_model.classifier(adapted_feat)
        return logits, adapted_feat, reg_feat


def fold_low_rank_adapter(adapter: LowRankPosthocAdapter) -> torch.nn.Module:
    """Fold z'=(I+BA)z into the ordinary classifier without runtime changes."""
    folded = copy.deepcopy(adapter.base_model)
    with torch.no_grad():
        feature_dim = int(folded.classifier.in_features)
        identity = torch.eye(
            feature_dim,
            dtype=adapter.up.weight.dtype,
            device=adapter.up.weight.device,
        )
        transform = identity + adapter.up.weight @ adapter.down.weight
        folded.classifier.weight.copy_(adapter.base_model.classifier.weight @ transform)
        folded.classifier.bias.copy_(adapter.base_model.classifier.bias)
    return folded


def relative_parameter_displacement(
    adapted: torch.nn.Module, source: torch.nn.Module
) -> float:
    numerator = torch.zeros((), dtype=torch.float64)
    denominator = torch.zeros((), dtype=torch.float64)
    adapted_params = dict(adapted.named_parameters())
    for name, source_parameter in source.named_parameters():
        target = adapted_params[name].detach().cpu().double()
        base = source_parameter.detach().cpu().double()
        numerator += torch.sum((target - base) ** 2)
        denominator += torch.sum(base**2)
    return float(torch.sqrt(numerator) / torch.sqrt(denominator).clamp_min(1e-12))


def _next_batch(iterator: Iterable, loader: DataLoader):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def supervised_ce_adapt(
    source_model: torch.nn.Module,
    calibration_loader: DataLoader,
    *,
    method: str,
    device: torch.device,
    steps: int = STEPS,
    lr: float = LR,
    seed: int = SEED,
) -> tuple[torch.nn.Module, list[dict[str, float]], dict[str, Any]]:
    if method not in {"a0t_full", "target_head", "classifier_only"}:
        raise ValueError("supervised CE adaptation supports a0t_full, target_head, or classifier_only")
    set_random_seed(seed)
    source = copy.deepcopy(source_model).to(device)
    adapted = copy.deepcopy(source_model).to(device)
    names = configure_trainable_parameters(adapted, method)
    parameters = [parameter for parameter in adapted.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=lr)
    iterator = iter(calibration_loader)
    rows: list[dict[str, float]] = []
    started = time.perf_counter()
    adapted.train()
    for step in range(1, steps + 1):
        batch, iterator = _next_batch(iterator, calibration_loader)
        x = batch[0].to(device)
        y = batch[1].to(device).long()
        optimizer.zero_grad()
        logits, _, _ = adapted(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 5.0)
        optimizer.step()
        rows.append(
            {
                "step": float(step),
                "target_ce": float(loss.detach().item()),
                "target_accuracy": float(
                    (logits.detach().argmax(dim=1) == y).float().mean().item()
                ),
            }
        )
    elapsed = time.perf_counter() - started
    total = sum(parameter.numel() for parameter in adapted.parameters())
    trainable = sum(parameter.numel() for parameter in parameters)
    return adapted, rows, {
        "method": method,
        "steps": int(steps),
        "optimizer": "Adam",
        "lr": float(lr),
        "seed": int(seed),
        "trainable_parameter_names": names,
        "trainable_parameters": int(trainable),
        "total_parameters": int(total),
        "trainable_parameter_ratio": float(trainable / total),
        "adaptation_seconds": float(elapsed),
        "relative_parameter_displacement": relative_parameter_displacement(
            adapted, source
        ),
    }


def low_rank_adapter_adapt(
    source_model: torch.nn.Module,
    calibration_loader: DataLoader,
    *,
    device: torch.device,
    rank: int = 4,
    steps: int = STEPS,
    lr: float = LR,
    seed: int = SEED,
) -> tuple[torch.nn.Module, list[dict[str, float]], dict[str, Any]]:
    """Train the fixed rank-4 residual adapter and return an exactly folded model."""
    set_random_seed(seed)
    source = copy.deepcopy(source_model).to(device)
    adapter = LowRankPosthocAdapter(source_model, rank=rank).to(device)
    parameters = [parameter for parameter in adapter.parameters() if parameter.requires_grad]
    names = [name for name, parameter in adapter.named_parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=lr)
    iterator = iter(calibration_loader)
    rows: list[dict[str, float]] = []
    started = time.perf_counter()
    adapter.train()
    for step in range(1, steps + 1):
        batch, iterator = _next_batch(iterator, calibration_loader)
        x = batch[0].to(device)
        y = batch[1].to(device).long()
        optimizer.zero_grad()
        logits, _, _ = adapter(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 5.0)
        optimizer.step()
        rows.append(
            {
                "step": float(step),
                "target_ce": float(loss.detach().item()),
                "target_accuracy": float(
                    (logits.detach().argmax(dim=1) == y).float().mean().item()
                ),
            }
        )
    elapsed = time.perf_counter() - started
    folded = fold_low_rank_adapter(adapter).to(device)
    total = sum(parameter.numel() for parameter in folded.parameters())
    trainable = sum(parameter.numel() for parameter in parameters)
    return folded, rows, {
        "method": "low_rank_adapter",
        "adapter_rank": int(rank),
        "adapter_initialization": "kaiming_down_zero_up",
        "deployment_form": "exact_classifier_fold",
        "steps": int(steps),
        "optimizer": "Adam",
        "lr": float(lr),
        "seed": int(seed),
        "trainable_parameter_names": names,
        "trainable_parameters": int(trainable),
        "total_parameters": int(total),
        "trainable_parameter_ratio": float(trainable / total),
        "adaptation_seconds": float(elapsed),
        "relative_parameter_displacement": relative_parameter_displacement(folded, source),
    }


def registered_a4_hyperparameters() -> dict[str, Any]:
    """Return the frozen A4 coefficients from CANONICAL-V1-A4-C5."""
    return {
        "USE_DEEP_CORAL": True,
        "USE_MMD_ALIGNMENT": True,
        "USE_ADVERSARIAL_DOMAIN": True,
        "MMD_OBJECTIVE": "mmd2",
        "STAGE_ALIGNMENT": "cross_domain_same_class_phase",
        "ADV_FEATURE_OBJECTIVE": "wasserstein_min",
        "CORAL_CLASS_CONDITIONAL": True,
        "LAMBDA_DEEP_CORAL": 0.5,
        "LAMBDA_GLOBAL_MMD": 0.5,
        "LAMBDA_CLASS_MMD": 0.5,
        "LAMBDA_STAGE_MMD": 0.2,
        "LAMBDA_ADV_DOMAIN": 0.5,
        "LAMBDA_PROTO_ANCHOR": 0.3,
        "LAMBDA_PROTO": 0.05,
        "LAMBDA_CONSISTENCY": 2.0,
        "LAMBDA_RES": 0.1,
        "LAMBDA_PROTO_MMD": 0.2,
        "LAMBDA_TARGET_CE": 0.0,
        "USE_ALIGN_REG_LEGACY": False,
        "LAMBDA_ALIGN_REG_LEGACY": 0.05,
        "USE_CONTRASTIVE_CONSISTENCY": True,
        "USE_PROTO_MMD": True,
        "USE_PROTO_DECOUPLING": True,
        "TARGET_CE_LABEL_SMOOTHING": 0.0,
        "TARGET_CE_CLASS_BALANCED": False,
        "SERVER_OPT_LR": LR,
        "HIDDEN_DIM2": 64,
        "NUM_CLASSES": 4,
        "MAX_VAL_BATCHES": 10,
        "ADV_DOMAIN_LR": 0.001,
        "ADV_CRITIC_ITERS": 3,
        "ADV_GRADIENT_PENALTY": 10.0,
        "ADV_CLASS_CONDITIONAL": True,
        "DA_LEARN_SEMANTIC_PROTOS": True,
        "RETURN_STEP_DIAGNOSTICS": True,
    }


def a4_posthoc_adapt(
    source_model: torch.nn.Module,
    source_loader: DataLoader,
    calibration_loader: DataLoader,
    *,
    device: torch.device,
    steps: int = STEPS,
    seed: int = SEED,
) -> tuple[torch.nn.Module, list[dict[str, float]], dict[str, Any]]:
    set_random_seed(seed)
    source = copy.deepcopy(source_model).to(device)
    model = copy.deepcopy(source_model).to(device)
    names = configure_trainable_parameters(model, "a4")
    trainer = ServerDomainAdaptation(
        model,
        source_loader,
        calibration_loader,
        semantic_protos={},
        device=device,
        hyperparams=registered_a4_hyperparameters(),
    )
    started = time.perf_counter()
    adapted, summary = trainer.run_adaptation(
        num_steps=steps,
        client_mus=[{}, {}],
        client_counts=[{}, {}],
        client_weights=torch.tensor([0.5, 0.5], device=device),
        client_ids=[1, 2],
        client_residuals=[None, None],
    )
    elapsed = time.perf_counter() - started
    per_step = summary.pop("step_diagnostics")
    rows = [
        {"step": float(index + 1), **{key: float(values[index]) for key, values in per_step.items()}}
        for index in range(steps)
    ]
    total = sum(parameter.numel() for parameter in adapted.parameters())
    summary.update(
        {
            "method": "a4",
            "steps": int(steps),
            "optimizer": "Adam",
            "lr": LR,
            "seed": int(seed),
            "trainable_parameter_names": names,
            "trainable_parameters": int(total),
            "total_parameters": int(total),
            "trainable_parameter_ratio": 1.0,
            "adaptation_seconds": float(elapsed),
            "relative_parameter_displacement": relative_parameter_displacement(
                adapted, source
            ),
            "interleaved_client_statistics_available": False,
        }
    )
    return adapted, rows, summary
