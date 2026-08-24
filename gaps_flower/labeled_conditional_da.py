"""Fixed C5 labeled and true-class-conditional DA diagnostics.

This module is deliberately separate from :mod:`gaps_flower.canonical_uda`:
the canonical E2 API remains target-x-only, while this diagnostic API receives
exactly target ``(x, class)`` pairs.
"""

from __future__ import annotations

import copy
import time
from collections import OrderedDict
from typing import Any

import torch
import torch.nn.functional as F

from gaps_flower.state_fingerprint import ordered_state_content_fingerprint
from utils import compute_mmd2, deep_coral_loss, set_random_seed


class _GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values: torch.Tensor) -> torch.Tensor:
        return values.view_as(values)

    @staticmethod
    def backward(ctx, gradient: torch.Tensor):
        return -gradient


def _grl(values: torch.Tensor) -> torch.Tensor:
    return _GradientReverse.apply(values)


class SharedDomainDiscriminator(torch.nn.Module):
    """The same-capacity discriminator used by the unconditional E2 DANN."""

    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(int(feature_dim), 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


def registered_method_specs() -> OrderedDict[str, dict[str, Any]]:
    common = {
        "steps": 100,
        "optimizer": "Adam",
        "optimizer_lr": 5e-4,
        "alignment_weight": 0.5,
        "seed": 42,
        "checkpoint_selection": "fixed_step_100",
        "hyperparameter_search": False,
        "target_fields": ["x", "class"],
    }
    rows = OrderedDict()
    for method, alignment in (
        ("coral_tce", "coral"),
        ("mmd_tce", "mmd2"),
        ("dann_tce", "dann_grl_binary_bce"),
    ):
        rows[method] = {
            **common,
            "alignment": alignment,
            "alignment_scope": "unconditional_global",
            "target_ce_weight": 1.0,
            "target_label_role": "classification_CE_and_available_training_labels",
        }
    for method, alignment in (
        ("cc_coral", "coral"),
        ("class_mmd", "mmd2"),
        ("cc_dann", "dann_grl_binary_bce"),
    ):
        rows[method] = {
            **common,
            "alignment": alignment,
            "alignment_scope": "true_class_conditional",
            "target_ce_weight": 0.0,
            "target_label_role": "class_grouping_only_no_target_CE",
        }
    return rows


def _next(iterator, loader):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def _source_xy(batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(batch, (tuple, list)) or len(batch) < 2:
        raise RuntimeError("FAIL_CLOSED source batch must provide x and class")
    return batch[0], batch[1]


def _target_xy(batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(batch, (tuple, list)) or len(batch) != 2:
        raise RuntimeError("FAIL_CLOSED target batch must contain exactly x and class")
    if not all(isinstance(value, torch.Tensor) for value in batch):
        raise RuntimeError("FAIL_CLOSED target x and class must be tensors")
    return batch[0], batch[1]


def _domain_bce(
    source_features: torch.Tensor,
    target_features: torch.Tensor,
    discriminator: SharedDomainDiscriminator,
) -> torch.Tensor:
    features = torch.cat([source_features, target_features], dim=0)
    targets = torch.cat(
        [
            torch.zeros(source_features.size(0), 1, device=features.device),
            torch.ones(target_features.size(0), 1, device=features.device),
        ],
        dim=0,
    )
    return F.binary_cross_entropy_with_logits(discriminator(_grl(features)), targets)


def class_conditional_alignment(
    method: str,
    source_features: torch.Tensor,
    source_labels: torch.Tensor,
    target_features: torch.Tensor,
    target_labels: torch.Tensor,
    discriminator: SharedDomainDiscriminator | None = None,
) -> tuple[torch.Tensor, list[int]]:
    """Average one alignment objective over true classes shared by the batch."""
    method_key = str(method).lower()
    if method_key not in {"cc_coral", "class_mmd", "cc_dann"}:
        raise ValueError(f"not a class-conditional method: {method}")
    shared = sorted(
        set(source_labels.detach().cpu().tolist())
        & set(target_labels.detach().cpu().tolist())
    )
    losses: list[torch.Tensor] = []
    valid: list[int] = []
    for class_id in shared:
        source_mask = source_labels == int(class_id)
        target_mask = target_labels == int(class_id)
        if method_key == "cc_coral":
            if int(source_mask.sum()) < 2 or int(target_mask.sum()) < 2:
                continue
            loss = deep_coral_loss(
                source_features[source_mask], target_features[target_mask]
            )
        elif method_key == "class_mmd":
            if int(source_mask.sum()) < 1 or int(target_mask.sum()) < 1:
                continue
            loss = compute_mmd2(
                source_features[source_mask], target_features[target_mask]
            )
        else:
            if discriminator is None:
                raise RuntimeError("FAIL_CLOSED CC-DANN discriminator missing")
            loss = _domain_bce(
                source_features[source_mask],
                target_features[target_mask],
                discriminator,
            )
        losses.append(loss)
        valid.append(int(class_id))
    if not losses:
        return (
            source_features.sum() * 0.0 + target_features.sum() * 0.0,
            [],
        )
    return torch.stack(losses).mean(), valid


def run_labeled_conditional_da(
    method: str,
    source_model: torch.nn.Module,
    source_loader,
    target_xy_loader,
    device: torch.device,
    *,
    num_steps: int = 100,
    model_lr: float = 5e-4,
    alignment_weight: float = 0.5,
    expected_source_fingerprint: str,
    seed: int = 42,
    formal: bool = True,
) -> tuple[torch.nn.Module, list[dict[str, Any]], float]:
    """Run one approved C5 diagnostic from an independently copied source model."""
    specs = registered_method_specs()
    method_key = str(method).lower()
    if method_key not in specs:
        raise ValueError(f"unsupported labeled-conditional DA method: {method}")
    spec = specs[method_key]
    if formal and (
        int(num_steps) != 100
        or float(model_lr) != 5e-4
        or float(alignment_weight) != 0.5
        or int(seed) != 42
    ):
        raise ValueError(
            "formal labeled-conditional DA requires steps100, Adam5e-4, "
            "alignment0.5, seed42"
        )
    if int(num_steps) <= 0:
        raise ValueError("num_steps must be positive")
    actual_fingerprint = ordered_state_content_fingerprint(source_model.state_dict())
    if actual_fingerprint != str(expected_source_fingerprint):
        raise RuntimeError("FAIL_CLOSED source fingerprint mismatch")

    set_random_seed(int(seed))
    model = copy.deepcopy(source_model).to(device)
    model.train()
    discriminator: SharedDomainDiscriminator | None = None
    optimizer: torch.optim.Optimizer | None = None
    source_iter, target_iter = iter(source_loader), iter(target_xy_loader)
    diagnostics: list[dict[str, Any]] = []
    started = time.perf_counter()

    for step in range(1, int(num_steps) + 1):
        source_batch, source_iter = _next(source_iter, source_loader)
        target_batch, target_iter = _next(target_iter, target_xy_loader)
        source_x, source_y = _source_xy(source_batch)
        target_x, target_y = _target_xy(target_batch)
        source_x = source_x.to(device)
        source_y = source_y.to(device).long()
        target_x = target_x.to(device)
        target_y = target_y.to(device).long()

        source_logits, source_features, _ = model(source_x)
        target_logits, target_features, _ = model(target_x)
        if optimizer is None:
            parameters = list(model.parameters())
            if method_key in {"dann_tce", "cc_dann"}:
                discriminator = SharedDomainDiscriminator(
                    int(source_features.shape[-1])
                ).to(device)
                parameters.extend(discriminator.parameters())
            optimizer = torch.optim.Adam(parameters, lr=float(model_lr))
        optimizer.zero_grad(set_to_none=True)

        source_ce = F.cross_entropy(source_logits, source_y)
        target_ce_active = float(spec["target_ce_weight"]) > 0.0
        raw_target_ce = (
            F.cross_entropy(target_logits, target_y)
            if target_ce_active
            else target_features.sum() * 0.0
        )
        if spec["alignment_scope"] == "true_class_conditional":
            raw_alignment, valid_classes = class_conditional_alignment(
                method_key,
                source_features,
                source_y,
                target_features,
                target_y,
                discriminator,
            )
        else:
            valid_classes = []
            if method_key == "coral_tce":
                raw_alignment = deep_coral_loss(source_features, target_features)
            elif method_key == "mmd_tce":
                raw_alignment = compute_mmd2(source_features, target_features)
            else:
                if discriminator is None:
                    raise RuntimeError("FAIL_CLOSED DANN discriminator missing")
                raw_alignment = _domain_bce(
                    source_features, target_features, discriminator
                )
        weighted_alignment = float(alignment_weight) * raw_alignment
        weighted_target_ce = float(spec["target_ce_weight"]) * raw_target_ce
        total = source_ce + weighted_alignment + weighted_target_ce
        if not torch.isfinite(total):
            raise RuntimeError("FAIL_CLOSED non-finite labeled-conditional objective")
        total.backward()
        torch.nn.utils.clip_grad_norm_(
            [parameter for group in optimizer.param_groups for parameter in group["params"]],
            max_norm=5.0,
        )
        optimizer.step()
        diagnostics.append(
            {
                "step": int(step),
                "method": method_key,
                "source_ce": float(source_ce.detach().item()),
                "raw_target_ce": float(raw_target_ce.detach().item()),
                "weighted_target_ce": float(weighted_target_ce.detach().item()),
                "raw_alignment": float(raw_alignment.detach().item()),
                "weighted_alignment": float(weighted_alignment.detach().item()),
                "total_loss": float(total.detach().item()),
                "alignment_scope": spec["alignment_scope"],
                "target_fields": ["x", "class"],
                "target_ce_active": bool(target_ce_active),
                "target_ce_weight": float(spec["target_ce_weight"]),
                "class_conditional_active": bool(
                    spec["alignment_scope"] == "true_class_conditional"
                ),
                "valid_class_count": int(len(valid_classes)),
                "valid_classes": ";".join(str(value) for value in valid_classes),
                "dann_objective": (
                    "GRL_binary_BCE_shared_discriminator"
                    if method_key in {"dann_tce", "cc_dann"}
                    else "UNAVAILABLE"
                ),
                "phase_field_status": "UNAVAILABLE",
                "concentration_field_status": "UNAVAILABLE",
                "pseudo_label_status": "DISABLED",
                "optimizer": "Adam",
                "optimizer_lr": float(model_lr),
                "alignment_weight": float(alignment_weight),
                "source_checkpoint_ordered_fingerprint": actual_fingerprint,
            }
        )
    return model, diagnostics, time.perf_counter() - started
