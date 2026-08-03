"""Canonical x-only CORAL, MMD, and DANN reference adaptation."""

from __future__ import annotations

import copy
import time
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


def _next(iterator, loader):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def _target_x(batch: Any) -> torch.Tensor:
    if isinstance(batch, torch.Tensor):
        return batch
    if isinstance(batch, (tuple, list)) and len(batch) == 1 and isinstance(batch[0], torch.Tensor):
        return batch[0]
    raise RuntimeError("FAIL_CLOSED E2 target batch is not x-only")


def run_canonical_uda(
    method: str,
    source_model: torch.nn.Module,
    source_loader,
    target_x_loader,
    device: torch.device,
    *,
    num_steps: int = 100,
    model_lr: float = 5e-4,
    alignment_weight: float = 0.5,
    expected_source_fingerprint: str,
    seed: int = 42,
    formal: bool = True,
) -> tuple[torch.nn.Module, list[dict[str, Any]], float]:
    """Run one fixed-endpoint canonical UDA branch with no target-label API."""
    method_key = str(method).lower()
    if method_key not in {"coral", "mmd", "dann"}:
        raise ValueError(f"Unsupported canonical UDA method: {method}")
    if formal and (
        int(num_steps) != 100
        or float(model_lr) != 5e-4
        or float(alignment_weight) != 0.5
        or int(seed) != 42
    ):
        raise ValueError("formal E2 requires steps100, Adam5e-4, weight0.5, seed42")
    actual_fingerprint = ordered_state_content_fingerprint(source_model.state_dict())
    if actual_fingerprint != expected_source_fingerprint:
        raise RuntimeError("FAIL_CLOSED source fingerprint mismatch")
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")

    set_random_seed(int(seed))
    model = copy.deepcopy(source_model).to(device)
    model.train()
    discriminator: torch.nn.Module | None = None
    optimizer: torch.optim.Optimizer | None = None
    source_iter, target_iter = iter(source_loader), iter(target_x_loader)
    diagnostics: list[dict[str, Any]] = []
    started = time.perf_counter()

    for step in range(1, int(num_steps) + 1):
        source_batch, source_iter = _next(source_iter, source_loader)
        target_batch, target_iter = _next(target_iter, target_x_loader)
        x_s = source_batch[0].to(device)
        y_s = source_batch[1].to(device).long()
        x_t = _target_x(target_batch).to(device)
        logits_s, feat_s, _ = model(x_s)
        _logits_t, feat_t, _ = model(x_t)
        if optimizer is None:
            parameters = list(model.parameters())
            if method_key == "dann":
                feat_dim = int(feat_s.shape[-1])
                discriminator = torch.nn.Sequential(
                    torch.nn.Linear(feat_dim, 32),
                    torch.nn.ReLU(),
                    torch.nn.Linear(32, 1),
                ).to(device)
                parameters.extend(discriminator.parameters())
            optimizer = torch.optim.Adam(parameters, lr=float(model_lr))
        optimizer.zero_grad(set_to_none=True)
        source_ce = F.cross_entropy(logits_s, y_s)
        if method_key == "coral":
            raw_alignment = deep_coral_loss(feat_s, feat_t)
            dann_objective = "UNAVAILABLE"
        elif method_key == "mmd":
            raw_alignment = compute_mmd2(feat_s, feat_t)
            dann_objective = "UNAVAILABLE"
        else:
            if discriminator is None:
                raise RuntimeError("FAIL_CLOSED DANN discriminator missing")
            features = torch.cat([feat_s, feat_t], dim=0)
            domain_targets = torch.cat(
                [
                    torch.zeros(feat_s.size(0), 1, device=device),
                    torch.ones(feat_t.size(0), 1, device=device),
                ],
                dim=0,
            )
            domain_logits = discriminator(_grl(features))
            raw_alignment = F.binary_cross_entropy_with_logits(
                domain_logits, domain_targets
            )
            dann_objective = "GRL_binary_BCE"
        weighted_alignment = float(alignment_weight) * raw_alignment
        total = source_ce + weighted_alignment
        if not torch.isfinite(total):
            raise RuntimeError("FAIL_CLOSED non-finite canonical UDA objective")
        total.backward()
        torch.nn.utils.clip_grad_norm_(
            [parameter for group in optimizer.param_groups for parameter in group["params"]],
            max_norm=5.0,
        )
        optimizer.step()
        diagnostics.append(
            {
                "step": step,
                "method": method_key,
                "source_ce": float(source_ce.detach().item()),
                "raw_alignment": float(raw_alignment.detach().item()),
                "weighted_alignment": float(weighted_alignment.detach().item()),
                "total_loss": float(total.detach().item()),
                "alignment_scope": "unconditional_global",
                "active_alignment": method_key,
                "target_fields": ["x"],
                "target_label_object_present": False,
                "target_ce_status": "UNAVAILABLE",
                "class_conditional_status": "UNAVAILABLE",
                "phase_conditional_status": "UNAVAILABLE",
                "concentration_status": "UNAVAILABLE",
                "pseudo_label_status": "DISABLED",
                "dann_objective": dann_objective,
                "optimizer": "Adam",
                "optimizer_lr": float(model_lr),
                "alignment_weight": float(alignment_weight),
                "source_checkpoint_ordered_fingerprint": actual_fingerprint,
            }
        )
    return model, diagnostics, time.perf_counter() - started
