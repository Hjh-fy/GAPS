"""Frozen, label-free UDA primitive used by the audited P0-U/P0-I studies."""

from __future__ import annotations

import copy
import hashlib
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from gaps_flower.domain_adaptation import wasserstein_feature_objective
from model import DomainDiscriminator
from utils import compute_mmd2, deep_coral_loss, set_random_seed

SEED = 42
MODEL_LR = 5e-4
BATCH_SIZE = 32
U1_WEIGHTS = {"source_ce": 1.0, "coral": 0.5, "global_mmd2": 0.5, "adversarial": 0.5}
CRITIC_LR = 1e-3
CRITIC_ITERS = 3
GRADIENT_PENALTY = 10.0


class FeatureOnlyCalibrationDataset(Dataset):
    """A public target API that loads and returns calibration features only."""

    def __init__(self, client_dir: str | Path, expected_rows: int = 320):
        self.feature_path = Path(client_dir) / "calibration_features.npy"
        self.features = np.load(self.feature_path, allow_pickle=False).astype(np.float32, copy=False)
        if self.features.shape != (expected_rows, 100, 8):
            raise RuntimeError(f"FAIL_CLOSED unexpected target feature shape: {self.features.shape}")

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> torch.Tensor:
        return torch.from_numpy(self.features[index])


def feature_only_loader(client_dir: str | Path, *, shuffle: bool, seed: int = SEED) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        FeatureOnlyCalibrationDataset(client_dir), batch_size=BATCH_SIZE, shuffle=shuffle,
        generator=generator if shuffle else None, num_workers=0,
    )


def require_x_only(batch: Any, *, method: str) -> torch.Tensor:
    if not isinstance(batch, torch.Tensor):
        raise RuntimeError(f"FAIL_CLOSED {method} target batch carries a non-feature object: {type(batch)!r}")
    if batch.ndim != 3 or tuple(batch.shape[1:]) != (100, 8):
        raise RuntimeError(f"FAIL_CLOSED {method} target feature shape: {tuple(batch.shape)}")
    return batch


def parameter_fingerprint(keys: Sequence[str], arrays: Iterable[np.ndarray]) -> str:
    """Deterministic content fingerprint independent of torch serialization metadata."""
    digest = hashlib.sha256()
    values = list(arrays)
    if len(keys) != len(values):
        raise RuntimeError("FAIL_CLOSED parameter key/value length mismatch")
    for key, value in zip(keys, values):
        contiguous = np.ascontiguousarray(value)
        digest.update(key.encode("utf-8")); digest.update(b"\0")
        digest.update(str(contiguous.dtype).encode("ascii")); digest.update(b"\0")
        digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _next(iterator, loader):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def _gradient_penalty(discriminator: torch.nn.Module, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    count = min(source.size(0), target.size(0)); source = source[:count]; target = target[:count]
    alpha = torch.rand(count, 1, device=source.device)
    interpolated = (alpha * source + (1.0 - alpha) * target).requires_grad_(True)
    score = discriminator(interpolated)
    gradients = torch.autograd.grad(score, interpolated, torch.ones_like(score), create_graph=True, retain_graph=True)[0]
    return ((gradients.norm(2, dim=1) - 1.0) ** 2).mean()


def run_frozen_u1(
    source_model: torch.nn.Module,
    source_loader: DataLoader,
    target_x_loader: DataLoader,
    device: torch.device,
    *,
    num_steps: int,
    seed: int = SEED,
    milestone_callback: Callable[[int, torch.nn.Module], None] | None = None,
) -> tuple[torch.nn.Module, list[dict[str, Any]], float]:
    """Exact P0-U U1 objective; the target argument is x-only by construction."""
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")
    set_random_seed(seed)
    model = copy.deepcopy(source_model).to(device)
    discriminator = DomainDiscriminator(feat_dim=64, hidden_dim=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=MODEL_LR)
    critic_optimizer = torch.optim.Adam(discriminator.parameters(), lr=CRITIC_LR)
    source_iter, target_iter = iter(source_loader), iter(target_x_loader)
    diagnostics: list[dict[str, Any]] = []
    if milestone_callback is not None:
        milestone_callback(0, model)
    started = time.perf_counter(); model.train(); discriminator.train()
    for step in range(1, num_steps + 1):
        source_batch, source_iter = _next(source_iter, source_loader)
        target_batch, target_iter = _next(target_iter, target_x_loader)
        x_s, y_s = source_batch[0].to(device), source_batch[1].to(device).long()
        x_t = require_x_only(target_batch, method="frozen-U1").to(device)
        optimizer.zero_grad(set_to_none=True)
        logits_s, feat_s, _ = model(x_s); _logits_t, feat_t, _ = model(x_t)
        for _ in range(CRITIC_ITERS):
            critic_optimizer.zero_grad(set_to_none=True)
            critic_loss = -(discriminator(feat_s.detach()).mean() - discriminator(feat_t.detach()).mean())
            critic_loss = critic_loss + GRADIENT_PENALTY * _gradient_penalty(discriminator, feat_s.detach(), feat_t.detach())
            critic_loss.backward(); torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0); critic_optimizer.step()
        source_ce = F.cross_entropy(logits_s, y_s)
        coral = deep_coral_loss(feat_s, feat_t); global_mmd2 = compute_mmd2(feat_s, feat_t)
        adversarial = wasserstein_feature_objective(discriminator, feat_s, feat_t)
        weighted_coral = U1_WEIGHTS["coral"] * coral
        weighted_mmd = U1_WEIGHTS["global_mmd2"] * global_mmd2
        weighted_adv = U1_WEIGHTS["adversarial"] * adversarial
        total = source_ce + weighted_coral + weighted_mmd + weighted_adv
        total.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
        diagnostics.append({
            "step": step, "source_ce": float(source_ce.detach()), "coral_loss": float(coral.detach()),
            "global_mmd2": float(global_mmd2.detach()), "adversarial_loss": float(adversarial.detach()),
            "weighted_coral": float(weighted_coral.detach()), "weighted_global_mmd2": float(weighted_mmd.detach()),
            "weighted_adversarial": float(weighted_adv.detach()), "total_loss": float(total.detach()),
            "target_batch_size": int(x_t.size(0)), "target_label_object_present": False,
            "target_ce_status": "UNAVAILABLE", "class_conditional_coral_status": "DISABLED",
            "class_mmd_status": "DISABLED", "stage_mmd_status": "DISABLED",
            "target_proto_anchor_status": "UNAVAILABLE", "pseudo_label_status": "DISABLED",
        })
        if milestone_callback is not None:
            milestone_callback(step, model)
    return model, diagnostics, time.perf_counter() - started
