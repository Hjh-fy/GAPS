"""Audited C5 semi-supervised commissioning primitives for Gate 3.

Training interfaces in this module intentionally separate unlabeled target X
from hidden ground truth.  Hidden labels are accepted only by the explicitly
post-hoc diagnostic function after endpoint locking.
"""

from __future__ import annotations

import copy
import itertools
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset

from gaps_flower.posthoc_commissioning import relative_parameter_displacement
from utils import set_random_seed


@dataclass(frozen=True)
class G3Config:
    steps: int = 100
    optimizer: str = "Adam"
    lr: float = 5e-4
    batch_size: int = 32
    seed: int = 42
    ema_alpha: float = 0.99
    lambda_proto: float = 0.05
    mme_lambda: float = 0.1
    taus: tuple[float, ...] = (0.90, 0.95)
    lambda_us: tuple[float, ...] = (0.25, 0.5, 1.0)

    def grid(self) -> tuple[tuple[float, float], ...]:
        return tuple(itertools.product(self.taus, self.lambda_us))


@dataclass(frozen=True)
class G3Request:
    source_checkpoint: Path
    calibration_manifest: Path
    labeled_manifest: Path
    target_test_manifest: Path | None = None

    def validate_static_boundary(self) -> None:
        if self.target_test_manifest is not None:
            raise ValueError("target test manifest must never enter G3 training API")


@dataclass(frozen=True)
class G3Partition:
    calibration_identities: tuple[str, ...]
    labeled_indices: tuple[int, ...]
    unlabeled_indices: tuple[int, ...]
    labeled_stratum_counts: dict[tuple[int, float], int]
    unlabeled_stratum_counts: dict[tuple[int, float], int]
    labeled_indices_by_stratum: dict[tuple[int, float], tuple[int, int]]


@dataclass(frozen=True)
class G3Fold:
    fold: int
    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    import json

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"manifest must be a non-empty JSON list: {path}")
    return payload


def _identity(row: dict[str, Any]) -> str:
    identity = str(row.get("physical_identity", "")).strip()
    if not identity:
        raise ValueError("manifest row has no physical_identity")
    return identity


def _stratum(row: dict[str, Any]) -> tuple[int, float]:
    class_id = int(row.get("class_id", row.get("classification_label", -1)))
    concentration = float(row.get("concentration", float("nan")))
    if class_id not in range(4) or not np.isfinite(concentration):
        raise ValueError("invalid class/concentration stratum")
    return class_id, concentration


def build_g3_partition(
    calibration_manifest: Path,
    labeled_manifest: Path,
) -> G3Partition:
    calibration = _read_manifest(calibration_manifest)
    labeled = _read_manifest(labeled_manifest)
    if len(calibration) != 320 or len(labeled) != 80:
        raise ValueError("G3 requires exactly 320 calibration and 80 labeled rows")
    for row in calibration + labeled:
        if int(row.get("client_id", -1)) != 5 or row.get("role") != "calibration":
            raise ValueError("G3 manifests must contain only C5 calibration rows")
    calibration_identities = [_identity(row) for row in calibration]
    labeled_identities = [_identity(row) for row in labeled]
    if len(set(calibration_identities)) != 320 or len(set(labeled_identities)) != 80:
        raise ValueError("G3 manifests contain duplicate identities")
    index_by_identity = {identity: index for index, identity in enumerate(calibration_identities)}
    if not set(labeled_identities).issubset(index_by_identity):
        raise ValueError("5% labeled manifest is not nested in canonical calibration")
    labeled_indices = tuple(index_by_identity[identity] for identity in labeled_identities)
    labeled_set = set(labeled_indices)
    unlabeled_indices = tuple(index for index in range(320) if index not in labeled_set)
    labeled_counts = Counter(_stratum(calibration[index]) for index in labeled_indices)
    unlabeled_counts = Counter(_stratum(calibration[index]) for index in unlabeled_indices)
    if len(labeled_counts) != 40 or set(labeled_counts.values()) != {2}:
        raise ValueError("5% labeled pool is not 2-per-40-stratum")
    if len(unlabeled_counts) != 40 or set(unlabeled_counts.values()) != {6}:
        raise ValueError("15% unlabeled pool is not 6-per-40-stratum")
    grouped: dict[tuple[int, float], list[int]] = defaultdict(list)
    for index in labeled_indices:
        grouped[_stratum(calibration[index])].append(index)
    grouped_fixed = {
        key: tuple(indices) for key, indices in sorted(grouped.items())
    }
    return G3Partition(
        calibration_identities=tuple(calibration_identities),
        labeled_indices=labeled_indices,
        unlabeled_indices=unlabeled_indices,
        labeled_stratum_counts=dict(labeled_counts),
        unlabeled_stratum_counts=dict(unlabeled_counts),
        labeled_indices_by_stratum=grouped_fixed,
    )


def deterministic_two_fold(partition: G3Partition) -> tuple[G3Fold, G3Fold]:
    folds: list[G3Fold] = []
    for fold in range(2):
        train: list[int] = []
        validation: list[int] = []
        for indices in partition.labeled_indices_by_stratum.values():
            if len(indices) != 2:
                raise ValueError("each labeled stratum must contain two windows")
            train.append(indices[fold])
            validation.append(indices[1 - fold])
        folds.append(
            G3Fold(
                fold=fold,
                train_indices=tuple(train),
                validation_indices=tuple(validation),
            )
        )
    return folds[0], folds[1]


class UnlabeledTargetDataset(Dataset):
    """X-only dataset: true class/phase/concentration are not retained."""

    def __init__(self, features: np.ndarray, identities: Sequence[str]) -> None:
        values = np.asarray(features, dtype=np.float32)
        if values.ndim != 3 or tuple(values.shape[1:]) != (50, 8):
            raise ValueError(f"unlabeled features must have shape (N,50,8), got {values.shape}")
        if len(values) != len(identities) or len(set(identities)) != len(identities):
            raise ValueError("unlabeled feature/identity mismatch or duplicate")
        if not np.all(np.isfinite(values)):
            raise ValueError("unlabeled features must be finite")
        self.features = torch.from_numpy(values.copy())
        self.identities = tuple(str(identity) for identity in identities)

    def __len__(self) -> int:
        return len(self.identities)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {"x": self.features[index], "identity": self.identities[index]}


def labeled_loader(
    features: np.ndarray,
    labels: np.ndarray,
    indices: Sequence[int],
    *,
    batch_size: int,
    seed: int,
    shuffle: bool = True,
) -> DataLoader:
    selected = np.asarray(tuple(indices), dtype=np.int64)
    dataset = TensorDataset(
        torch.from_numpy(np.asarray(features[selected], dtype=np.float32)),
        torch.from_numpy(np.asarray(labels[selected], dtype=np.int64)),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
    )


def unlabeled_loader(
    features: np.ndarray,
    identities: Sequence[str],
    *,
    batch_size: int,
    seed: int,
    shuffle: bool = True,
) -> DataLoader:
    return DataLoader(
        UnlabeledTargetDataset(features, identities),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
    )


class _GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value: torch.Tensor, scale: float) -> torch.Tensor:
        ctx.scale = float(scale)
        return value.view_as(value)

    @staticmethod
    def backward(ctx, gradient: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.scale * gradient, None


def gradient_reverse(value: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    return _GradientReverse.apply(value, float(scale))


@torch.no_grad()
def update_ema_teacher(
    teacher: torch.nn.Module,
    student: torch.nn.Module,
    *,
    alpha: float,
) -> None:
    for teacher_parameter, student_parameter in zip(
        teacher.parameters(), student.parameters(), strict=True
    ):
        teacher_parameter.mul_(alpha).add_(student_parameter.detach(), alpha=1.0 - alpha)
        teacher_parameter.requires_grad_(False)
    for teacher_buffer, student_buffer in zip(
        teacher.buffers(), student.buffers(), strict=True
    ):
        if teacher_buffer.is_floating_point():
            teacher_buffer.mul_(alpha).add_(student_buffer.detach(), alpha=1.0 - alpha)
        else:
            teacher_buffer.copy_(student_buffer)


def _next(iterator: Iterable, loader: DataLoader):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


@torch.no_grad()
def compute_frozen_class_prototypes(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    num_classes: int = 4,
) -> torch.Tensor:
    model.eval()
    sums: list[torch.Tensor | None] = [None] * num_classes
    counts = [0] * num_classes
    for batch in loader:
        x = batch[0].to(device)
        y = batch[1].to(device).long()
        _, features, _ = model(x)
        for class_id in range(num_classes):
            mask = y == class_id
            if mask.any():
                value = features[mask].sum(dim=0)
                sums[class_id] = value if sums[class_id] is None else sums[class_id] + value
                counts[class_id] += int(mask.sum().item())
    if any(value is None or count == 0 for value, count in zip(sums, counts)):
        raise ValueError("source prototype loader does not cover every class")
    prototypes = torch.stack(
        [value / count for value, count in zip(sums, counts) if value is not None]
    )
    return F.normalize(prototypes, dim=1).detach()


def _system_summary(
    model: torch.nn.Module,
    source: torch.nn.Module,
    *,
    method: str,
    config: G3Config,
    seconds: float,
) -> dict[str, Any]:
    return {
        "method": method,
        "steps": config.steps,
        "optimizer": config.optimizer,
        "lr": config.lr,
        "seed": config.seed,
        "trainable_parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "total_parameters": int(sum(p.numel() for p in model.parameters())),
        "adaptation_seconds": float(seconds),
        "relative_parameter_displacement": relative_parameter_displacement(model, source),
    }


def mme_compatible_adapt(
    source_model: torch.nn.Module,
    labeled: DataLoader,
    unlabeled: DataLoader,
    *,
    device: torch.device,
    config: G3Config,
) -> tuple[torch.nn.Module, list[dict[str, float]], dict[str, Any]]:
    """MME objective on the frozen experiment's existing linear head.

    This is deliberately named compatible rather than an exact reproduction:
    the canonical GAPS classifier is retained instead of replacing it with the
    original paper's temperature-scaled cosine classifier.
    """
    set_random_seed(config.seed)
    source = copy.deepcopy(source_model).to(device)
    model = copy.deepcopy(source_model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    labeled_iterator = iter(labeled)
    unlabeled_iterator = iter(unlabeled)
    rows: list[dict[str, float]] = []
    started = time.perf_counter()
    model.train()
    for step in range(1, config.steps + 1):
        labeled_batch, labeled_iterator = _next(labeled_iterator, labeled)
        unlabeled_batch, unlabeled_iterator = _next(unlabeled_iterator, unlabeled)
        x_l, y_l = labeled_batch[0].to(device), labeled_batch[1].to(device).long()
        x_u = unlabeled_batch["x"].to(device)
        optimizer.zero_grad()
        logits_l, _, _ = model(x_l)
        _, features_u, _ = model(x_u)
        logits_u = model.classifier(gradient_reverse(features_u))
        probabilities_u = torch.softmax(logits_u, dim=1)
        entropy_negative = torch.sum(
            probabilities_u * torch.log(probabilities_u.clamp_min(1e-8)), dim=1
        ).mean()
        supervised = F.cross_entropy(logits_l, y_l)
        weighted_mme = config.mme_lambda * entropy_negative
        loss = supervised + weighted_mme
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        rows.append(
            {
                "step": float(step),
                "supervised_ce": float(supervised.detach().item()),
                "mme_negative_entropy_raw": float(entropy_negative.detach().item()),
                "mme_weighted": float(weighted_mme.detach().item()),
                "unlabeled_mean_confidence": float(probabilities_u.max(dim=1).values.mean().item()),
            }
        )
    seconds = time.perf_counter() - started
    summary = _system_summary(
        model, source, method="mme_compatible_linear_head", config=config, seconds=seconds
    )
    summary.update(
        {
            "mme_lambda": config.mme_lambda,
            "exact_mme_reproduction": False,
            "architecture_note": "existing normalized-feature linear classifier with bias retained",
            "optimizer_updates": config.steps,
        }
    )
    return model, rows, summary


def gaps_ssda_adapt(
    source_model: torch.nn.Module,
    labeled: DataLoader,
    unlabeled: DataLoader,
    frozen_source_prototypes: torch.Tensor,
    *,
    tau: float,
    lambda_u: float,
    device: torch.device,
    config: G3Config,
) -> tuple[torch.nn.Module, torch.nn.Module, list[dict[str, float]], dict[str, Any]]:
    set_random_seed(config.seed)
    source = copy.deepcopy(source_model).to(device)
    student = copy.deepcopy(source_model).to(device)
    teacher = copy.deepcopy(source_model).to(device)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    prototypes = F.normalize(frozen_source_prototypes.detach().to(device), dim=1)
    if prototypes.shape[0] != 4:
        raise ValueError("GAPS-SSDA requires four frozen class prototypes")
    optimizer = torch.optim.Adam(student.parameters(), lr=config.lr)
    labeled_iterator = iter(labeled)
    unlabeled_iterator = iter(unlabeled)
    rows: list[dict[str, float]] = []
    started = time.perf_counter()
    student.train()
    for step in range(1, config.steps + 1):
        labeled_batch, labeled_iterator = _next(labeled_iterator, labeled)
        unlabeled_batch, unlabeled_iterator = _next(unlabeled_iterator, unlabeled)
        x_l, y_l = labeled_batch[0].to(device), labeled_batch[1].to(device).long()
        x_u = unlabeled_batch["x"].to(device)
        with torch.no_grad():
            teacher_logits, _, _ = teacher(x_u)
            teacher_probabilities = torch.softmax(teacher_logits, dim=1)
            confidence, pseudo_labels = teacher_probabilities.max(dim=1)
            accepted = confidence >= float(tau)
        optimizer.zero_grad()
        logits_l, features_l, _ = student(x_l)
        logits_u, features_u, _ = student(x_u)
        supervised = F.cross_entropy(logits_l, y_l)
        if accepted.any():
            pseudo_ce = F.cross_entropy(logits_u[accepted], pseudo_labels[accepted])
            anchor_features = torch.cat((features_l, features_u[accepted]), dim=0)
            anchor_labels = torch.cat((y_l, pseudo_labels[accepted]), dim=0)
        else:
            pseudo_ce = logits_u.sum() * 0.0
            anchor_features = features_l
            anchor_labels = y_l
        prototype_raw = torch.sum(
            (F.normalize(anchor_features, dim=1) - prototypes[anchor_labels]) ** 2,
            dim=1,
        ).mean()
        weighted_pseudo = float(lambda_u) * pseudo_ce
        weighted_prototype = config.lambda_proto * prototype_raw
        loss = supervised + weighted_pseudo + weighted_prototype
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 5.0)
        optimizer.step()
        update_ema_teacher(teacher, student, alpha=config.ema_alpha)
        teacher.eval()
        rows.append(
            {
                "step": float(step),
                "supervised_ce": float(supervised.detach().item()),
                "pseudo_ce_raw": float(pseudo_ce.detach().item()),
                "pseudo_ce_weighted": float(weighted_pseudo.detach().item()),
                "prototype_raw": float(prototype_raw.detach().item()),
                "prototype_weighted": float(weighted_prototype.detach().item()),
                "pseudo_acceptance_rate": float(accepted.float().mean().item()),
                "pseudo_mean_confidence": float(confidence.mean().item()),
            }
        )
    seconds = time.perf_counter() - started
    summary = _system_summary(
        student, source, method="gaps_ssda", config=config, seconds=seconds
    )
    summary.update(
        {
            "tau": float(tau),
            "lambda_u": float(lambda_u),
            "lambda_proto": config.lambda_proto,
            "ema_alpha": config.ema_alpha,
            "source_prototype_scope": "class_only",
            "optimizer_updates": config.steps,
        }
    )
    return student, teacher, rows, summary


@torch.no_grad()
def predict_probabilities(
    model: torch.nn.Module,
    features: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    loader = DataLoader(
        TensorDataset(torch.from_numpy(np.asarray(features, dtype=np.float32))),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    model.eval()
    parts: list[np.ndarray] = []
    for (x,) in loader:
        logits, _, _ = model(x.to(device))
        parts.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(parts, axis=0)


def macro_f1_nll(labels: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predictions = probabilities.argmax(axis=1)
    f1_values: list[float] = []
    for class_id in range(4):
        true_positive = float(np.sum((labels == class_id) & (predictions == class_id)))
        false_positive = float(np.sum((labels != class_id) & (predictions == class_id)))
        false_negative = float(np.sum((labels == class_id) & (predictions != class_id)))
        denominator = 2 * true_positive + false_positive + false_negative
        f1_values.append(2 * true_positive / denominator if denominator else 0.0)
    true_probability = np.clip(probabilities[np.arange(len(labels)), labels], 1e-12, 1.0)
    return float(np.mean(f1_values)), float(-np.log(true_probability).mean())


def posthoc_hidden_pseudo_diagnostic(
    teacher: torch.nn.Module,
    unlabeled_dataset: UnlabeledTargetDataset,
    hidden_labels: np.ndarray,
    *,
    tau: float,
    device: torch.device,
    batch_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Offline-only truth audit; never call during training or selection."""
    labels = np.asarray(hidden_labels, dtype=np.int64)
    if labels.shape != (len(unlabeled_dataset),):
        raise ValueError("hidden diagnostic label count mismatch")
    probabilities = predict_probabilities(
        teacher,
        unlabeled_dataset.features.numpy(),
        device=device,
        batch_size=batch_size,
    )
    confidence = probabilities.max(axis=1)
    pseudo = probabilities.argmax(axis=1)
    accepted = confidence >= float(tau)
    rows: list[dict[str, Any]] = []
    for index, identity in enumerate(unlabeled_dataset.identities):
        rows.append(
            {
                "physical_identity": identity,
                "pseudo_class": int(pseudo[index]),
                "confidence": float(confidence[index]),
                "accepted": bool(accepted[index]),
                "hidden_true_class_posthoc": int(labels[index]),
                "pseudo_correct_posthoc": bool(pseudo[index] == labels[index]),
            }
        )
    per_class: dict[str, dict[str, float | int | None]] = {}
    for class_id in range(4):
        predicted_mask = pseudo == class_id
        accepted_mask = accepted & predicted_mask
        per_class[str(class_id)] = {
            "predicted": int(predicted_mask.sum()),
            "accepted": int(accepted_mask.sum()),
            "coverage_of_unlabeled_pool": float(accepted_mask.mean()),
            "precision_posthoc": (
                float((pseudo[accepted_mask] == labels[accepted_mask]).mean())
                if accepted_mask.any()
                else None
            ),
        }
    summary = {
        "scope": "POST_HOC_DIAGNOSTIC_ONLY",
        "tau": float(tau),
        "N": int(len(labels)),
        "accepted": int(accepted.sum()),
        "acceptance_rate": float(accepted.mean()),
        "mean_confidence": float(confidence.mean()),
        "accepted_mean_confidence": float(confidence[accepted].mean()) if accepted.any() else None,
        "pseudo_label_precision_posthoc": (
            float((pseudo[accepted] == labels[accepted]).mean()) if accepted.any() else None
        ),
        "confidence_quantiles": {
            str(quantile): float(np.quantile(confidence, quantile))
            for quantile in (0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0)
        },
        "per_predicted_class": per_class,
    }
    return rows, summary
