"""Fit gated per-class specialist regressors for deployment.

This is a deployment-focused subset of the single-machine auto_v2_specialist
idea. It keeps the existing affine calibration as the general baseline, trains
specialist regressors only for requested classes, and enables a specialist route
only when validation metrics improve.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import FLConfig
from gaps_deploy.calibration import PhaseAffineCalibrator
from gaps_flower.calibration_fit import (
    GAS_NAMES,
    _create_target_calibration_loader,
    _denormalize_by_class_torch,
    build_routing_config,
)
from gaps_flower.regression_task import create_regression_model, make_regression_config
from utils import create_model_by_config, normalize_concentration, set_random_seed

CONC_RANGES: Dict[int, Tuple[float, float]] = {
    0: (12.5, 125.0),
    1: (25.0, 250.0),
    2: (12.5, 125.0),
    3: (25.0, 250.0),
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("specialist_calibration_fit")


def _parse_classes(raw: str) -> List[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def _extract_model_state(checkpoint: Any) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        if "model_state" in checkpoint:
            return checkpoint["model_state"]
        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]
    return checkpoint


def _make_regression_config_from_checkpoint(
    args: argparse.Namespace,
    device: torch.device,
    checkpoint: Any,
):
    model_config = checkpoint.get("model_config", {}) if isinstance(checkpoint, dict) else {}
    reg_head_depth = (
        args.reg_head_depth
        if args.reg_head_depth is not None
        else model_config.get("reg_head_depth")
    )
    reg_output_mode = args.reg_output_mode or model_config.get("reg_output_mode")
    use_reg_window_stats = (
        args.reg_window_stats
        if args.reg_window_stats
        else model_config.get("reg_window_stats")
    )
    reg_window_stats_mode = args.reg_window_stats_mode or model_config.get("reg_window_stats_mode")
    reg_window_stats_dim = (
        args.reg_window_stats_dim
        if args.reg_window_stats_dim is not None
        else model_config.get("reg_window_stats_dim")
    )
    reg_response_branch = args.reg_response_branch or model_config.get("reg_response_branch")
    reg_dct_k = args.reg_dct_k if args.reg_dct_k is not None else model_config.get("reg_dct_k")
    reg_dct_gamma_init = (
        args.reg_dct_gamma_init
        if args.reg_dct_gamma_init is not None
        else model_config.get("reg_dct_gamma_init")
    )
    reg_dct_dropout = (
        args.reg_dct_dropout
        if args.reg_dct_dropout is not None
        else model_config.get("reg_dct_dropout")
    )
    reg_msconv_channels = (
        args.reg_msconv_channels
        if args.reg_msconv_channels is not None
        else model_config.get("reg_msconv_channels")
    )
    reg_msconv_kernels = args.reg_msconv_kernels or model_config.get("reg_msconv_kernels")
    reg_msconv_gamma_init = (
        args.reg_msconv_gamma_init
        if args.reg_msconv_gamma_init is not None
        else model_config.get("reg_msconv_gamma_init")
    )
    reg_msconv_dropout = (
        args.reg_msconv_dropout
        if args.reg_msconv_dropout is not None
        else model_config.get("reg_msconv_dropout")
    )
    use_reg_tcn_adapter = (
        args.reg_tcn_adapter
        if args.reg_tcn_adapter
        else model_config.get("reg_tcn_adapter")
    )
    reg_tcn_adapter_kernel = (
        args.reg_tcn_adapter_kernel
        if args.reg_tcn_adapter_kernel is not None
        else model_config.get("reg_tcn_adapter_kernel")
    )
    reg_tcn_adapter_gamma_init = (
        args.reg_tcn_adapter_gamma_init
        if args.reg_tcn_adapter_gamma_init is not None
        else model_config.get("reg_tcn_adapter_gamma_init")
    )
    reg_tcn_adapter_dropout = (
        args.reg_tcn_adapter_dropout
        if args.reg_tcn_adapter_dropout is not None
        else model_config.get("reg_tcn_adapter_dropout")
    )
    use_reg_shared_trunk = (
        args.reg_use_shared_trunk
        if hasattr(args, 'reg_use_shared_trunk') and args.reg_use_shared_trunk
        else model_config.get("reg_use_shared_trunk")
    )
    reg_shared_trunk_dim = (
        args.reg_shared_trunk_dim
        if hasattr(args, 'reg_shared_trunk_dim') and args.reg_shared_trunk_dim is not None
        else model_config.get("reg_shared_trunk_dim")
    )
    reg_gas_emb_dim = (
        args.reg_gas_emb_dim
        if hasattr(args, 'reg_gas_emb_dim') and args.reg_gas_emb_dim is not None
        else model_config.get("reg_gas_emb_dim")
    )
    reg_residual_head_depth = (
        args.reg_residual_head_depth
        if hasattr(args, 'reg_residual_head_depth') and args.reg_residual_head_depth is not None
        else model_config.get("reg_residual_head_depth")
    )
    use_reg_ratio_branch = (
        args.use_reg_ratio_branch
        if hasattr(args, 'use_reg_ratio_branch') and args.use_reg_ratio_branch
        else model_config.get("use_reg_ratio_branch")
    )
    reg_ratio_gamma_init = (
        args.reg_ratio_gamma_init
        if hasattr(args, 'reg_ratio_gamma_init') and args.reg_ratio_gamma_init is not None
        else model_config.get("reg_ratio_gamma_init")
    )
    reg_ratio_dropout = (
        args.reg_ratio_dropout
        if hasattr(args, 'reg_ratio_dropout') and args.reg_ratio_dropout is not None
        else model_config.get("reg_ratio_dropout")
    )
    return make_regression_config(
        device=device.type,
        batch_size=args.batch_size,
        lr=args.lr,
        reg_head_depth=reg_head_depth,
        reg_output_mode=reg_output_mode,
        use_reg_window_stats=use_reg_window_stats,
        reg_window_stats_mode=reg_window_stats_mode,
        reg_window_stats_dim=reg_window_stats_dim,
        reg_response_branch=reg_response_branch,
        reg_dct_k=reg_dct_k,
        reg_dct_gamma_init=reg_dct_gamma_init,
        reg_dct_dropout=reg_dct_dropout,
        reg_msconv_channels=reg_msconv_channels,
        reg_msconv_kernels=reg_msconv_kernels,
        reg_msconv_gamma_init=reg_msconv_gamma_init,
        reg_msconv_dropout=reg_msconv_dropout,
        use_reg_tcn_adapter=use_reg_tcn_adapter,
        reg_tcn_adapter_kernel=reg_tcn_adapter_kernel,
        reg_tcn_adapter_gamma_init=reg_tcn_adapter_gamma_init,
        reg_tcn_adapter_dropout=reg_tcn_adapter_dropout,
        use_reg_shared_trunk=use_reg_shared_trunk,
        reg_shared_trunk_dim=reg_shared_trunk_dim,
        reg_gas_emb_dim=reg_gas_emb_dim,
        reg_residual_head_depth=reg_residual_head_depth,
        use_reg_ratio_branch=use_reg_ratio_branch,
        reg_ratio_gamma_init=reg_ratio_gamma_init,
        reg_ratio_dropout=reg_ratio_dropout,
    )


def _response_signature_descriptor(x: np.ndarray) -> np.ndarray:
    """Return compact 40-D response signatures for deployment QC.

    Each 100 x 8 window is represented by five per-sensor summaries:
    mean, std, amplitude, late-minus-early slope, and short-term noise.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 3:
        raise ValueError(f"Expected [batch,time,sensors] features, got {x.shape}")
    edge = max(1, int(round(x.shape[1] * 0.10)))
    mean_ch = x.mean(axis=1)
    std_ch = x.std(axis=1)
    amp_ch = x.max(axis=1) - x.min(axis=1)
    slope_ch = x[:, -edge:, :].mean(axis=1) - x[:, :edge, :].mean(axis=1)
    diff = np.diff(x, axis=1)
    noise_ch = diff.std(axis=1) if diff.size else np.zeros_like(mean_ch)
    return np.concatenate([mean_ch, std_ch, amp_ch, slope_ch, noise_ch], axis=1)


def _robust_center_scale(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(arr, dtype=np.float64)
    center = np.median(arr, axis=0)
    q75 = np.percentile(arr, 75, axis=0)
    q25 = np.percentile(arr, 25, axis=0)
    scale = q75 - q25
    std = arr.std(axis=0)
    scale = np.where(np.abs(scale) > 1e-8, scale, std)
    scale = np.where(np.abs(scale) > 1e-8, scale, 1.0)
    return center, scale


def _build_response_refs(loader: DataLoader, num_classes: int = 4) -> Dict[str, Dict[str, Any]]:
    """Build per-class response signature references for deployment QC.

    The signature matches ``RiskScoreComputer``. New packages use a compact
    40-D response descriptor per window: mean, std, amplitude, slope, and
    short-term noise for each sensor channel. Distances are computed in robust
    z-scored signature space and normalized by the LOOCV p90 nearest-neighbor
    distance inside each class.
    """
    sigs_by_class: Dict[int, List[np.ndarray]] = {c: [] for c in range(num_classes)}
    rows_by_class: Dict[int, List[Dict[str, Any]]] = {c: [] for c in range(num_classes)}

    for batch in loader:
        x = batch[0].detach().cpu().numpy()
        y_cls = batch[1].detach().cpu().numpy().astype(int).reshape(-1)
        y_reg_full = batch[2].detach().cpu().numpy()
        y_phase = batch[3].detach().cpu().numpy().astype(int).reshape(-1) if len(batch) > 3 else np.full_like(y_cls, -1)
        sigs = _response_signature_descriptor(x)
        for i, cls_id in enumerate(y_cls.tolist()):
            if cls_id < 0 or cls_id >= num_classes:
                continue
            concentration = float(y_reg_full[i, cls_id]) if y_reg_full.ndim == 2 else float(y_reg_full[i])
            sigs_by_class[cls_id].append(np.asarray(sigs[i], dtype=np.float64))
            rows_by_class[cls_id].append(
                {
                    "concentration": concentration,
                    "phase": int(y_phase[i]) if i < len(y_phase) else -1,
                }
            )

    refs: Dict[str, Dict[str, Any]] = {}
    for cls_id, sig_list in sigs_by_class.items():
        if not sig_list:
            continue
        arr = np.vstack(sig_list).astype(np.float64)
        center, scale = _robust_center_scale(arr)
        z_sigs = (arr - center.reshape(1, -1)) / scale.reshape(1, -1)

        if len(z_sigs) >= 2:
            nearest = []
            for i in range(len(z_sigs)):
                dists = np.linalg.norm(z_sigs - z_sigs[i].reshape(1, -1), axis=1)
                dists[i] = np.inf
                nearest.append(float(np.min(dists)))
            loocv_p90 = float(np.percentile(nearest, 90))
        else:
            loocv_p90 = 1.0
        if not np.isfinite(loocv_p90) or loocv_p90 < 1e-8:
            loocv_p90 = 1.0

        refs[str(cls_id)] = {
            "center": center.tolist(),
            "scale": scale.tolist(),
            "z_sigs": z_sigs.tolist(),
            "loocv_p90": loocv_p90,
            "rows": rows_by_class[cls_id],
            "signature": "mean_std_amp_slope_noise",
            "signature_dim": int(arr.shape[1]),
            "center_stat": "median",
            "scale_stat": "iqr_std_fallback",
            "response_ranking_enabled": True,
            "qc_signal_version": "response_rank_v2_40d",
            "n_samples": len(sig_list),
        }
    return refs


def _split_loader(
    loader: DataLoader,
    batch_size: int,
    val_ratio: float,
    seed: int,
    split_by: str = "class",
) -> Tuple[DataLoader, DataLoader]:
    """Calibration train/validation split, optionally by class and concentration."""
    dataset = loader.dataset
    labels = np.asarray(getattr(dataset, "classification_labels", []), dtype=np.int64)
    if len(labels) != len(dataset):
        raise RuntimeError("Calibration dataset must expose classification_labels.")
    reg_labels = getattr(dataset, "regression_labels", None)

    rng = np.random.default_rng(seed)
    train_indices: List[int] = []
    val_indices: List[int] = []

    groups: Dict[Tuple[int, str], List[int]] = {}
    for idx, cls_id in enumerate(labels.tolist()):
        key: Tuple[int, str]
        if split_by == "class_concentration" and reg_labels is not None:
            conc = float(reg_labels[idx, int(cls_id)])
            key = (int(cls_id), f"{conc:.6f}")
        else:
            key = (int(cls_id), "all")
        groups.setdefault(key, []).append(idx)

    for indices in groups.values():
        rng.shuffle(indices)
        if len(indices) < 2:
            train_indices.extend(indices)
            continue
        n_val = max(1, int(round(len(indices) * val_ratio)))
        n_val = min(n_val, len(indices) - 1)
        val_indices.extend(indices[:n_val])
        train_indices.extend(indices[n_val:])

    if not val_indices:
        for cls_id in sorted(np.unique(labels).tolist()):
            cls_indices = np.where(labels == cls_id)[0].tolist()
            rng.shuffle(cls_indices)
            if len(cls_indices) >= 2:
                val_indices.append(cls_indices[0])
                train_indices.extend(cls_indices[1:])

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    train_loader = DataLoader(Subset(dataset, train_indices), batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(Subset(dataset, val_indices), batch_size=batch_size, shuffle=False, num_workers=0)
    logger.info(
        "Calibration split: train=%d, val=%d, val_ratio=%.2f, split_by=%s",
        len(train_indices),
        len(val_indices),
        val_ratio,
        split_by,
    )
    return train_loader, val_loader


def _train_specialist(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_id: Optional[int],
    steps: int,
    lr: float,
    class_weight: float,
    huber_delta: float,
    reg_range_penalty: float = 0.0,
) -> float:
    """Train regression branch with an extra weight for one target class."""
    for param in model.parameters():
        param.requires_grad = False

    trainable: List[torch.nn.Parameter] = []
    for name in ["reg_proj", "reg_transformer", "reg_attn", "reg_attn_linear", "reg_stats_proj"]:
        module = getattr(model, name, None)
        if module is not None:
            trainable.extend(module.parameters())
    if getattr(model, "reg_heads", None) is not None:
        trainable.extend(model.reg_heads.parameters())
    for name in ["proto_scale", "proto_bias", "proto_conc", "conc_directions", "conc_scale", "conc_bias"]:
        param = getattr(model, name, None)
        if isinstance(param, torch.nn.Parameter):
            trainable.append(param)

    for param in trainable:
        param.requires_grad = True
    if not trainable:
        raise RuntimeError("No trainable regression parameters found.")

    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=1e-3)
    reg_output_mode = str(getattr(model, "reg_output_mode", "sigmoid")).lower()
    use_unclamped_linear_train = reg_output_mode == "linear"
    iterator = iter(loader)
    running = 0.0
    model.train()
    for _ in range(max(1, steps)):
        try:
            x, y_cls, y_reg_full, y_phase = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            x, y_cls, y_reg_full, y_phase = next(iterator)

        x = x.to(device)
        y_cls = y_cls.to(device).long()
        y_reg_full = y_reg_full.to(device)
        y_phase = y_phase.to(device).long()
        y_true = y_reg_full[torch.arange(y_cls.size(0), device=device), y_cls].unsqueeze(1)
        y_norm = normalize_concentration(y_true, y_cls)

        optimizer.zero_grad()
        _, _, reg_feat = model(x)
        pred_norm = model.forward_reg(
            reg_feat,
            y_cls=y_cls,
            y_phase=y_phase,
            clamp_output=not use_unclamped_linear_train,
        )
        losses = F.smooth_l1_loss(pred_norm, y_norm, beta=huber_delta, reduction="none").view(-1)
        weights = torch.ones_like(losses)
        if class_id is not None and int(class_id) >= 0:
            weights[y_cls.view(-1) == int(class_id)] = float(class_weight)
        loss = (losses * weights).mean()
        if use_unclamped_linear_train and reg_range_penalty > 0:
            range_violation = F.relu(-pred_norm).pow(2) + F.relu(pred_norm - 1.0).pow(2)
            loss = loss + float(reg_range_penalty) * range_violation.mean()
        loss.backward()
        optimizer.step()
        running += float(loss.item())

    avg = running / max(1, steps)
    if class_id is None or int(class_id) < 0:
        logger.info("full calibration train: steps=%d, avg_loss=%.6f", steps, avg)
    else:
        logger.info(
            "specialist class %d train: steps=%d, weight=%.2f, avg_loss=%.6f",
            class_id,
            steps,
            class_weight,
            avg,
        )
    return avg


def _apply_affine(pred: np.ndarray, cls_ids: np.ndarray, affine_params: Dict[int, Dict[str, Any]]) -> np.ndarray:
    out = pred.copy()
    for cls_id, params in affine_params.items():
        mask = cls_ids == int(cls_id)
        if mask.any():
            out[mask] = out[mask] * float(params.get("a", 1.0)) + float(params.get("b", 0.0))
    return out


def _apply_phase_affine(
    pred: np.ndarray,
    cls_ids: np.ndarray,
    phase_ids: np.ndarray,
    phase_params: Dict[int, Dict[str, Any]],
) -> np.ndarray:
    out = pred.copy()
    for cls_id, params in phase_params.items():
        class_mask = cls_ids == int(cls_id)
        if not class_mask.any():
            continue
        phase_calibrators = params.get("phase_calibrators", {})
        for phase_key, calib in phase_calibrators.items():
            mask = class_mask & (phase_ids == int(phase_key))
            if mask.any():
                out[mask] = out[mask] * float(calib.get("a", 1.0)) + float(calib.get("b", 0.0))
    return out


def _collect_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_true: List[float] = []
    all_pred: List[float] = []
    all_cls: List[int] = []
    all_phase: List[int] = []
    with torch.no_grad():
        for x, y_cls, y_reg_full, y_phase in loader:
            x = x.to(device)
            y_cls = y_cls.to(device).long()
            y_reg_full = y_reg_full.to(device)
            y_phase = y_phase.to(device).long()
            _, _, reg_feat = model(x)
            pred_norm = model.forward_reg(reg_feat, y_cls=y_cls, y_phase=y_phase)
            y_true = y_reg_full[torch.arange(y_cls.size(0), device=device), y_cls]
            pred_ppm = _denormalize_by_class_torch(pred_norm, y_cls)
            all_true.extend(y_true.cpu().numpy().tolist())
            all_pred.extend(pred_ppm.cpu().numpy().tolist())
            all_cls.extend(y_cls.cpu().numpy().astype(int).tolist())
            all_phase.extend(y_phase.cpu().numpy().astype(int).tolist())
    return (
        np.asarray(all_true, dtype=np.float64),
        np.asarray(all_pred, dtype=np.float64),
        np.asarray(all_cls, dtype=int),
        np.asarray(all_phase, dtype=int),
    )


def _metrics_from_arrays(
    true_arr: np.ndarray,
    pred_arr: np.ndarray,
    cls_arr: np.ndarray,
    num_classes: int,
) -> Dict[str, Any]:
    per_class: Dict[int, Dict[str, float]] = {}
    for cls_id in range(num_classes):
        mask = cls_arr == cls_id
        if mask.sum() < 2:
            per_class[cls_id] = {"n": int(mask.sum()), "R2": None, "MAE": None}
            continue
        err = pred_arr[mask] - true_arr[mask]
        ss_res = float(np.sum(err ** 2))
        ss_tot = float(np.sum((true_arr[mask] - true_arr[mask].mean()) ** 2))
        per_class[cls_id] = {
            "n": int(mask.sum()),
            "R2": float(1.0 - ss_res / max(ss_tot, 1e-12)),
            "MAE": float(np.mean(np.abs(err))),
            "RMSE": float(np.sqrt(np.mean(err ** 2))),
            "NRMSE_range": _nrmse_range(err, cls_arr[mask]),
            "P90AE": float(np.percentile(np.abs(err), 90)),
            "Bias": float(np.mean(err)),
        }
    err_all = pred_arr - true_arr
    ss_res_all = float(np.sum(err_all ** 2))
    ss_tot_all = float(np.sum((true_arr - true_arr.mean()) ** 2))
    return {
        "overall": {
            "R2": float(1.0 - ss_res_all / max(ss_tot_all, 1e-12)),
            "MAE": float(np.mean(np.abs(err_all))),
            "RMSE": float(np.sqrt(np.mean(err_all ** 2))),
            "NRMSE_range": _nrmse_range(err_all, cls_arr),
        },
        "per_class": {str(k): v for k, v in per_class.items()},
    }


def _evaluate_routed(
    base_model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    routing_config: Dict[str, Any],
    full_model: Optional[torch.nn.Module] = None,
    specialist_models: Optional[Dict[int, torch.nn.Module]] = None,
) -> Dict[str, Any]:
    true_arr, pred_arr, cls_arr, phase_arr = _collect_predictions(base_model, loader, device)
    selected_modes = {int(k): v for k, v in routing_config.get("selected_modes", {}).items()}

    if full_model is not None and any(mode == "full" for mode in selected_modes.values()):
        _, full_pred, full_cls, _ = _collect_predictions(full_model, loader, device)
        for cls_id, mode in selected_modes.items():
            if mode == "full":
                mask = full_cls == int(cls_id)
                pred_arr[mask] = full_pred[mask]

    for cls_id, specialist in (specialist_models or {}).items():
        mode = selected_modes.get(int(cls_id), "none")
        if mode not in {"specialist", "specialist_full"}:
            continue
        _, spec_pred, spec_cls, _ = _collect_predictions(specialist, loader, device)
        mask = spec_cls == int(cls_id)
        pred_arr[mask] = spec_pred[mask]

    affine_params = {int(k): v for k, v in routing_config.get("affine_params", {}).items()}
    phase_params = {int(k): v for k, v in routing_config.get("phase_affine_params", {}).items()}
    affine_for_active = {c: p for c, p in affine_params.items() if selected_modes.get(c) in {"bias_only", "affine_only"}}
    phase_for_active = {c: p for c, p in phase_params.items() if selected_modes.get(c) == "phase_affine_only"}
    if affine_for_active:
        pred_arr = _apply_affine(pred_arr, cls_arr, affine_for_active)
    if phase_for_active:
        pred_arr = _apply_phase_affine(pred_arr, cls_arr, phase_arr, phase_for_active)
    return _metrics_from_arrays(true_arr, pred_arr, cls_arr, num_classes)


def _fit_affine_params_on_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    mode: str = "affine_only",
) -> Dict[int, Dict[str, Any]]:
    """Fit per-class bias/affine parameters on an existing loader."""
    raw_metrics = _evaluate_oracle(model, loader, device, num_classes, affine_params=None)
    params: Dict[int, Dict[str, Any]] = {}

    stores: Dict[int, Dict[str, List[float]]] = {
        cls_id: {"true": [], "pred": []} for cls_id in range(num_classes)
    }
    model.eval()
    with torch.no_grad():
        for x, y_cls, y_reg_full, y_phase in loader:
            x = x.to(device)
            y_cls = y_cls.to(device).long()
            y_reg_full = y_reg_full.to(device)
            y_phase = y_phase.to(device).long()
            _, _, reg_feat = model(x)
            pred_norm = model.forward_reg(reg_feat, y_cls=y_cls, y_phase=y_phase)
            pred_ppm = _denormalize_by_class_torch(pred_norm, y_cls)
            y_true = y_reg_full[torch.arange(y_cls.size(0), device=device), y_cls]
            for cls_id in range(num_classes):
                mask = y_cls == cls_id
                if mask.any():
                    stores[cls_id]["true"].extend(y_true[mask].cpu().numpy().tolist())
                    stores[cls_id]["pred"].extend(pred_ppm[mask].cpu().numpy().tolist())

    for cls_id in range(num_classes):
        y_true = np.asarray(stores[cls_id]["true"], dtype=np.float64)
        y_pred = np.asarray(stores[cls_id]["pred"], dtype=np.float64)
        if len(y_true) < 2:
            params[cls_id] = {"a": 1.0, "b": 0.0, "mode": mode, "n_samples": int(len(y_true)), "calib_r2": 0.0, "calib_mae": 0.0}
            continue
        residual = y_true - y_pred
        if mode == "bias_only" or float(np.var(y_pred)) < 1e-12:
            a = 1.0
            b = float(np.mean(residual))
        else:
            coeffs, _, _, _ = np.linalg.lstsq(
                np.column_stack([y_pred, np.ones_like(y_pred)]),
                y_true,
                rcond=None,
            )
            a, b = float(coeffs[0]), float(coeffs[1])
        y_adj = a * y_pred + b
        err = y_adj - y_true
        ss_res = float(np.sum(err ** 2))
        ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
        params[cls_id] = {
            "a": a,
            "b": b,
            "mode": mode,
            "n_samples": int(len(y_true)),
            "calib_r2": float(1.0 - ss_res / max(ss_tot, 1e-12)),
            "calib_mae": float(np.mean(np.abs(err))),
            "raw_val_r2": raw_metrics.get("per_class", {}).get(str(cls_id), {}).get("R2"),
        }
    return params


def _fit_phase_affine_params_on_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    num_phases: int = 3,
    mode: str = "affine_only",
) -> Dict[int, Dict[str, Any]]:
    """Fit per-class, per-phase affine calibrators on ppm predictions."""
    true_arr, pred_arr, cls_arr, phase_arr = _collect_predictions(model, loader, device)
    params: Dict[int, Dict[str, Any]] = {}
    for cls_id in range(num_classes):
        mask = cls_arr == cls_id
        calibrator = PhaseAffineCalibrator(num_phases=num_phases)
        if mask.sum() >= 2:
            calibrator.fit(true_arr[mask], pred_arr[mask], phase_arr[mask], mode=mode)
        else:
            calibrator.fit(np.asarray([0.0, 1.0]), np.asarray([0.0, 1.0]), np.asarray([0, 0]), mode=mode)
        params[cls_id] = calibrator.to_dict()
        params[cls_id]["n_samples"] = int(mask.sum())
    return params


def _evaluate_oracle(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    affine_params: Dict[int, Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    model.eval()
    stores: Dict[int, Dict[str, List[float]]] = {
        cls_id: {"true": [], "pred": []} for cls_id in range(num_classes)
    }
    all_true: List[float] = []
    all_pred: List[float] = []
    all_cls: List[int] = []
    with torch.no_grad():
        for x, y_cls, y_reg_full, y_phase in loader:
            x = x.to(device)
            y_cls = y_cls.to(device).long()
            y_reg_full = y_reg_full.to(device)
            y_phase = y_phase.to(device).long()
            _, _, reg_feat = model(x)
            pred_norm = model.forward_reg(reg_feat, y_cls=y_cls, y_phase=y_phase)
            y_true = y_reg_full[torch.arange(y_cls.size(0), device=device), y_cls]
            pred_ppm = _denormalize_by_class_torch(pred_norm, y_cls)
            all_true.extend(y_true.cpu().numpy().tolist())
            all_pred.extend(pred_ppm.cpu().numpy().tolist())
            all_cls.extend(y_cls.cpu().numpy().astype(int).tolist())

    true_arr = np.asarray(all_true, dtype=np.float64)
    pred_arr = np.asarray(all_pred, dtype=np.float64)
    cls_arr = np.asarray(all_cls, dtype=int)
    if affine_params:
        pred_arr = _apply_affine(pred_arr, cls_arr, affine_params)

    per_class: Dict[int, Dict[str, float]] = {}
    for cls_id in range(num_classes):
        mask = cls_arr == cls_id
        if mask.sum() < 2:
            per_class[cls_id] = {"n": int(mask.sum()), "R2": None, "MAE": None}
            continue
        err = pred_arr[mask] - true_arr[mask]
        ss_res = float(np.sum(err ** 2))
        ss_tot = float(np.sum((true_arr[mask] - true_arr[mask].mean()) ** 2))
        per_class[cls_id] = {
            "n": int(mask.sum()),
            "R2": float(1.0 - ss_res / max(ss_tot, 1e-12)),
            "MAE": float(np.mean(np.abs(err))),
            "RMSE": float(np.sqrt(np.mean(err ** 2))),
            "NRMSE_range": _nrmse_range(err, cls_arr[mask]),
            "P90AE": float(np.percentile(np.abs(err), 90)),
            "Bias": float(np.mean(err)),
        }
    err_all = pred_arr - true_arr
    ss_res_all = float(np.sum(err_all ** 2))
    ss_tot_all = float(np.sum((true_arr - true_arr.mean()) ** 2))
    return {
        "overall": {
            "R2": float(1.0 - ss_res_all / max(ss_tot_all, 1e-12)),
            "MAE": float(np.mean(np.abs(err_all))),
            "RMSE": float(np.sqrt(np.mean(err_all ** 2))),
            "NRMSE_range": _nrmse_range(err_all, cls_arr),
        },
        "per_class": {str(k): v for k, v in per_class.items()},
    }


def _score(metrics: Dict[str, Any], class_id: int, metric: str) -> float:
    value = metrics.get("per_class", {}).get(str(class_id), {}).get(metric)
    if value is None:
        return -float("inf")
    if metric.upper() in {"MAE", "P90AE", "RMSE", "NRMSE_RANGE"}:
        return -float(value)
    return float(value)


def _nrmse_range(err: np.ndarray, cls_ids: np.ndarray) -> float:
    ranges = []
    for cls_id in cls_ids.astype(int).tolist():
        lo, hi = CONC_RANGES.get(int(cls_id), (0.0, 1.0))
        ranges.append(max(float(hi) - float(lo), 1e-12))
    range_arr = np.asarray(ranges, dtype=np.float64)
    return float(np.sqrt(np.mean((err / range_arr) ** 2)))


def _metric_value(metrics: Dict[str, Any], key: str) -> float | None:
    value = metrics.get(key)
    return None if value is None else float(value)


def _gate_accept(
    baseline_metrics: Dict[str, Any],
    specialist_metrics: Dict[str, Any],
    baseline_score: float,
    specialist_score: float,
    args: argparse.Namespace,
) -> Tuple[bool, Dict[str, Any]]:
    """Return gate decision with optional MAE/P90/Bias guardrails."""
    primary_ok = specialist_score >= baseline_score + float(args.min_delta)
    details: Dict[str, Any] = {
        "gate_mode": args.gate_mode,
        "primary_ok": bool(primary_ok),
    }
    if args.gate_mode == "metric":
        details.update({"p90_ok": None, "bias_ok": None})
        return bool(primary_ok), details

    baseline_p90 = _metric_value(baseline_metrics, "P90AE")
    specialist_p90 = _metric_value(specialist_metrics, "P90AE")
    baseline_bias = _metric_value(baseline_metrics, "Bias")
    specialist_bias = _metric_value(specialist_metrics, "Bias")

    if args.use_p90_guard:
        p90_ok = (
            baseline_p90 is None
            or specialist_p90 is None
            or specialist_p90 <= baseline_p90 + float(args.p90_max_worsen)
        )
    else:
        p90_ok = True
    if args.use_bias_guard:
        bias_ok = (
            baseline_bias is None
            or specialist_bias is None
            or abs(specialist_bias) <= abs(baseline_bias) + float(args.bias_max_worsen)
        )
    else:
        bias_ok = True
    details.update({
        "use_p90_guard": bool(args.use_p90_guard),
        "use_bias_guard": bool(args.use_bias_guard),
        "p90_ok": bool(p90_ok),
        "bias_ok": bool(bias_ok),
        "baseline_p90": baseline_p90,
        "specialist_p90": specialist_p90,
        "baseline_bias": baseline_bias,
        "specialist_bias": specialist_bias,
        "p90_max_worsen": args.p90_max_worsen,
        "bias_max_worsen": args.bias_max_worsen,
    })
    return bool(primary_ok and p90_ok and bias_ok), details


def _maybe_refit_specialist(
    base_model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_id: int,
    args: argparse.Namespace,
) -> Tuple[torch.nn.Module, float | None]:
    """Optionally refit an accepted specialist on the full calibration split."""
    if not args.refit_full_calib:
        return base_model, None

    refit_model = copy.deepcopy(base_model).to(device)
    refit_steps = int(args.refit_steps or args.steps)
    avg_loss = _train_specialist(
        refit_model,
        loader,
        device,
        class_id,
        steps=refit_steps,
        lr=args.lr,
        class_weight=args.class_weight,
        huber_delta=args.huber_delta,
        reg_range_penalty=args.reg_range_penalty,
    )
    return refit_model, avg_loss


def _build_auto_v2_routing(
    candidate_metrics: Dict[str, Dict[str, Any]],
    affine_params: Dict[int, Dict[str, Any]],
    bias_params: Dict[int, Dict[str, Any]],
    phase_params: Dict[int, Dict[str, Any]],
    gate_metric: str,
    min_delta: float,
    num_classes: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    selected_modes: Dict[str, str] = {}
    routed_affine: Dict[str, Dict[str, Any]] = {}
    routed_phase: Dict[str, Dict[str, Any]] = {}
    class_candidate_scores: Dict[str, Dict[str, float]] = {}

    for cls_id in range(num_classes):
        best_mode = "none"
        best_score = _score(candidate_metrics["none"], cls_id, gate_metric)
        class_candidate_scores[str(cls_id)] = {}
        for mode in ["none", "bias_only", "affine_only", "phase_affine_only", "full"]:
            score = _score(candidate_metrics[mode], cls_id, gate_metric)
            class_candidate_scores[str(cls_id)][mode] = score
            if score >= best_score + float(min_delta):
                best_mode = mode
                best_score = score

        selected_modes[str(cls_id)] = best_mode
        if best_mode == "bias_only":
            routed_affine[str(cls_id)] = bias_params[cls_id]
        elif best_mode == "affine_only":
            routed_affine[str(cls_id)] = affine_params[cls_id]
        elif best_mode == "phase_affine_only":
            routed_phase[str(cls_id)] = phase_params[cls_id]

    routing_config = {
        "selected_modes": selected_modes,
        "affine_params": routed_affine,
        "phase_affine_params": routed_phase,
        "routing_mode": "auto_v2",
        "num_classes": num_classes,
    }
    diagnostics = {
        "selected_modes": selected_modes,
        "class_candidate_scores": class_candidate_scores,
        "candidate_val_metrics": candidate_metrics,
    }
    return routing_config, diagnostics


def _run_auto_v2_specialist(args: argparse.Namespace, output_dir: Path, device: torch.device) -> None:
    loader = _create_target_calibration_loader(args.calib_data_dir, args.batch_size)
    train_loader, val_loader = _split_loader(
        loader,
        args.batch_size,
        args.val_ratio,
        args.seed,
        split_by=args.split_by,
    )
    response_refs = _build_response_refs(train_loader, num_classes=4)

    classifier = create_model_by_config(FLConfig(), with_reg_head=False).to(device)
    classifier_state = _extract_model_state(torch.load(args.classifier_ckpt, map_location=device, weights_only=False))
    classifier.load_state_dict(classifier_state, strict=False)
    classifier.eval()

    reg_ckpt = torch.load(args.regression_ckpt, map_location=device, weights_only=False)
    reg_config = _make_regression_config_from_checkpoint(args, device, reg_ckpt)
    base_model = create_regression_model(reg_config).to(device)
    reg_state = _extract_model_state(reg_ckpt)
    base_model.load_state_dict(reg_state, strict=False)
    base_model.eval()

    full_model = copy.deepcopy(base_model).to(device)
    _train_specialist(
        full_model,
        train_loader,
        device,
        class_id=None,
        steps=args.full_steps,
        lr=args.lr,
        class_weight=1.0,
        huber_delta=args.huber_delta,
        reg_range_penalty=args.reg_range_penalty,
    )

    bias_params = _fit_affine_params_on_loader(base_model, train_loader, device, num_classes=4, mode="bias_only")
    affine_params = _fit_affine_params_on_loader(base_model, train_loader, device, num_classes=4, mode="affine_only")
    phase_params = _fit_phase_affine_params_on_loader(
        base_model,
        train_loader,
        device,
        num_classes=4,
        num_phases=3,
        mode="affine_only",
    )

    candidate_metrics = {
        "none": _evaluate_oracle(base_model, val_loader, device, 4),
        "bias_only": _evaluate_routed(
            base_model,
            val_loader,
            device,
            4,
            {
                "selected_modes": {str(c): "bias_only" for c in range(4)},
                "affine_params": {str(k): v for k, v in bias_params.items()},
                "phase_affine_params": {},
            },
        ),
        "affine_only": _evaluate_routed(
            base_model,
            val_loader,
            device,
            4,
            build_routing_config(affine_params, mode="affine_only"),
        ),
        "phase_affine_only": _evaluate_routed(
            base_model,
            val_loader,
            device,
            4,
            {
                "selected_modes": {str(c): "phase_affine_only" for c in range(4)},
                "affine_params": {},
                "phase_affine_params": {str(k): v for k, v in phase_params.items()},
            },
        ),
        "full": _evaluate_routed(
            base_model,
            val_loader,
            device,
            4,
            {
                "selected_modes": {str(c): "full" for c in range(4)},
                "affine_params": {},
                "phase_affine_params": {},
            },
            full_model=full_model,
        ),
    }

    routing_config, auto_diag = _build_auto_v2_routing(
        candidate_metrics,
        affine_params,
        bias_params,
        phase_params,
        args.gate_metric,
        args.min_delta,
        num_classes=4,
    )

    specialist_dir = output_dir / "specialists"
    specialist_dir.mkdir(parents=True, exist_ok=True)
    selected_specialists: List[int] = []
    specialist_models: Dict[int, torch.nn.Module] = {}
    specialist_decisions: Dict[str, Any] = {}
    general_selected_metrics = _evaluate_routed(
        base_model,
        val_loader,
        device,
        4,
        routing_config,
        full_model=full_model,
    )

    for class_id in _parse_classes(args.classes):
        specialist = copy.deepcopy(base_model).to(device)
        _train_specialist(
            specialist,
            train_loader,
            device,
            class_id=class_id,
            steps=args.steps,
            lr=args.lr,
            class_weight=args.class_weight,
            huber_delta=args.huber_delta,
            reg_range_penalty=args.reg_range_penalty,
        )
        specialist_metrics = _evaluate_oracle(specialist, val_loader, device, 4)
        baseline_score = _score(general_selected_metrics, class_id, args.gate_metric)
        specialist_score = _score(specialist_metrics, class_id, args.gate_metric)
        baseline_cls_metrics = general_selected_metrics["per_class"].get(str(class_id), {})
        specialist_cls_metrics = specialist_metrics["per_class"].get(str(class_id), {})
        accepted, gate_details = _gate_accept(
            baseline_cls_metrics,
            specialist_cls_metrics,
            baseline_score,
            specialist_score,
            args,
        )
        specialist_decisions[str(class_id)] = {
            "metric": args.gate_metric,
            "gate_mode": args.gate_mode,
            "baseline_score": baseline_score,
            "specialist_score": specialist_score,
            "min_delta": args.min_delta,
            "gate_details": gate_details,
            "accepted": accepted,
            "baseline": baseline_cls_metrics,
            "specialist": specialist_cls_metrics,
        }
        logger.info(
            "auto_v2 class %d specialist gate: baseline=%.4f, specialist=%.4f, accepted=%s",
            class_id,
            baseline_score,
            specialist_score,
            accepted,
        )
        if accepted:
            specialist, refit_loss = _maybe_refit_specialist(specialist, loader, device, class_id, args)
            specialist_decisions[str(class_id)]["refit_loss"] = refit_loss
            routing_config["selected_modes"][str(class_id)] = "specialist_full"
            specialist_models[class_id] = specialist
            torch.save(
                {
                    "model_state": specialist.state_dict(),
                    "class_id": class_id,
                    "gate": specialist_decisions[str(class_id)],
                },
                specialist_dir / f"class_{class_id}.pth",
            )
            selected_specialists.append(class_id)

    selected_metrics = _evaluate_routed(
        base_model,
        val_loader,
        device,
        4,
        routing_config,
        full_model=full_model,
        specialist_models=specialist_models,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": full_model.state_dict(), "mode": "full"}, output_dir / "full_model.pth")
    (output_dir / "routing_config.json").write_text(
        json.dumps(routing_config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    stats = {
        "mode": "auto_v2_specialist",
        "split_by": args.split_by,
        "classes": _parse_classes(args.classes),
        "gate_metric": args.gate_metric,
        "gate_mode": args.gate_mode,
        "selected_specialists": selected_specialists,
        "auto_v2": auto_diag,
        "general_selected_val_metrics": general_selected_metrics,
        "specialist_gate_decisions": specialist_decisions,
        "selected_val_metrics": selected_metrics,
        "routing_config": routing_config,
        "full_model_path": "full_model.pth",
        "response_refs": response_refs,
    }
    (output_dir / "calibration_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logger.info("auto_v2_specialist calibration written to %s", output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit gated specialist regression calibrators.")
    parser.add_argument("--classifier-ckpt", required=True)
    parser.add_argument("--regression-ckpt", required=True)
    parser.add_argument("--calib-data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--calibration-mode",
        default="specialist_gated",
        choices=["specialist_gated", "auto_v2_specialist"],
    )
    parser.add_argument("--classes", default="1,2")
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--full-steps", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--class-weight", type=float, default=2.0)
    parser.add_argument("--huber-delta", type=float, default=0.2)
    parser.add_argument("--val-ratio", type=float, default=0.3)
    parser.add_argument("--split-by", default="class", choices=["class", "class_concentration"])
    parser.add_argument("--gate-metric", default="R2", choices=["R2", "MAE", "RMSE", "NRMSE_range", "P90AE"])
    parser.add_argument("--gate-mode", default="metric", choices=["metric", "guarded"])
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--use-p90-guard", action="store_true")
    parser.add_argument("--p90-max-worsen", type=float, default=0.0)
    parser.add_argument("--use-bias-guard", action="store_true")
    parser.add_argument("--bias-max-worsen", type=float, default=10.0)
    parser.add_argument("--refit-affine-full-calib", action="store_true")
    parser.add_argument("--refit-full-calib", action="store_true")
    parser.add_argument("--refit-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reg-head-depth", type=int, default=None)
    parser.add_argument("--reg-output-mode", default="", choices=["", "sigmoid", "linear"])
    parser.add_argument("--reg-range-penalty", type=float, default=0.0)
    parser.add_argument("--reg-window-stats", action="store_true")
    parser.add_argument("--reg-window-stats-mode", default="", choices=["", "global", "per_channel"])
    parser.add_argument("--reg-window-stats-dim", type=int, default=None)
    parser.add_argument("--reg-response-branch", default="", choices=["", "none", "dct", "msconv"])
    parser.add_argument("--reg-dct-k", type=int, default=None)
    parser.add_argument("--reg-dct-gamma-init", type=float, default=None)
    parser.add_argument("--reg-dct-dropout", type=float, default=None)
    parser.add_argument("--reg-msconv-channels", type=int, default=None)
    parser.add_argument("--reg-msconv-kernels", default="")
    parser.add_argument("--reg-msconv-gamma-init", type=float, default=None)
    parser.add_argument("--reg-msconv-dropout", type=float, default=None)
    parser.add_argument("--reg-tcn-adapter", action="store_true")
    parser.add_argument("--reg-tcn-adapter-kernel", type=int, default=None)
    parser.add_argument("--reg-tcn-adapter-gamma-init", type=float, default=None)
    parser.add_argument("--reg-tcn-adapter-dropout", type=float, default=None)
    parser.add_argument("--reg-use-shared-trunk", action="store_true",
                        help="use shared concentration trunk + class residual heads")
    parser.add_argument("--reg-shared-trunk-dim", type=int, default=None,
                        help="shared trunk hidden dim (default 128)")
    parser.add_argument("--reg-gas-emb-dim", type=int, default=None,
                        help="gas embedding dim (default 16)")
    parser.add_argument("--reg-residual-head-depth", type=int, default=None,
                        help="residual head depth (default 2)")
    parser.add_argument("--use-reg-ratio-branch", action="store_true",
                        help="use cross-channel ratio response branch")
    parser.add_argument("--reg-ratio-gamma-init", type=float, default=None)
    parser.add_argument("--reg-ratio-dropout", type=float, default=None)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    set_random_seed(args.seed)
    device = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    output_dir = Path(args.output_dir)

    if args.calibration_mode == "auto_v2_specialist":
        _run_auto_v2_specialist(args, output_dir, device)
        return

    specialist_dir = output_dir / "specialists"
    specialist_dir.mkdir(parents=True, exist_ok=True)

    loader = _create_target_calibration_loader(args.calib_data_dir, args.batch_size)
    train_loader, val_loader = _split_loader(
        loader,
        args.batch_size,
        args.val_ratio,
        args.seed,
        split_by=args.split_by,
    )
    response_refs = _build_response_refs(train_loader, num_classes=4)

    classifier = create_model_by_config(FLConfig(), with_reg_head=False).to(device)
    classifier_state = _extract_model_state(torch.load(args.classifier_ckpt, map_location=device, weights_only=False))
    classifier.load_state_dict(classifier_state, strict=False)
    classifier.eval()

    reg_ckpt = torch.load(args.regression_ckpt, map_location=device, weights_only=False)
    reg_config = _make_regression_config_from_checkpoint(args, device, reg_ckpt)
    base_model = create_regression_model(reg_config).to(device)
    reg_state = _extract_model_state(reg_ckpt)
    base_model.load_state_dict(reg_state, strict=False)
    base_model.eval()

    affine_params = _fit_affine_params_on_loader(
        base_model,
        train_loader,
        device,
        num_classes=4,
        mode="affine_only",
    )
    routing_config = build_routing_config(affine_params, mode="affine_only")
    baseline_metrics = _evaluate_oracle(base_model, val_loader, device, 4, affine_params=affine_params)

    decisions: Dict[str, Any] = {}
    selected_specialists: List[int] = []
    for class_id in _parse_classes(args.classes):
        specialist = copy.deepcopy(base_model).to(device)
        _train_specialist(
            specialist,
            train_loader,
            device,
            class_id,
            steps=args.steps,
            lr=args.lr,
            class_weight=args.class_weight,
            huber_delta=args.huber_delta,
            reg_range_penalty=args.reg_range_penalty,
        )
        specialist_metrics = _evaluate_oracle(specialist, val_loader, device, 4, affine_params=None)
        baseline_score = _score(baseline_metrics, class_id, args.gate_metric)
        specialist_score = _score(specialist_metrics, class_id, args.gate_metric)
        baseline_cls_metrics = baseline_metrics["per_class"].get(str(class_id), {})
        specialist_cls_metrics = specialist_metrics["per_class"].get(str(class_id), {})
        accepted, gate_details = _gate_accept(
            baseline_cls_metrics,
            specialist_cls_metrics,
            baseline_score,
            specialist_score,
            args,
        )
        decisions[str(class_id)] = {
            "metric": args.gate_metric,
            "gate_mode": args.gate_mode,
            "baseline_score": baseline_score,
            "specialist_score": specialist_score,
            "min_delta": args.min_delta,
            "gate_details": gate_details,
            "refit_full_calib": bool(args.refit_full_calib),
            "accepted": accepted,
            "baseline": baseline_cls_metrics,
            "specialist": specialist_cls_metrics,
        }
        logger.info(
            "class %d specialist gate: baseline=%.4f, specialist=%.4f, accepted=%s",
            class_id,
            baseline_score,
            specialist_score,
            accepted,
        )
        if accepted:
            specialist, refit_loss = _maybe_refit_specialist(
                specialist,
                loader,
                device,
                class_id,
                args,
            )
            decisions[str(class_id)]["refit_loss"] = refit_loss
            routing_config["selected_modes"][str(class_id)] = "specialist_full"
            torch.save(
                {
                    "model_state": specialist.state_dict(),
                    "class_id": class_id,
                    "gate": decisions[str(class_id)],
                },
                specialist_dir / f"class_{class_id}.pth",
            )
            selected_specialists.append(class_id)

    if args.refit_affine_full_calib:
        full_affine_params = _fit_affine_params_on_loader(
            base_model,
            loader,
            device,
            num_classes=4,
            mode="affine_only",
        )
        selected_modes = dict(routing_config.get("selected_modes", {}))
        routing_config = build_routing_config(full_affine_params, mode="affine_only")
        for class_id in selected_specialists:
            routing_config["selected_modes"][str(class_id)] = selected_modes.get(
                str(class_id),
                "specialist_full",
            )

    stats = {
        "mode": "specialist_gated",
        "classes": _parse_classes(args.classes),
        "selected_specialists": selected_specialists,
        "gate_metric": args.gate_metric,
        "gate_mode": args.gate_mode,
        "use_p90_guard": bool(args.use_p90_guard),
        "p90_max_worsen": args.p90_max_worsen,
        "use_bias_guard": bool(args.use_bias_guard),
        "bias_max_worsen": args.bias_max_worsen,
        "refit_affine_full_calib": bool(args.refit_affine_full_calib),
        "refit_full_calib": bool(args.refit_full_calib),
        "refit_steps": args.refit_steps,
        "baseline_val_metrics": baseline_metrics,
        "gate_decisions": decisions,
        "routing_config": routing_config,
        "response_refs": response_refs,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "routing_config.json").write_text(
        json.dumps(routing_config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "calibration_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logger.info("specialist calibration written to %s", output_dir)


if __name__ == "__main__":
    main()
