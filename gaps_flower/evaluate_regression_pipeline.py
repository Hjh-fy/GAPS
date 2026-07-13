"""Evaluate the GAPS classification -> regression -> calibration pipeline.

This evaluator is designed for the main C1/C2 -> C3/C4/C5 workflow after the
strong classification checkpoint has been frozen. It keeps the classification
checkpoint fixed as the routing model and evaluates a regression checkpoint on
held-out target splits.

Key outputs:
- per-window prediction records CSV
- overall / per-client / per-class / per-client-class regression metrics
- route accuracy and route-correct regression metrics

Examples
--------
python -m gaps_flower.evaluate_regression_pipeline \
  --classifier-ckpt results/.../classifier_deploy_expB_strong_da.pth \
  --regression-ckpt results/.../reg_dct16/regression_fedavg_global.pt \
  --data-root dataset/client_data_federated_window_fullgrid_src12_tgt345 \
  --client-ids 3,4,5 \
  --split test \
  --routing-config results/.../routing_config.json \
  --route-source predicted \
  --output-dir results/.../regression_eval \
  --device cpu
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import FLConfig
from gaps_flower.evaluate_checkpoint import load_checkpoint_model
from gaps_flower.regression_task import create_regression_model, make_regression_config
from utils import CONC_STATS

GAS_NAMES = ["Ethanol", "CO", "Ethylene", "Methane"]


class SplitArrayDataset(Dataset):
    def __init__(
        self,
        features: np.ndarray,
        cls_labels: np.ndarray,
        reg_labels: np.ndarray,
        phase_labels: np.ndarray,
        client_ids: np.ndarray,
        row_ids: np.ndarray,
    ) -> None:
        self.features = torch.from_numpy(features.astype(np.float32, copy=False))
        self.cls_labels = torch.from_numpy(cls_labels.astype(np.int64, copy=False))
        self.reg_labels = torch.from_numpy(reg_labels.astype(np.float32, copy=False))
        self.phase_labels = torch.from_numpy(phase_labels.astype(np.int64, copy=False))
        self.client_ids = torch.from_numpy(client_ids.astype(np.int64, copy=False))
        self.row_ids = torch.from_numpy(row_ids.astype(np.int64, copy=False))

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, idx: int):
        return (
            self.features[idx],
            self.cls_labels[idx],
            self.reg_labels[idx],
            self.phase_labels[idx],
            self.client_ids[idx],
            self.row_ids[idx],
        )


def parse_client_ids(text: str) -> list[int]:
    return [int(item.strip()) for item in str(text).split(",") if item.strip()]


def resolve_device(text: str) -> torch.device:
    if text == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if text.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(text)


def _extract_state(ckpt: Any) -> dict[str, torch.Tensor]:
    if isinstance(ckpt, dict):
        state = ckpt.get("model_state", ckpt)
        if isinstance(state, dict) and "model_state" in state and "optimizer_state" in state:
            state = state["model_state"]
        if isinstance(state, dict):
            return state
    raise ValueError("Checkpoint does not contain a model state dictionary")


def _model_config_from_ckpt(ckpt: Any) -> dict[str, Any]:
    if isinstance(ckpt, dict):
        value = ckpt.get("model_config", {})
        if isinstance(value, dict):
            return value
    return {}


def make_regression_config_from_checkpoint(
    ckpt: Any,
    device: torch.device,
    batch_size: int,
    args: argparse.Namespace,
) -> FLConfig:
    cfg = _model_config_from_ckpt(ckpt)

    def pick(arg_name: str, cfg_name: str, default: Any = None) -> Any:
        value = getattr(args, arg_name, None)
        if value is not None:
            return value
        return cfg.get(cfg_name, default)

    return make_regression_config(
        device=device.type,
        batch_size=batch_size,
        reg_head_depth=pick("reg_head_depth", "reg_head_depth"),
        reg_output_mode=pick("reg_output_mode", "reg_output_mode"),
        use_reg_window_stats=(True if args.reg_window_stats else cfg.get("reg_window_stats")),
        reg_window_stats_mode=pick("reg_window_stats_mode", "reg_window_stats_mode"),
        reg_window_stats_dim=pick("reg_window_stats_dim", "reg_window_stats_dim"),
        reg_response_branch=pick("reg_response_branch", "reg_response_branch"),
        reg_dct_k=pick("reg_dct_k", "reg_dct_k"),
        reg_dct_gamma_init=pick("reg_dct_gamma_init", "reg_dct_gamma_init"),
        reg_dct_dropout=pick("reg_dct_dropout", "reg_dct_dropout"),
        reg_msconv_channels=pick("reg_msconv_channels", "reg_msconv_channels"),
        reg_msconv_kernels=pick("reg_msconv_kernels", "reg_msconv_kernels"),
        reg_msconv_gamma_init=pick("reg_msconv_gamma_init", "reg_msconv_gamma_init"),
        reg_msconv_dropout=pick("reg_msconv_dropout", "reg_msconv_dropout"),
        use_reg_tcn_adapter=(True if args.reg_tcn_adapter else cfg.get("reg_tcn_adapter")),
        reg_tcn_adapter_kernel=pick("reg_tcn_adapter_kernel", "reg_tcn_adapter_kernel"),
        reg_tcn_adapter_gamma_init=pick("reg_tcn_adapter_gamma_init", "reg_tcn_adapter_gamma_init"),
        reg_tcn_adapter_dropout=pick("reg_tcn_adapter_dropout", "reg_tcn_adapter_dropout"),
        use_reg_shared_trunk=(True if args.reg_use_shared_trunk else cfg.get("reg_use_shared_trunk")),
        reg_shared_trunk_dim=pick("reg_shared_trunk_dim", "reg_shared_trunk_dim"),
        reg_gas_emb_dim=pick("reg_gas_emb_dim", "reg_gas_emb_dim"),
        reg_residual_head_depth=pick("reg_residual_head_depth", "reg_residual_head_depth"),
        use_reg_ratio_branch=(True if args.use_reg_ratio_branch else cfg.get("use_reg_ratio_branch")),
        reg_ratio_gamma_init=pick("reg_ratio_gamma_init", "reg_ratio_gamma_init"),
        reg_ratio_dropout=pick("reg_ratio_dropout", "reg_ratio_dropout"),
    )


def load_regression_model(
    checkpoint_path: str,
    device: torch.device,
    batch_size: int,
    args: argparse.Namespace,
) -> tuple[torch.nn.Module, FLConfig, dict[str, Any]]:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = make_regression_config_from_checkpoint(ckpt, device, batch_size, args)
    model = create_regression_model(config).to(device)
    state = _extract_state(ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[regression] missing keys: {len(missing)}")
    if unexpected:
        print(f"[regression] unexpected keys: {len(unexpected)}")
    model.eval()
    return model, config, ckpt if isinstance(ckpt, dict) else {}


def load_specialist_models(
    specialist_dir: str | None,
    selected_modes: dict[int, str],
    device: torch.device,
    config: FLConfig,
) -> dict[int, torch.nn.Module]:
    if not specialist_dir:
        return {}
    root = Path(specialist_dir)
    out: dict[int, torch.nn.Module] = {}
    for class_id, mode in selected_modes.items():
        if mode not in {"specialist", "specialist_full"}:
            continue
        path = root / f"class_{class_id}.pth"
        if not path.exists():
            print(f"[warning] selected specialist class {class_id} but checkpoint missing: {path}")
            continue
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model = create_regression_model(config).to(device)
        model.load_state_dict(_extract_state(ckpt), strict=False)
        model.eval()
        out[int(class_id)] = model
    return out


def load_full_model(
    full_model_path: str | None,
    device: torch.device,
    config: FLConfig,
) -> torch.nn.Module | None:
    if not full_model_path:
        return None
    path = Path(full_model_path)
    if not path.exists():
        print(f"[warning] full model checkpoint missing: {path}")
        return None
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = create_regression_model(config).to(device)
    model.load_state_dict(_extract_state(ckpt), strict=False)
    model.eval()
    return model


def load_split_arrays(data_root: str | Path, client_ids: list[int], split: str) -> SplitArrayDataset:
    root = Path(data_root)
    features_list: list[np.ndarray] = []
    cls_list: list[np.ndarray] = []
    reg_list: list[np.ndarray] = []
    phase_list: list[np.ndarray] = []
    client_id_list: list[np.ndarray] = []
    row_id_list: list[np.ndarray] = []

    for cid in client_ids:
        cdir = root / f"client_{cid}"
        paths = {
            "features": cdir / f"{split}_features.npy",
            "classification_labels": cdir / f"{split}_classification_labels.npy",
            "regression_labels": cdir / f"{split}_regression_labels.npy",
            "phase_labels": cdir / f"{split}_phase_labels.npy",
        }
        missing = [str(p) for p in paths.values() if not p.exists()]
        if missing:
            raise FileNotFoundError("Missing split files:\n" + "\n".join(missing))
        x = np.load(paths["features"], allow_pickle=True)
        y_cls = np.load(paths["classification_labels"], allow_pickle=True)
        y_reg = np.load(paths["regression_labels"], allow_pickle=True)
        y_phase = np.load(paths["phase_labels"], allow_pickle=True)
        n = int(len(x))
        features_list.append(x)
        cls_list.append(y_cls)
        reg_list.append(y_reg)
        phase_list.append(y_phase)
        client_id_list.append(np.full(n, cid, dtype=np.int64))
        row_id_list.append(np.arange(n, dtype=np.int64))

    return SplitArrayDataset(
        features=np.concatenate(features_list, axis=0),
        cls_labels=np.concatenate(cls_list, axis=0),
        reg_labels=np.concatenate(reg_list, axis=0),
        phase_labels=np.concatenate(phase_list, axis=0),
        client_ids=np.concatenate(client_id_list, axis=0),
        row_ids=np.concatenate(row_id_list, axis=0),
    )


def denormalize_by_class(pred_norm: torch.Tensor, class_ids: torch.Tensor) -> torch.Tensor:
    pred_norm = pred_norm.view(-1)
    class_ids = class_ids.view(-1).long()
    ppm = torch.zeros_like(pred_norm)
    for cls_id, stats in CONC_STATS.items():
        mask = class_ids == int(cls_id)
        if mask.any():
            ppm[mask] = pred_norm[mask] * float(stats["max"] - stats["min"]) + float(stats["min"])
    return ppm


def class_range(class_id: int) -> float:
    stats = CONC_STATS.get(int(class_id), {"min": 0.0, "max": 1.0})
    return max(float(stats["max"] - stats["min"]), 1e-12)


def deployment_risk_components(
    *,
    probabilities: np.ndarray,
    margin: float,
    route_class: int,
    base_raw_ppm: float,
    routed_ppm: float,
    calibrated_ppm: float,
) -> dict[str, float]:
    """Compute risk features using only values available at deployment."""
    probs = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if probs.size == 0 or not np.isfinite(probs).all():
        raise ValueError("probabilities must be a finite non-empty vector")
    route_range = class_range(int(route_class))
    entropy = float(-(probs * np.log(np.maximum(probs, 1e-12))).sum())
    entropy_risk = float(
        entropy / max(math.log(max(int(probs.size), 2)), 1e-12)
    )
    margin_risk = float(max(0.0, 1.0 - float(margin)))
    response_gap = float(abs(float(calibrated_ppm) - float(routed_ppm)) / route_range)
    route_gap = float(abs(float(routed_ppm) - float(base_raw_ppm)) / route_range)
    route_response = float(max(response_gap, route_gap))
    return {
        "deployment_route_range_ppm": route_range,
        "deployment_risk_classifier_entropy": entropy_risk,
        "deployment_risk_margin": margin_risk,
        "deployment_risk_response_gap": response_gap,
        "deployment_risk_route_gap": route_gap,
        "deployment_risk_route_response": route_response,
        "deployment_risk_composite": float(
            max(entropy_risk, margin_risk, route_response)
        ),
    }


def load_routing_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {"selected_modes": {}, "affine_params": {}, "phase_affine_params": {}}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload


def apply_routing_calibration(
    pred_ppm: np.ndarray,
    route_cls: np.ndarray,
    phase_ids: np.ndarray,
    routing_config: dict[str, Any],
) -> np.ndarray:
    out = pred_ppm.astype(np.float64).copy()
    selected = {int(k): str(v) for k, v in routing_config.get("selected_modes", {}).items()}
    affine = {int(k): v for k, v in routing_config.get("affine_params", {}).items()}
    phase_params = {int(k): v for k, v in routing_config.get("phase_affine_params", {}).items()}

    for cls_id, params in affine.items():
        mode = selected.get(int(cls_id), params.get("mode", "none"))
        if mode not in {"bias_only", "affine_only"}:
            continue
        mask = route_cls == int(cls_id)
        if not mask.any():
            continue
        out[mask] = out[mask] * float(params.get("a", 1.0)) + float(params.get("b", 0.0))

    for cls_id, params in phase_params.items():
        mode = selected.get(int(cls_id), "none")
        if mode != "phase_affine_only":
            continue
        class_mask = route_cls == int(cls_id)
        for phase_key, calib in params.get("phase_calibrators", {}).items():
            mask = class_mask & (phase_ids == int(phase_key))
            if mask.any():
                out[mask] = out[mask] * float(calib.get("a", 1.0)) + float(calib.get("b", 0.0))
    return out


def regression_metrics(true: np.ndarray, pred: np.ndarray, true_cls: np.ndarray, route_correct: np.ndarray) -> dict[str, Any]:
    n = int(len(true))
    if n == 0:
        return {
            "n": 0,
            "RMSE": None,
            "MAE": None,
            "NRMSE_range": None,
            "R2": None,
            "Acc@10%": None,
            "Acc@10%range": None,
            "P90AE": None,
            "P95AE": None,
            "Bias": None,
            "route_accuracy": None,
        }
    err = pred.astype(np.float64) - true.astype(np.float64)
    ae = np.abs(err)
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(ae))
    ranges = np.asarray([class_range(int(c)) for c in true_cls], dtype=np.float64)
    nrmse = float(np.sqrt(np.mean((err / ranges) ** 2)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((true - true.mean()) ** 2))
    r2 = float(1.0 - ss_res / max(ss_tot, 1e-12))
    acc10 = float(np.mean(ae <= 0.10 * np.maximum(np.abs(true), 1e-6)))
    acc10_range = float(np.mean(ae <= 0.10 * ranges))
    return {
        "n": n,
        "RMSE": rmse,
        "MAE": mae,
        "NRMSE_range": nrmse,
        "R2": r2,
        "Acc@10%": acc10,
        "Acc@10%range": acc10_range,
        "P90AE": float(np.percentile(ae, 90)),
        "P95AE": float(np.percentile(ae, 95)),
        "Bias": float(np.mean(err)),
        "route_accuracy": float(np.mean(route_correct.astype(bool))),
    }


def grouped_metrics(records: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for rec in records:
        gkey = tuple(rec[k] for k in keys)
        groups.setdefault(gkey, []).append(rec)
    rows: list[dict[str, Any]] = []
    for gkey, items in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        true = np.asarray([r["true_ppm"] for r in items], dtype=np.float64)
        pred = np.asarray([r["pred_cal_ppm"] for r in items], dtype=np.float64)
        true_cls = np.asarray([r["true_cls"] for r in items], dtype=int)
        route_correct = np.asarray([r["route_correct"] for r in items], dtype=bool)
        row = {key: value for key, value in zip(keys, gkey)}
        row.update(regression_metrics(true, pred, true_cls, route_correct))
        rows.append(row)
    return rows


def metrics_for_records(records: list[dict[str, Any]], pred_key: str) -> dict[str, Any]:
    true = np.asarray([r["true_ppm"] for r in records], dtype=np.float64)
    pred = np.asarray([r[pred_key] for r in records], dtype=np.float64)
    true_cls = np.asarray([r["true_cls"] for r in records], dtype=int)
    route_correct = np.asarray([r["route_correct"] for r in records], dtype=bool)
    return regression_metrics(true, pred, true_cls, route_correct)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    for row in rows[1:]:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def collect_records(
    classifier: torch.nn.Module,
    reg_model: torch.nn.Module,
    full_model: torch.nn.Module | None,
    specialist_models: dict[int, torch.nn.Module],
    loader: DataLoader,
    device: torch.device,
    route_source: str,
    routing_config: dict[str, Any],
) -> list[dict[str, Any]]:
    classifier.eval()
    reg_model.eval()
    selected_modes = {int(k): str(v) for k, v in routing_config.get("selected_modes", {}).items()}
    records: list[dict[str, Any]] = []

    with torch.no_grad():
        for batch in loader:
            x, true_cls, y_reg_full, phase, client_ids, row_ids = batch
            x = x.to(device)
            true_cls = true_cls.to(device).long()
            y_reg_full = y_reg_full.to(device)
            phase = phase.to(device).long()
            logits, _, _ = classifier(x)
            probs = F.softmax(logits, dim=1)
            confidence, pred_cls = probs.max(dim=1)
            top2 = torch.topk(probs, k=min(2, probs.size(1)), dim=1).values
            margin = top2[:, 0] - top2[:, 1] if top2.size(1) == 2 else confidence
            route_cls = true_cls if route_source == "oracle" else pred_cls

            _, _, reg_feat = reg_model(x)
            pred_norm = reg_model.forward_reg(reg_feat, y_cls=route_cls, y_phase=phase)
            base_raw_ppm = denormalize_by_class(pred_norm, route_cls)
            routed_ppm = base_raw_ppm

            # Match gaps_deploy.DeployPredictor semantics: selected mode
            # "full" uses one full target-calibrated model before any
            # specialist replacement.
            if full_model is not None:
                routed_ppm_np = routed_ppm.detach().cpu().numpy().astype(np.float64)
                for cls_id, mode in selected_modes.items():
                    if mode != "full":
                        continue
                    mask = route_cls == int(cls_id)
                    if not mask.any():
                        continue
                    _, _, full_feat = full_model(x[mask])
                    full_norm = full_model.forward_reg(
                        full_feat,
                        y_cls=route_cls[mask],
                        y_phase=phase[mask],
                    )
                    full_ppm = denormalize_by_class(full_norm, route_cls[mask]).detach().cpu().numpy()
                    routed_ppm_np[mask.detach().cpu().numpy()] = full_ppm
                routed_ppm = torch.from_numpy(routed_ppm_np).to(device=device, dtype=torch.float32)

            # Optional specialist replacement by routed class.
            if specialist_models:
                routed_ppm_np = routed_ppm.detach().cpu().numpy().astype(np.float64)
                route_np_tmp = route_cls.detach().cpu().numpy().astype(int)
                for cls_id, specialist in specialist_models.items():
                    if selected_modes.get(int(cls_id)) not in {"specialist", "specialist_full"}:
                        continue
                    mask_np = route_np_tmp == int(cls_id)
                    if not np.any(mask_np):
                        continue
                    _, _, spec_feat = specialist(x)
                    spec_norm = specialist.forward_reg(spec_feat, y_cls=route_cls, y_phase=phase)
                    spec_ppm = denormalize_by_class(spec_norm, route_cls).detach().cpu().numpy()
                    routed_ppm_np[mask_np] = spec_ppm[mask_np]
                routed_ppm = torch.from_numpy(routed_ppm_np).to(device=device, dtype=torch.float32)

            true_ppm = y_reg_full[torch.arange(true_cls.size(0), device=device), true_cls]

            base_raw_np = base_raw_ppm.detach().cpu().numpy().astype(np.float64)
            routed_np = routed_ppm.detach().cpu().numpy().astype(np.float64)
            true_np = true_ppm.detach().cpu().numpy().astype(np.float64)
            true_cls_np = true_cls.detach().cpu().numpy().astype(int)
            route_np = route_cls.detach().cpu().numpy().astype(int)
            pred_cls_np = pred_cls.detach().cpu().numpy().astype(int)
            phase_np = phase.detach().cpu().numpy().astype(int)
            client_np = client_ids.numpy().astype(int)
            row_np = row_ids.numpy().astype(int)
            conf_np = confidence.detach().cpu().numpy().astype(float)
            margin_np = margin.detach().cpu().numpy().astype(float)
            prob_np = probs.detach().cpu().numpy().astype(float)
            pred_cal_np = apply_routing_calibration(routed_np, route_np, phase_np, routing_config)

            for i in range(len(true_np)):
                err = float(pred_cal_np[i] - true_np[i])
                entropy = float(-(prob_np[i] * np.log(np.maximum(prob_np[i], 1e-12))).sum())
                routed_delta = float(routed_np[i] - base_raw_np[i])
                final_delta = float(pred_cal_np[i] - routed_np[i])
                range_ppm = class_range(int(true_cls_np[i]))
                response_gap = float(abs(final_delta) / range_ppm)
                route_gap = float(abs(routed_delta) / range_ppm)
                margin_risk = float(max(0.0, 1.0 - margin_np[i]))
                entropy_risk = float(entropy / max(math.log(max(prob_np.shape[1], 2)), 1e-12))
                route_response_risk = float(max(response_gap, route_gap))
                deployment_risks = deployment_risk_components(
                    probabilities=prob_np[i],
                    margin=float(margin_np[i]),
                    route_class=int(route_np[i]),
                    base_raw_ppm=float(base_raw_np[i]),
                    routed_ppm=float(routed_np[i]),
                    calibrated_ppm=float(pred_cal_np[i]),
                )
                records.append(
                    {
                        "client_id": int(client_np[i]),
                        "client": f"C{int(client_np[i])}",
                        "row_id": int(row_np[i]),
                        "true_cls": int(true_cls_np[i]),
                        "true_class": int(true_cls_np[i]),
                        "true_gas": GAS_NAMES[int(true_cls_np[i])] if int(true_cls_np[i]) < len(GAS_NAMES) else str(true_cls_np[i]),
                        "pred_cls": int(pred_cls_np[i]),
                        "route_cls": int(route_np[i]),
                        "route_gas": GAS_NAMES[int(route_np[i])] if int(route_np[i]) < len(GAS_NAMES) else str(route_np[i]),
                        "route_source": route_source,
                        "route_correct": bool(int(route_np[i]) == int(true_cls_np[i])),
                        "class_correct": bool(int(pred_cls_np[i]) == int(true_cls_np[i])),
                        "phase": int(phase_np[i]),
                        "true_ppm": float(true_np[i]),
                        "base_raw_ppm": float(base_raw_np[i]),
                        "routed_pred_ppm": float(routed_np[i]),
                        "final_calibrated_ppm": float(pred_cal_np[i]),
                        # Backward-compatible aliases. From this point on,
                        # pred_raw_ppm means the pure base regressor output.
                        "pred_raw_ppm": float(base_raw_np[i]),
                        "pred_cal_ppm": float(pred_cal_np[i]),
                        "routed_minus_base_ppm": routed_delta,
                        "calibrated_minus_routed_ppm": final_delta,
                        "error_ppm": err,
                        "abs_error_ppm": abs(err),
                        "squared_error": err * err,
                        "class_confidence": float(conf_np[i]),
                        "class_margin": float(margin_np[i]),
                        "class_entropy": entropy,
                        "range_ppm": range_ppm,
                        "rel_abs_error": float(abs(err) / max(abs(true_np[i]), 1e-6)),
                        "range_norm_abs_error": float(abs(err) / range_ppm),
                        "response_mean_conc_gap_norm": response_gap,
                        "route_neural_gap_norm": route_gap,
                        "class_response_margin_risk": margin_risk,
                        "route_response_risk": route_response_risk,
                        "classifier_entropy_risk": entropy_risk,
                        "composite_response_risk": float(max(response_gap, route_gap, margin_risk, entropy_risk)),
                        **deployment_risks,
                    }
                )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate classification-routed regression pipeline")
    parser.add_argument("--classifier-ckpt", required=True)
    parser.add_argument("--regression-ckpt", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--client-ids", default="3,4,5")
    parser.add_argument("--split", default="test", choices=("train", "calibration", "test"))
    parser.add_argument("--routing-config", default="")
    parser.add_argument("--full-model", default="", help="Optional full target-calibrated model checkpoint for routing mode=full")
    parser.add_argument("--specialist-dir", default="", help="Optional directory containing specialists/class_<id>.pth or class_<id>.pth")
    parser.add_argument("--route-source", choices=("predicted", "oracle"), default="predicted")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-dir", required=True)
    # Regression architecture overrides, kept aligned with client/server scripts.
    parser.add_argument("--reg-head-depth", type=int, default=None)
    parser.add_argument("--reg-output-mode", choices=("sigmoid", "linear"), default=None)
    parser.add_argument("--reg-window-stats", action="store_true")
    parser.add_argument("--reg-window-stats-mode", choices=("global", "per_channel"), default=None)
    parser.add_argument("--reg-window-stats-dim", type=int, default=None)
    parser.add_argument("--reg-response-branch", choices=("none", "dct", "msconv"), default=None)
    parser.add_argument("--reg-dct-k", type=int, default=None)
    parser.add_argument("--reg-dct-gamma-init", type=float, default=None)
    parser.add_argument("--reg-dct-dropout", type=float, default=None)
    parser.add_argument("--reg-msconv-channels", type=int, default=None)
    parser.add_argument("--reg-msconv-kernels", default=None)
    parser.add_argument("--reg-msconv-gamma-init", type=float, default=None)
    parser.add_argument("--reg-msconv-dropout", type=float, default=None)
    parser.add_argument("--reg-tcn-adapter", action="store_true")
    parser.add_argument("--reg-tcn-adapter-kernel", type=int, default=None)
    parser.add_argument("--reg-tcn-adapter-gamma-init", type=float, default=None)
    parser.add_argument("--reg-tcn-adapter-dropout", type=float, default=None)
    parser.add_argument("--reg-use-shared-trunk", action="store_true")
    parser.add_argument("--reg-shared-trunk-dim", type=int, default=None)
    parser.add_argument("--reg-gas-emb-dim", type=int, default=None)
    parser.add_argument("--reg-residual-head-depth", type=int, default=None)
    parser.add_argument("--use-reg-ratio-branch", action="store_true")
    parser.add_argument("--reg-ratio-gamma-init", type=float, default=None)
    parser.add_argument("--reg-ratio-dropout", type=float, default=None)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    device = torch.device("cpu") if args.cpu else resolve_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    classifier, clf_config, clf_ckpt = load_checkpoint_model(
        args.classifier_ckpt,
        device,
        args.batch_size,
    )
    classifier.to(device).eval()

    reg_model, reg_config, reg_ckpt = load_regression_model(
        args.regression_ckpt,
        device,
        args.batch_size,
        args,
    )
    routing_config = load_routing_config(args.routing_config or None)
    selected_modes = {int(k): str(v) for k, v in routing_config.get("selected_modes", {}).items()}
    specialist_root = ""
    if args.specialist_dir:
        root = Path(args.specialist_dir)
        specialist_root = str(root / "specialists") if (root / "specialists").exists() else str(root)
    full_model = load_full_model(args.full_model or None, device, reg_config)
    specialist_models = load_specialist_models(specialist_root or None, selected_modes, device, reg_config)

    client_ids = parse_client_ids(args.client_ids)
    dataset = load_split_arrays(args.data_root, client_ids, args.split)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    records = collect_records(
        classifier,
        reg_model,
        full_model,
        specialist_models,
        loader,
        device,
        args.route_source,
        routing_config,
    )

    true = np.asarray([r["true_ppm"] for r in records], dtype=np.float64)
    pred = np.asarray([r["pred_cal_ppm"] for r in records], dtype=np.float64)
    true_cls = np.asarray([r["true_cls"] for r in records], dtype=int)
    route_correct = np.asarray([r["route_correct"] for r in records], dtype=bool)

    summary = {
        "classifier_ckpt": str(args.classifier_ckpt),
        "regression_ckpt": str(args.regression_ckpt),
        "routing_config": str(args.routing_config),
        "full_model": str(args.full_model),
        "specialist_dir": specialist_root,
        "data_root": str(args.data_root),
        "client_ids": client_ids,
        "split": args.split,
        "route_source": args.route_source,
        "device": str(device),
        "n_records": int(len(records)),
        "overall": regression_metrics(true, pred, true_cls, route_correct),
        "stage_overall": {
            "base_raw": metrics_for_records(records, "base_raw_ppm"),
            "routed": metrics_for_records(records, "routed_pred_ppm"),
            "final_calibrated": metrics_for_records(records, "final_calibrated_ppm"),
        },
        "by_client": grouped_metrics(records, ["client_id"]),
        "by_class": grouped_metrics(records, ["true_cls", "true_gas"]),
        "by_client_class": grouped_metrics(records, ["client_id", "true_cls", "true_gas"]),
        "routing_selected_modes": routing_config.get("selected_modes", {}),
        "regression_model_config": _model_config_from_ckpt(reg_ckpt),
    }

    records_path = output_dir / f"{args.split}_records.csv"
    write_csv(records_path, records)
    write_csv(output_dir / "metrics_by_client.csv", summary["by_client"])
    write_csv(output_dir / "metrics_by_class.csv", summary["by_class"])
    write_csv(output_dir / "metrics_by_client_class.csv", summary["by_client_class"])
    (output_dir / "regression_metrics_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary["overall"], indent=2, ensure_ascii=False))
    print(f"records: {records_path}")
    print(f"summary: {output_dir / 'regression_metrics_summary.json'}")


if __name__ == "__main__":
    main()
