"""Classic regression-head ablations on fixed route predictions.

The first implemented experiment is a source-only per-gas Ridge head over
deployment-visible rich window statistics. It does not touch the classifier or
backbone. Target evaluation uses the existing fixed-DA predicted class as the
hard route, then compares no-QC full-set ppm metrics against the current
R3aK16/auto_v2 final_ppm baseline.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


CLASS_NAMES = {0: "Ethanol", 1: "CO", 2: "Ethylene", 3: "Methane"}
CLASS_RANGES = {0: 112.5, 1: 225.0, 2: 112.5, 3: 225.0}
CO_CLASS = 1


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fnum(value: Any, default: float = np.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def inum(value: Any, default: int = -1) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def client_num(client: str | int) -> int:
    return int(str(client).upper().replace("CLIENT_", "").replace("C", ""))


def client_name(client: str | int) -> str:
    return f"C{client_num(client)}"


def co_bin(true_ppm: float) -> str:
    if true_ppm <= 100.0:
        return "CO_low_25_100"
    if true_ppm <= 175.0:
        return "CO_mid_125_175"
    return "CO_high_200_250"


def safe_ratio(a: float, b: float) -> float:
    return float(a / b) if abs(b) > 1e-9 else 0.0


def rich_feature_dict(window: np.ndarray, phase: int = -1, meta: dict[str, Any] | None = None) -> dict[str, float]:
    x = np.asarray(window, dtype=np.float64)
    meta = meta or {}
    diff = np.diff(x, axis=0)
    ch_mean = x.mean(axis=0)
    ch_std = x.std(axis=0)
    ch_min = x.min(axis=0)
    ch_max = x.max(axis=0)
    ch_amp = ch_max - ch_min
    ch_slope = (x[-1] - x[0]) / max(x.shape[0] - 1, 1)
    ch_absdiff_mean = np.abs(diff).mean(axis=0)
    ch_absdiff_max = np.abs(diff).max(axis=0)

    row: dict[str, float] = {}
    for name, values in [
        ("mean", ch_mean),
        ("std", ch_std),
        ("min", ch_min),
        ("max", ch_max),
        ("amp", ch_amp),
        ("slope", ch_slope),
        ("absdiff_mean", ch_absdiff_mean),
        ("absdiff_max", ch_absdiff_max),
    ]:
        for idx, value in enumerate(values):
            row[f"ch{idx}_{name}"] = float(value)

    amp_order = np.argsort(-ch_amp)
    top_amp = ch_amp[amp_order]
    top_slope = ch_slope[amp_order]
    row.update(
        {
            "global_mean": float(x.mean()),
            "global_std": float(x.std()),
            "global_min": float(x.min()),
            "global_max": float(x.max()),
            "global_amp": float(x.max() - x.min()),
            "global_absdiff_mean": float(np.abs(diff).mean()),
            "global_absdiff_max": float(np.abs(diff).max()),
            "slope_mean": float(ch_slope.mean()),
            "slope_std": float(ch_slope.std()),
            "amp_mean": float(ch_amp.mean()),
            "amp_std": float(ch_amp.std()),
            "amp_top1": float(top_amp[0]),
            "amp_top2": float(top_amp[1]) if len(top_amp) > 1 else 0.0,
            "amp_top3": float(top_amp[2]) if len(top_amp) > 2 else 0.0,
            "amp_top4": float(top_amp[3]) if len(top_amp) > 3 else 0.0,
            "amp_top1_top2_ratio": safe_ratio(float(top_amp[0]), float(top_amp[1]) if len(top_amp) > 1 else 0.0),
            "amp_top1_top3_ratio": safe_ratio(float(top_amp[0]), float(top_amp[2]) if len(top_amp) > 2 else 0.0),
            "amp_top1_top4_ratio": safe_ratio(float(top_amp[0]), float(top_amp[3]) if len(top_amp) > 3 else 0.0),
            "slope_top1_top2_ratio": safe_ratio(float(top_slope[0]), float(top_slope[1]) if len(top_slope) > 1 else 0.0),
        }
    )

    window_start = fnum(meta.get("window_start_s"), 0.0)
    window_end = fnum(meta.get("window_end_s"), 0.0)
    window_center = fnum(meta.get("window_center_s"), (window_start + window_end) / 2.0)
    onset = fnum(meta.get("t_onset"), 0.0)
    t_min = fnum(meta.get("t_min"), 0.0)
    row.update(
        {
            "window_start_s": window_start,
            "window_end_s": window_end,
            "window_center_s": window_center,
            "window_len_s": window_end - window_start,
            "t_onset": onset,
            "t_min": t_min,
            "center_minus_onset": window_center - onset,
            "center_minus_t_min": window_center - t_min,
            "interpolated_ratio": fnum(meta.get("interpolated_ratio"), 0.0),
            "max_gap_inside_window": fnum(meta.get("max_gap_inside_window"), 0.0),
        }
    )
    response_phase = str(meta.get("response_phase", "unknown"))
    for value in ["main_response", "recovery", "unknown"]:
        row[f"response_phase_{value}"] = float(response_phase == value)
    phase_label = str(meta.get("phase_label", "unknown"))
    for value in ["early", "middle", "late", "unknown"]:
        row[f"phase_label_{value}"] = float(phase_label == value)
    phase_int = int(phase) if int(phase) in {0, 1, 2} else -1
    for value in [0, 1, 2]:
        row[f"phase_id_{value}"] = float(phase_int == value)
    row["phase_id_unknown"] = float(phase_int < 0)
    return row


def load_split(data_root: Path, client: str, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    cdir = data_root / f"client_{client_num(client)}"
    features = np.load(cdir / f"{split}_features.npy").astype(np.float32)
    cls = np.load(cdir / f"{split}_classification_labels.npy").astype(np.int64).reshape(-1)
    reg = np.load(cdir / f"{split}_regression_labels.npy").astype(np.float64)
    phase = np.load(cdir / f"{split}_phase_labels.npy").astype(np.int64).reshape(-1)
    meta_path = cdir / f"{split}_experiment_info.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else [{} for _ in range(len(features))]
    return features, cls, reg, phase, meta


def build_oracle_rows(data_root: Path, clients: Sequence[str], split: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for client in clients:
        features, cls, reg, phase, meta = load_split(data_root, client, split)
        for idx in range(len(features)):
            true_class = int(cls[idx])
            feature_dict = rich_feature_dict(features[idx], int(phase[idx]), meta[idx] if idx < len(meta) else {})
            rows.append(
                {
                    "client": client_name(client),
                    "split": split,
                    "sample_index": idx,
                    "true_class": true_class,
                    "route_class": true_class,
                    "true_ppm": float(reg[idx, true_class]),
                    "phase": int(phase[idx]),
                    "feature_dict": feature_dict,
                }
            )
    return rows


def add_target_features(rows: list[dict[str, str]], data_root: Path) -> list[dict[str, Any]]:
    cache: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]] = {}
    out: list[dict[str, Any]] = []
    for row in rows:
        client = client_name(row.get("client") or row.get("client_id"))
        split = str(row.get("split"))
        key = (client, split)
        if key not in cache:
            cache[key] = load_split(data_root, client, split)
        features, _cls, _reg, phase, meta = cache[key]
        idx = inum(row.get("sample_index"))
        item: dict[str, Any] = dict(row)
        pred_class = inum(item.get("pred_class"))
        true_class = inum(item.get("true_class"))
        item.update(
            {
                "client": client,
                "split": split,
                "sample_index": idx,
                "pred_class": pred_class,
                "route_class": pred_class,
                "true_class": true_class,
                "true_ppm": fnum(item.get("true_ppm")),
                "final_ppm": fnum(item.get("final_ppm")),
                "feature_dict": rich_feature_dict(features[idx], int(phase[idx]), meta[idx] if idx < len(meta) else {}),
            }
        )
        out.append(item)
    return out


@dataclass
class RidgeHead:
    alpha: float
    feature_names: list[str]
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    clip_min: float
    clip_max: float

    def predict(self, rows: Sequence[dict[str, Any]], clip: bool = True) -> np.ndarray:
        x = matrix_from_rows(rows, self.feature_names)
        x = np.where(np.isfinite(x), x, self.mean)
        scale = np.where(np.abs(self.scale) < 1e-9, 1.0, self.scale)
        design = np.concatenate([np.ones((x.shape[0], 1)), (x - self.mean) / scale], axis=1)
        pred = design @ self.coef
        if clip:
            pred = np.clip(pred, self.clip_min, self.clip_max)
        return pred

    def to_json(self) -> dict[str, Any]:
        return {
            "alpha": self.alpha,
            "feature_names": self.feature_names,
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "coef": self.coef.tolist(),
            "clip_min": self.clip_min,
            "clip_max": self.clip_max,
        }


def matrix_from_rows(rows: Sequence[dict[str, Any]], feature_names: Sequence[str]) -> np.ndarray:
    return np.asarray(
        [[fnum(row["feature_dict"].get(name), 0.0) for name in feature_names] for row in rows],
        dtype=np.float64,
    )


def fit_ridge(rows: Sequence[dict[str, Any]], feature_names: Sequence[str], alpha: float) -> RidgeHead:
    x = matrix_from_rows(rows, feature_names)
    y = np.asarray([fnum(row["true_ppm"]) for row in rows], dtype=np.float64)
    mean = np.nanmean(x, axis=0)
    x = np.where(np.isfinite(x), x, mean)
    scale = np.nanstd(x, axis=0)
    scale = np.where(np.abs(scale) < 1e-9, 1.0, scale)
    z = (x - mean) / scale
    design = np.concatenate([np.ones((z.shape[0], 1)), z], axis=1)
    reg = np.eye(design.shape[1]) * float(alpha)
    reg[0, 0] = 0.0
    coef = np.linalg.pinv(design.T @ design + reg) @ design.T @ y
    return RidgeHead(
        alpha=float(alpha),
        feature_names=list(feature_names),
        mean=mean,
        scale=scale,
        coef=coef,
        clip_min=float(np.min(y)),
        clip_max=float(np.max(y)),
    )


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2))) if y_true.size else float("inf")


def fit_select_refit(
    train_rows: Sequence[dict[str, Any]],
    val_rows: Sequence[dict[str, Any]],
    feature_names: Sequence[str],
    alphas: Sequence[float],
) -> tuple[RidgeHead, dict[str, Any]]:
    best_alpha = float(alphas[0])
    best_rmse = float("inf")
    val_y = np.asarray([fnum(row["true_ppm"]) for row in val_rows], dtype=np.float64)
    audit_rows: list[dict[str, Any]] = []
    for alpha in alphas:
        model = fit_ridge(train_rows, feature_names, float(alpha))
        pred = model.predict(val_rows, clip=True)
        score = rmse(val_y, pred)
        audit_rows.append({"alpha": float(alpha), "val_RMSE": score})
        if score < best_rmse:
            best_rmse = score
            best_alpha = float(alpha)
    model = fit_ridge([*train_rows, *val_rows], feature_names, best_alpha)
    return model, {"best_alpha": best_alpha, "best_val_RMSE": best_rmse, "alpha_audit": audit_rows}


def deterministic_train_val(rows: Sequence[dict[str, Any]], val_ratio: float = 0.25) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Deterministic class/client/concentration-stratified calibration split.

    Earlier selector experiments used a simple tail holdout by sample index.
    That is reproducible, but it can under-cover specific concentration levels
    inside the calibration-validation split. Since target direct heads are
    selected from small calibration sets, the holdout should cover concentration
    levels as evenly as possible.
    """
    ordered = sorted(rows, key=lambda row: (client_num(row["client"]), inum(row.get("sample_index"))))
    if len(ordered) <= 2:
        n_val = max(1, len(ordered) // 2)
        return ordered[:-n_val], ordered[-n_val:]

    buckets: dict[tuple[int, int, float], list[dict[str, Any]]] = {}
    for row in ordered:
        key = (
            client_num(row["client"]),
            inum(row.get("true_class", row.get("route_class"))),
            round(fnum(row.get("true_ppm")), 6),
        )
        buckets.setdefault(key, []).append(row)

    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    carry: list[dict[str, Any]] = []
    for key in sorted(buckets):
        bucket = sorted(buckets[key], key=lambda row: inum(row.get("sample_index")))
        if len(bucket) == 1:
            carry.extend(bucket)
            continue
        n_val = max(1, int(round(len(bucket) * val_ratio)))
        n_val = min(n_val, len(bucket) - 1)
        train_rows.extend(bucket[:-n_val])
        val_rows.extend(bucket[-n_val:])

    if carry:
        carry = sorted(carry, key=lambda row: (client_num(row["client"]), inum(row.get("sample_index"))))
        n_val = max(1, int(round(len(carry) * val_ratio))) if len(carry) >= 4 else max(0, len(carry) // 2)
        train_rows.extend(carry[:-n_val])
        val_rows.extend(carry[-n_val:])

    if not val_rows:
        n_val = max(1, int(round(len(ordered) * val_ratio)))
        return ordered[:-n_val], ordered[-n_val:]
    if not train_rows:
        return ordered[:-1], ordered[-1:]
    return (
        sorted(train_rows, key=lambda row: (client_num(row["client"]), inum(row.get("sample_index")))),
        sorted(val_rows, key=lambda row: (client_num(row["client"]), inum(row.get("sample_index")))),
    )


def metrics(rows: Sequence[dict[str, Any]], pred_key: str) -> dict[str, Any]:
    pred = np.asarray([fnum(row.get(pred_key)) for row in rows], dtype=np.float64)
    true = np.asarray([fnum(row.get("true_ppm")) for row in rows], dtype=np.float64)
    cls = np.asarray([inum(row.get("true_class")) for row in rows], dtype=np.int64)
    mask = np.isfinite(pred) & np.isfinite(true)
    pred = pred[mask]
    true = true[mask]
    cls = cls[mask]
    if pred.size == 0:
        return {"N": 0, "RMSE": None, "MAE": None, "NRMSE": None, "Bias": None, "P90AE": None, "R2": None}
    err = pred - true
    ranges = np.asarray([CLASS_RANGES.get(int(c), np.nan) for c in cls], dtype=np.float64)
    range_mask = np.isfinite(ranges) & (ranges > 0)
    centered = true - np.mean(true)
    ss_tot = float(np.sum(centered * centered))
    return {
        "N": int(pred.size),
        "RMSE": float(np.sqrt(np.mean(err * err))),
        "MAE": float(np.mean(np.abs(err))),
        "NRMSE": float(np.sqrt(np.mean((err[range_mask] / ranges[range_mask]) ** 2))) if np.any(range_mask) else None,
        "Bias": float(np.mean(err)),
        "P90AE": float(np.percentile(np.abs(err), 90)),
        "R2": float(1.0 - np.sum(err * err) / ss_tot) if ss_tot > 1e-12 else None,
    }


def summarize(rows: Sequence[dict[str, Any]], pred_key: str, mode: str, split: str) -> list[dict[str, Any]]:
    split_rows = [row for row in rows if row["split"] == split]
    scopes: list[tuple[str, list[dict[str, Any]]]] = [("ALL", split_rows)]
    for client in sorted({row["client"] for row in split_rows}, key=client_num):
        c_rows = [row for row in split_rows if row["client"] == client]
        scopes.append((client, c_rows))
        co_rows = [row for row in c_rows if inum(row["true_class"]) == CO_CLASS]
        scopes.append((f"{client}-CO", co_rows))
        for bin_name in ["CO_low_25_100", "CO_mid_125_175", "CO_high_200_250"]:
            scopes.append((f"{client}-{bin_name}", [row for row in co_rows if co_bin(fnum(row["true_ppm"])) == bin_name]))
        nonco_rows = [row for row in c_rows if inum(row["true_class"]) != CO_CLASS]
        scopes.append((f"{client}-nonCO", nonco_rows))
    all_nonco = [row for row in split_rows if inum(row["true_class"]) != CO_CLASS]
    scopes.append(("nonCO_ALL", all_nonco))
    out: list[dict[str, Any]] = []
    for scope, scope_rows in scopes:
        if scope_rows:
            out.append({"mode": mode, "split": split, "scope": scope, "pred_key": pred_key, **metrics(scope_rows, pred_key)})
    return out


def apply_models(rows: list[dict[str, Any]], models: dict[int, RidgeHead], prefix: str) -> list[dict[str, Any]]:
    out = [dict(row) for row in rows]
    for cls_id, model in models.items():
        idxs = [idx for idx, row in enumerate(out) if inum(row.get("route_class")) == cls_id]
        if not idxs:
            continue
        pred_rows = [out[idx] for idx in idxs]
        pred_raw = model.predict(pred_rows, clip=False)
        pred_clip = model.predict(pred_rows, clip=True)
        for idx, raw, clipped in zip(idxs, pred_raw, pred_clip):
            out[idx][f"{prefix}_raw_ppm"] = float(raw)
            out[idx][f"{prefix}_ppm"] = float(clipped)
            out[idx][f"{prefix}_delta_vs_final"] = float(clipped - fnum(out[idx].get("final_ppm"), clipped))
    return out


def apply_client_models(rows: list[dict[str, Any]], models: dict[tuple[str, int], RidgeHead], prefix: str) -> list[dict[str, Any]]:
    out = [dict(row) for row in rows]
    for (client, cls_id), model in models.items():
        idxs = [
            idx for idx, row in enumerate(out)
            if str(row.get("client")) == client and inum(row.get("route_class")) == cls_id
        ]
        if not idxs:
            continue
        pred_rows = [out[idx] for idx in idxs]
        pred_raw = model.predict(pred_rows, clip=False)
        pred_clip = model.predict(pred_rows, clip=True)
        for idx, raw, clipped in zip(idxs, pred_raw, pred_clip):
            out[idx][f"{prefix}_raw_ppm"] = float(raw)
            out[idx][f"{prefix}_ppm"] = float(clipped)
            out[idx][f"{prefix}_delta_vs_final"] = float(clipped - fnum(out[idx].get("final_ppm"), clipped))
    return out


def write_report(
    out: Path,
    target_summary: list[dict[str, Any]],
    source_summary: list[dict[str, Any]],
    fit_audit: list[dict[str, Any]],
    fit_mode: str,
) -> None:
    def get(rows: list[dict[str, Any]], mode: str, split: str, scope: str, field: str = "RMSE") -> float | None:
        for row in rows:
            if row["mode"] == mode and row["split"] == split and row["scope"] == scope:
                return row.get(field)
        return None

    scopes = ["ALL", "C3-CO", "C4-CO", "C5-CO", "C3-CO_high_200_250", "C4-CO_high_200_250", "C5-CO_high_200_250", "nonCO_ALL"]
    lines = ["| mode | " + " | ".join(scopes) + " |", "|---|" + "|".join(["---:"] * len(scopes)) + "|"]
    target_modes = sorted({row["mode"] for row in target_summary}, key=lambda x: (x != "baseline_final_ppm", x))
    for mode in target_modes:
        vals = []
        for scope in scopes:
            val = get(target_summary, mode, "test", scope)
            vals.append("" if val is None else f"{float(val):.2f}")
        lines.append("| " + mode + " | " + " | ".join(vals) + " |")

    source_scopes = ["ALL", "C1-CO", "C2-CO", "C1-CO_high_200_250", "C2-CO_high_200_250", "nonCO_ALL"]
    source_lines = ["| mode | " + " | ".join(source_scopes) + " |", "|---|" + "|".join(["---:"] * len(source_scopes)) + "|"]
    source_modes = sorted({row["mode"] for row in source_summary})
    for mode in source_modes:
        vals = []
        for scope in source_scopes:
            val = get(source_summary, mode, "test", scope)
            vals.append("" if val is None else f"{float(val):.2f}")
        source_lines.append("| " + mode + " | " + " | ".join(vals) + " |")

    fit_lines = ["| class | gas | train N | val N | best alpha | val RMSE |", "|---:|---|---:|---:|---:|---:|"]
    for row in fit_audit:
        fit_lines.append(
            f"| {row['class_id']} | {row['gas']} | {row['train_N']} | {row['val_N']} | "
            f"{row['best_alpha']:.6g} | {row['best_val_RMSE']:.2f} |"
        )

    if fit_mode == "source_global":
        protocol = """- Fit data: source C1/C2 train.
- Alpha selection: source C1/C2 calibration.
- Refit: source C1/C2 train + calibration.
- Source evaluation: C1/C2 test with oracle gas route.
- Target evaluation: C3/C4/C5 test with existing fixed-DA predicted gas route."""
    else:
        protocol = """- Fit data: target C3/C4/C5 calibration, per client and per gas.
- Alpha selection: deterministic holdout from the same target calibration split.
- Refit: target calibration train + holdout for the selected client/gas head.
- Source evaluation: not applicable.
- Target evaluation: C3/C4/C5 test with existing fixed-DA predicted gas route."""

    text = f"""# Regression Head Ablation: H1 Ridge Rich Stats

This experiment freezes the fixed-DA classification route and tests a classic per-gas Ridge regression head over rich window statistics.

Training protocol:

{protocol}
- QC filtering: none.

## Target Test RMSE

{chr(10).join(lines)}

## Source Test RMSE

{chr(10).join(source_lines)}

## Ridge Fit Audit

{chr(10).join(fit_lines)}

## Reading

- If H1 beats `baseline_final_ppm` on target ALL/CO, the current neural regression head is likely over-complex or poorly constrained.
- If H1 is strong on source but weak on target, the bottleneck is likely cross-client calibration transfer rather than head capacity.
- If H1 is weak even on source, rich stats alone are not enough for a source-only replacement head.
"""
    (out / "reg_head_ablation_report.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run classic regression head ablations.")
    parser.add_argument("--data-root", default="dataset/client_data_c12src_c345tgt_2080_timeaware_60_170_window_fullgrid")
    parser.add_argument("--target-predictions", default="results/timeaware_2080_c12src_c345tgt_fixed_da_r25_r3ak16_auto_v2_eval/ppm_layer_co_audit/target_layer_predictions.csv")
    parser.add_argument("--train-clients", default="1,2")
    parser.add_argument("--target-clients", default="3,4,5")
    parser.add_argument("--alphas", default="0,0.01,0.1,1,10,100,1000")
    parser.add_argument("--fit-mode", default="source_global", choices=["source_global", "target_client_calibration"])
    parser.add_argument("--output-dir", default="results/reg_head_ablation_20260624/h1_ridge_rich_direct")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    train_clients = [client_name(item.strip()) for item in args.train_clients.split(",") if item.strip()]
    target_clients = [client_name(item.strip()) for item in args.target_clients.split(",") if item.strip()]
    alphas = [float(item.strip()) for item in args.alphas.split(",") if item.strip()]

    fit_audit: list[dict[str, Any]] = []
    target_base_rows = [
        row for row in read_csv(args.target_predictions)
        if client_name(row.get("client") or row.get("client_id")) in set(target_clients)
    ]
    target_all_rows = add_target_features(target_base_rows, data_root)
    target_test_rows = [row for row in target_all_rows if row["split"] == "test"]
    source_summary: list[dict[str, Any]] = []

    if args.fit_mode == "source_global":
        source_train = build_oracle_rows(data_root, train_clients, "train")
        source_val = build_oracle_rows(data_root, train_clients, "calibration")
        source_test = build_oracle_rows(data_root, train_clients, "test")
        feature_names = sorted(source_train[0]["feature_dict"].keys())
        models: dict[int, RidgeHead] = {}
        for cls_id in sorted(CLASS_NAMES):
            train_rows = [row for row in source_train if row["true_class"] == cls_id]
            val_rows = [row for row in source_val if row["true_class"] == cls_id]
            model, audit = fit_select_refit(train_rows, val_rows, feature_names, alphas)
            models[cls_id] = model
            fit_audit.append(
                {
                    "fit_mode": args.fit_mode,
                    "client": "source_global",
                    "class_id": cls_id,
                    "gas": CLASS_NAMES[cls_id],
                    "train_N": len(train_rows),
                    "val_N": len(val_rows),
                    "best_alpha": audit["best_alpha"],
                    "best_val_RMSE": audit["best_val_RMSE"],
                    "alpha_audit": json.dumps(audit["alpha_audit"], ensure_ascii=False),
                }
            )
        source_pred = apply_models(source_test, models, "h1_ridge_rich")
        for row in source_pred:
            row["h1_ridge_rich_oracle_ppm"] = row["h1_ridge_rich_ppm"]
        source_summary = summarize(source_pred, "h1_ridge_rich_oracle_ppm", "h1_ridge_rich_oracle", "test")
        target_pred = apply_models(target_test_rows, models, "h1_ridge_rich")
        model_json = {str(cls_id): model.to_json() for cls_id, model in models.items()}
        target_mode = "h1_ridge_rich_direct"
    else:
        feature_names = sorted(target_all_rows[0]["feature_dict"].keys())
        client_models: dict[tuple[str, int], RidgeHead] = {}
        calibration_rows = [dict(row) for row in target_all_rows if row["split"] == "calibration"]
        for row in calibration_rows:
            row["route_class"] = row["true_class"]
        for client in target_clients:
            for cls_id in sorted(CLASS_NAMES):
                cls_rows = [row for row in calibration_rows if row["client"] == client and row["true_class"] == cls_id]
                train_rows, val_rows = deterministic_train_val(cls_rows, val_ratio=0.25)
                model, audit = fit_select_refit(train_rows, val_rows, feature_names, alphas)
                client_models[(client, cls_id)] = model
                fit_audit.append(
                    {
                        "fit_mode": args.fit_mode,
                        "client": client,
                        "class_id": cls_id,
                        "gas": CLASS_NAMES[cls_id],
                        "train_N": len(train_rows),
                        "val_N": len(val_rows),
                        "best_alpha": audit["best_alpha"],
                        "best_val_RMSE": audit["best_val_RMSE"],
                        "alpha_audit": json.dumps(audit["alpha_audit"], ensure_ascii=False),
                    }
                )
        target_pred = apply_client_models(target_test_rows, client_models, "h1_ridge_rich")
        model_json = {f"{client}:{cls_id}": model.to_json() for (client, cls_id), model in client_models.items()}
        target_mode = "h1_ridge_rich_target_cal_direct"

    for row in target_pred:
        row["baseline_final_ppm"] = fnum(row.get("final_ppm"))
        row[f"{target_mode}_ppm"] = fnum(row.get("h1_ridge_rich_ppm"))
    target_summary: list[dict[str, Any]] = []
    target_summary.extend(summarize(target_pred, "baseline_final_ppm", "baseline_final_ppm", "test"))
    target_summary.extend(summarize(target_pred, f"{target_mode}_ppm", target_mode, "test"))

    write_csv(out / "target_predictions.csv", [{k: v for k, v in row.items() if k != "feature_dict"} for row in target_pred])
    if args.fit_mode == "source_global":
        write_csv(out / "source_predictions.csv", [{k: v for k, v in row.items() if k != "feature_dict"} for row in source_pred])
    else:
        write_csv(out / "source_predictions.csv", [])
    write_csv(out / "target_summary.csv", target_summary)
    write_csv(out / "source_summary.csv", source_summary)
    write_csv(out / "fit_audit.csv", fit_audit)
    (out / "ridge_models.json").write_text(
        json.dumps(
            {
                "feature_names": feature_names,
                "models": model_json,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    write_report(out, target_summary, source_summary, fit_audit, args.fit_mode)
    (out / "manifest.json").write_text(
        json.dumps(
            {
                "data_root": args.data_root,
                "target_predictions": args.target_predictions,
                "train_clients": train_clients,
                "target_clients": target_clients,
                "fit_mode": args.fit_mode,
                "alphas": alphas,
                "outputs": {
                    "target_summary": str(out / "target_summary.csv"),
                    "source_summary": str(out / "source_summary.csv"),
                    "fit_audit": str(out / "fit_audit.csv"),
                    "report": str(out / "reg_head_ablation_report.md"),
                },
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote regression head ablation to {out}")


if __name__ == "__main__":
    main()
