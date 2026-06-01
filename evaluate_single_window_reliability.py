"""Single-window reliability analysis for gas regression predictions.

The main deployment protocol is one window in, one prediction out. This script
does not aggregate windows from the same raw file. It adds a deployment-visible
risk score to each single-window prediction and evaluates whether high-risk
windows explain large concentration errors.

Labels are used only for evaluation. Risk scores use model confidence, the
predicted class/concentration, and the target calibration set.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


GAS_NAMES = ["Ethanol", "CO", "Ethylene", "Methane"]

BASE_SCORES = [
    "classifier_uncertainty",
    "margin_risk",
    "response_signature_norm",
    "response_conc_gap_norm",
    "response_mean_conc_gap_norm",
    "class_response_rank_risk",
    "class_response_margin_risk",
    "route_response_risk",
    "composite_response_risk",
]


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fnum(value, default=np.nan):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def inum(value, default=0):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def parse_clients(text):
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def parse_coverages(text):
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def gas_name(cls):
    cls = inum(cls)
    if 0 <= cls < len(GAS_NAMES):
        return GAS_NAMES[cls]
    return f"Class{cls}"


def client_sort_key(name):
    text = str(name)
    if text.startswith("C"):
        text = text[1:]
    try:
        return int(float(text))
    except ValueError:
        return 10**9


def primary_groups(rows):
    clients = sorted({r["client"] for r in rows}, key=client_sort_key)
    return ["ALL"] + clients


def feature_descriptors(features):
    """Return compact response descriptors for each window.

    features are expected to be shaped [n, time, channels]. The descriptor uses
    physically interpretable summaries: mean response, dynamic range, slope, and
    short-term variability for each sensor channel.
    """
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 3:
        raise ValueError(f"Expected feature array [n,time,channels], got {x.shape}")
    n, t, _ = x.shape
    edge = max(1, int(round(t * 0.10)))
    mean_ch = x.mean(axis=1)
    std_ch = x.std(axis=1)
    amp_ch = x.max(axis=1) - x.min(axis=1)
    slope_ch = x[:, -edge:, :].mean(axis=1) - x[:, :edge, :].mean(axis=1)
    diff = np.diff(x, axis=1)
    noise_ch = diff.std(axis=1) if diff.size else np.zeros_like(mean_ch)
    sig = np.concatenate([mean_ch, std_ch, amp_ch, slope_ch, noise_ch], axis=1)
    return {
        "signature": sig,
        "mean_all": mean_ch.mean(axis=1),
        "std_all": std_ch.mean(axis=1),
        "amp_all": amp_ch.mean(axis=1),
        "abs_slope_all": np.abs(slope_ch).mean(axis=1),
        "n_windows": n,
    }


def robust_center_scale(arr):
    arr = np.asarray(arr, dtype=np.float64)
    center = np.median(arr, axis=0)
    q75 = np.percentile(arr, 75, axis=0)
    q25 = np.percentile(arr, 25, axis=0)
    scale = q75 - q25
    scale = np.where(scale > 1e-8, scale, arr.std(axis=0))
    scale = np.where(scale > 1e-8, scale, 1.0)
    return center, scale


def concentration_from_info(info, cls):
    if "concentration" in info:
        return fnum(info.get("concentration"))
    reg = info.get("regression_label", [])
    if isinstance(reg, list) and 0 <= cls < len(reg):
        return fnum(reg[cls])
    return np.nan


def build_calibration_reference(client_dir):
    client_dir = Path(client_dir)
    calib_features = np.load(client_dir / "calibration_features.npy")
    calib_info = load_json(client_dir / "calibration_experiment_info.json")
    desc = feature_descriptors(calib_features)
    refs = {}
    for cls in sorted({inum(item.get("classification_label")) for item in calib_info}):
        idxs = [i for i, item in enumerate(calib_info) if inum(item.get("classification_label")) == cls]
        if not idxs:
            continue
        sigs = desc["signature"][idxs]
        center, scale = robust_center_scale(sigs)
        z_sigs = (sigs - center.reshape(1, -1)) / scale.reshape(1, -1)
        loocv = []
        for i, sig in enumerate(z_sigs):
            if len(z_sigs) <= 1:
                continue
            others = np.delete(z_sigs, i, axis=0)
            loocv.append(float(np.min(np.linalg.norm(others - sig.reshape(1, -1), axis=1))))
        rows = []
        for local_i, global_i in enumerate(idxs):
            item = calib_info[global_i]
            rows.append({
                "global_index": global_i,
                "filename": item.get("filename", ""),
                "class": cls,
                "gas": gas_name(cls),
                "concentration": concentration_from_info(item, cls),
                "mean_all": float(desc["mean_all"][global_i]),
                "amp_all": float(desc["amp_all"][global_i]),
                "abs_slope_all": float(desc["abs_slope_all"][global_i]),
                "z_signature": z_sigs[local_i],
            })
        refs[cls] = {
            "class": cls,
            "center": center,
            "scale": scale,
            "rows": rows,
            "z_sigs": z_sigs,
            "loocv_p90": float(np.percentile(loocv, 90)) if loocv else 1.0,
            "loocv_median": float(np.median(loocv)) if loocv else 1.0,
        }
    return refs


def nearest_signature(sig, ref):
    z = (sig - ref["center"]) / ref["scale"]
    dists = np.linalg.norm(ref["z_sigs"] - z.reshape(1, -1), axis=1)
    if dists.size > 1:
        zero_mask = dists <= 1e-12
        if np.any(zero_mask) and np.any(~zero_mask):
            dists = np.where(zero_mask, np.inf, dists)
    idx = int(np.argmin(dists))
    return ref["rows"][idx], float(dists[idx])


def class_response_consistency(sig, refs, pred_cls):
    """Compare the predicted class with calibration response prototypes.

    A class can be highly confident in softmax space but still inconsistent with
    the sensor response. This score asks whether the test signature is closer to
    the predicted-class calibration cloud than to other class clouds.
    """
    items = []
    for cls, ref in refs.items():
        nearest, dist = nearest_signature(sig, ref)
        scale = max(fnum(ref.get("loocv_p90"), 1.0), 1e-8)
        items.append({
            "class": int(cls),
            "gas": gas_name(cls),
            "nearest": nearest,
            "dist": dist,
            "norm": float(dist / scale),
        })
    items.sort(key=lambda item: item["norm"])
    pred_item = next((item for item in items if item["class"] == pred_cls), None)
    if pred_item is None or not items:
        return {
            "best_response_class": None,
            "best_response_gas": "",
            "best_response_norm": np.nan,
            "pred_response_norm": np.nan,
            "class_response_rank": np.nan,
            "class_response_rank_risk": np.nan,
            "class_response_margin": np.nan,
            "class_response_margin_risk": np.nan,
        }
    rank = 1 + [item["class"] for item in items].index(pred_cls)
    best = items[0]
    pred_norm = pred_item["norm"]
    best_norm = best["norm"]
    margin = pred_norm - best_norm
    return {
        "best_response_class": int(best["class"]),
        "best_response_gas": best["gas"],
        "best_response_nearest_calib_filename": best["nearest"].get("filename", ""),
        "best_response_nearest_calib_conc": fnum(best["nearest"].get("concentration")),
        "best_response_norm": float(best_norm),
        "pred_response_norm": float(pred_norm),
        "class_response_rank": int(rank),
        "class_response_rank_risk": float(rank - 1),
        "class_response_margin": float(margin),
        "class_response_margin_risk": float(max(0.0, margin)),
    }


def nearest_scalar(value, ref, key):
    rows = ref["rows"]
    distances = np.asarray([abs(fnum(row.get(key)) - value) for row in rows], dtype=np.float64)
    if distances.size > 1:
        zero_mask = distances <= 1e-12
        if np.any(zero_mask) and np.any(~zero_mask):
            distances = np.where(zero_mask, np.inf, distances)
    idx = int(np.argmin(distances))
    return rows[idx], float(abs(fnum(rows[idx].get(key)) - value))


def load_pipeline_predictions(pred_file, mode="pipeline"):
    rows = []
    for row in read_csv(pred_file):
        if row.get("mode") != mode:
            continue
        r = dict(row)
        for key in [
            "client_id", "sample_index", "true_class", "pred_class", "phase",
            "class_correct", "soft_route_used",
        ]:
            if key in r:
                r[key] = inum(r[key])
        for key in [
            "true_ppm", "pred_ppm", "error_ppm", "abs_error_ppm",
            "top1_confidence", "top2_confidence", "confidence_margin",
        ]:
            if key in r:
                r[key] = fnum(r[key])
        rows.append(r)
    rows.sort(key=lambda r: inum(r.get("sample_index")))
    return rows


def enrich_client(experiment_dir, data_dir, client_id, mode="pipeline", split="test"):
    experiment_dir = Path(experiment_dir)
    data_dir = Path(data_dir)
    split = str(split).strip().lower()
    if split == "test":
        pred_name = f"target_client{client_id}_predictions.csv"
        feature_prefix = "test"
    elif split == "calibration":
        pred_name = f"target_client{client_id}_calibration_predictions.csv"
        feature_prefix = "calibration"
    else:
        raise ValueError(f"Unsupported split: {split}")
    pred_file = experiment_dir / "checkpoints" / "separate_regression" / "prediction_exports" / pred_name
    client_dir = data_dir / f"client_{client_id}"
    rows = load_pipeline_predictions(pred_file, mode=mode)
    split_features = np.load(client_dir / f"{feature_prefix}_features.npy")
    split_info = load_json(client_dir / f"{feature_prefix}_experiment_info.json")
    desc = feature_descriptors(split_features)
    refs = build_calibration_reference(client_dir)
    if len(rows) != desc["n_windows"]:
        raise ValueError(f"C{client_id} {split}: predictions={len(rows)} but windows={desc['n_windows']}")

    enriched = []
    for row in rows:
        idx = inum(row.get("sample_index"))
        info = split_info[idx]
        pred_cls = inum(row.get("pred_class"))
        ref = refs.get(pred_cls)
        r = dict(row)
        r.update({
            "client": f"C{client_id}",
            "split": split,
            "true_gas": row.get("true_gas") or gas_name(row.get("true_class")),
            "pred_gas": row.get("pred_gas") or gas_name(row.get("pred_class")),
            "filename": info.get("filename", ""),
            "repeat_id": info.get("repeat_id", ""),
            "concentration_code": info.get("concentration_code", ""),
            "feature_mean_all": float(desc["mean_all"][idx]),
            "feature_amp_all": float(desc["amp_all"][idx]),
            "feature_abs_slope_all": float(desc["abs_slope_all"][idx]),
        })
        r["classifier_uncertainty"] = float(1.0 - fnum(row.get("top1_confidence"), 1.0))
        r["margin_risk"] = float(1.0 - fnum(row.get("confidence_margin"), 1.0))
        r.update(class_response_consistency(desc["signature"][idx], refs, pred_cls))
        best_ref = refs.get(inum(r.get("best_response_class"), -1))
        if best_ref is not None:
            best_mean, best_mean_dist = nearest_scalar(float(desc["mean_all"][idx]), best_ref, "mean_all")
            r["best_response_mean_nearest_calib_filename"] = best_mean.get("filename", "")
            r["best_response_mean_nearest_calib_conc"] = fnum(best_mean.get("concentration"))
            r["best_response_mean_distance"] = best_mean_dist
        else:
            r["best_response_mean_nearest_calib_filename"] = ""
            r["best_response_mean_nearest_calib_conc"] = np.nan
            r["best_response_mean_distance"] = np.nan
        if ref is not None:
            nearest, sig_dist = nearest_signature(desc["signature"][idx], ref)
            nearest_mean, mean_dist = nearest_scalar(float(desc["mean_all"][idx]), ref, "mean_all")
            pred_ppm = fnum(row.get("pred_ppm"))
            sig_scale = max(fnum(ref.get("loocv_p90"), 1.0), 1e-8)
            conc_gap = abs(pred_ppm - fnum(nearest.get("concentration")))
            mean_conc_gap = abs(pred_ppm - fnum(nearest_mean.get("concentration")))
            r.update({
                "nearest_calib_filename": nearest.get("filename", ""),
                "nearest_calib_conc": fnum(nearest.get("concentration")),
                "nearest_mean_calib_filename": nearest_mean.get("filename", ""),
                "nearest_mean_calib_conc": fnum(nearest_mean.get("concentration")),
                "response_signature_dist": sig_dist,
                "response_signature_norm": float(sig_dist / sig_scale),
                "response_mean_distance": mean_dist,
                "response_conc_gap": conc_gap,
                "response_mean_conc_gap": mean_conc_gap,
                "response_conc_gap_norm": float(conc_gap / 25.0),
                "response_mean_conc_gap_norm": float(mean_conc_gap / 25.0),
            })
        else:
            for key in [
                "nearest_calib_filename", "nearest_calib_conc", "nearest_mean_calib_filename",
                "nearest_mean_calib_conc", "response_signature_dist", "response_signature_norm",
                "response_mean_distance", "response_conc_gap", "response_mean_conc_gap",
                "response_conc_gap_norm", "response_mean_conc_gap_norm",
            ]:
                r[key] = np.nan
        r["route_response_risk"] = float(np.nanmax([
            fnum(r.get("class_response_rank_risk")),
            fnum(r.get("class_response_margin_risk")),
            fnum(r.get("classifier_uncertainty")) * 10.0,
        ]))
        r["composite_response_risk"] = float(np.nanmax([
            fnum(r.get("response_signature_norm")),
            fnum(r.get("response_conc_gap_norm")),
            fnum(r.get("response_mean_conc_gap_norm")),
            fnum(r.get("class_response_rank_risk")),
            fnum(r.get("class_response_margin_risk")),
            fnum(r.get("classifier_uncertainty")) * 10.0,
        ]))
        r["high_error_25ppm"] = int(fnum(r.get("abs_error_ppm")) > 25.0)
        r["high_error_40ppm"] = int(fnum(r.get("abs_error_ppm")) > 40.0)
        enriched.append(r)
    return enriched


def metrics(rows):
    if not rows:
        return {
            "n": 0, "R2": None, "RMSE": None, "MAE": None, "MedAE": None,
            "P90AE": None, "P95AE": None, "Bias": None, "class_acc": None,
        }
    y = np.asarray([fnum(r.get("true_ppm")) for r in rows], dtype=np.float64)
    p = np.asarray([fnum(r.get("pred_ppm")) for r in rows], dtype=np.float64)
    err = p - y
    ae = np.abs(err)
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else None
    return {
        "n": int(len(rows)),
        "R2": r2,
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "MAE": float(np.mean(ae)),
        "MedAE": float(np.median(ae)),
        "P90AE": float(np.percentile(ae, 90)),
        "P95AE": float(np.percentile(ae, 95)),
        "Bias": float(np.mean(err)),
        "class_acc": float(np.mean([inum(r.get("class_correct")) for r in rows])),
    }


def rankdata(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=np.float64)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def spearman(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return None
    rx = rankdata(x[mask])
    ry = rankdata(y[mask])
    if float(rx.std()) < 1e-12 or float(ry.std()) < 1e-12:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def stratify(rows, score, coverages, group):
    scored = []
    for row in rows:
        value = fnum(row.get(score), np.inf)
        scored.append((value, row))
    scored.sort(key=lambda item: (not np.isfinite(item[0]), item[0]))
    out = []
    n = len(scored)
    for cov in coverages:
        keep_n = max(1, int(round(n * cov)))
        keep = [row for _, row in scored[:keep_n]]
        item = metrics(keep)
        finite_scores = [value for value, _ in scored[:keep_n] if np.isfinite(value)]
        item.update({
            "group": group,
            "score": score,
            "target_coverage": float(cov),
            "accepted_coverage": float(keep_n / max(1, n)),
            "risk_threshold": float(max(finite_scores)) if finite_scores else None,
            "mean_risk_score": float(np.mean(finite_scores)) if finite_scores else None,
        })
        out.append(item)
    return out


def build_summaries(rows, coverages):
    groups = {"ALL": rows}
    for client in sorted({r["client"] for r in rows}):
        groups[client] = [r for r in rows if r["client"] == client]
    for client in sorted({r["client"] for r in rows}):
        for cls in sorted({inum(r.get("true_class")) for r in rows if r["client"] == client}):
            key = f"{client}_{gas_name(cls)}"
            groups[key] = [r for r in rows if r["client"] == client and inum(r.get("true_class")) == cls]

    baseline_rows = []
    curve_rows = []
    corr_rows = []
    for group, group_rows in groups.items():
        m = metrics(group_rows)
        cond = metrics([r for r in group_rows if inum(r.get("class_correct")) == 1])
        baseline_rows.append({"group": group, "subset": "full", **m})
        baseline_rows.append({"group": group, "subset": "conditional_class_correct", **cond})
        abs_err = [fnum(r.get("abs_error_ppm")) for r in group_rows]
        high25 = [inum(r.get("high_error_25ppm")) for r in group_rows]
        for score in BASE_SCORES:
            vals = [fnum(r.get(score)) for r in group_rows]
            corr_rows.append({
                "group": group,
                "score": score,
                "spearman_abs_error": spearman(vals, abs_err),
                "spearman_high_error_25ppm": spearman(vals, high25),
                "n": len(group_rows),
            })
            curve_rows.extend(stratify(group_rows, score, coverages, group))
    return baseline_rows, curve_rows, corr_rows


def fmt(value, digits=3):
    if value is None:
        return ""
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return ""
        return f"{float(value):.{digits}f}"
    return str(value)


def best_score_rows(curve_rows, coverage=0.90, groups=None):
    if groups is None:
        groups = sorted({r["group"] for r in curve_rows if r["group"] == "ALL" or "_" not in r["group"]}, key=lambda x: -1 if x == "ALL" else client_sort_key(x))
    out = []
    for group in groups:
        candidates = [r for r in curve_rows if r["group"] == group and abs(fnum(r.get("target_coverage")) - coverage) < 1e-9]
        if not candidates:
            continue
        best = min(candidates, key=lambda r: fnum(r.get("P90AE"), np.inf))
        out.append(best)
    return out


def write_markdown(path, baseline_rows, curve_rows, corr_rows, enriched_rows, experiment_dir, data_dir):
    display_groups = primary_groups(enriched_rows)
    lines = ["# Single-Window Reliability Analysis", ""]
    lines.append(f"- experiment_dir: `{experiment_dir}`")
    lines.append(f"- data_dir: `{data_dir}`")
    lines.append("- protocol: single-window prediction; no file-level voting or median aggregation")
    lines.append("- risk scores are deployment-visible; labels are used only for evaluation")
    lines.append("")

    lines.append("## Full-Coverage Baseline")
    cols = ["group", "subset", "n", "R2", "MAE", "P90AE", "P95AE", "class_acc"]
    focus = [r for r in baseline_rows if r["group"] in display_groups]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for row in focus:
        lines.append("| " + " | ".join(fmt(row.get(c)) for c in cols) + " |")
    lines.append("")

    lines.append("## Score Correlation with Absolute Error")
    corr_cols = ["group", "score", "spearman_abs_error", "spearman_high_error_25ppm", "n"]
    focus_corr = [r for r in corr_rows if r["group"] in display_groups]
    lines.append("| " + " | ".join(corr_cols) + " |")
    lines.append("|" + "|".join(["---"] * len(corr_cols)) + "|")
    for row in focus_corr:
        lines.append("| " + " | ".join(fmt(row.get(c)) for c in corr_cols) + " |")
    lines.append("")

    lines.append("## Best 90% Coverage Workpoints by P90AE")
    wp_cols = ["group", "score", "accepted_coverage", "R2", "MAE", "P90AE", "P95AE", "risk_threshold"]
    lines.append("| " + " | ".join(wp_cols) + " |")
    lines.append("|" + "|".join(["---"] * len(wp_cols)) + "|")
    for row in best_score_rows(curve_rows, coverage=0.90, groups=display_groups):
        lines.append("| " + " | ".join(fmt(row.get(c)) for c in wp_cols) + " |")
    lines.append("")

    lines.append("## Composite Response-Risk Workpoints")
    comp = [
        r for r in curve_rows
        if r["score"] == "composite_response_risk"
        and r["group"] in display_groups
        and fnum(r.get("target_coverage")) in (1.0, 0.95, 0.9, 0.8, 0.7)
    ]
    comp_cols = ["group", "accepted_coverage", "R2", "MAE", "P90AE", "P95AE", "class_acc", "risk_threshold"]
    lines.append("| " + " | ".join(comp_cols) + " |")
    lines.append("|" + "|".join(["---"] * len(comp_cols)) + "|")
    for row in comp:
        lines.append("| " + " | ".join(fmt(row.get(c)) for c in comp_cols) + " |")
    lines.append("")

    lines.append("## Worst Single Windows")
    worst = sorted(enriched_rows, key=lambda r: fnum(r.get("abs_error_ppm")), reverse=True)[:18]
    worst_cols = [
        "client", "filename", "true_gas", "true_ppm", "pred_gas", "pred_ppm",
        "abs_error_ppm", "class_correct", "top1_confidence", "composite_response_risk",
        "route_response_risk", "best_response_gas", "class_response_rank",
        "response_signature_norm", "response_conc_gap", "nearest_calib_conc",
    ]
    lines.append("| " + " | ".join(worst_cols) + " |")
    lines.append("|" + "|".join(["---"] * len(worst_cols)) + "|")
    for row in worst:
        lines.append("| " + " | ".join(fmt(row.get(c)) for c in worst_cols) + " |")
    lines.append("")

    lines.append("## Interpretation")
    lines.append("- Full-coverage metrics remain the main result. Selective workpoints are a deployment reliability layer, not a replacement for the main score.")
    lines.append("- If a score has positive correlation with absolute error and lower MAE/P90AE at lower coverage, it can support a retest or warning policy.")
    lines.append("- Response-based scores are more physically interpretable than classifier confidence when the classifier is already near saturation.")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_curves(curve_rows, out_dir):
    out_dir = Path(out_dir)
    plot_scores = ["composite_response_risk", "route_response_risk", "response_mean_conc_gap_norm"]
    metrics_to_plot = [("MAE", "MAE (ppm)"), ("P90AE", "P90AE (ppm)"), ("P95AE", "P95AE (ppm)")]
    groups = sorted({r["group"] for r in curve_rows if r["group"] == "ALL" or "_" not in r["group"]}, key=lambda x: -1 if x == "ALL" else client_sort_key(x))
    for metric_name, ylabel in metrics_to_plot:
        fig, axes = plt.subplots(1, len(groups), figsize=(3.85 * len(groups), 3.5), sharex=True)
        if len(groups) == 1:
            axes = [axes]
        for ax, group in zip(axes, groups):
            for score in plot_scores:
                rows = sorted(
                    [r for r in curve_rows if r["group"] == group and r["score"] == score],
                    key=lambda r: r["accepted_coverage"],
                )
                if not rows:
                    continue
                ax.plot(
                    [r["accepted_coverage"] for r in rows],
                    [r[metric_name] for r in rows],
                    marker="o",
                    linewidth=1.5,
                    markersize=3.5,
                    label=score,
                )
            ax.set_title(group)
            ax.set_xlabel("Accepted coverage")
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.22)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.invert_yaxis()
        axes[0].legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f"single_window_reliability_{metric_name}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_risk_scatter(rows, out_dir):
    out_dir = Path(out_dir)
    clients = sorted({r["client"] for r in rows}, key=client_sort_key)
    fig, axes = plt.subplots(1, len(clients), figsize=(4.05 * len(clients), 3.6), sharey=True)
    if len(clients) == 1:
        axes = [axes]
    colors = {0: "#4C78A8", 1: "#F58518", 2: "#E45756", 3: "#54A24B"}
    for ax, client in zip(axes, clients):
        sub = [r for r in rows if r["client"] == client]
        for cls in sorted({inum(r.get("true_class")) for r in sub}):
            cls_rows = [r for r in sub if inum(r.get("true_class")) == cls]
            ax.scatter(
                [fnum(r.get("composite_response_risk")) for r in cls_rows],
                [fnum(r.get("abs_error_ppm")) for r in cls_rows],
                s=14,
                alpha=0.62,
                color=colors.get(cls, "#888888"),
                label=gas_name(cls),
            )
        ax.set_title(client)
        ax.set_xlabel("Composite response risk")
        ax.grid(alpha=0.22)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("Absolute error (ppm)")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "single_window_risk_vs_error.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Single-window reliability analysis")
    parser.add_argument("--experiment_dir", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--clients", default="1,2,3")
    parser.add_argument("--mode", default="pipeline", choices=["pipeline", "oracle"])
    parser.add_argument("--split", default="test", choices=["test", "calibration"],
                        help="Prediction/data split to enrich. Calibration requires exported calibration predictions.")
    parser.add_argument("--coverages", default="1.0,0.95,0.9,0.8,0.7,0.6,0.5")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    clients = parse_clients(args.clients)
    coverages = parse_coverages(args.coverages)
    all_rows = []
    for client_id in clients:
        all_rows.extend(enrich_client(args.experiment_dir, args.data_dir, client_id, args.mode, args.split))

    baseline_rows, curve_rows, corr_rows = build_summaries(all_rows, coverages)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "single_window_reliability_records.csv", all_rows)
    write_csv(out_dir / "single_window_reliability_baseline.csv", baseline_rows)
    write_csv(out_dir / "single_window_reliability_curves.csv", curve_rows)
    write_csv(out_dir / "single_window_reliability_correlations.csv", corr_rows)
    summary = {
        "settings": vars(args),
        "baseline": baseline_rows,
        "correlations": corr_rows,
        "best_90pct_workpoints": best_score_rows(curve_rows, coverage=0.90, groups=primary_groups(all_rows)),
    }
    (out_dir / "single_window_reliability_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown(
        out_dir / "single_window_reliability_summary.md",
        baseline_rows,
        curve_rows,
        corr_rows,
        all_rows,
        args.experiment_dir,
        args.data_dir,
    )
    plot_curves(curve_rows, out_dir)
    plot_risk_scatter(all_rows, out_dir)
    print((out_dir / "single_window_reliability_summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
