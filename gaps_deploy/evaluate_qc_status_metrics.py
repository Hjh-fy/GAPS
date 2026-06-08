"""Evaluate deployed QC decisions from prediction CSV files.

Unlike a threshold sweep, this script uses the actual `qc_status` written by
`predict_client_file` after loading a deployment policy.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

from .calibration import DEFAULT_CONC_RANGES


def _read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _to_float(value: Any, default: float = np.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value: Any, default: int = -1) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _error_arrays(rows: Sequence[Dict[str, str]]) -> Dict[str, np.ndarray]:
    y_true = np.asarray([_to_float(row.get("true_ppm")) for row in rows], dtype=np.float64)
    y_pred = np.asarray([_to_float(row.get("calibrated_ppm")) for row in rows], dtype=np.float64)
    true_cls = np.asarray([_to_int(row.get("true_class")) for row in rows], dtype=np.int64)
    pred_cls = np.asarray([_to_int(row.get("pred_class")) for row in rows], dtype=np.int64)
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "true_cls": true_cls,
        "pred_cls": pred_cls,
        "valid": valid,
    }


def _metrics(rows: Sequence[Dict[str, str]]) -> Dict[str, Any]:
    arrays = _error_arrays(rows)
    valid = arrays["valid"]
    if int(valid.sum()) == 0:
        return {
            "n": 0,
            "R2": None,
            "MAE": None,
            "RMSE": None,
            "NRMSE_range": None,
            "Bias": None,
            "route_accuracy": None,
        }

    y_true = arrays["y_true"][valid]
    y_pred = arrays["y_pred"][valid]
    true_cls = arrays["true_cls"][valid]
    pred_cls = arrays["pred_cls"][valid]
    err = y_pred - y_true
    abs_err = np.abs(err)
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))

    ranges = []
    for cls in true_cls:
        lo, hi = DEFAULT_CONC_RANGES.get(int(cls), (float(np.min(y_true)), float(np.max(y_true))))
        ranges.append(max(float(hi) - float(lo), 1e-12))
    range_arr = np.asarray(ranges, dtype=np.float64)

    route_valid = (true_cls >= 0) & (pred_cls >= 0)
    return {
        "n": int(valid.sum()),
        "R2": float(1.0 - ss_res / max(ss_tot, 1e-12)) if int(valid.sum()) >= 2 else None,
        "MAE": float(np.mean(abs_err)),
        "RMSE": float(np.sqrt(np.mean(err**2))),
        "NRMSE_range": float(np.sqrt(np.mean((err / range_arr) ** 2))),
        "MedAE": float(np.median(abs_err)),
        "P90AE": float(np.percentile(abs_err, 90)),
        "P95AE": float(np.percentile(abs_err, 95)),
        "Bias": float(np.mean(err)),
        "route_accuracy": float(np.mean(true_cls[route_valid] == pred_cls[route_valid])) if int(route_valid.sum()) else None,
    }


def _route_wrong_mask(rows: Sequence[Dict[str, str]]) -> np.ndarray:
    arrays = _error_arrays(rows)
    true_cls = arrays["true_cls"]
    pred_cls = arrays["pred_cls"]
    valid = (true_cls >= 0) & (pred_cls >= 0)
    return valid & (true_cls != pred_cls)


def _abs_errors(rows: Sequence[Dict[str, str]]) -> np.ndarray:
    arrays = _error_arrays(rows)
    err = np.abs(arrays["y_pred"] - arrays["y_true"])
    err[~arrays["valid"]] = np.nan
    return err


def evaluate_rows(
    rows: Sequence[Dict[str, str]],
    high_error_quantile: float,
) -> Dict[str, Any]:
    total = len(rows)
    statuses = np.asarray([row.get("qc_status", "accept") for row in rows], dtype=object)
    flagged = statuses != "accept"
    route_wrong = _route_wrong_mask(rows)
    abs_err = _abs_errors(rows)
    valid_errors = abs_err[np.isfinite(abs_err)]
    if valid_errors.size:
        high_error_threshold = float(np.quantile(valid_errors, high_error_quantile))
        high_error = np.isfinite(abs_err) & (abs_err >= high_error_threshold)
    else:
        high_error_threshold = float("nan")
        high_error = np.zeros(total, dtype=bool)

    def select(mask: np.ndarray) -> List[Dict[str, str]]:
        return [row for row, keep in zip(rows, mask.tolist()) if keep]

    report: Dict[str, Any] = {
        "total": total,
        "accept": int(np.sum(statuses == "accept")),
        "review": int(np.sum(statuses == "review")),
        "reject": int(np.sum(statuses == "reject")),
        "flagged": int(np.sum(flagged)),
        "accept_rate": float(np.mean(statuses == "accept")) if total else 0.0,
        "review_rate": float(np.mean(statuses == "review")) if total else 0.0,
        "reject_rate": float(np.mean(statuses == "reject")) if total else 0.0,
        "flag_rate": float(np.mean(flagged)) if total else 0.0,
        "route_wrong_total": int(route_wrong.sum()),
        "route_wrong_flagged": int((route_wrong & flagged).sum()),
        "route_wrong_recall": float((route_wrong & flagged).sum() / max(int(route_wrong.sum()), 1)),
        "high_error_quantile": float(high_error_quantile),
        "high_error_threshold": high_error_threshold,
        "high_error_total": int(high_error.sum()),
        "high_error_flagged": int((high_error & flagged).sum()),
        "high_error_recall": float((high_error & flagged).sum() / max(int(high_error.sum()), 1)),
        "all_metrics": _metrics(rows),
        "accept_metrics": _metrics(select(statuses == "accept")),
        "review_metrics": _metrics(select(statuses == "review")),
        "reject_metrics": _metrics(select(statuses == "reject")),
        "flagged_metrics": _metrics(select(flagged)),
    }
    return report


def _flatten(client_id: str, report: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "client_id": client_id,
        "total": report["total"],
        "accept": report["accept"],
        "review": report["review"],
        "reject": report["reject"],
        "flagged": report["flagged"],
        "accept_rate": report["accept_rate"],
        "flag_rate": report["flag_rate"],
        "route_wrong_total": report["route_wrong_total"],
        "route_wrong_flagged": report["route_wrong_flagged"],
        "route_wrong_recall": report["route_wrong_recall"],
        "high_error_threshold": report["high_error_threshold"],
        "high_error_total": report["high_error_total"],
        "high_error_flagged": report["high_error_flagged"],
        "high_error_recall": report["high_error_recall"],
    }
    for prefix in ["all_metrics", "accept_metrics", "flagged_metrics"]:
        for key, value in report[prefix].items():
            out[f"{prefix}_{key}"] = value
    return out


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _infer_client_id(path: Path, rows: Sequence[Dict[str, str]]) -> str:
    if rows and rows[0].get("client_id"):
        return str(rows[0]["client_id"])
    stem = path.stem.lower()
    for token in stem.replace("-", "_").split("_"):
        if token.startswith("c") and token[1:].isdigit():
            return token.upper()
        if token.startswith("client") and token[len("client"):].isdigit():
            return "C" + token[len("client"):]
    return path.stem


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate actual deployment QC statuses.")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--high-error-quantile", type=float, default=0.90)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-csv", default="")
    args = parser.parse_args()

    reports: Dict[str, Any] = {}
    flat_rows: List[Dict[str, Any]] = []
    for raw_path in args.inputs:
        path = Path(raw_path)
        rows = _read_rows(path)
        client_id = _infer_client_id(path, rows)
        report = evaluate_rows(rows, args.high_error_quantile)
        reports[client_id] = report
        flat_rows.append(_flatten(client_id, report))
        accept = report["accept_metrics"]
        print(
            f"{client_id}: accept={report['accept_rate']:.2%}, "
            f"flag={report['flag_rate']:.2%}, "
            f"accept_MAE={accept.get('MAE')}, "
            f"accept_RMSE={accept.get('RMSE')}, "
            f"route_wrong_recall={report['route_wrong_recall']:.4f}, "
            f"high_error_recall={report['high_error_recall']:.4f}"
        )

    if args.output_json:
        out_json = Path(args.output_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(reports, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Saved JSON: {out_json}")
    if args.output_csv:
        out_csv = Path(args.output_csv)
        _write_csv(out_csv, flat_rows)
        print(f"Saved CSV: {out_csv}")


if __name__ == "__main__":
    main()
