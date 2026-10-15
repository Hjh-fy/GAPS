"""Fixed-estimator evaluation helpers for the final IoT-J classification suite."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np


def _validate_windows(values: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 3:
        raise ValueError(f"{name} must have shape [windows, time, channels]")
    if array.shape[0] < 1 or array.shape[1] < 1 or array.shape[2] < 1:
        raise ValueError(f"{name} must have shape [windows, time, channels]")
    if not np.issubdtype(array.dtype, np.number) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains invalid sensor values")
    return array.astype(np.float64, copy=False)


def sensor_channel_shift_rows(
    source_windows: np.ndarray,
    target_windows: np.ndarray,
    *,
    target_id: str,
) -> list[dict]:
    """Compute raw sensor-space distribution shifts independently per channel."""
    source = _validate_windows(source_windows, name="source")
    target = _validate_windows(target_windows, name="target")
    if source.shape[2] != target.shape[2]:
        raise ValueError("source and target channel counts differ")
    source_flat = source.reshape(-1, source.shape[2])
    target_flat = target.reshape(-1, target.shape[2])
    rows: list[dict] = []
    for channel in range(source.shape[2]):
        src = source_flat[:, channel]
        tgt = target_flat[:, channel]
        src_mean = float(np.mean(src))
        tgt_mean = float(np.mean(tgt))
        src_std = float(np.std(src, ddof=0))
        tgt_std = float(np.std(tgt, ddof=0))
        src_median = float(np.median(src))
        tgt_median = float(np.median(tgt))
        src_q05, src_q25, src_q75, src_q95 = np.quantile(
            src, [0.05, 0.25, 0.75, 0.95]
        )
        tgt_q05, tgt_q25, tgt_q75, tgt_q95 = np.quantile(
            tgt, [0.05, 0.25, 0.75, 0.95]
        )
        src_iqr = float(src_q75 - src_q25)
        tgt_iqr = float(tgt_q75 - tgt_q25)
        rows.append(
            {
                "target_id": str(target_id),
                "channel": int(channel),
                "source_mean": src_mean,
                "target_mean": tgt_mean,
                "mean_shift": tgt_mean - src_mean,
                "source_std": src_std,
                "target_std": tgt_std,
                "std_shift": tgt_std - src_std,
                "source_median": src_median,
                "target_median": tgt_median,
                "median_shift": tgt_median - src_median,
                "source_iqr": src_iqr,
                "target_iqr": tgt_iqr,
                "iqr_shift": tgt_iqr - src_iqr,
                "source_q05": float(src_q05),
                "target_q05": float(tgt_q05),
                "q05_shift": float(tgt_q05 - src_q05),
                "source_q95": float(src_q95),
                "target_q95": float(tgt_q95),
                "q95_shift": float(tgt_q95 - src_q95),
                "standardized_mean_difference": (
                    (tgt_mean - src_mean) / src_std if src_std > 0.0 else 0.0
                ),
                "estimator_scope": "raw_sensor_space",
            }
        )
    return rows


def sensor_covariance_diagnostics(
    source_windows: np.ndarray,
    target_windows: np.ndarray,
    *,
    target_id: str,
) -> dict:
    source = _validate_windows(source_windows, name="source")
    target = _validate_windows(target_windows, name="target")
    if source.shape[2] != target.shape[2]:
        raise ValueError("source and target channel counts differ")
    source_cov = np.atleast_2d(np.cov(source.reshape(-1, source.shape[2]), rowvar=False))
    target_cov = np.atleast_2d(np.cov(target.reshape(-1, target.shape[2]), rowvar=False))
    delta = target_cov - source_cov
    shift = float(np.linalg.norm(delta, ord="fro"))
    source_norm = float(np.linalg.norm(source_cov, ord="fro"))
    return {
        "target_id": str(target_id),
        "num_channels": int(source.shape[2]),
        "source_covariance_frobenius_norm": source_norm,
        "target_covariance_frobenius_norm": float(
            np.linalg.norm(target_cov, ord="fro")
        ),
        "covariance_frobenius_shift": shift,
        "covariance_relative_frobenius_shift": (
            shift / source_norm if source_norm > 0.0 else 0.0
        ),
        "covariance_max_abs_shift": float(np.max(np.abs(delta))),
        "estimator_scope": "raw_sensor_space",
    }


def classification_metrics(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    num_classes: int = 4,
    ece_bins: int = 15,
) -> dict:
    probs = np.asarray(probabilities, dtype=np.float64)
    y_true = np.asarray(labels, dtype=np.int64).reshape(-1)
    if probs.ndim != 2 or probs.shape != (len(y_true), int(num_classes)):
        raise ValueError("probabilities must have shape [examples, num_classes]")
    if not np.all(np.isfinite(probs)) or np.any(probs < 0.0):
        raise ValueError("invalid probabilities")
    totals = probs.sum(axis=1, keepdims=True)
    if np.any(totals <= 0.0):
        raise ValueError("probability rows must have positive mass")
    probs = probs / totals
    if np.any(y_true < 0) or np.any(y_true >= num_classes):
        raise ValueError("labels outside fixed class order")
    predictions = np.argmax(probs, axis=1)
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(confusion, (y_true, predictions), 1)
    per_class = []
    f1_values = []
    for class_id in range(num_classes):
        tp = int(confusion[class_id, class_id])
        support = int(confusion[class_id, :].sum())
        predicted = int(confusion[:, class_id].sum())
        recall = tp / support if support else 0.0
        precision = tp / predicted if predicted else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall > 0.0
            else 0.0
        )
        f1_values.append(f1)
        per_class.append(
            {
                "class_id": class_id,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
            }
        )
    confidences = np.max(probs, axis=1)
    correct = (predictions == y_true).astype(np.float64)
    ece = 0.0
    for bin_index in range(int(ece_bins)):
        lower = bin_index / ece_bins
        upper = (bin_index + 1) / ece_bins
        mask = (
            (confidences >= lower) & (confidences <= upper)
            if bin_index == ece_bins - 1
            else (confidences >= lower) & (confidences < upper)
        )
        if np.any(mask):
            ece += float(np.mean(mask)) * abs(
                float(np.mean(correct[mask])) - float(np.mean(confidences[mask]))
            )
    true_prob = np.clip(probs[np.arange(len(y_true)), y_true], 1e-12, 1.0)
    return {
        "num_examples": int(len(y_true)),
        "accuracy": float(np.mean(correct)) if len(y_true) else 0.0,
        "macro_f1": float(np.mean(f1_values)),
        "nll": float(np.mean(-np.log(true_prob))) if len(y_true) else 0.0,
        "ece": float(ece),
        "ece_bins": int(ece_bins),
        "class_order": list(range(num_classes)),
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
    }


def add_source_target_f1_gaps(
    target_rows: Iterable[dict], *, source_macro_f1: float
) -> list[dict]:
    source_value = float(source_macro_f1)
    rows = []
    for item in target_rows:
        row = dict(item)
        target_value = float(row["macro_f1"])
        row["source_macro_f1"] = source_value
        row["source_target_f1_gap"] = source_value - target_value
        row["source_population"] = "combined_registered_C1_C2_test"
        rows.append(row)
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("refusing to write empty CSV")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-windows", required=True)
    parser.add_argument("--target-windows", required=True)
    parser.add_argument("--target-id", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    source = np.load(args.source_windows, allow_pickle=False)
    target = np.load(args.target_windows, allow_pickle=False)
    output = Path(args.output_dir)
    rows = sensor_channel_shift_rows(source, target, target_id=args.target_id)
    _write_csv(output / "sensor_channel_shift.csv", rows)
    (output / "sensor_covariance_shift.json").write_text(
        json.dumps(
            sensor_covariance_diagnostics(source, target, target_id=args.target_id),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
