"""Read-only canonical-v1 routing, FedRidge-bootstrap, and QC analyses."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CANONICAL = ROOT / "results/iotj_canonical_v1_final_20260808"
OUTPUT = ROOT / "results/iotj_canonical_v1_scientific_validation_20260809"
TARGETS = ("C3", "C4", "C5")
CLASS_RANGES = {0: 112.5, 1: 225.0, 2: 112.5, 3: 225.0}


def routing_gap_row(
    *, s_all_rmse: float, s_cc_rmse: float, oracle_rmse: float
) -> dict[str, float]:
    return {
        "routing_gap_ppm": float(s_all_rmse - s_cc_rmse),
        "oracle_gap_ppm": float(s_cc_rmse - oracle_rmse),
    }


def _rmse(true: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(pred - true))))


def _mae(true: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - true)))


def _nrmse(frame: pd.DataFrame, pred_col: str) -> float:
    error = frame[pred_col].to_numpy(float) - frame["true_ppm"].to_numpy(float)
    scale = frame["true_class"].map(CLASS_RANGES).to_numpy(float)
    return float(np.sqrt(np.mean(np.square(error / scale))))


def _metrics(frame: pd.DataFrame, pred_col: str) -> dict[str, float | int]:
    true = frame["true_ppm"].to_numpy(float)
    pred = frame[pred_col].to_numpy(float)
    return {
        "N": int(len(frame)),
        "RMSE": _rmse(true, pred),
        "NRMSE_range": _nrmse(frame, pred_col),
        "MAE": _mae(true, pred),
    }


def grouped_paired_rmse_bootstrap(
    frame: pd.DataFrame,
    group_col: str,
    *,
    repeats: int = 5000,
    seed: int = 20260809,
) -> dict[str, Any]:
    if repeats < 1:
        raise ValueError("bootstrap repeats must be positive")
    required = {group_col, "true_ppm", "pred_83d_ppm", "pred_84d_h1_ppm"}
    missing = required.difference(frame.columns)
    if missing or frame.empty:
        raise ValueError(f"paired grouped bootstrap inputs are incomplete: {sorted(missing)}")
    grouped = []
    for _group, part in frame.groupby(group_col, sort=True):
        true = part["true_ppm"].to_numpy(float)
        grouped.append((
            len(part),
            float(np.square(part["pred_83d_ppm"].to_numpy(float) - true).sum()),
            float(np.square(part["pred_84d_h1_ppm"].to_numpy(float) - true).sum()),
        ))
    values = np.asarray(grouped, dtype=np.float64)
    group_count = len(values)
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, group_count, size=(repeats, group_count))
    counts = values[sampled, 0].sum(axis=1)
    rmse83 = np.sqrt(values[sampled, 1].sum(axis=1) / counts)
    rmse84 = np.sqrt(values[sampled, 2].sum(axis=1) / counts)
    deltas = rmse84 - rmse83
    true = frame["true_ppm"].to_numpy(float)
    point83 = _rmse(true, frame["pred_83d_ppm"].to_numpy(float))
    point84 = _rmse(true, frame["pred_84d_h1_ppm"].to_numpy(float))
    return {
        "N": int(len(frame)),
        "group_count": group_count,
        "bootstrap_unit": "raw_file_group",
        "bootstrap_repeats": repeats,
        "bootstrap_seed": seed,
        "RMSE_83D": point83,
        "RMSE_84D": point84,
        "delta_rmse_ppm": point84 - point83,
        "delta_ci025_ppm": float(np.quantile(deltas, 0.025)),
        "delta_ci975_ppm": float(np.quantile(deltas, 0.975)),
        "bootstrap_probability_delta_lt_zero": float(np.mean(deltas < 0.0)),
    }


def risk_coverage_curve(
    frame: pd.DataFrame, *, coverages: Sequence[float]
) -> list[dict[str, Any]]:
    ordered = frame.sort_values("qc_risk_score_final", kind="stable")
    rows: list[dict[str, Any]] = []
    for requested in coverages:
        if not 0 < requested <= 1:
            raise ValueError("coverage must be in (0,1]")
        count = max(1, int(np.floor(len(ordered) * requested)))
        accepted = ordered.iloc[:count]
        metrics = _metrics(accepted, "pred_84d_h1_ppm")
        rows.append({
            "requested_coverage": float(requested),
            "coverage": count / len(ordered),
            **metrics,
            "max_accepted_risk": float(accepted["qc_risk_score_final"].max()),
        })
    return rows


def capture_summary(frame: pd.DataFrame, workpoint: str) -> dict[str, Any]:
    decision_col = f"{workpoint}_decision"
    high_risk = frame[decision_col] != "accepted"
    misroute = frame["route_correct"].astype(int) == 0
    large_error = frame["abs_error"].astype(float) > 40.0
    def rate(event: pd.Series) -> float:
        denominator = int(event.sum())
        return float((high_risk & event).sum() / denominator) if denominator else float("nan")
    return {
        "workpoint": workpoint,
        "N": int(len(frame)),
        "accepted_N": int((~high_risk).sum()),
        "high_risk_N": int(high_risk.sum()),
        "misroute_N": int(misroute.sum()),
        "misroute_captured_N": int((high_risk & misroute).sum()),
        "misroute_capture_rate": rate(misroute),
        "error_gt40_N": int(large_error.sum()),
        "error_gt40_captured_N": int((high_risk & large_error).sum()),
        "error_gt40_capture_rate": rate(large_error),
    }


def _load_records() -> pd.DataFrame:
    frames = []
    for target in TARGETS:
        path = CANONICAL / f"evidence_closure/derived_records/{target}/test_records.csv"
        frame = pd.read_csv(path)
        frame["target"] = target
        frame["raw_file_group"] = target + "|" + frame["filename"].astype(str)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    fields: list[str] = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def routing_analysis(records: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    scopes: list[tuple[str, pd.DataFrame]] = [("ALL", records)]
    scopes.extend((target, records[records["target"] == target]) for target in TARGETS)
    expanded: list[tuple[str, str, pd.DataFrame]] = []
    for scope, frame in scopes:
        expanded.append((scope, "ALL", frame))
        for gas, gas_frame in frame.groupby("gas", sort=True):
            expanded.append((scope, str(gas), gas_frame))
    for scope, gas, frame in expanded:
        correct = frame[frame["route_correct"].astype(int) == 1]
        s_all = _metrics(frame, "pred_84d_h1_ppm")
        s_cc = _metrics(correct, "pred_84d_h1_ppm")
        oracle = _metrics(frame, "oracle_route_pred_84d_h1_ppm")
        rows.append({
            "scope": scope, "gas": gas,
            "S_ALL_N": s_all["N"], "S_ALL_RMSE": s_all["RMSE"],
            "S_ALL_NRMSE": s_all["NRMSE_range"], "S_ALL_MAE": s_all["MAE"],
            "S_CC_N": s_cc["N"], "S_CC_RMSE": s_cc["RMSE"],
            "S_CC_NRMSE": s_cc["NRMSE_range"], "S_CC_MAE": s_cc["MAE"],
            "oracle_N": oracle["N"], "oracle_RMSE": oracle["RMSE"],
            "oracle_NRMSE": oracle["NRMSE_range"], "oracle_MAE": oracle["MAE"],
            **routing_gap_row(
                s_all_rmse=float(s_all["RMSE"]),
                s_cc_rmse=float(s_cc["RMSE"]),
                oracle_rmse=float(oracle["RMSE"]),
            ),
            "misroute_N": int((frame["route_correct"].astype(int) == 0).sum()),
        })
    return rows


def bootstrap_analysis(records: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    scopes: list[tuple[str, pd.DataFrame]] = [("ALL", records)]
    scopes.extend((target, records[records["target"] == target]) for target in TARGETS)
    for scope, frame in scopes:
        subsets = [("ALL", frame)] + [
            (str(gas), gas_frame) for gas, gas_frame in frame.groupby("gas", sort=True)
        ]
        for gas, subset in subsets:
            rows.append({
                "scope": scope, "gas": gas,
                **grouped_paired_rmse_bootstrap(
                    subset, "raw_file_group", repeats=5000, seed=20260809
                ),
            })
    return rows


def qc_analysis(records: pd.DataFrame) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    risk_rows: list[dict[str, Any]] = []
    capture_rows: list[dict[str, Any]] = []
    coverages = np.round(np.arange(0.10, 1.001, 0.01), 2)
    scopes: list[tuple[str, pd.DataFrame]] = [("ALL", records)]
    scopes.extend((target, records[records["target"] == target]) for target in TARGETS)
    for scope, frame in scopes:
        curve = risk_coverage_curve(frame, coverages=coverages)
        for row in curve:
            row["scope"] = scope
        risk_rows.extend(curve)
        for workpoint in ("HC90", "HC95"):
            capture_rows.append({"scope": scope, "analysis": "fixed_workpoint", **capture_summary(frame, workpoint)})
        ordered = frame.sort_values("qc_risk_score_final", ascending=False, kind="stable")
        for top_fraction in (0.01, 0.025, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30):
            count = max(1, int(np.ceil(len(ordered) * top_fraction)))
            selected = ordered.index[:count]
            selected_mask = frame.index.isin(selected)
            misroute = frame["route_correct"].astype(int) == 0
            large_error = frame["abs_error"].astype(float) > 40.0
            capture_rows.append({
                "scope": scope,
                "analysis": "top_risk_fraction",
                "top_risk_fraction": top_fraction,
                "high_risk_N": count,
                "misroute_N": int(misroute.sum()),
                "misroute_capture_rate": float((selected_mask & misroute).sum() / misroute.sum()) if misroute.any() else float("nan"),
                "error_gt40_N": int(large_error.sum()),
                "error_gt40_capture_rate": float((selected_mask & large_error).sum() / large_error.sum()) if large_error.any() else float("nan"),
            })
    return risk_rows, capture_rows


def _aurc(rows: list[dict[str, Any]], scope: str) -> float:
    selected = sorted((row for row in rows if row["scope"] == scope), key=lambda row: row["coverage"])
    x = np.asarray([row["coverage"] for row in selected], dtype=float)
    y = np.asarray([row["NRMSE_range"] for row in selected], dtype=float)
    return float(np.trapz(y, x) / (x[-1] - x[0]))


def generate() -> None:
    outputs = [
        OUTPUT / "routing_scope_summary.csv",
        OUTPUT / "ROUTING_ERROR_PROPAGATION_ANALYSIS.md",
        OUTPUT / "fedridge_bootstrap_summary.csv",
        OUTPUT / "FEDRIDGE_GROUPED_BOOTSTRAP.md",
        OUTPUT / "qc_risk_coverage_final.csv",
        OUTPUT / "qc_error_capture_summary.csv",
        OUTPUT / "QC_CLAIM_VALIDATION.md",
    ]
    existing = [path for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite analysis outputs: {existing}")
    records = _load_records()
    routing = routing_analysis(records)
    bootstrap = bootstrap_analysis(records)
    risk_rows, capture_rows = qc_analysis(records)
    _write_csv(outputs[0], routing)
    _write_csv(outputs[2], bootstrap)
    _write_csv(outputs[4], risk_rows)
    _write_csv(outputs[5], capture_rows)
    all_route = next(row for row in routing if row["scope"] == "ALL" and row["gas"] == "ALL")
    outputs[1].write_text(
        "# Routing error propagation analysis\n\n"
        f"Canonical ALL S_ALL/S_CC/oracle RMSE = {all_route['S_ALL_RMSE']:.4f}/"
        f"{all_route['S_CC_RMSE']:.4f}/{all_route['oracle_RMSE']:.4f} ppm. "
        f"The defined routing gap is {all_route['routing_gap_ppm']:.4f} ppm and "
        f"the defined oracle gap is {all_route['oracle_gap_ppm']:.4f} ppm. "
        "S_CC is a selected route-correct population, whereas oracle uses all windows; "
        "therefore the gap is descriptive propagation evidence rather than a controlled causal effect.\n",
        encoding="utf-8",
    )
    all_boot = next(row for row in bootstrap if row["scope"] == "ALL" and row["gas"] == "ALL")
    outputs[3].write_text(
        "# FedRidge raw-file-grouped bootstrap\n\n"
        f"Using 5,000 paired raw-file-group bootstrap replicates (seed 20260809), "
        f"overall DeltaRMSE = RMSE84-RMSE83 = {all_boot['delta_rmse_ppm']:.4f} ppm, "
        f"95% percentile CI [{all_boot['delta_ci025_ppm']:.4f}, {all_boot['delta_ci975_ppm']:.4f}]. "
        "Windows were never treated as independent bootstrap units. C4 degradation and every "
        "per-target/per-gas interval remain in the CSV.\n",
        encoding="utf-8",
    )
    capture90 = next(row for row in capture_rows if row["scope"] == "ALL" and row.get("workpoint") == "HC90")
    capture95 = next(row for row in capture_rows if row["scope"] == "ALL" and row.get("workpoint") == "HC95")
    outputs[6].write_text(
        "# QC claim validation\n\n"
        "Thresholds and scores are unchanged. Existing calibration-locked HC90/HC95 and "
        "1,000-repeat same-budget random results are supplemented with fixed-score capture curves. "
        f"HC90 captures {capture90['misroute_capture_rate']:.2%} of misroutes and "
        f"{capture90['error_gt40_capture_rate']:.2%} of >40 ppm errors; HC95 captures "
        f"{capture95['misroute_capture_rate']:.2%} and {capture95['error_gt40_capture_rate']:.2%}, respectively. "
        f"The normalized trapezoidal NRMSE risk-coverage AURC over achieved coverage 0.10-1.00 is {_aurc(risk_rows, 'ALL'):.6f}. "
        "AURC is descriptive and was not used to select QC.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    generate()
