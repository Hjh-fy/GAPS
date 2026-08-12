"""Read-only consolidation of the frozen canonical-v1 regression and QC evidence.

The script deliberately performs no checkpoint inference, training, policy search, or
modification of historical result roots.  It creates one new, fail-closed summary
directory from the sealed R1 predictions and Q1 confidence records.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = REPO_ROOT / "results" / "iotj_canonical_v1_final"
DEFAULT_OUTPUT = RESULTS_ROOT / "current_authoritative_summary_20260812"
R1_ROOT = RESULTS_ROOT / "canonical_r1_83d_vs_r84_20260812"
Q0_ROOT = RESULTS_ROOT / "canonical_q0_qc_necessity_20260812"
Q1_ROOT = RESULTS_ROOT / "canonical_q1_conformal_qc_v2_20260812"
R2_ROOT = RESULTS_ROOT / "canonical_r2_transfer_safe_v2_20260812"
R0_ROOT = RESULTS_ROOT / "canonical_fedridge_r0_v2_20260812"
C0_ROOT = RESULTS_ROOT / "canonical_regression_reconstruction_qc_20260811" / "C0"


def metric_row(truth: np.ndarray, prediction: np.ndarray, gas_range: float | np.ndarray, **extra: Any) -> dict[str, Any]:
    """Compute micro-population regression metrics from a sealed row subset."""
    truth = np.asarray(truth, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    if truth.ndim != 1 or prediction.shape != truth.shape or truth.size == 0:
        raise ValueError("truth and prediction must be same-shaped non-empty 1D arrays")
    ranges = np.asarray(gas_range, dtype=float)
    if ranges.ndim == 0:
        ranges = np.full(truth.shape, float(ranges))
    if ranges.shape != truth.shape or np.any(ranges <= 0):
        raise ValueError("gas_range must be positive and scalar or truth-shaped")
    error = prediction - truth
    absolute_error = np.abs(error)
    sst = float(np.sum((truth - np.mean(truth)) ** 2))
    result = {
        **extra,
        "n": int(truth.size),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(absolute_error)),
        "nrmse_range": float(np.sqrt(np.mean((error / ranges) ** 2))),
        "r2": float(1.0 - np.sum(error**2) / sst) if sst > 0.0 else float("nan"),
        "bias": float(np.mean(error)),
        "p50_absolute_error": float(np.quantile(absolute_error, 0.50)),
        "p90_absolute_error": float(np.quantile(absolute_error, 0.90)),
        "p95_absolute_error": float(np.quantile(absolute_error, 0.95)),
    }
    return result


def calibration_threshold(risks: np.ndarray, identities: Sequence[str], nominal: float) -> dict[str, Any]:
    """Lock the upper empirical risk quantile using stable (risk, identity) order."""
    risks = np.asarray(risks, dtype=float)
    if risks.ndim != 1 or risks.size == 0 or len(identities) != risks.size:
        raise ValueError("risks and identities must be non-empty and have equal length")
    if not 0.0 < nominal <= 1.0:
        raise ValueError("nominal must lie in (0, 1]")
    ordered = sorted(enumerate(zip(risks.tolist(), identities)), key=lambda item: (item[1][0], item[1][1]))
    rank = int(math.ceil(nominal * len(ordered))) - 1
    index, (threshold, identity) = ordered[rank]
    return {
        "nominal_coverage": float(nominal),
        "threshold": float(threshold),
        "selected_index": int(index),
        "selected_identity": str(identity),
        "calibration_n": int(len(ordered)),
        "rank": int(rank + 1),
        "ties_at_threshold": int(np.sum(risks == threshold)),
        "source": "calibration_only",
        "tie_rule": "stable sort by (risk, identity); select ceil(nominal*n)-1",
    }


def summarize_selective(
    truth: np.ndarray,
    prediction: np.ndarray,
    risks: np.ndarray,
    threshold: float,
    gas_range: float,
) -> dict[str, Any]:
    """Evaluate ACCEPT iff confidence risk is no greater than the locked threshold."""
    risks = np.asarray(risks, dtype=float)
    accepted = risks <= threshold
    if risks.shape != np.asarray(truth).shape or not np.any(accepted):
        raise ValueError("risks must align with truth and retain at least one row")
    ranges = np.asarray(gas_range, dtype=float)
    if ranges.ndim == 0:
        accepted_ranges: float | np.ndarray = float(ranges)
    else:
        if ranges.shape != np.asarray(truth).shape:
            raise ValueError("gas_range must align with truth before selective filtering")
        accepted_ranges = ranges[accepted]
    accepted_metrics = metric_row(
        np.asarray(truth)[accepted], np.asarray(prediction)[accepted], accepted_ranges
    )
    return {
        "threshold": float(threshold),
        "accepted_n": int(np.sum(accepted)),
        "rejected_n": int(np.sum(~accepted)),
        "coverage": float(np.mean(accepted)),
        **{f"accepted_{key}": value for key, value in accepted_metrics.items()},
    }


def normalize_evidence_status(status: str) -> str:
    normalized = status.strip().upper()
    if normalized not in {"HISTORICAL_SUPERSEDED", "INVALID_DO_NOT_CITE", "AUTHORITATIVE"}:
        raise ValueError(f"unsupported evidence status: {status}")
    return normalized


def scoped_prediction(row: dict[str, str], scope: str) -> float:
    """Return the sealed R1 prediction prescribed by a diagnostic routing scope."""
    if scope in {"S_ALL", "S_CC"}:
        return float(row["prediction"])
    if scope in {"Oracle_ALL", "Oracle_CC"}:
        return float(row[f"route_{row['true_class']}"])
    raise ValueError(f"unsupported scope: {scope}")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    materialized = list(rows)
    if not materialized:
        raise ValueError(f"refusing to write empty CSV: {path.name}")
    keys = list(materialized[0])
    for row in materialized[1:]:
        keys.extend(key for key in row if key not in keys)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(materialized)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _r1_metrics(predictions: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_target_scope: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    by_target_scope_gas: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in predictions:
        if row["method"] != "R84_CONCAT":
            continue
        target = row["target"]
        scopes = {
            "S_ALL": True,
            "S_CC": row["true_class"] == row["predicted_class"],
            "Oracle_ALL": True,
            "Oracle_CC": row["true_class"] == row["predicted_class"],
        }
        for scope, include in scopes.items():
            if include:
                by_target_scope[(target, scope)].append(row)
                by_target_scope_gas[(target, scope, row["gas_code"])].append(row)
    metric_rows: list[dict[str, Any]] = []
    gas_rows: list[dict[str, Any]] = []
    for (target, scope), rows in sorted(by_target_scope.items()):
        metric_rows.append(metric_row(
            np.array([float(row["truth"]) for row in rows]),
            np.array([scoped_prediction(row, scope) for row in rows]),
            gas_range=np.array([float(row["gas_range"]) for row in rows]),
            target=target,
            method="R84_CONCAT",
            scope=scope,
            aggregation="micro",
            source="sealed_R1_predictions",
        ))
    for scope in ("S_ALL", "S_CC", "Oracle_ALL", "Oracle_CC"):
        scoped = [row for (target, row_scope), rows in by_target_scope.items() if row_scope == scope for row in rows]
        metric_rows.append(metric_row(
            np.array([float(row["truth"]) for row in scoped]),
            np.array([scoped_prediction(row, scope) for row in scoped]),
            gas_range=np.array([float(row["gas_range"]) for row in scoped]),
            target="POOLED_C3_C4_C5",
            method="R84_CONCAT",
            scope=scope,
            aggregation="micro",
            source="sealed_R1_predictions",
        ))
    for (target, scope, gas), rows in sorted(by_target_scope_gas.items()):
        gas_rows.append(metric_row(
            np.array([float(row["truth"]) for row in rows]),
            np.array([scoped_prediction(row, scope) for row in rows]),
            gas_range=float(rows[0]["gas_range"]),
            target=target,
            method="R84_CONCAT",
            scope=scope,
            gas=gas,
            aggregation="micro",
            source="sealed_R1_predictions",
        ))
    return metric_rows, gas_rows


def _comparison_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    methods = {"TARGET_ONLY_83D_RIDGE": "83D", "R84_CONCAT": "R84_CONCAT"}
    out: list[dict[str, Any]] = []
    for target in ("C3", "C4", "C5"):
        for scope in ("S_ALL", "S_CC", "Oracle_ALL", "Oracle_CC"):
            for method, label in methods.items():
                selected = [r for r in rows if r["target"] == target and r["method"] == method and (scope in {"S_ALL", "Oracle_ALL"} or r["true_class"] == r["predicted_class"])]
                out.append(metric_row(
                    np.array([float(r["truth"]) for r in selected]),
                    np.array([scoped_prediction(r, scope) for r in selected]),
                    gas_range=np.array([float(r["gas_range"]) for r in selected]),
                    target=target, method=label, scope=scope, aggregation="micro", source="sealed_R1_predictions",
                ))
    for scope in ("S_ALL", "S_CC", "Oracle_ALL", "Oracle_CC"):
        for method, label in methods.items():
            selected = [r for r in rows if r["method"] == method and (scope in {"S_ALL", "Oracle_ALL"} or r["true_class"] == r["predicted_class"])]
            out.append(metric_row(
                np.array([float(r["truth"]) for r in selected]),
                np.array([scoped_prediction(r, scope) for r in selected]),
                gas_range=np.array([float(r["gas_range"]) for r in selected]),
                target="POOLED_C3_C4_C5", method=label, scope=scope, aggregation="micro", source="sealed_R1_predictions",
            ))
    return out


def _historical_rows() -> list[dict[str, Any]]:
    """Index actual preserved legacy results without promoting them to authority."""
    candidates = [RESULTS_ROOT / "a0t_vs_a4_regression" / "regression_comparison.csv"]
    rows: list[dict[str, Any]] = []
    for path in candidates:
        if not path.is_file():
            raise FileNotFoundError(f"registered historical comparison source missing: {path}")
        source_hash = _sha256(path)
        for row in _read_csv(path):
            if row["method"] != "A4" or row["scope"] != "S_ALL":
                continue
            rows.append({
                "evidence_status": "HISTORICAL_SUPERSEDED",
                "source_path": str(path.relative_to(REPO_ROOT)),
                "source_sha256": source_hash,
                "role": "comparison_only_not_final_authority",
                "experiment_id": row["experiment_id"],
                "method": row["method"],
                "target": row["target"],
                "scope": row["scope"],
                "n": row["N"],
                "rmse": row["RMSE"],
                "mae": row["MAE"],
                "r2": row["R2"],
                "nrmse_range": row["NRMSE_range"],
                "bias": row["Bias"],
            })
    if len(rows) != 4:
        raise ValueError(f"expected four historical A4 S_ALL rows, found {len(rows)}")
    return rows


def _risk_aurc_rows() -> list[dict[str, Any]]:
    rows = []
    for source, status in ((Q0_ROOT / "qc_aurc.csv", "HISTORICAL_SUPERSEDED"), (Q1_ROOT / "q1_aurc.csv", "AUTHORITATIVE")):
        for row in _read_csv(source):
            rows.append({**row, "evidence_status": status, "source_path": str(source.relative_to(REPO_ROOT)), "source_sha256": _sha256(source)})
    return rows


def _historical_qc_rows() -> list[dict[str, Any]]:
    """Preserve the registered legacy multi-signal HC90/HC95 evidence verbatim."""
    source = REPO_ROOT / "results" / "iotj_final_end_to_end_a4_20260804" / "qc" / "qc_operating_points.csv"
    if not source.is_file():
        raise FileNotFoundError(f"registered historical QC source missing: {source}")
    source_hash = _sha256(source)
    rows = []
    for row in _read_csv(source):
        workpoint = f"HC{int(round(float(row['target_coverage']) * 100))}"
        rows.append({
            "policy": "historical_multisignal_equal_mean",
            "scope": "C5",
            "workpoint": workpoint,
            "coverage": row["test_coverage"],
            "rmse": row["RMSE"],
            "nrmse": row["NRMSE"],
            "status": "HISTORICAL_SUPERSEDED",
            "reason_not_final": "legacy quantitative provenance / canonical multisignal inputs unavailable",
            "source_path": str(source.relative_to(REPO_ROOT)),
            "source_sha256": source_hash,
        })
    if {row["workpoint"] for row in rows} != {"HC90", "HC95"}:
        raise ValueError("historical QC must provide HC90 and HC95")
    return rows


def _q1_thresholds_and_metrics(
    r1_predictions: list[dict[str, str]], interval_rows: list[dict[str, str]], lock: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Join Q1 classifier confidence records to R1 sealed predictions by identity.

    The Q1 table contributes *only* classifier confidence risk.  Truth and the
    final R84 prediction always come from R1, never from a conformal interval row.
    """
    thresholds: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    risk_by_identity = {(row["target"], row["physical_identity"]): float(row["confidence_risk"]) for row in interval_rows}
    for target in ("C3", "C4", "C5"):
        calibration = lock["calibration_cdf_references"][target]["confidence_risk"]
        calibration_ids = [f"{target}:calibration:{index:06d}" for index in range(len(calibration))]
        selected_rows = [row for row in r1_predictions if row["target"] == target and row["method"] == "R84_CONCAT"]
        if any((target, row["physical_identity"]) not in risk_by_identity for row in selected_rows):
            raise ValueError(f"missing Q1 classifier risk for sealed R1 {target} prediction")
        for nominal in (0.90, 0.95):
            threshold = calibration_threshold(np.asarray(calibration), calibration_ids, nominal)
            thresholds.append({"target": target, "policy": "CONFIDENCE_QC_FINAL", **threshold})
            selective = summarize_selective(
                np.array([float(row["truth"]) for row in selected_rows]),
                np.array([float(row["prediction"]) for row in selected_rows]),
                np.array([risk_by_identity[(target, row["physical_identity"])] for row in selected_rows]),
                threshold=threshold["threshold"], gas_range=np.array([float(row["gas_range"]) for row in selected_rows]),
            )
            metrics.append({"target": target, "policy": "CONFIDENCE_QC_FINAL", "nominal_coverage": nominal, **selective})
    # Pooled is micro-all-sample, while retaining each target's independently
    # calibration-locked threshold; no pooled test-time threshold is selected.
    for nominal in (0.90, 0.95):
        pooled_rows = [
            row for row in r1_predictions
            if row["method"] == "R84_CONCAT" and row["target"] in {"C3", "C4", "C5"}
        ]
        threshold_by_target = {
            target: calibration_threshold(
                np.asarray(lock["calibration_cdf_references"][target]["confidence_risk"]),
                [f"{target}:calibration:{index:06d}" for index in range(len(lock["calibration_cdf_references"][target]["confidence_risk"]))],
                nominal,
            )["threshold"]
            for target in ("C3", "C4", "C5")
        }
        risks = np.array([risk_by_identity[(row["target"], row["physical_identity"])] for row in pooled_rows])
        accepted = np.array([risk <= threshold_by_target[row["target"]] for row, risk in zip(pooled_rows, risks)])
        all_truth = np.array([float(row["truth"]) for row in pooled_rows])
        all_prediction = np.array([float(row["prediction"]) for row in pooled_rows])
        all_ranges = np.array([float(row["gas_range"]) for row in pooled_rows])
        full = metric_row(all_truth, all_prediction, all_ranges)
        accepted_metrics = metric_row(all_truth[accepted], all_prediction[accepted], all_ranges[accepted])
        metrics.append({
            "target": "POOLED_C3_C4_C5", "policy": "CONFIDENCE_QC_FINAL", "nominal_coverage": nominal,
            "threshold": "per_target_calibration_locked", "accepted_n": int(accepted.sum()), "rejected_n": int((~accepted).sum()),
            "coverage": float(accepted.mean()), **{f"accepted_{key}": value for key, value in accepted_metrics.items()},
            "full_rmse": full["rmse"], "full_nrmse_range": full["nrmse_range"],
        })
    return thresholds, metrics


def _source_artifacts() -> list[Path]:
    return [
        C0_ROOT / "C0_DECISION.json",
        R0_ROOT / "R0_V2_DECISION.json",
        R1_ROOT / "predictions.csv", R1_ROOT / "canonical_regression_bootstrap.csv", R1_ROOT / "R1_DECISION.json",
        Q0_ROOT / "Q0_DECISION.json", Q0_ROOT / "qc_aurc.csv", Q0_ROOT / "qc_risk_coverage_curves.csv",
        Q1_ROOT / "Q1_DECISION.json", Q1_ROOT / "conformal_policy_lock.json", Q1_ROOT / "conformal_intervals.csv",
        Q1_ROOT / "q1_aurc.csv", Q1_ROOT / "q1_same_count_metrics.csv", R2_ROOT / "R2_DECISION.json",
    ]


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"destination already exists; refusing to overwrite: {output}")
    sources = _source_artifacts()
    missing = [path for path in sources if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"required authoritative source missing: {missing}")
    output.mkdir(parents=True)
    try:
        predictions = _read_csv(R1_ROOT / "predictions.csv")
        final_metrics, _ = _r1_metrics(predictions)
        comparisons = _comparison_rows(predictions)
        interval_rows = _read_csv(Q1_ROOT / "conformal_intervals.csv")
        with (Q1_ROOT / "conformal_policy_lock.json").open(encoding="utf-8") as handle:
            lock = json.load(handle)
        thresholds, qc_metrics = _q1_thresholds_and_metrics(predictions, interval_rows, lock)
        _write_csv(output / "FINAL_R84_REGRESSION_METRICS.csv", final_metrics)
        _write_csv(output / "FINAL_83D_VS_R84_COMPARISON.csv", comparisons)
        _write_csv(output / "FINAL_CONFIDENCE_QC_THRESHOLDS.csv", thresholds)
        _write_csv(output / "FINAL_CONFIDENCE_QC_METRICS.csv", qc_metrics)
        _write_csv(output / "HISTORICAL_VS_AUTHORITATIVE_REGRESSION.csv", [
            *_historical_rows(),
            {"evidence_status": "AUTHORITATIVE", "source_path": str((R1_ROOT / "predictions.csv").relative_to(REPO_ROOT)), "source_sha256": _sha256(R1_ROOT / "predictions.csv"), "role": "recomputed_final_metrics_source"},
        ])
        _write_csv(output / "HISTORICAL_QC_EVIDENCE_INDEX.csv", _historical_qc_rows())
        _write_csv(output / "FINAL_QC_RISK_COVERAGE_SUMMARY.csv", _risk_aurc_rows())
        inventory = [{"study": "C0" if path.is_relative_to(C0_ROOT) else "R0" if path.is_relative_to(R0_ROOT) else "R1" if path.is_relative_to(R1_ROOT) else "Q0" if path.is_relative_to(Q0_ROOT) else "Q1" if path.is_relative_to(Q1_ROOT) else "R2", "source_path": str(path.relative_to(REPO_ROOT)), "sha256": _sha256(path), "status": "AUTHORITATIVE"} for path in sources]
        _write_csv(output / "CURRENT_AUTHORITATIVE_EVIDENCE_INVENTORY.csv", inventory)
        provenance = {"created_utc": datetime.now(timezone.utc).isoformat(), "source_artifacts": inventory, "no_checkpoint_inference": True, "final_regression": "R84_CONCAT", "final_qc": "CONFIDENCE_QC_FINAL"}
        (output / "RESULT_PROVENANCE_INDEX.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
        bootstrap_path = R1_ROOT / "canonical_regression_bootstrap.csv"
        report = "# Current Authoritative Results Summary\n\n## Authority boundary\n\nFinal regression metrics are recomputed from sealed R1 `R84_CONCAT` predictions; the comparison table includes S_ALL, S_CC, Oracle_ALL, and Oracle_CC. Existing grouped bootstrap intervals are retained from `" + str(bootstrap_path.relative_to(REPO_ROOT)) + "`, not recomputed.\n\nFinal QC is `CONFIDENCE_QC_FINAL`: risk is `1-max_g p(g|x)`, and ACCEPT iff risk is at or below a deterministic HC90/HC95 calibration-only empirical threshold. Q1 classifier records contribute confidence risks only; all test truth and R84 predictions are joined from R1 sealed predictions. Q1 conformal interval coverage is not selective-output coverage.\n\n## Historical preservation\n\n`HISTORICAL_SUPERSEDED` means comparison-only and is distinct from `INVALID_DO_NOT_CITE`. No historical asset was modified. R2-v2 retains `RETAIN_R84_DEVICE_DEPENDENT`; Q0 retains `MULTISIGNAL_QC_NOT_ESTABLISHED`; Q1-v2 retains `CONFIDENCE_QC_FINAL`.\n"
        (output / "CURRENT_AUTHORITATIVE_RESULTS_SUMMARY_20260812.md").write_text(report, encoding="utf-8")
        index = {str(path.relative_to(output)): _sha256(path) for path in sorted(output.iterdir()) if path.is_file()}
        (output / "sha256_index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    except Exception:
        shutil.rmtree(output)
        raise


if __name__ == "__main__":
    main()
