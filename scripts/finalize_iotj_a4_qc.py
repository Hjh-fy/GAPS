"""Finalize label-free QC coverage operating points for the A4+84D/H1 pipeline."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.finalize_iotj_a4_end_to_end import regression_metrics, write_json


COVERAGE_TARGETS = tuple(round(0.70 + 0.025 * index, 3) for index in range(13))
PREDICTION_KEY = "pred_84d_h1_ppm"
RISK_COMPONENTS = (
    "classification_uncertainty_risk",
    "regression_disagreement_risk",
    "source_prior_disagreement_risk",
)
RISK_FORMULA = "equal_mean_of_calibration_p95_normalized_components"


def _risk_scales(calibration: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    scales = {}
    for key in RISK_COMPONENTS:
        values = np.asarray([float(row[key]) for row in calibration], dtype=np.float64)
        if len(values) == 0 or not np.isfinite(values).all():
            raise RuntimeError(f"FAIL_CLOSED calibration component invalid: {key}")
        scale = float(np.quantile(values, 0.95))
        if scale <= 0:
            raise RuntimeError(f"FAIL_CLOSED calibration component scale is zero: {key}")
        scales[key] = scale
    return scales


def _combined_risk(row: Mapping[str, Any], scales: Mapping[str, float]) -> float:
    normalized = [
        min(max(float(row[key]) / float(scales[key]), 0.0), 1.0)
        for key in RISK_COMPONENTS
    ]
    risk = float(np.mean(normalized))
    if not math.isfinite(risk):
        raise RuntimeError("FAIL_CLOSED combined QC risk is non-finite")
    return risk


def fit_qc_thresholds(
    calibration: Sequence[Mapping[str, Any]],
    coverages: Sequence[float] = COVERAGE_TARGETS,
) -> list[dict[str, Any]]:
    """Fit robust scales and thresholds from label-free calibration fields."""
    scales = _risk_scales(calibration)
    risks = np.asarray([_combined_risk(row, scales) for row in calibration])
    if len(risks) == 0 or not np.isfinite(risks).all():
        raise RuntimeError("FAIL_CLOSED calibration risk is empty/non-finite")
    ordered = np.sort(risks)
    thresholds: list[dict[str, Any]] = []
    for coverage in coverages:
        if not 0 < coverage <= 1:
            raise ValueError("coverage must be in (0, 1]")
        retained_n = int(math.ceil(float(coverage) * len(ordered)))
        threshold = float("inf") if coverage == 1.0 else float(ordered[retained_n - 1])
        thresholds.append(
            {
                "target_coverage": float(coverage),
                "threshold": threshold,
                "calibration_N": len(ordered),
                "calibration_retained_N": retained_n,
                "selection_split": "C5_calibration_x_only_risk",
                "target_test_used_for_selection": False,
                "risk_formula": RISK_FORMULA,
                **{f"p95_scale_{key}": value for key, value in scales.items()},
            }
        )
    return thresholds


def _capture_rate(event: np.ndarray, accepted: np.ndarray) -> float:
    total = int(event.sum())
    return float((event & ~accepted).sum() / total) if total else float("nan")


def evaluate_qc_curve(
    records: Sequence[Mapping[str, Any]],
    thresholds: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if not thresholds:
        raise RuntimeError("FAIL_CLOSED QC threshold policy is empty")
    scales = {
        key: float(thresholds[0][f"p95_scale_{key}"]) for key in RISK_COMPONENTS
    }
    risks = np.asarray([_combined_risk(row, scales) for row in records])
    true = np.asarray([float(row["true_ppm"]) for row in records])
    pred = np.asarray([float(row[PREDICTION_KEY]) for row in records])
    misroute = np.asarray(
        [int(row["true_class"]) != int(row["pred_class"]) for row in records]
    )
    absolute_error = np.abs(pred - true)
    large_error = absolute_error >= 40.0
    top_error = absolute_error >= float(np.quantile(absolute_error, 0.90, method="higher"))
    if not np.isfinite(risks).all():
        raise RuntimeError("FAIL_CLOSED test risk is non-finite")
    curve: list[dict[str, Any]] = []
    for policy in thresholds:
        accepted = risks <= float(policy["threshold"])
        metrics = regression_metrics(records, PREDICTION_KEY, accepted.tolist())
        curve.append(
            {
                **dict(policy),
                "accepted_N": int(accepted.sum()),
                "test_N": len(records),
                "test_coverage": float(accepted.mean()),
                **metrics,
                "misroute_events": int(misroute.sum()),
                "misroute_capture_rate": _capture_rate(misroute, accepted),
                "error_ge_40ppm_events": int(large_error.sum()),
                "error_ge_40ppm_capture_rate": _capture_rate(large_error, accepted),
                "top10pct_error_events": int(top_error.sum()),
                "top10pct_error_capture_rate": _capture_rate(top_error, accepted),
            }
        )
    return curve


def annotate_operating_point(
    records: Sequence[Mapping[str, Any]],
    thresholds: Sequence[Mapping[str, Any]],
    target_coverage: float,
) -> list[dict[str, Any]]:
    policy = next(
        row
        for row in thresholds
        if math.isclose(float(row["target_coverage"]), target_coverage, abs_tol=1e-9)
    )
    threshold = float(policy["threshold"])
    scales = {key: float(policy[f"p95_scale_{key}"]) for key in RISK_COMPONENTS}
    return [
        {
            **dict(row),
            "operating_point": f"HC{int(round(target_coverage * 100))}",
            "target_coverage": target_coverage,
            "qc_threshold": threshold,
            "qc_risk_score_final": _combined_risk(row, scales),
            "accepted": int(_combined_risk(row, scales) <= threshold),
        }
        for row in records
    ]


def random_reference(
    records: Sequence[Mapping[str, Any]],
    curve: Sequence[Mapping[str, Any]],
    repeats: int = 1000,
    seed: int = 20260804,
) -> list[dict[str, Any]]:
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    rng = np.random.default_rng(seed)
    n = len(records)
    output: list[dict[str, Any]] = []
    for point in curve:
        retained = int(point["accepted_N"])
        values = []
        for _ in range(repeats):
            indexes = rng.choice(n, size=retained, replace=False)
            mask = np.zeros(n, dtype=bool)
            mask[indexes] = True
            values.append(regression_metrics(records, PREDICTION_KEY, mask.tolist())["NRMSE"])
        array = np.asarray(values, dtype=np.float64)
        output.append(
            {
                "target_coverage": float(point["target_coverage"]),
                "accepted_N": retained,
                "test_coverage": float(point["test_coverage"]),
                "repeats": repeats,
                "seed": seed,
                "random_NRMSE_mean": float(array.mean()),
                "random_NRMSE_sample_std": float(array.std(ddof=1)) if repeats > 1 else 0.0,
                "random_NRMSE_p025": float(np.quantile(array, 0.025)),
                "random_NRMSE_p975": float(np.quantile(array, 0.975)),
            }
        )
    return output


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"FAIL_CLOSED refusing empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fields} for row in rows)


def run_qc(regression_root: Path, output: Path, repeats: int) -> None:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    calibration = _read_csv(regression_root / "final_calibration_records.csv")
    test = _read_csv(regression_root / "final_test_records.csv")
    thresholds = fit_qc_thresholds(calibration)
    _write_csv(output / "qc_threshold_lock.csv", thresholds)
    curve = evaluate_qc_curve(test, thresholds)
    random = random_reference(test, curve, repeats=repeats, seed=20260804)
    _write_csv(output / "qc_coverage_curve.csv", curve)
    _write_csv(output / "qc_random_reference.csv", random)
    operating = [
        row
        for row in curve
        if any(
            math.isclose(float(row["target_coverage"]), value, abs_tol=1e-9)
            for value in (0.90, 0.95)
        )
    ]
    _write_csv(output / "qc_operating_points.csv", operating)
    for coverage in (0.90, 0.95):
        _write_csv(
            output / f"test_hc{int(coverage * 100)}_records.csv",
            annotate_operating_point(test, thresholds, coverage),
        )
    write_json(
        output / "protocol_manifest.json",
        {
            "schema_version": "iotj.final_a4_qc.v1",
            "status": "complete",
            "prediction": PREDICTION_KEY,
            "risk_formula": RISK_FORMULA,
            "risk_normalization": "component-wise C5 calibration p95, clipped to [0,1], then equal mean",
            "risk_inputs_are_label_free": True,
            "threshold_selection": "C5 calibration risk empirical quantile",
            "coverage_targets": list(COVERAGE_TARGETS),
            "target_test_used_for_threshold_selection": False,
            "random_reference_repeats": repeats,
            "random_reference_seed": 20260804,
            "hc90_hc95_meaning": "calibration-targeted retained coverage, not accuracy",
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--regression-root",
        default="results/iotj_final_end_to_end_a4_20260804/regression",
    )
    parser.add_argument(
        "--output", default="results/iotj_final_end_to_end_a4_20260804/qc"
    )
    parser.add_argument("--random-repeats", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_qc(Path(args.regression_root), Path(args.output), args.random_repeats)


if __name__ == "__main__":
    main()
