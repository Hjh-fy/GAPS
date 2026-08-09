"""Compare canonical-v1 with the preregistered strict non-overlap robustness run."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gaps_deploy.c5_h8_runtime import SerializedRidge
from scripts import run_gaps_cross_target_r84_full as common
from scripts import run_iotj_canonical_v1_r84 as r84


TARGETS = ("C3", "C4", "C5")
STRICT_DATA = ROOT / "dataset/iotj_canonical_v1_strict_nonoverlap"
STRICT_RUN = ROOT / "results/iotj_canonical_v1_scientific_validation_20260809/strict_nonoverlap/run"
CANONICAL_ROUTING = ROOT / "docs/experiments/iotj_canonical_v1_final/routing_scope_summary.csv"
CANONICAL_CLASSIFICATION = ROOT / "docs/experiments/iotj_canonical_v1_final/03_classification_final.csv"
DEFAULT_OUTPUT = ROOT / "results/iotj_canonical_v1_scientific_validation_20260809/strict_nonoverlap"


def collapse_flags(canonical_f1: float, strict_f1: float, canonical_rmse: float, strict_rmse: float) -> dict[str, bool]:
    """Pre-result sensitivity flag, not a model-selection threshold."""
    return {
        "classification": canonical_f1 - strict_f1 >= 0.20,
        "regression": strict_rmse >= 2.0 * canonical_rmse,
    }


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows({key: row.get(key, "") for key in fields} for row in rows)


def strict_oracle_metrics(target: str) -> dict[str, Any]:
    models_payload = json.loads((STRICT_RUN / f"regression/{target}/regression_models.json").read_text(encoding="utf-8"))
    models = {int(key): SerializedRidge.from_json(value) for key, value in models_payload.items()}
    h1 = common.load_h1()
    original_data = r84.DATA_ROOT
    try:
        r84.DATA_ROOT = STRICT_DATA
        base = r84.enriched_oracle_rows(target, "test")
    finally:
        r84.DATA_ROOT = original_data
    oracle = []
    for row in base:
        true_class = int(row["true_class"])
        item = {**row, "pred_class": true_class}
        item["H1_federated_source_ridge_ppm"] = h1[true_class].predict(row["feature_dict"])
        oracle.append(item)
    records = common.apply_r84_models(oracle, models)
    return common.metrics(records)


def build(output: Path) -> dict[str, Any]:
    output = output.resolve()
    strict_regression = STRICT_RUN / "regression"
    if not (strict_regression / "protocol_manifest.json").is_file():
        raise RuntimeError("FAIL_CLOSED strict R84 endpoint missing")
    canonical_cls = {row["scope"]: row for row in read_csv(CANONICAL_CLASSIFICATION)}
    canonical_route = {row["scope"]: row for row in read_csv(CANONICAL_ROUTING) if row["gas"] == "ALL" and row["scope"] in TARGETS}
    rows: list[dict[str, Any]] = []
    decisions = []
    for target in TARGETS:
        target_manifest = json.loads((strict_regression / target / "target_manifest.json").read_text(encoding="utf-8"))
        strict_cls = target_manifest["test_classification"]
        strict_scopes = {row["evaluation_scope"]: row for row in read_csv(strict_regression / target / "regression_summary.csv")}
        oracle = strict_oracle_metrics(target)
        canonical = canonical_route[target]
        canonical_scopes = {
            "S_ALL": {"N": canonical["S_ALL_N"], "RMSE": canonical["S_ALL_RMSE"], "NRMSE": canonical["S_ALL_NRMSE"], "MAE": canonical["S_ALL_MAE"]},
            "S_CC": {"N": canonical["S_CC_N"], "RMSE": canonical["S_CC_RMSE"], "NRMSE": canonical["S_CC_NRMSE"], "MAE": canonical["S_CC_MAE"]},
            "Oracle": {"N": canonical["oracle_N"], "RMSE": canonical["oracle_RMSE"], "NRMSE": canonical["oracle_NRMSE"], "MAE": canonical["oracle_MAE"]},
        }
        for protocol, classification, scopes in (
            ("canonical_window_level", canonical_cls[target], canonical_scopes),
            ("strict_grouped_nonoverlap", strict_cls, {**strict_scopes, "Oracle": oracle}),
        ):
            for scope, metric in scopes.items():
                rows.append({
                    "protocol": protocol, "target": target,
                    "test_N": int(classification["N"]), "accuracy": float(classification["accuracy"]),
                    "macro_f1": float(classification["macro_f1"]), "regression_scope": scope,
                    "regression_N": int(metric["N"]), "RMSE": float(metric["RMSE"]),
                    "NRMSE": float(metric["NRMSE"]), "MAE": float(metric["MAE"]),
                    "seed": 42,
                })
        flags = collapse_flags(
            float(canonical_cls[target]["macro_f1"]), float(strict_cls["macro_f1"]),
            float(canonical_scopes["S_ALL"]["RMSE"]), float(strict_scopes["S_ALL"]["RMSE"]),
        )
        decisions.append({
            "target": target,
            "macro_f1_delta_strict_minus_canonical": float(strict_cls["macro_f1"]) - float(canonical_cls[target]["macro_f1"]),
            "s_all_rmse_delta_strict_minus_canonical": float(strict_scopes["S_ALL"]["RMSE"]) - float(canonical_scopes["S_ALL"]["RMSE"]),
            "s_all_rmse_ratio": float(strict_scopes["S_ALL"]["RMSE"]) / float(canonical_scopes["S_ALL"]["RMSE"]),
            "classification_collapse_flag": flags["classification"],
            "regression_collapse_flag": flags["regression"],
        })
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "strict_non_overlap_summary.csv", rows)
    write_csv(output / "strict_non_overlap_deltas.csv", decisions)
    any_collapse = any(row["classification_collapse_flag"] or row["regression_collapse_flag"] for row in decisions)
    lines = [
        "# Strict non-overlap robustness analysis", "",
        "This supplementary sensitivity changes target membership only; canonical-v1 remains the primary protocol. The strict split has zero exact-window, raw-file, and raw-time overlap and retains the frozen calibration N.", "",
        "The preregistered descriptive collapse flags are an absolute Macro-F1 loss of at least 0.20 or an S_ALL RMSE ratio of at least 2.0. They are reporting flags, not tuning or acceptance criteria.", "",
        "| Target | ΔMacro-F1 | ΔS_ALL RMSE (ppm) | RMSE ratio | Classification flag | Regression flag |",
        "|---|---:|---:|---:|---|---|",
    ]
    for row in decisions:
        lines.append(f"| {row['target']} | {row['macro_f1_delta_strict_minus_canonical']:.6f} | {row['s_all_rmse_delta_strict_minus_canonical']:.3f} | {row['s_all_rmse_ratio']:.3f} | {row['classification_collapse_flag']} | {row['regression_collapse_flag']} |")
    lines += ["", f"Overall collapse flag: **{any_collapse}**.", "", "All drops and improvements are retained. No retraining, hyperparameter change, outlier removal, or replacement of the canonical main results is authorized by this analysis."]
    (output / "STRICT_NON_OVERLAP_ANALYSIS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"status": "PASS_WITH_LIMITATION" if any_collapse else "PASS", "collapse_flag": any_collapse, "targets": decisions}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(build(args.output), indent=2))


if __name__ == "__main__":
    main()
