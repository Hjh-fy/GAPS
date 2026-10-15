"""Validate and consolidate formal A6/B5 C5 regression and QC evidence."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence


MODES = tuple(f"R{index}" for index in range(8))
SCOPES = ("S_ALL", "S_CC", "S_CW", "gas_0", "gas_1", "gas_2", "gas_3")
METRICS = ("RMSE", "NRMSE", "MAE", "P90AE")
QC_WORKPOINTS = ("FULL", "HC95", "HC90")
QC_ORACLE_EXTENSION_FILES = (
    "h8_no_rescue/target_predictions_plus_source_preds.csv",
    "h8_no_rescue/target_predictions_plus_source_preds_oracle_route.csv",
    "high_coverage_qc/manifest.json",
    "high_coverage_qc/operational_summary.json",
    "high_coverage_qc/risk_policy.json",
    "high_coverage_qc/risk_selection.json",
    "high_coverage_qc/test_full_records.csv",
    "high_coverage_qc/test_hc95_records.csv",
    "high_coverage_qc/test_hc90_records.csv",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    payload = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in payload for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(payload)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_qc_oracle_sources(root: Path) -> dict[str, dict[str, Any]]:
    sources: dict[str, dict[str, Any]] = {}
    for relative in QC_ORACLE_EXTENSION_FILES:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        sources[relative] = {
            "path": str(path),
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
        }
    return sources


def validate_ladder_summary(rows: Sequence[dict[str, Any]], classifier_id: str) -> None:
    expected = {(mode, scope) for mode in MODES for scope in SCOPES}
    found: set[tuple[str, str]] = set()
    for row in rows:
        key = (str(row.get("mode")), str(row.get("scope")))
        if key in found:
            raise ValueError(f"{classifier_id}: duplicate ladder row {key}")
        found.add(key)
        for metric in METRICS:
            raw = row.get(metric)
            if raw in (None, ""):
                raise ValueError(f"{classifier_id}: empty {metric} for {key}")
            if not math.isfinite(float(raw)):
                raise ValueError(f"{classifier_id}: non-finite {metric} for {key}")
        truth_flag = int(float(row.get("uses_test_truth_at_runtime", 0)))
        if truth_flag != int(key[0] == "R7"):
            raise ValueError(f"{classifier_id}: invalid oracle truth flag for {key}")
        if key[1] == "S_ALL" and int(float(row.get("N", 0))) != 1360:
            raise ValueError(f"{classifier_id}: S_ALL must contain 1360 rows for {key}")
    missing = expected - found
    extra = found - expected
    if missing or extra:
        raise ValueError(
            f"{classifier_id}: incomplete R0-R7 contract; missing={sorted(missing)}, extra={sorted(extra)}"
        )


def flatten_operational_qc(
    classifier_id: str,
    operational: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    missing = [workpoint for workpoint in QC_WORKPOINTS if workpoint not in operational]
    if missing:
        raise ValueError(f"{classifier_id}: missing QC workpoints: {missing}")
    extra = sorted(set(operational) - set(QC_WORKPOINTS))
    if extra:
        raise ValueError(f"{classifier_id}: unexpected QC workpoints: {extra}")
    output: list[dict[str, Any]] = []
    for workpoint in QC_WORKPOINTS:
        item = operational[workpoint]
        total = int(item["N"])
        if total != 1360:
            raise ValueError(
                f"{classifier_id}/{workpoint}: expected N=1360, found {total}"
            )
        accept_n = int(item["accept_N"])
        review_n = int(item["review_N"])
        reject_n = int(item["reject_N"])
        nonreject_n = int(item["nonreject_N"])
        if min(accept_n, review_n, reject_n, nonreject_n) < 0:
            raise ValueError(f"{classifier_id}/{workpoint}: negative decision count")
        if accept_n + review_n + reject_n != total:
            raise ValueError(
                f"{classifier_id}/{workpoint}: decision counts do not sum to N"
            )
        if nonreject_n != accept_n + review_n:
            raise ValueError(
                f"{classifier_id}/{workpoint}: nonreject_N does not equal accept_N + review_N"
            )
        expected_metric_n = {
            "full_metrics": total,
            "accept_metrics": accept_n,
            "nonreject_metrics": nonreject_n,
            "review_metrics": review_n,
            "reject_metrics": reject_n,
            "oracle_accept_metrics": accept_n,
            "oracle_nonreject_metrics": nonreject_n,
        }
        for metric_key, expected_n in expected_metric_n.items():
            metric_n = int(item[metric_key]["N"])
            if metric_n != expected_n:
                raise ValueError(
                    f"{classifier_id}/{workpoint}: {metric_key}.N={metric_n}, expected {expected_n}"
                )
        expected_yield = accept_n / total
        if not math.isclose(
            float(item["automatic_yield"]), expected_yield, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(f"{classifier_id}/{workpoint}: automatic_yield mismatch")
        expected_nonreject = nonreject_n / total
        if not math.isclose(
            float(item["nonreject_coverage"]),
            expected_nonreject,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"{classifier_id}/{workpoint}: nonreject_coverage mismatch")
        accepted = item["accept_metrics"]
        nonreject = item["nonreject_metrics"]
        oracle_accept = item["oracle_accept_metrics"]
        oracle_nonreject = item["oracle_nonreject_metrics"]
        random_control = item["random_control"]
        output.append(
            {
                "classifier_id": classifier_id,
                "workpoint": workpoint,
                "N": item["N"],
                "accept_N": item["accept_N"],
                "nonreject_N": item["nonreject_N"],
                "review_N": item["review_N"],
                "reject_N": item["reject_N"],
                "automatic_yield": item["automatic_yield"],
                "nonreject_coverage": item["nonreject_coverage"],
                "accept_RMSE": accepted["RMSE"],
                "accept_NRMSE": accepted["NRMSE"],
                "accept_MAE": accepted["MAE"],
                "accept_P90AE": accepted["P90AE"],
                "nonreject_RMSE": nonreject["RMSE"],
                "nonreject_NRMSE": nonreject["NRMSE"],
                "oracle_accept_RMSE": oracle_accept["RMSE"],
                "oracle_accept_NRMSE": oracle_accept["NRMSE"],
                "oracle_nonreject_RMSE": oracle_nonreject["RMSE"],
                "oracle_nonreject_NRMSE": oracle_nonreject["NRMSE"],
                "route_wrong_recall": item["route_wrong_recall"],
                "high_error_recall": item["high_error_recall"],
                "class_correct_false_flag_rate": item["class_correct_false_flag_rate"],
                "random_accept_RMSE_mean": random_control["accept_RMSE"]["mean"],
                "random_route_wrong_recall_mean": random_control["route_wrong_recall"]["mean"],
                "random_high_error_recall_mean": random_control["high_error_recall"]["mean"],
            }
        )
    return output


def _report(ladder: Sequence[dict[str, Any]], qc: Sequence[dict[str, Any]]) -> str:
    lines = [
        "# Formal C5 Regression and QC Summary",
        "",
        "All model fitting and policy fitting represented here ran on Alibaba Cloud ECS.",
        "R7 is an offline per-row oracle and is not deployment-visible.",
        "QC oracle columns are an offline forced-true-class routing diagnostic under frozen QC masks; they retain the actual-route accept/review/reject decisions and are not deployable performance.",
        "",
        "## Coverage-1 Regression",
        "",
        "| Classifier | Mode | Scope | N | RMSE | NRMSE | MAE | P90AE |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in ladder:
        if row["scope"] not in {"S_ALL", "S_CC"}:
            continue
        lines.append(
            "| {classifier_id} | {mode} | {scope} | {N} | {RMSE:.4f} | {NRMSE:.4f} | {MAE:.4f} | {P90AE:.4f} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Operational QC",
            "",
            "| Classifier | Workpoint | Accept/Review/Reject | Yield | Nonreject | Actual Accepted RMSE | Actual Accepted NRMSE | Actual Nonreject RMSE | Actual Nonreject NRMSE | Oracle Accepted RMSE | Oracle Accepted NRMSE | Oracle Nonreject RMSE | Oracle Nonreject NRMSE | Route-error recall | High-error recall | Random RMSE |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in qc:
        lines.append(
            "| {classifier_id} | {workpoint} | {accept_N}/{review_N}/{reject_N} | {automatic_yield:.4f} | {nonreject_coverage:.4f} | {accept_RMSE:.4f} | {accept_NRMSE:.4f} | {nonreject_RMSE:.4f} | {nonreject_NRMSE:.4f} | {oracle_accept_RMSE:.4f} | {oracle_accept_NRMSE:.4f} | {oracle_nonreject_RMSE:.4f} | {oracle_nonreject_NRMSE:.4f} | {route_wrong_recall:.4f} | {high_error_recall:.4f} | {random_accept_RMSE_mean:.4f} |".format(
                **row
            )
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--classifiers", default="A6,B5")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--code-revision", default="unrecorded")
    args = parser.parse_args()
    classifier_ids = [item.strip() for item in args.classifiers.split(",") if item.strip()]
    ladder_rows: list[dict[str, Any]] = []
    qc_rows: list[dict[str, Any]] = []
    sources: dict[str, dict[str, str]] = {}
    for classifier_id in classifier_ids:
        root = args.run_root / classifier_id
        ladder_path = root / "r0_r7" / "r0_r7_summary.csv"
        qc_path = root / "high_coverage_qc" / "operational_summary.json"
        suite_path = root / "suite_manifest.json"
        for path in (ladder_path, qc_path, suite_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        loaded = _read_csv(ladder_path)
        validate_ladder_summary(loaded, classifier_id)
        for row in loaded:
            item: dict[str, Any] = {"classifier_id": classifier_id, **row}
            for metric in METRICS:
                item[metric] = float(item[metric])
            item["N"] = int(float(item["N"]))
            item["coverage"] = float(item["coverage"])
            ladder_rows.append(item)
        operational = json.loads(qc_path.read_text(encoding="utf-8"))
        qc_rows.extend(flatten_operational_qc(classifier_id, operational))
        sources[classifier_id] = {
            "ladder": str(ladder_path),
            "ladder_sha256": _sha256(ladder_path),
            "qc": str(qc_path),
            "qc_sha256": _sha256(qc_path),
            "suite_manifest": str(suite_path),
            "suite_manifest_sha256": _sha256(suite_path),
            "qc_oracle_extension": collect_qc_oracle_sources(root),
        }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "r0_r7_comparison.csv", ladder_rows)
    _write_csv(args.output_dir / "qc_operational_comparison.csv", qc_rows)
    (args.output_dir / "formal_regression_report.md").write_text(
        _report(ladder_rows, qc_rows), encoding="utf-8"
    )
    manifest = {
        "schema_version": 2,
        "protocol": {"source_clients": [1, 2], "target_clients": [5]},
        "classifiers": classifier_ids,
        "training_location": "Alibaba Cloud ECS",
        "code_revision": args.code_revision,
        "ladder_contract": {"modes": list(MODES), "scopes": list(SCOPES)},
        "qc_oracle_contract": {
            "scope": "offline forced-true-class routing diagnostic under frozen QC masks",
            "qc_masks": "actual-route accept/review/reject decisions",
            "uses_test_truth_at_runtime": True,
            "workpoints": list(QC_WORKPOINTS),
            "expected_test_rows": 1360,
        },
        "sources": sources,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"ladder_rows": len(ladder_rows), "qc_rows": len(qc_rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
