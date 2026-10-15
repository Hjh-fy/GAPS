"""Validate the final deployment bundle before Raspberry Pi transfer."""

from __future__ import annotations

import argparse
import csv
import gc
import inspect
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gaps_deploy.final_runtime import CO_GATE_FIELDS, OUTPUT_FIELDS, FinalDeployRuntime


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def fnum(value: Any) -> float:
    return float(value)


def rmse(rows: Iterable[dict[str, Any]], pred: str, truth: str = "true_ppm") -> float | None:
    selected = list(rows)
    if not selected:
        return None
    errors = np.asarray([fnum(row[pred]) - fnum(row[truth]) for row in selected], dtype=np.float64)
    return float(np.sqrt(np.mean(errors * errors)))


def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    accepted = [row for row in rows if row["qc_decision"] == "accept"]
    nonreject = [row for row in rows if row["qc_decision"] in {"accept", "review"}]
    triggers = [row for row in rows if abs(fnum(row["co_corrected_ppm"]) - fnum(row["final_ppm"])) > 1e-9]
    return {
        "n": len(rows),
        "classification_accuracy": float(np.mean([int(row["gas_class"]) == int(row["true_class"]) for row in rows])),
        "full_rmse": rmse(rows, "final_ppm"),
        "accepted_only_rmse": rmse(accepted, "final_ppm"),
        "coverage_review_rmse": rmse(nonreject, "final_ppm"),
        "accepted_coverage": len(accepted) / max(1, len(rows)),
        "coverage_review": len(nonreject) / max(1, len(rows)),
        "co_correction_trigger_count": len(triggers),
        "accepted_correction_trigger_count": sum(row["qc_decision"] == "accept" for row in triggers),
        "non_co_correction_trigger_count": sum(int(row["gas_class"]) != 1 for row in triggers),
    }


def reference_metrics(rows: list[dict[str, str]]) -> dict[str, Any]:
    normalized = []
    for row in rows:
        normalized.append({
            "gas_class": int(float(row["pred_class"])),
            "true_class": int(float(row["true_class"])),
            "true_ppm": float(row["true_ppm"]),
            "final_ppm": float(row["final_ppm"]),
            "qc_decision": row["qc_decision"],
            "co_corrected_ppm": float(row["final_ppm"]),
        })
    return metrics(normalized)


def rss_mb() -> float | None:
    try:
        import psutil
        return float(psutil.Process().memory_info().rss / (1024 * 1024))
    except Exception:
        return None


def aggregate_stage_timings(items: list[dict[str, float]]) -> dict[str, float]:
    keys = sorted({key for item in items for key in item if key != "batch_size"})
    return {key: float(sum(float(item.get(key, 0.0)) for item in items)) for key in keys}


def performance_check(runtime: FinalDeployRuntime, features: np.ndarray, phases: np.ndarray) -> dict[str, Any]:
    runtime.predict_batch(features[:16], phase=phases[:16])
    single_timings = []
    for idx in range(20):
        runtime.predict_single(features[idx], phase=int(phases[idx]))
        single_timings.append(dict(runtime.last_timing_ms))

    batch_count = min(1000, len(features))
    runtime.predict_batch(features[:batch_count], phase=phases[:batch_count])
    batch_timing = dict(runtime.last_timing_ms)
    single_total = [item["runtime_total_ms"] for item in single_timings]
    stage_medians = {}
    for key in single_timings[0]:
        if key == "batch_size":
            continue
        stage_medians[key] = float(statistics.median(float(item[key]) for item in single_timings))
    return {
        "model_load_ms": float(runtime.model_load_ms),
        "single_window_repeats": len(single_timings),
        "single_window_total_ms_median": float(statistics.median(single_total)),
        "single_window_total_ms_p90": float(np.percentile(single_total, 90)),
        "single_window_stage_ms_median": stage_medians,
        "batch_windows": batch_count,
        "batch_timing_ms": batch_timing,
        "batch_windows_per_second": float(batch_count / max(batch_timing["runtime_total_ms"] / 1000.0, 1e-12)),
        "rss_mb_after_benchmark": rss_mb(),
    }


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Final deployment bundle validation",
        "",
        f"Overall passed: `{report['passed']}`",
        "",
        "## Replay",
        "",
        "| Client | N | Accuracy | Full RMSE | Accepted RMSE | Coverage+Review RMSE | Accept coverage | A+R coverage | CO triggers |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for client, item in report["clients"].items():
        m = item["runtime_metrics"]
        lines.append(
            f"| {client} | {m['n']} | {m['classification_accuracy']:.6f} | {m['full_rmse']:.3f} | "
            f"{m['accepted_only_rmse']:.3f} | {m['coverage_review_rmse']:.3f} | "
            f"{m['accepted_coverage']:.2%} | {m['coverage_review']:.2%} | {m['co_correction_trigger_count']} |"
        )
    m = report["all_runtime_metrics"]
    lines.append(
        f"| ALL | {m['n']} | {m['classification_accuracy']:.6f} | {m['full_rmse']:.3f} | "
        f"{m['accepted_only_rmse']:.3f} | {m['coverage_review_rmse']:.3f} | "
        f"{m['accepted_coverage']:.2%} | {m['coverage_review']:.2%} | {m['co_correction_trigger_count']} |"
    )
    perf = report["performance"]
    lines.extend([
        "",
        "## Performance",
        "",
        f"- Model load: `{perf['model_load_ms']:.2f} ms`",
        f"- Single window median: `{perf['single_window_total_ms_median']:.3f} ms`",
        f"- Single window p90: `{perf['single_window_total_ms_p90']:.3f} ms`",
        f"- {perf['batch_windows']} windows: `{perf['batch_timing_ms']['runtime_total_ms']:.2f} ms`",
        f"- Throughput: `{perf['batch_windows_per_second']:.1f} windows/s`",
        f"- RSS after benchmark: `{perf['rss_mb_after_benchmark']} MB`",
        "",
        "## Field Audit",
        "",
        f"- Output fields exact: `{report['field_audit']['output_fields_exact']}`",
        f"- Gate fields allowed: `{report['field_audit']['gate_fields_allowed']}`",
        f"- Forbidden label fields absent from gate: `{report['field_audit']['forbidden_fields_absent']}`",
        f"- Accepted correction triggers: `{m['accepted_correction_trigger_count']}`",
        f"- Non-CO correction triggers: `{m['non_co_correction_trigger_count']}`",
        "",
        "## Alignment",
        "",
        f"- Classification mismatches: `{report['alignment']['classification_mismatches']}`",
        f"- QC mismatches: `{report['alignment']['qc_mismatches']}`",
        f"- Max routed ppm delta: `{report['alignment']['max_routed_ppm_delta']:.6g}`",
        f"- Max final ppm delta: `{report['alignment']['max_final_ppm_delta']:.6g}`",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate final deployment bundle")
    parser.add_argument("--bundle", default="results/deployment_fixed_da_c12_c345_final")
    parser.add_argument("--data-root", default="dataset/client_data_c12src_c345tgt_2080_timeaware_60_170_window_fullgrid")
    parser.add_argument("--reference", default="results/timeaware_2080_c12src_c345tgt_fixed_da_r25_r3ak16_auto_v2_eval/fixed_da_r25/qc_test_records.csv")
    parser.add_argument("--output-dir", default="results/deployment_fixed_da_c12_c345_validation")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    bundle = Path(args.bundle)
    data_root = Path(args.data_root)
    output = Path(args.output_dir)
    with Path(args.reference).open("r", encoding="utf-8-sig", newline="") as handle:
        reference = list(csv.DictReader(handle))

    client_reports = {}
    all_runtime_rows: list[dict[str, Any]] = []
    all_reference_rows: list[dict[str, str]] = []
    class_mismatches = 0
    qc_mismatches = 0
    max_routed_delta = 0.0
    max_final_delta = 0.0
    performance = None

    for client_num in (3, 4, 5):
        client = f"C{client_num}"
        cdir = data_root / f"client_{client_num}"
        features = np.load(cdir / "test_features.npy", allow_pickle=True).astype(np.float32)
        phases = np.load(cdir / "test_phase_labels.npy", allow_pickle=True).astype(np.int64)
        true_class = np.load(cdir / "test_classification_labels.npy").astype(np.int64)
        true_reg = np.load(cdir / "test_regression_labels.npy").astype(np.float64)
        refs = [row for row in reference if row["client"] == client]
        if len(refs) != len(features):
            raise ValueError(f"{client}: reference rows {len(refs)} != features {len(features)}")

        rss_before = rss_mb()
        runtime = FinalDeployRuntime(bundle, client, device=args.device)
        if client == "C3":
            performance = performance_check(runtime, features, phases)

        rows: list[dict[str, Any]] = []
        timing_chunks = []
        for start in range(0, len(features), args.batch_size):
            end = min(start + args.batch_size, len(features))
            batch_rows = runtime.predict_batch(features[start:end], phase=phases[start:end])
            timing_chunks.append(dict(runtime.last_timing_ms))
            for offset, row in enumerate(batch_rows):
                idx = start + offset
                row["true_class"] = int(true_class[idx])
                row["true_ppm"] = float(true_reg[idx, true_class[idx]])
                rows.append(row)

        for row, ref in zip(rows, refs):
            class_mismatches += int(int(row["gas_class"]) != int(float(ref["pred_class"])))
            qc_mismatches += int(row["qc_decision"] != ref["qc_decision"])
            max_routed_delta = max(max_routed_delta, abs(float(row["routed_pred_ppm"]) - float(ref["auto_v2_ppm"])))
            max_final_delta = max(max_final_delta, abs(float(row["final_ppm"]) - float(ref["final_ppm"])))

        public_rows = [{field: row[field] for field in OUTPUT_FIELDS} for row in rows]
        write_csv(output / f"{client}_test_runtime_outputs.csv", public_rows)
        client_reports[client] = {
            "runtime_metrics": metrics(rows),
            "reference_metrics": reference_metrics(refs),
            "model_load_ms": runtime.model_load_ms,
            "rss_mb_before_load": rss_before,
            "rss_mb_after_replay": rss_mb(),
            "replay_stage_timing_ms": aggregate_stage_timings(timing_chunks),
        }
        all_runtime_rows.extend(rows)
        all_reference_rows.extend(refs)
        del runtime
        gc.collect()

    gate_source = inspect.getsource(FinalDeployRuntime._co_gate)
    forbidden = {"true_ppm", "true_class", "oracle", "test_label"}
    field_audit = {
        "output_fields": OUTPUT_FIELDS,
        "output_fields_exact": OUTPUT_FIELDS == json.loads((bundle / "runtime_config.json").read_text(encoding="utf-8"))["output_fields"],
        "gate_fields": sorted(CO_GATE_FIELDS),
        "gate_fields_allowed": CO_GATE_FIELDS <= {"client_id", "pred_class", "routed_pred_ppm", "qc_decision", "risk_score"},
        "forbidden_fields_absent": not any(name in gate_source for name in forbidden),
    }
    alignment = {
        "classification_mismatches": class_mismatches,
        "qc_mismatches": qc_mismatches,
        "max_routed_ppm_delta": max_routed_delta,
        "max_final_ppm_delta": max_final_delta,
        "tolerance_ppm": 2e-3,
    }
    all_metrics = metrics(all_runtime_rows)
    ref_metrics = reference_metrics(all_reference_rows)
    passed = bool(
        field_audit["output_fields_exact"]
        and field_audit["gate_fields_allowed"]
        and field_audit["forbidden_fields_absent"]
        and class_mismatches == 0
        and qc_mismatches == 0
        and max_routed_delta <= 2e-3
        and max_final_delta <= 2e-3
        and all_metrics["accepted_correction_trigger_count"] == 0
        and all_metrics["non_co_correction_trigger_count"] == 0
    )
    report = {
        "passed": passed,
        "bundle": str(bundle),
        "clients": client_reports,
        "all_runtime_metrics": all_metrics,
        "all_reference_metrics": ref_metrics,
        "alignment": alignment,
        "field_audit": field_audit,
        "performance": performance,
    }
    write_json(output / "validation_report.json", report)
    (output / "validation_report.md").write_text(markdown(report), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
