"""Build frozen IoT-J system evidence tables, figures, manifests, and audit indexes."""

from __future__ import annotations

import hashlib
import math
import argparse
import csv
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GAS_NAMES = {0: "Ethanol", 1: "CO", 2: "Ethylene", 3: "Methane"}


class FrozenAssetError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_frozen_assets(records: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for name, record in records.items():
        path = Path(str(record.get("path", "")))
        try:
            expected_bytes = int(record["bytes"])
            expected_sha = str(record["sha256"]).lower()
        except (KeyError, TypeError, ValueError) as error:
            raise FrozenAssetError(f"frozen asset descriptor is invalid: {name}") from error
        if not path.is_file():
            raise FrozenAssetError(f"frozen asset is missing: {name}")
        actual_sha = sha256_file(path)
        actual_bytes = path.stat().st_size
        if actual_bytes != expected_bytes or actual_sha != expected_sha:
            raise FrozenAssetError(f"frozen asset identity differs: {name}")
        output[name] = {"path": str(path.resolve()), "bytes": actual_bytes, "sha256": actual_sha, "status": "PASS"}
    return output


def _metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, float | None]:
    if not rows:
        return {"RMSE": None, "MAE": None, "NRMSE": None}
    true = np.asarray([float(row["true_ppm"]) for row in rows], dtype=np.float64)
    pred = np.asarray([float(row["prediction_ppm"]) for row in rows], dtype=np.float64)
    if not np.isfinite(true).all() or not np.isfinite(pred).all():
        raise ValueError("QC rows contain NaN/Inf")
    errors = pred - true
    class_ranges = {0: 112.5, 1: 225.0, 2: 112.5, 3: 225.0}
    normalized = np.asarray([errors[index] / class_ranges[int(row["true_class"])] for index, row in enumerate(rows)], dtype=np.float64)
    return {"RMSE": float(np.sqrt(np.mean(errors**2))), "MAE": float(np.mean(np.abs(errors))), "NRMSE": float(np.sqrt(np.mean(normalized**2)))}


def summarize_selective_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    runtime: str,
    regression_structure: str,
    workpoint: str,
    deployment_status: str,
) -> dict[str, Any]:
    if deployment_status not in {"FORMAL_BASELINE", "VALID_CANDIDATE_NOT_PROMOTED", "NO_QC_REFERENCE"}:
        raise ValueError("deployment status is outside the frozen vocabulary")
    if not rows or len({str(row["row_key"]) for row in rows}) != len(rows):
        raise ValueError("QC row keys are missing or duplicated")
    decisions = [str(row["qc_decision"]) for row in rows]
    if not set(decisions) <= {"accept", "review", "reject"}:
        raise ValueError("QC decision is outside accept/review/reject")
    accepted = [row for row in rows if row["qc_decision"] == "accept"]
    reviewed = [row for row in rows if row["qc_decision"] == "review"]
    rejected = [row for row in rows if row["qc_decision"] == "reject"]
    accepted_review = accepted + reviewed
    co = [row for row in rows if int(row["true_class"]) == 1]
    co_accept = [row for row in co if row["qc_decision"] == "accept"]
    co_high = [row for row in co if 200 <= float(row["true_ppm"]) <= 250]
    co_high_accept = [row for row in co_high if row["qc_decision"] == "accept"]
    full_m, accept_m, ar_m = _metrics(rows), _metrics(accepted), _metrics(accepted_review)
    return {
        "runtime": runtime,
        "regression_structure": regression_structure,
        "QC_workpoint": workpoint,
        "total_N": len(rows),
        "accept_N": len(accepted), "review_N": len(reviewed), "reject_N": len(rejected),
        "accepted_yield": len(accepted) / len(rows),
        "accepted_plus_review_yield": len(accepted_review) / len(rows),
        "full_RMSE": full_m["RMSE"], "full_MAE": full_m["MAE"],
        "accepted_RMSE": accept_m["RMSE"], "accepted_MAE": accept_m["MAE"], "accepted_NRMSE": accept_m["NRMSE"],
        "accepted_plus_review_RMSE": ar_m["RMSE"],
        "review_RMSE": _metrics(reviewed)["RMSE"], "reject_RMSE": _metrics(rejected)["RMSE"],
        "misclassified_accept_N": sum(int(row["pred_class"]) != int(row["true_class"]) for row in accepted),
        "misclassified_review_N": sum(int(row["pred_class"]) != int(row["true_class"]) for row in reviewed),
        "misclassified_reject_N": sum(int(row["pred_class"]) != int(row["true_class"]) for row in rejected),
        "CO_N": len(co), "CO_accepted_yield": len(co_accept) / len(co) if co else None, "CO_accepted_RMSE": _metrics(co_accept)["RMSE"],
        "CO_high_N": len(co_high), "CO_high_accepted_yield": len(co_high_accept) / len(co_high) if co_high else None, "CO_high_accepted_RMSE": _metrics(co_high_accept)["RMSE"],
        "deployment_status": deployment_status,
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _normalize_v4(path: Path, *, no_qc: bool = False) -> list[dict[str, Any]]:
    rows = []
    for index, source in enumerate(_read_csv(path)):
        rows.append({
            "row_key": f"C5:test:{index}",
            "true_class": int(float(source["true_class"])),
            "pred_class": int(float(source["pred_class"])),
            "true_ppm": float(source["true_ppm"]),
            "prediction_ppm": float(source["target_ridge_plus_source_preds_ppm"]),
            "qc_decision": "accept" if no_qc else source["qc_decision"],
        })
    return rows


def _normalize_v5(path: Path, *, no_qc: bool = False) -> list[dict[str, Any]]:
    rows = []
    for source in _read_csv(path):
        rows.append({
            "row_key": source["row_key"],
            "true_class": int(source["true_class"]),
            "pred_class": int(source["pred_class"]),
            "true_ppm": float(source["true_ppm"]),
            "prediction_ppm": float(source["prediction_ppm"]),
            "qc_decision": "accept" if no_qc else source["qc_decision"],
        })
    return rows


def _per_gas_rows(rows: Sequence[Mapping[str, Any]], runtime: str, workpoint: str) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for gas_id, gas in GAS_NAMES.items():
        subset = [row for row in rows if int(row["true_class"]) == gas_id]
        accepted = [row for row in subset if row["qc_decision"] == "accept"]
        reviewed = [row for row in subset if row["qc_decision"] == "review"]
        rejected = [row for row in subset if row["qc_decision"] == "reject"]
        output.append({
            "runtime": runtime, "workpoint": workpoint, "gas": gas,
            "total_N": len(subset), "accept_N": len(accepted), "accepted_yield": len(accepted) / len(subset),
            "accepted_RMSE": _metrics(accepted)["RMSE"], "accepted_MAE": _metrics(accepted)["MAE"],
            "review_N": len(reviewed), "reject_N": len(rejected), "reject_rate": len(rejected) / len(subset),
        })
    co_high = [row for row in rows if int(row["true_class"]) == 1 and 200 <= float(row["true_ppm"]) <= 250]
    accepted = [row for row in co_high if row["qc_decision"] == "accept"]
    reviewed = [row for row in co_high if row["qc_decision"] == "review"]
    rejected = [row for row in co_high if row["qc_decision"] == "reject"]
    output.append({
        "runtime": runtime, "workpoint": workpoint, "gas": "CO-high 200–250 ppm",
        "total_N": len(co_high), "accept_N": len(accepted), "accepted_yield": len(accepted) / len(co_high),
        "accepted_RMSE": _metrics(accepted)["RMSE"], "accepted_MAE": _metrics(accepted)["MAE"],
        "review_N": len(reviewed), "reject_N": len(rejected), "reject_rate": len(rejected) / len(co_high),
    })
    return output


def _markdown_table(rows: Sequence[Mapping[str, Any]]) -> str:
    fields = list(rows[0])
    lines = ["| " + " | ".join(fields) + " |", "|" + "|".join(["---"] * len(fields)) + "|"]
    for row in rows:
        values = []
        for field in fields:
            value = row[field]
            values.append("" if value is None else (f"{value:.6g}" if isinstance(value, float) else str(value)))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def _latex_table(rows: Sequence[Mapping[str, Any]]) -> str:
    fields = list(rows[0])
    escape = lambda value: str(value).replace("_", "\\_").replace("%", "\\%")
    body = [" & ".join(escape(field) for field in fields) + r" \\ \midrule"]
    for row in rows:
        body.append(" & ".join("" if row[field] is None else (f"{row[field]:.6g}" if isinstance(row[field], float) else escape(row[field])) for field in fields) + r" \\")
    return "\\begin{tabular}{" + "l" * len(fields) + "}\n\\toprule\n" + "\n".join(body) + "\n\\bottomrule\n\\end{tabular}\n"


def _write_table_set(root: Path, stem: str, rows: Sequence[Mapping[str, Any]], title_en: str | None = None, title_zh: str | None = None) -> None:
    _write_csv(root / f"{stem}.csv", rows)
    prefix = ""
    if title_en or title_zh:
        prefix = f"# {title_en}\n\n中文标题：{title_zh}\n\n"
    (root / f"{stem}.md").write_text(prefix + _markdown_table(rows), encoding="utf-8")
    (root / f"{stem}.tex").write_text(_latex_table(rows), encoding="utf-8")


def _candidate_table(path: Path) -> list[dict[str, Any]]:
    components = {
        "QC1": "confidence",
        "QC2": "confidence + prototype/support distance",
        "QC3": "confidence + prototype/support distance + regression consistency",
    }
    output = []
    for row in _read_csv(path):
        candidate = row["candidate"]
        output.append({
            "candidate": candidate,
            "risk_components": components[candidate],
            "OOF_Spearman": float(row["spearman_risk_vs_abs_error"]),
            "lowest_decile_RMSE": float(row["lowest_risk_decile_RMSE"]),
            "highest_decile_RMSE": float(row["highest_risk_decile_RMSE"]),
            "tail_enrichment_ratio": float(row["tail_enrichment_ratio"]),
            "HC95_accepted_RMSE": float(row["HC95_accepted_RMSE"]),
            "HC95_yield": float(row["HC95_accepted_yield"]),
            "HC90_accepted_RMSE": float(row["HC90_accepted_RMSE"]),
            "HC90_yield": float(row["HC90_accepted_yield"]),
            "risk_direction": "PASS" if row["risk_direction_pass"] == "True" else "FAIL",
            "tail_enrichment": "PASS" if float(row["tail_enrichment_ratio"]) > 1.0 else "FAIL",
            "selection": "SELECTED" if candidate == "QC2" else "NOT_SELECTED",
        })
    return output


def _plot_qc(overall: Sequence[Mapping[str, Any]], per_gas: Sequence[Mapping[str, Any]], output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = [row for row in overall if row["QC_workpoint"] in {"HC95", "HC90"}]
    colors = {"V4": "#0072B2", "V5": "#D55E00"}
    markers = {"HC95": "o", "HC90": "s"}
    fig, ax = plt.subplots(figsize=(6.4, 4.4), constrained_layout=True)
    for row in selected:
        ax.scatter(100 * float(row["accepted_yield"]), float(row["accepted_RMSE"]), s=75, color=colors[row["runtime"]], marker=markers[row["QC_workpoint"]], edgecolor="black", linewidth=0.6)
        ax.annotate(f"{row['runtime']} {row['QC_workpoint']}", (100 * float(row["accepted_yield"]), float(row["accepted_RMSE"])), xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Accepted yield (%)")
    ax.set_ylabel("Accepted RMSE (ppm)")
    ax.grid(alpha=0.25)
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"qc_quality_coverage_tradeoff.{suffix}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    gases = ["Overall", "CO", "CO-high 200–250 ppm"]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2), sharey=True, constrained_layout=True)
    for axis, workpoint in zip(axes, ("HC95", "HC90")):
        x = np.arange(len(gases)); width = 0.34
        for offset, runtime in ((-width / 2, "V4"), (width / 2, "V5")):
            overall_row = next(row for row in selected if row["runtime"] == runtime and row["QC_workpoint"] == workpoint)
            gas_map = {(row["runtime"], row["workpoint"], row["gas"]): row for row in per_gas}
            values = [overall_row["accepted_yield"], gas_map[(runtime, workpoint, "CO")]["accepted_yield"], gas_map[(runtime, workpoint, "CO-high 200–250 ppm")]["accepted_yield"]]
            axis.bar(x + offset, np.asarray(values) * 100, width, label=runtime, color=colors[runtime], edgecolor="black", linewidth=0.5)
        axis.set_title(workpoint); axis.set_xticks(x, gases, rotation=18, ha="right"); axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Accepted yield (%)"); axes[1].legend(frameon=False)
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"qc_per_gas_yield_comparison.{suffix}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    (output / "qc_figure_captions.en.md").write_text("**Quality–coverage trade-off.** Only the four frozen HC95/HC90 operating points are shown; no curve or post-test threshold is fitted. The second panel compares overall, CO, and CO-high accepted yield.\n", encoding="utf-8")
    (output / "qc_figure_captions.zh.md").write_text("**质量—覆盖率权衡。** 图中仅展示四个冻结的 HC95/HC90 工作点，不拟合曲线，也不进行 test 后阈值搜索。第二图比较整体、CO 与 CO-high 的 accepted yield。\n", encoding="utf-8")


def _package_sizes() -> list[dict[str, Any]]:
    v4 = ROOT / "results/iotj_b5_c5_deployment_p1_20260722/bundle_candidate"
    v5 = ROOT / "results/iotj_b5_c5_runtime_v5_candidate_20260724/runtime_v5"
    v5_qc = ROOT / "results/iotj_b5_c5_runtime_v5_qc_20260725/runtime_v5_qc_bundle"
    rows = []
    for runtime, roots, status in (("RUNTIME_V4_FULL", (v4,), "FORMAL_BASELINE"), ("RUNTIME_V5_REGRESSION_CORE", (v5,), "FINAL_SIMPLIFIED_REGRESSION"), ("RUNTIME_V5_QC2_CANDIDATE", (v5, v5_qc), "VALID_CANDIDATE_NOT_PROMOTED")):
        # The v5 QC bundle is an overlay, so its deployable footprint includes
        # the complete v5 regression core as well as the frozen QC references.
        assets = [path for root in roots for path in root.rglob("*") if path.is_file()]
        rows.append({"runtime": runtime, "file_count": len(assets), "total_bytes": sum(path.stat().st_size for path in assets), "deployment_status": status})
    return rows


def build(args: argparse.Namespace) -> None:
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=False)
    paper = output / "paper_tables"; figures = output / "figures"; metrics = output / "system_metrics"
    paper.mkdir(); figures.mkdir(); metrics.mkdir()

    decision = json.loads((ROOT / "results/iotj_b5_c5_runtime_v5_qc_20260725/decision_gate.json").read_text(encoding="utf-8"))
    frozen_v5 = json.loads((ROOT / "results/iotj_b5_c5_runtime_v5_qc_20260725/frozen_runtime_v5_assets.json").read_text(encoding="utf-8"))
    v4_paths = {
        "bundle_manifest": ROOT / "results/iotj_b5_c5_deployment_p1_20260722/bundle_candidate/manifest.json",
        "c5_test_features": ROOT / "dataset/client_data_c1234src_c5tgt_2080_timeaware_60_170_window_fullgrid/client_5/test_features.npy",
        "c5_test_metadata": ROOT / "dataset/client_data_c1234src_c5tgt_2080_timeaware_60_170_window_fullgrid/client_5/test_experiment_info.json",
        "c5_test_phase_labels": ROOT / "dataset/client_data_c1234src_c5tgt_2080_timeaware_60_170_window_fullgrid/client_5/test_phase_labels.npy",
        "hc95_reference": ROOT / "results/iotj_b5_c5_deployment_p1_20260722/high_coverage_qc/test_hc95_records.csv",
        "hc90_reference": ROOT / "results/iotj_b5_c5_deployment_p1_20260722/high_coverage_qc/test_hc90_records.csv",
    }
    v4_records = {name: {"path": str(path), "bytes": path.stat().st_size, "sha256": decision["runtime_v4_hc_frozen_sha256"][name]} for name, path in v4_paths.items()}
    verified = {"runtime_v4": verify_frozen_assets(v4_records), "runtime_v5": verify_frozen_assets(frozen_v5["assets"])}
    _write_json(output / "frozen_asset_manifest.json", {"schema_version": "iotj.final_system_frozen_assets.v1", "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), "status": "PASS", "assets": verified, "runtime_v4_six_sha_unchanged": True})
    _write_json(output / "benchmark_protocol.json", {"schema_version": "iotj.final_system_benchmark_protocol.v1", "runtimes": ["RUNTIME_V4_FULL", "RUNTIME_V5_REGRESSION_CORE", "RUNTIME_V5_QC2_CANDIDATE"], "batch_size": 1, "warmup": 50, "runs": {"PC": 500, "Pi": 500}, "pi_permitted_reduction": 200, "threads": 1, "device": "cpu", "steady_state_excludes_disk_io": True, "cold_start_separate": True, "row_universe": "fixed C5 test 1360 in canonical order", "threshold_search": False})
    _write_json(output / "environment_manifest.json", {"schema_version": "iotj.final_system_environment.v1", "PC": {"hostname": platform.node(), "os": platform.platform(), "machine": platform.machine(), "processor": platform.processor(), "python": sys.version, "numpy": np.__version__, "cpu_count": os.cpu_count()}, "Pi": "populated_by_formal_pi_benchmark", "thread_configuration": {"inference_threads": 1, "interop_threads": 1}})

    v4_hc95_path = v4_paths["hc95_reference"]; v4_hc90_path = v4_paths["hc90_reference"]
    v5_hc95_path = ROOT / "results/iotj_b5_c5_runtime_v5_qc_20260725/hc95_test_rows.csv"
    v5_hc90_path = ROOT / "results/iotj_b5_c5_runtime_v5_qc_20260725/hc90_test_rows.csv"
    row_sets = {
        ("V4", "NO_QC"): _normalize_v4(v4_hc95_path, no_qc=True),
        ("V4", "HC95"): _normalize_v4(v4_hc95_path), ("V4", "HC90"): _normalize_v4(v4_hc90_path),
        ("V5", "NO_QC"): _normalize_v5(v5_hc95_path, no_qc=True),
        ("V5", "HC95"): _normalize_v5(v5_hc95_path), ("V5", "HC90"): _normalize_v5(v5_hc90_path),
    }
    for rows in row_sets.values():
        if len(rows) != 1360 or [row["row_key"] for row in rows] != [f"C5:test:{i}" for i in range(1360)]:
            raise ValueError("QC table input row universe differs")
    overall = []
    for runtime, workpoint, structure, status in (
        ("V4", "NO_QC", "B5 + H1/H2/H3 + C5 Ridge", "NO_QC_REFERENCE"),
        ("V4", "HC95", "B5 + H1/H2/H3 + C5 Ridge", "FORMAL_BASELINE"),
        ("V4", "HC90", "B5 + H1/H2/H3 + C5 Ridge", "FORMAL_BASELINE"),
        ("V5", "NO_QC", "B5 + Federated H1 + C5 105D Ridge", "NO_QC_REFERENCE"),
        ("V5", "HC95", "B5 + Federated H1 + C5 105D Ridge", "VALID_CANDIDATE_NOT_PROMOTED"),
        ("V5", "HC90", "B5 + Federated H1 + C5 105D Ridge", "VALID_CANDIDATE_NOT_PROMOTED"),
    ):
        overall.append(summarize_selective_rows(row_sets[(runtime, workpoint)], runtime=runtime, regression_structure=structure, workpoint=workpoint, deployment_status=status))
    _write_table_set(paper, "table_qc_overall", overall, "Selective-output quality–coverage trade-off on the C5 target device", "C5目标设备上的选择性输出质量—覆盖率权衡")
    notes = "\nNotes: Runtime v5 has lower accepted RMSE but lower accepted yield. Its HC90 CO yield and accepted-RMSE promotion guard failed. Runtime v4 therefore remains the formal baseline; runtime v5 QC2 is valid but not globally superior.\n"
    with (paper / "table_qc_overall.md").open("a", encoding="utf-8") as handle: handle.write(notes)

    per_gas = []
    for runtime, workpoint in (("V4", "HC95"), ("V5", "HC95"), ("V4", "HC90"), ("V5", "HC90")):
        per_gas.extend(_per_gas_rows(row_sets[(runtime, workpoint)], runtime, workpoint))
    _write_table_set(paper, "table_qc_per_gas", per_gas)
    candidates = _candidate_table(ROOT / "results/iotj_b5_c5_runtime_v5_qc_20260725/qc_candidate_calibration_summary.csv")
    _write_table_set(paper, "table_qc_oof_candidates", candidates)
    with (paper / "table_qc_oof_candidates.md").open("a", encoding="utf-8") as handle: handle.write("\nQC2 was selected using calibration OOF evidence only. QC3 failed tail enrichment. The C5 test set was not used for candidate selection.\n")
    _plot_qc(overall, per_gas, figures)

    sizes = _package_sizes(); _write_csv(metrics / "package_size_summary.csv", sizes)
    b5 = json.loads((ROOT / "results/iotj_ecs_c2_b5_canonical_analysis_20260721/b5_canonical_system_metrics.json").read_text(encoding="utf-8"))
    communication = b5["communication"]
    model_params = 22765
    theoretical_per_direction = model_params * 4 * 2
    _write_csv(metrics / "b5_fl_communication_summary.csv", [{
        "rounds": 25, "clients_per_round": 2, "actual_clients": "C1;C2", "serialized_model_parameters": model_params,
        "theoretical_server_to_clients_bytes_per_round": theoretical_per_direction,
        "theoretical_clients_to_server_bytes_per_round": theoretical_per_direction,
        "theoretical_model_payload_25round_bytes": theoretical_per_direction * 2 * 25,
        "measured_application_downlink_25round_bytes": communication["application_downlink_25round_total_bytes"],
        "measured_application_uplink_25round_bytes": communication["application_uplink_25round_total_bytes"],
        "measured_application_total_25round_bytes": communication["application_25round_total_bytes"],
        "protocol_overhead_in_measured_application_bytes": True, "transport_bytes_collected": False,
    }])
    h1_payload = json.loads((ROOT / "results/iotj_b5_c5_runtime_v5_candidate_20260724/federated_h1/communication_payload_summary.json").read_text(encoding="utf-8"))
    h1_rows = [{"direction": item["direction"], "kind": item["kind"], "bytes": item["bytes"], "sha256": item["sha256"]} for item in h1_payload["payloads"]]
    h1_rows.append({"direction": "TOTAL_SERIALIZED_ARTIFACTS", "kind": "one_shot_sum_counting_server_broadcast_artifact_once", "bytes": sum(int(item["bytes"]) for item in h1_payload["payloads"]), "sha256": "not_applicable"})
    _write_csv(metrics / "federated_h1_communication_summary.csv", h1_rows)
    target = json.loads((ROOT / "results/iotj_b5_c5_runtime_v5_candidate_20260724/target_ridge/regression_reference_summary.json").read_text(encoding="utf-8"))
    _write_csv(metrics / "target_ridge_construction_summary.csv", [{"calibration_rows": 320, "calibration_validation_rows": 80, "input_dimension": 105, "target_ridge_parameters": 424, "calibration_validation_RMSE": target["calibration_validation_RMSE"], "feature_time_seconds": "not_instrumented_in_original_construction", "alpha_selection_seconds": "not_instrumented_in_original_construction", "full_refit_seconds": "not_instrumented_in_original_construction", "serialization_seconds": "not_instrumented_in_original_construction", "evidence_status": "construction provenance complete; timing unknown"}])
    _write_json(output / "evidence_boundary.json", {"runtime_v4": "formal selective-output baseline", "runtime_v5_qc2": "valid candidate not promoted because yield/per-gas guards failed", "runtime_v5_regression": "selected simplified Federated-H1 regression", "quality_coverage_interpretation": "lower accepted RMSE must be interpreted together with lower yield", "filename_grouping_scope": "calibration OOF folds only", "historical_calibration_test_split": "window-level; original-file independence is not claimed", "post_test_threshold_change": False, "low_calibration_started": False})
    _write_json(output / "build_receipt.json", {"schema_version": "iotj.final_system_evidence_build.v1", "status": "TABLES_AND_FIGURES_READY_BENCHMARK_PENDING", "decision": decision["decision"], "manual_numeric_entry": False, "source_tables": [str(v4_hc95_path), str(v4_hc90_path), str(v5_hc95_path), str(v5_hc90_path)]})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/iotj_final_system_benchmark_20260725")
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
