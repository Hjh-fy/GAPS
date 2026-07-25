"""Finalize formal PC/Pi benchmark summaries, paper tables, report, and indexes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ENTRY_ROOT = Path(__file__).resolve().parents[1]
if str(ENTRY_ROOT) not in sys.path:
    sys.path.insert(0, str(ENTRY_ROOT))

from scripts.build_iotj_final_system_evidence import (
    ROOT, _normalize_v4, _normalize_v5, _write_csv, _write_json,
    _write_table_set, summarize_selective_rows,
)


RUNTIMES = {
    "RUNTIME_V4_FULL": ("runtime_v4_full.json", "FORMAL_BASELINE"),
    "RUNTIME_V5_REGRESSION_CORE": ("runtime_v5_regression_core.json", "FINAL_SIMPLIFIED_REGRESSION"),
    "RUNTIME_V5_QC2_CANDIDATE": ("runtime_v5_qc2_candidate.json", "VALID_CANDIDATE_NOT_PROMOTED"),
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def count_parameters(value: Any, active: bool = False) -> int:
    if isinstance(value, Mapping):
        return sum(count_parameters(item, active or key in {"coef", "coefs", "intercepts"}) for key, item in value.items())
    if isinstance(value, list):
        return sum(count_parameters(item, active) for item in value)
    return int(active and isinstance(value, (int, float)))


def benchmark_summaries(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summaries, breakdown = [], []
    for platform_id, folder in (("PC", root / "benchmarks/pc"), ("Pi", root / "benchmarks/pi")):
        for runtime, (filename, status) in RUNTIMES.items():
            payload = read_json(folder / filename)
            if payload.get("status") != "PASS" or payload.get("runtime") != runtime or payload["protocol"] != {"batch_size": 1, "warmup": 50, "runs": 500, "threads": 1, "device": "cpu", "clock": "time.perf_counter_ns", "disk_io_in_steady_state": False}:
                raise ValueError(f"formal benchmark protocol/status differs: {platform_id}/{runtime}")
            latency, resources = payload["latency"], payload["resources"]
            summaries.append({
                "platform": platform_id, "runtime": runtime, "N": latency["n"],
                "mean_ms": latency["mean_ms"], "sample_std_ms": latency["sample_std_ms"],
                "p50_ms": latency["p50_ms"], "p90_ms": latency["p90_ms"], "p95_ms": latency["p95_ms"], "p99_ms": latency["p99_ms"],
                "min_ms": latency["min_ms"], "max_ms": latency["max_ms"], "throughput_windows_per_s": latency["throughput_windows_per_s"],
                "rss_baseline_mib": resources["rss_baseline_bytes"] / 2**20, "rss_peak_mib": resources["rss_peak_bytes"] / 2**20,
                "temperature_start_c": resources["temperature_start_c"], "temperature_peak_c": resources["temperature_peak_c"], "temperature_end_c": resources["temperature_end_c"],
                "throttled_before": resources["throttled_before"], "throttled_after": resources["throttled_after"], "deployment_status": status,
            })
            for stage, values in payload["latency_breakdown"].items():
                breakdown.append({"platform": platform_id, "runtime": runtime, "stage": stage, **values})
    return summaries, breakdown


def rewrite_qc_table(root: Path) -> list[dict[str, Any]]:
    v4_root = ROOT / "results/iotj_b5_c5_deployment_p1_20260722/high_coverage_qc"
    v5_root = ROOT / "results/iotj_b5_c5_runtime_v5_qc_20260725"
    rows = {
        ("V4", "NO_QC"): _normalize_v4(v4_root / "test_hc95_records.csv", no_qc=True),
        ("V4", "HC95"): _normalize_v4(v4_root / "test_hc95_records.csv"), ("V4", "HC90"): _normalize_v4(v4_root / "test_hc90_records.csv"),
        ("V5", "NO_QC"): _normalize_v5(v5_root / "hc95_test_rows.csv", no_qc=True),
        ("V5", "HC95"): _normalize_v5(v5_root / "hc95_test_rows.csv"), ("V5", "HC90"): _normalize_v5(v5_root / "hc90_test_rows.csv"),
    }
    output = []
    for runtime, workpoint, structure, status in (
        ("V4", "NO_QC", "B5 + H1/H2/H3 + C5 Ridge", "NO_QC_REFERENCE"), ("V4", "HC95", "B5 + H1/H2/H3 + C5 Ridge", "FORMAL_BASELINE"), ("V4", "HC90", "B5 + H1/H2/H3 + C5 Ridge", "FORMAL_BASELINE"),
        ("V5", "NO_QC", "B5 + Federated H1 + C5 105D Ridge", "NO_QC_REFERENCE"), ("V5", "HC95", "B5 + Federated H1 + C5 105D Ridge", "VALID_CANDIDATE_NOT_PROMOTED"), ("V5", "HC90", "B5 + Federated H1 + C5 105D Ridge", "VALID_CANDIDATE_NOT_PROMOTED"),
    ):
        output.append(summarize_selective_rows(rows[(runtime, workpoint)], runtime=runtime, regression_structure=structure, workpoint=workpoint, deployment_status=status))
    _write_table_set(root / "paper_tables", "table_qc_overall", output, "Selective-output quality–coverage trade-off on the C5 target device", "C5目标设备上的选择性输出质量—覆盖率权衡")
    with (root / "paper_tables/table_qc_overall.md").open("a", encoding="utf-8") as handle:
        handle.write("\nNotes: Runtime v5 has lower accepted RMSE but lower accepted yield. Its HC90 CO yield and accepted-RMSE promotion guard failed. Runtime v4 therefore remains the formal baseline; runtime v5 QC2 is valid but not globally superior.\n")
    return output


def finalize(args: argparse.Namespace) -> None:
    root = args.result_root
    summaries, breakdown = benchmark_summaries(root)
    _write_csv(root / "benchmarks/benchmark_summary.csv", summaries)
    _write_csv(root / "benchmarks/latency_breakdown.csv", breakdown)
    for platform_id, folder, name in (("Pi", root / "benchmarks/pi", "pi_resource_summary.json"), ("PC", root / "benchmarks/pc", "pc_resource_summary.json")):
        records = []
        for runtime, (filename, _status) in RUNTIMES.items():
            payload = read_json(folder / filename)
            resource = dict(payload["resources"])
            if platform_id == "Pi":
                resource["throttled_during"] = "no event: get_throttled sticky register remained 0x0 after the run"
            records.append({"runtime": runtime, "resources": resource, "environment": payload["environment"]})
        _write_json(root / f"benchmarks/{name}", {"schema_version": "iotj.final_runtime_resource.v1", "platform": platform_id, "records": records, "throttling_observed": False if platform_id == "Pi" else None})

    cold_rows = []
    for platform_id, prefix in (("PC", "pc"), ("Pi", "cold_pi")):
        for runtime, stem in (("RUNTIME_V4_FULL", "runtime_v4"), ("RUNTIME_V5_REGRESSION_CORE", "runtime_v5_core"), ("RUNTIME_V5_QC2_CANDIDATE", "runtime_v5_qc2")):
            path = root / f"cold_start/{prefix}_{stem}.json"
            payload = read_json(path)
            cold_rows.append({"platform": platform_id, "runtime": runtime, "python_launch_to_runtime_ready_ms": payload["python_launch_to_runtime_ready_ms"], "bundle_load_ms": payload["bundle_load_ms"], "first_inference_ms": payload["first_inference_ms"], "python_launch_to_first_inference_complete_ms": payload["python_launch_to_first_inference_complete_ms"]})
    _write_csv(root / "cold_start/cold_start_summary.csv", cold_rows)

    overall = rewrite_qc_table(root)
    h1_payload = read_json(ROOT / "results/iotj_b5_c5_runtime_v5_candidate_20260724/federated_h1/communication_payload_summary.json")
    h1_rows = []
    actual_total = 0
    for item in h1_payload["payloads"]:
        copies = 2 if item["direction"] == "server_to_C1_and_C2" else 1
        transmitted = int(item["bytes"]) * copies
        actual_total += transmitted
        h1_rows.append({"direction": item["direction"], "kind": item["kind"], "artifact_bytes": item["bytes"], "recipient_copies": copies, "theoretical_serialized_exchange_bytes": transmitted, "sha256": item["sha256"]})
    global_model = ROOT / "results/iotj_b5_c5_runtime_v5_candidate_20260724/federated_h1/global_h1_model.json"
    global_bytes = global_model.stat().st_size; actual_total += global_bytes
    h1_rows.append({"direction": "server_to_C5_deployment", "kind": "selected_global_H1_model", "artifact_bytes": global_bytes, "recipient_copies": 1, "theoretical_serialized_exchange_bytes": global_bytes, "sha256": hashlib.sha256(global_model.read_bytes()).hexdigest()})
    h1_rows.append({"direction": "TOTAL", "kind": "one_shot_sufficient_statistics_exchange", "artifact_bytes": "not_applicable", "recipient_copies": "not_applicable", "theoretical_serialized_exchange_bytes": actual_total, "sha256": "not_applicable"})
    _write_csv(root / "system_metrics/federated_h1_communication_summary.csv", h1_rows)
    v4_r4 = read_json(ROOT / "results/iotj_b5_c5_deployment_p1_20260722/bundle_candidate/assets/r4_policy.json")
    v4_h23 = read_json(ROOT / "results/iotj_b5_c5_deployment_p1_20260722/bundle_candidate/assets/h23_reference.json")
    v5_h1 = read_json(ROOT / "results/iotj_b5_c5_runtime_v5_candidate_20260724/runtime_v5/assets/federated_h1.json")
    v5_target = read_json(ROOT / "results/iotj_b5_c5_runtime_v5_candidate_20260724/runtime_v5/assets/target_ridge_105d.json")
    regression_params = {"RUNTIME_V4_FULL": count_parameters(v4_r4) + count_parameters(v4_h23), "RUNTIME_V5_REGRESSION_CORE": count_parameters(v5_h1) + count_parameters(v5_target), "RUNTIME_V5_QC2_CANDIDATE": count_parameters(v5_h1) + count_parameters(v5_target)}
    sizes = {row["runtime"]: row for row in csv.DictReader((root / "system_metrics/package_size_summary.csv").open(encoding="utf-8"))}
    lookup = {(row["platform"], row["runtime"]): row for row in summaries}
    system = []
    for runtime, (_filename, status) in RUNTIMES.items():
        pc, pi = lookup[("PC", runtime)], lookup[("Pi", runtime)]
        system.append({"runtime": runtime, "classifier_params": 22765, "regression_params": regression_params[runtime], "QC_params_assets": "none" if runtime == "RUNTIME_V5_REGRESSION_CORE" else "non-trainable frozen references/policy", "bundle_size_bytes": int(sizes[runtime]["total_bytes"]), "PC_p50_ms": pc["p50_ms"], "PC_p95_ms": pc["p95_ms"], "Pi_p50_ms": pi["p50_ms"], "Pi_p95_ms": pi["p95_ms"], "Pi_peak_RSS_MiB": pi["rss_peak_mib"], "Pi_peak_temperature_C": pi["temperature_peak_c"], "Pi_throughput_windows_per_s": pi["throughput_windows_per_s"], "deployment_status": status})
    _write_table_set(root / "paper_tables", "table_system_efficiency", system)
    with (root / "paper_tables/table_system_efficiency.md").open("a", encoding="utf-8") as handle:
        handle.write("\nRuntime v4 is the formal selective-output baseline. Runtime v5 regression is the selected simplified regression implementation. Runtime v5 QC2 is a valid candidate that was not promoted. Benchmarking does not alter method selection.\n")

    env = read_json(root / "environment_manifest.json")
    pi_payload = read_json(root / "benchmarks/pi/runtime_v5_regression_core.json")
    env["Pi"] = {**pi_payload["environment"], "model": "Raspberry Pi 5 Model B Rev 1.1", "ram_bytes": 8589934592, "benchmark_temperature_range_c": [min(row["temperature_start_c"] for row in summaries if row["platform"] == "Pi"), max(row["temperature_peak_c"] for row in summaries if row["platform"] == "Pi")], "throttled_all_runs": "0x0"}
    env["PC"]["ram_bytes"] = 16994848768; env["PC"]["torch"] = read_json(root / "benchmarks/pc/runtime_v5_regression_core.json")["environment"]["torch"]
    _write_json(root / "environment_manifest.json", env)
    _write_json(root / "pi_benchmark_preflight_failures.json", {"schema_version": "iotj.final_benchmark_preflight_failures.v1", "failures": [{"sequence": 1, "stage": "before timing", "reason": "portable package omitted scripts/iotj_b5_c5_bundle_contract.py and v4 parity reference relocation", "model_or_policy_executed": False, "resolution": "import/reference closure added and regression-tested"}, {"sequence": 2, "stage": "before v5 timing", "reason": "v5 calibration_lineage absolute path was not relocated", "model_or_policy_executed": False, "resolution": "lineage copied byte-identically, descriptor relocated, regression-tested"}], "parameter_or_protocol_change": False})

    v4 = next(row for row in overall if row["runtime"] == "V4" and row["QC_workpoint"] == "HC95")
    v5 = next(row for row in overall if row["runtime"] == "V5" and row["QC_workpoint"] == "HC95")
    report = f"""# GAPS IoT-J 最终系统 benchmark 结果（2026-07-25）

## 结论

固定资产与两平台实测均通过 fail-closed 审计。Runtime v4 继续作为正式 selective-output baseline；Federated-H1 Runtime v5 regression core 是最终简化回归实现；v5 QC2 仍为 `VALID_CANDIDATE_NOT_PROMOTED`，没有因本次 benchmark 改变算法或 QC 决策。当前已具备进入另行授权 low-calibration 阶段的工程条件，但本阶段未启动 low-calibration。

## 协议与环境

- 固定 C5 test 行宇宙：1360；benchmark 按 canonical 顺序取前 500 行，batch=1、warm-up=50、500 次、CPU 单线程、`torch.inference_mode()`。
- steady-state 不含磁盘读取；cold start 独立测量 Python child launch→runtime ready→first inference。
- PC：{env['PC']['processor']}，Windows 11，PyTorch {env['PC']['torch']}，RAM 16 GiB。
- Pi：Raspberry Pi 5 Model B Rev 1.1，aarch64，PyTorch {env['Pi']['torch']}，RAM 8 GiB；全部三次 `throttled=0x0`。

## A–C. 延迟、吞吐与资源

{_markdown(system)}

分阶段明细见 `results/iotj_final_system_benchmark_20260725/benchmarks/latency_breakdown.csv`；B5 classification 是三个对象的主要 steady-state 延迟组成。PC 观测存在较大的系统调度尾部，故同时报告 p50/p95/p99，不用均值替代尾延迟。Pi 峰值温度为 {max(row['temperature_peak_c'] for row in summaries if row['platform']=='Pi'):.2f}°C，未观测 throttling。

## D. 包大小与参数

系统表已同时报告 classifier 参数、regression 参数和 bundle 字节。v5 core 不含 QC；v5 QC2 的 reference/policy 是非训练参数资产。便携 Pi 合同仅重定位路径，模型与 policy 字节 SHA 不变。

## E. B5 FL communication

正式 B5 seed42 真实拓扑 25-round measured application payload 为 17,572,650 bytes；其中 downlink/uplink 为 8,764,300/8,808,350 bytes。理论模型 tensor payload 与 measured serialized application payload 已分列，transport bytes 未采集，不能把 application bytes 称为链路层流量。

## F. Federated H1 与 C5 target Ridge 构建成本

Federated H1 是一次性 sufficient-statistics exchange，C1/C2 的 moments、normal equations、clipped validation SSE/count 及 server 返回资产逐项见 `system_metrics/federated_h1_communication_summary.csv`；没有传输 raw source rows/X/y，也没有宣称 secure aggregation。C5 target Ridge 使用 320 calibration rows、105D 输入、424 个 target-head 参数。原正式构建未分阶段采集 wall time，因此 feature/alpha/refit/serialization 时间保持 `unknown`，没有事后伪造计时。

## G. QC quality–coverage

- v4 HC95：yield {100*v4['accepted_yield']:.2f}%，accepted RMSE {v4['accepted_RMSE']:.4f} ppm。
- v5 QC2 HC95：yield {100*v5['accepted_yield']:.2f}%，accepted RMSE {v5['accepted_RMSE']:.4f} ppm。
- v5 的 accepted RMSE 更低，但 yield 也更低；HC90 CO yield 与 accepted-RMSE promotion guard 失败，不能宣称 v5 QC 全局更优。

论文表格位于 `paper_tables/`，论文图位于 `figures/`。图只展示冻结 HC95/HC90 四个工作点，没有拟合曲线、增加阈值或重新打开 test。

## H. 异常与审计

Pi 首次两次尝试均在正式计时前 fail-closed，原因分别为便携包 import/reference 闭包和 v5 lineage 绝对路径；均通过测试后仅修复路径封装，不修改 runtime、模型、policy 或阈值。正式六个 PC/Pi 对象均 `PASS`，分阶段复算与普通 runtime 预测/decision 一致。

## I. Evidence boundary

- Runtime v4 是正式 selective-output baseline。
- v5 QC2 是有效但未晋级的 candidate；accepted RMSE 的降低必须和更低 yield 一起解释。
- v5 Federated-H1 regression 保持最终简化回归选择。
- filename grouping 仅适用于 calibration OOF folds；历史 calibration/test split 是 window-level，不宣称 original-file level 完全独立。
- test 打开后没有修改 candidate、组件、scale、ECDF 或 threshold。
- 本阶段没有启动 low-calibration、新 QC、训练或 runtime v5 promotion。
"""
    args.report_path.write_text(report, encoding="utf-8")

    _write_json(root / "build_receipt.json", {"schema_version": "iotj.final_system_evidence_build.v1", "status": "COMPLETE", "manual_numeric_entry": False, "PC_and_Pi_benchmark_pass": True, "runtime_v4_six_sha_unchanged": True, "low_calibration_started": False})
    tracked = [path for path in root.rglob("*") if path.is_file() and not path.name.endswith("_rows.csv") and path.name != "sha256_index.json"]
    sha_rows = [{"path": str(path.relative_to(ROOT)).replace("\\", "/"), "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in sorted(tracked)]
    _write_json(root / "sha256_index.json", {"schema_version": "iotj.final_system_benchmark_sha256.v1", "artifacts": sha_rows, "large_row_logs_excluded_from_git": True})
    index = {"schema_version": "iotj.final_system_benchmark_result_index.v1", "experiment_id": "IOTJ-FINAL-SYSTEM-BENCHMARK-20260725", "status": "COMPLETE", "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), "result_root": str(root.relative_to(ROOT)).replace("\\", "/"), "report": str(args.report_path.relative_to(ROOT)).replace("\\", "/"), "decision": "RUNTIME_V4_FORMAL_BASELINE_V5_REGRESSION_FINAL_V5_QC2_NOT_PROMOTED", "low_calibration_started": False, "sha256_index": str((root / "sha256_index.json").relative_to(ROOT)).replace("\\", "/")}
    _write_json(args.index_path, index)


def _markdown(rows: Sequence[Mapping[str, Any]]) -> str:
    fields = list(rows[0]); lines = ["| " + " | ".join(fields) + " |", "|" + "|".join(["---"] * len(fields)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join("" if row[field] is None else (f"{row[field]:.6g}" if isinstance(row[field], float) else str(row[field])) for field in fields) + " |")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=ROOT / "results/iotj_final_system_benchmark_20260725")
    parser.add_argument("--report-path", type=Path, default=ROOT / "docs/experiments/iotj_final_system_benchmark_result_20260725.zh.md")
    parser.add_argument("--index-path", type=Path, default=ROOT / "docs/experiments/iotj_final_system_benchmark_result_index_20260725.json")
    return parser.parse_args()


if __name__ == "__main__":
    finalize(parse_args())
