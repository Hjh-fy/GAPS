"""Benchmark frozen IoT-J runtime objects without changing model decisions.

The command is intentionally inference-only.  It validates the fixed C5 row
universe, measures batch-one steady-state latency, records resources, and
checks every measured output against the ordinary runtime path.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import psutil
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gaps_deploy.c5_federated_source_ridge_qc_runtime import C5FederatedSourceRidgeQCRuntime
from gaps_deploy.c5_federated_source_ridge_runtime import C5FederatedSourceRidgeRuntime
from gaps_deploy.c5_h8_runtime import C5H8Runtime
from gaps_deploy.rich_residual import target_ridge_features


def latency_statistics(values_ms: Sequence[float]) -> dict[str, float | int]:
    values = np.asarray(values_ms, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("latency values must be a non-empty finite non-negative vector")
    return {
        "n": int(len(values)),
        "mean_ms": float(np.mean(values)),
        "sample_std_ms": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "p50_ms": float(np.percentile(values, 50)),
        "p90_ms": float(np.percentile(values, 90)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "min_ms": float(np.min(values)),
        "max_ms": float(np.max(values)),
    }


def require_same_row_universe(expected: Sequence[str], observed: Sequence[str]) -> None:
    if list(expected) != list(observed) or len(set(expected)) != len(expected):
        raise ValueError("runtime benchmark row universe differs")


def _thermal_celsius() -> float | None:
    path = Path("/sys/class/thermal/thermal_zone0/temp")
    try:
        return float(path.read_text(encoding="utf-8").strip()) / 1000.0
    except (OSError, ValueError):
        return None


def _throttled() -> str | None:
    try:
        import subprocess
        result = subprocess.run(["vcgencmd", "get_throttled"], check=False, capture_output=True, text=True, timeout=5)
        return result.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _load_inputs(data_root: Path) -> tuple[np.ndarray, list[Mapping[str, Any]], np.ndarray, list[str]]:
    windows = np.load(data_root / "test_features.npy", mmap_mode="r")
    phases = np.load(data_root / "test_phase_labels.npy", mmap_mode="r")
    metadata = json.loads((data_root / "test_experiment_info.json").read_text(encoding="utf-8"))
    if windows.shape != (1360, 100, 8) or phases.shape != (1360,) or not isinstance(metadata, list) or len(metadata) != 1360:
        raise ValueError("fixed C5 test input schema differs")
    row_keys = [f"C5:test:{index}" for index in range(1360)]
    return windows, metadata, phases, row_keys


def _runtime_factory(kind: str, contract: Path, device: str) -> tuple[Any, Callable[..., list[dict[str, Any]]]]:
    if kind == "RUNTIME_V4_FULL":
        runtime = C5H8Runtime.from_runtime_contract(contract, device=device)
        return runtime, lambda w, m, p: runtime.predict_batch(w, list(m), p, workpoint="HC95")
    if kind == "RUNTIME_V5_REGRESSION_CORE":
        runtime = C5FederatedSourceRidgeRuntime.from_runtime_contract(contract, device=device)
        return runtime, runtime.infer
    if kind == "RUNTIME_V5_QC2_CANDIDATE":
        runtime = C5FederatedSourceRidgeQCRuntime.from_runtime_contract(contract, device=device)
        return runtime, runtime.infer
    raise ValueError(f"unsupported runtime kind: {kind}")


def _signature(kind: str, row: Mapping[str, Any]) -> tuple[Any, ...]:
    if kind == "RUNTIME_V4_FULL":
        return (int(row["pred_class"]), float(row["final_ppm"]), str(row["qc_decision"]))
    if kind == "RUNTIME_V5_REGRESSION_CORE":
        return (int(row["pred_class"]), float(row["prediction_ppm"]))
    return (int(row["pred_class"]), float(row["prediction_ppm"]), str(row["qc_decision"]))


def _tick(stage: str, started: int, timings: dict[str, list[float]]) -> int:
    now = time.perf_counter_ns()
    timings.setdefault(stage, []).append((now - started) / 1e6)
    return now


def _profile_once(kind: str, runtime: Any, window: np.ndarray, metadata: Mapping[str, Any], phase: int, timings: dict[str, list[float]]) -> dict[str, Any]:
    end_to_end = time.perf_counter_ns()
    tick = end_to_end
    values = np.asarray(window, dtype=np.float32).reshape(1, 100, 8)
    phase_values = np.asarray([phase])
    if not np.isfinite(values).all() or phase not in (0, 1, 2):
        raise ValueError("profile input validation failed")
    tick = _tick("input_validation", tick, timings)
    if kind == "RUNTIME_V4_FULL":
        logits, probabilities, predicted, cls_features, reg_features = runtime.extract_backbone(values)
        tick = _tick("B5_classification", tick, timings)
        meta = dict(metadata); meta["phase"] = phase
        rich = target_ridge_features(values[0], meta)
        rich.update({f"reg_feat_{j:03d}": float(value) for j, value in enumerate(reg_features[0])})
        tick = _tick("rich_feature_generation", tick, timings)
        route = int(predicted[0]); augmented = dict(rich); augmented["route_class"] = route
        h1 = runtime.h8_policy.source_ridge[route].predict(augmented); tick = _tick("H1", tick, timings)
        h2 = runtime.h8_policy.source_mlp[route].predict(augmented); tick = _tick("H2", tick, timings)
        h3 = runtime.h8_policy.shared_mlp.predict(augmented); tick = _tick("H3", tick, timings)
        augmented["srcpred_H1_source_ridge_ppm"] = h1
        augmented["srcpred_H2_source_per_gas_mlp_ppm"] = h2
        augmented["srcpred_H3_source_shared_mlp_ppm"] = h3
        prediction = runtime.h8_policy.target_ridge[route].predict(augmented); tick = _tick("target_Ridge", tick, timings)
        h23 = runtime.h23_policy.predict_components(rich, route); tick = _tick("H2.3_auxiliary", tick, timings)
        ordered = np.sort(probabilities[0]); entropy = float(-(probabilities[0] * np.log(np.maximum(probabilities[0], 1e-12))).sum())
        row = {"route_class": route, "phase": phase, "deployment_risk_classifier_entropy": entropy / math.log(4.0), "deployment_risk_margin": max(0.0, 1.0 - float(ordered[-1] - ordered[-2])), "H1_source_ridge_ppm": h1, "H2_source_per_gas_mlp_ppm": h2, "H3_source_shared_mlp_ppm": h3, "target_ridge_plus_source_preds_ppm": prediction, **h23}
        row.update({f"cls_feat_{j:03d}": float(value) for j, value in enumerate(cls_features[0])})
        row.update(runtime.risk_policy.score(row)); tick = _tick("risk_component", tick, timings)
        decision = runtime.risk_policy.decide(row["deployment_risk_full"], "HC95"); _tick("QC_decision", tick, timings)
        result = {"pred_class": route, "final_ppm": prediction, "qc_decision": decision}
    else:
        base = runtime if kind == "RUNTIME_V5_REGRESSION_CORE" else runtime.base
        with torch.inference_mode():
            logits_tensor, representation_tensor, _regression = base.model(torch.from_numpy(values).to(base.device))
            probabilities_tensor = torch.softmax(logits_tensor, dim=1)
        probabilities = probabilities_tensor.detach().cpu().numpy().astype(np.float64)
        representations = representation_tensor.detach().cpu().numpy().astype(np.float64)
        route = int(np.argmax(probabilities[0])); tick = _tick("B5_classification", tick, timings)
        meta = dict(metadata); meta["phase"] = phase
        rich = base.feature_extractor(values[0], meta); tick = _tick("rich_feature_generation", tick, timings)
        h1 = base.source_h1[route].predict(rich); tick = _tick("H1", tick, timings)
        target = dict(rich); target["srcpred_H1_source_ridge_ppm"] = h1
        prediction = base.target_ridge[route].predict(target); tick = _tick("target_Ridge", tick, timings)
        result = {"pred_class": route, "prediction_ppm": prediction}
        if kind == "RUNTIME_V5_QC2_CANDIDATE":
            score = runtime.policy.score(probabilities=probabilities[0], representation=representations[0], pred_class=route, source_h1_ppm=h1, prediction_ppm=prediction)
            tick = _tick("risk_component", tick, timings)
            decision, _auto = runtime.policy.decision(score["deployment_risk"], prediction, runtime.policy.payload["workpoints"][runtime.workpoint]); _tick("QC_decision", tick, timings)
            result["qc_decision"] = decision
    timings.setdefault("end_to_end_profiled", []).append((time.perf_counter_ns() - end_to_end) / 1e6)
    return result


def benchmark(args: argparse.Namespace) -> None:
    if args.warmup != 50 or args.runs not in (200, 500) or args.batch_size != 1:
        raise ValueError("formal protocol requires warmup=50, runs=500 (or preregistered Pi 200), batch_size=1")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    windows, metadata, phases, row_keys = _load_inputs(args.data_root)
    process = psutil.Process(os.getpid())
    temp_start, throttle_start = _thermal_celsius(), _throttled()
    rss_baseline = process.memory_info().rss
    load_tick = time.perf_counter_ns()
    runtime, infer = _runtime_factory(args.runtime, args.contract, args.device)
    bundle_load_ms = (time.perf_counter_ns() - load_tick) / 1e6
    first_tick = time.perf_counter_ns()
    first = infer(windows[0:1], metadata[0:1], phases[0:1])
    first_inference_ms = (time.perf_counter_ns() - first_tick) / 1e6
    if len(first) != 1:
        raise ValueError("runtime first inference row count differs")

    for index in range(args.warmup):
        item = index % len(windows)
        infer(windows[item : item + 1], metadata[item : item + 1], phases[item : item + 1])

    latencies: list[float] = []
    measured_keys: list[str] = []
    rss_peak = process.memory_info().rss
    temp_peak = temp_start
    cpu_before = process.cpu_times()
    wall_tick = time.perf_counter_ns()
    with torch.inference_mode():
        for index in range(args.runs):
            item = index % len(windows)
            tick = time.perf_counter_ns()
            rows = infer(windows[item : item + 1], metadata[item : item + 1], phases[item : item + 1])
            latencies.append((time.perf_counter_ns() - tick) / 1e6)
            if len(rows) != 1:
                raise ValueError("runtime measured inference row count differs")
            measured_keys.append(row_keys[item])
            rss_peak = max(rss_peak, process.memory_info().rss)
            current_temp = _thermal_celsius()
            if current_temp is not None:
                temp_peak = current_temp if temp_peak is None else max(temp_peak, current_temp)
    elapsed_s = (time.perf_counter_ns() - wall_tick) / 1e9
    cpu_after = process.cpu_times()
    cpu_seconds = (cpu_after.user - cpu_before.user) + (cpu_after.system - cpu_before.system)
    breakdown: dict[str, list[float]] = {}
    for index in range(args.runs):
        item = index % len(windows)
        ordinary = infer(windows[item : item + 1], metadata[item : item + 1], phases[item : item + 1])[0]
        profiled = _profile_once(args.runtime, runtime, windows[item], metadata[item], int(phases[item]), breakdown)
        if _signature(args.runtime, ordinary) != _signature(args.runtime, profiled):
            raise ValueError("benchmark breakdown path changed runtime prediction or decision")
    stats = latency_statistics(latencies)
    stats["throughput_windows_per_s"] = args.runs / elapsed_s
    output = {
        "schema_version": "iotj.final_runtime_benchmark.v1",
        "status": "PASS",
        "runtime": args.runtime,
        "platform_id": args.platform_id,
        "protocol": {"batch_size": 1, "warmup": args.warmup, "runs": args.runs, "threads": args.threads, "device": args.device, "clock": "time.perf_counter_ns", "disk_io_in_steady_state": False},
        "latency": stats,
        "latency_breakdown": {stage: latency_statistics(values) for stage, values in breakdown.items()},
        "cold_start": {"bundle_load_ms": bundle_load_ms, "first_inference_ms": first_inference_ms},
        "resources": {
            "rss_baseline_bytes": rss_baseline,
            "rss_peak_bytes": rss_peak,
            "rss_end_bytes": process.memory_info().rss,
            "process_cpu_core_percent": 100.0 * cpu_seconds / elapsed_s,
            "available_ram_end_bytes": psutil.virtual_memory().available,
            "thread_count": process.num_threads(),
            "temperature_start_c": temp_start,
            "temperature_peak_c": temp_peak,
            "temperature_end_c": _thermal_celsius(),
            "throttled_before": throttle_start,
            "throttled_after": _throttled(),
        },
        "environment": {"hostname": platform.node(), "os": platform.platform(), "machine": platform.machine(), "python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__, "torch_threads": torch.get_num_threads()},
        "row_universe": {"source_N": len(row_keys), "measured_N": len(measured_keys), "order": "fixed C5 test order modulo N", "unique_measured_rows": len(set(measured_keys))},
        "prediction_stability_check": "PASS",
    }
    if any(not math.isfinite(float(value)) for value in latencies):
        raise ValueError("benchmark produced NaN/Inf latency")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.rows_output:
        args.rows_output.parent.mkdir(parents=True, exist_ok=True)
        with args.rows_output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["run", "row_key", "latency_ms"])
            writer.writeheader()
            writer.writerows({"run": i, "row_key": key, "latency_ms": value} for i, (key, value) in enumerate(zip(measured_keys, latencies)))
    print(json.dumps(output, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", required=True, choices=["RUNTIME_V4_FULL", "RUNTIME_V5_REGRESSION_CORE", "RUNTIME_V5_QC2_CANDIDATE"])
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--platform-id", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--runs", type=int, default=500)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--rows-output", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    benchmark(parse_args())
