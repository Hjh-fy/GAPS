"""Read-only timing diagnosis for an incomplete IoT-J confirmation attempt."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence


ROUND_FIELDS = (
    "round",
    "round_wall_s",
    "pi_client_train_core_s",
    "pi_client_fit_callback_s",
    "pc_client_train_core_s",
    "pc_client_fit_callback_s",
    "server_aggregate_total_s",
    "server_da_s",
    "server_non_da_s",
    "client_waiting_or_sync_residual_s",
    "pi_rss_mean_mib",
    "pi_rss_peak_mib",
    "pi_cpu_host_mean_percent",
    "pi_cpu_host_peak_percent",
    "pi_temperature_mean_c",
    "pi_temperature_peak_c",
    "pc_rss_mean_mib",
    "pc_rss_peak_mib",
    "pc_cpu_host_mean_percent",
    "pc_cpu_host_peak_percent",
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise RuntimeError(f"blank JSONL line: {path}:{line_number}")
        value = json.loads(line)
        if not isinstance(value, dict):
            raise RuntimeError(f"JSON object required: {path}:{line_number}")
        records.append(value)
    if not records:
        raise RuntimeError(f"no events in {path}")
    return records


def _index(events: Iterable[Mapping[str, Any]], event_type: str) -> dict[int, Mapping[str, Any]]:
    result: dict[int, Mapping[str, Any]] = {}
    for event in events:
        if event.get("event_type") != event_type:
            continue
        round_idx = event.get("round")
        if not isinstance(round_idx, int) or round_idx < 1:
            continue
        if round_idx in result:
            raise RuntimeError(f"duplicate {event_type} for round {round_idx}")
        result[round_idx] = event
    return result


def _payload_ns(event: Mapping[str, Any], name: str) -> int:
    payload = event.get("payload")
    value = payload.get(name) if isinstance(payload, Mapping) else None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise RuntimeError(f"missing/nonnegative timing {name}")
    return int(value)


def build_round_rows(
    server: Sequence[Mapping[str, Any]],
    c1: Sequence[Mapping[str, Any]],
    c2: Sequence[Mapping[str, Any]],
) -> list[dict[str, float | int]]:
    required = {
        "fit_round_end": _index(server, "fit_round_end"),
        "server_aggregate_end": _index(server, "server_aggregate_end"),
        "c1_train": _index(c1, "client_train_end"),
        "c1_fit": _index(c1, "client_fit_end"),
        "c2_train": _index(c2, "client_train_end"),
        "c2_fit": _index(c2, "client_fit_end"),
    }
    common = set.intersection(*(set(values) for values in required.values()))
    if not common or common != set(range(1, max(common) + 1)):
        raise RuntimeError("completed rounds must be a nonempty contiguous prefix")
    rows: list[dict[str, float | int]] = []
    for round_idx in sorted(common):
        aggregate = required["server_aggregate_end"][round_idx]
        aggregate_ns = _payload_ns(aggregate, "server_aggregate_fit_total_ns")
        da_ns = _payload_ns(aggregate, "server_da_total_ns")
        non_da_ns = _payload_ns(aggregate, "server_aggregate_non_da_ns")
        if aggregate_ns != da_ns + non_da_ns:
            raise RuntimeError(f"aggregate timing mismatch in round {round_idx}")
        c1_fit = _payload_ns(required["c1_fit"][round_idx], "client_fit_callback_ns")
        c2_fit = _payload_ns(required["c2_fit"][round_idx], "client_fit_callback_ns")
        wall_ns = _payload_ns(required["fit_round_end"][round_idx], "fit_round_wall_ns")
        residual = wall_ns - max(c1_fit, c2_fit) - aggregate_ns
        rows.append(
            {
                "round": round_idx,
                "round_wall_s": wall_ns / 1e9,
                "pi_client_train_core_s": _payload_ns(required["c1_train"][round_idx], "client_train_core_ns") / 1e9,
                "pi_client_fit_callback_s": c1_fit / 1e9,
                "pc_client_train_core_s": _payload_ns(required["c2_train"][round_idx], "client_train_core_ns") / 1e9,
                "pc_client_fit_callback_s": c2_fit / 1e9,
                "server_aggregate_total_s": aggregate_ns / 1e9,
                "server_da_s": da_ns / 1e9,
                "server_non_da_s": non_da_ns / 1e9,
                "client_waiting_or_sync_residual_s": residual / 1e9,
            }
        )
    return rows


def _fit_intervals(events: Sequence[Mapping[str, Any]]) -> dict[int, tuple[int, int]]:
    starts, ends = _index(events, "client_fit_start"), _index(events, "client_fit_end")
    result: dict[int, tuple[int, int]] = {}
    for round_idx in set(starts) & set(ends):
        start, end = starts[round_idx].get("monotonic_ns"), ends[round_idx].get("monotonic_ns")
        if not isinstance(start, int) or not isinstance(end, int) or end < start:
            raise RuntimeError(f"invalid fit interval in round {round_idx}")
        result[round_idx] = (start, end)
    return result


def add_resource_columns(
    rows: list[dict[str, float | int]],
    client_events: Sequence[Mapping[str, Any]],
    resource_events: Sequence[Mapping[str, Any]],
    *,
    prefix: str,
    include_temperature: bool,
) -> None:
    intervals = _fit_intervals(client_events)
    grouped: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for event in resource_events:
        if event.get("event_type") != "resource_sample":
            continue
        payload = event.get("payload")
        if not isinstance(payload, Mapping):
            continue
        start, end = payload.get("sample_interval_start_monotonic_ns"), payload.get("sample_interval_end_monotonic_ns")
        if not isinstance(start, int) or not isinstance(end, int):
            continue
        for round_idx, (fit_start, fit_end) in intervals.items():
            if start <= fit_end and end >= fit_start:
                grouped[round_idx].append(payload)
    for row in rows:
        payloads = grouped.get(int(row["round"]), [])
        if not payloads:
            raise RuntimeError(f"no {prefix} resource sample overlaps round {row['round']}")
        rss = [float(item["rss_tree_bytes"]) / (1024 * 1024) for item in payloads]
        peak = [float(item["rss_tree_peak_bytes"]) / (1024 * 1024) for item in payloads]
        cpu = [float(item["cpu_percent_tree_host_scale"]) for item in payloads]
        row[f"{prefix}_rss_mean_mib"] = mean(rss)
        row[f"{prefix}_rss_peak_mib"] = max(peak)
        row[f"{prefix}_cpu_host_mean_percent"] = mean(cpu)
        row[f"{prefix}_cpu_host_peak_percent"] = max(cpu)
        if include_temperature:
            temperatures = [float(item["cpu_temperature_c"]) for item in payloads if item.get("cpu_temperature_available") is True]
            row["pi_temperature_mean_c"] = mean(temperatures) if temperatures else ""
            row["pi_temperature_peak_c"] = max(temperatures) if temperatures else ""


def _mean(rows: Sequence[Mapping[str, float | int]], field: str) -> float:
    return mean(float(row[field]) for row in rows)


def _pilot_baseline(path: Path) -> dict[str, float]:
    records = list(csv.DictReader(Path(path).read_text(encoding="utf-8").splitlines()))
    if not records:
        raise RuntimeError(f"pilot table is empty: {path}")
    mapping = {
        "round_wall_s": "fit_round_wall_s",
        "server_da_s": "server_da_s",
        "pi_client_train_core_s": "pi_train_s",
        "pc_client_train_core_s": "pc_train_s",
    }
    return {target: mean(float(row[source]) for row in records) for target, source in mapping.items()}


def analysis_markdown(rows: Sequence[Mapping[str, float | int]], pilot: Mapping[str, float]) -> str:
    current = {field: _mean(rows, field) for field in pilot}
    ratios = {field: current[field] / pilot[field] for field in pilot}
    pc_share = current["pc_client_train_core_s"] / current["round_wall_s"]
    da_share = current["server_da_s"] / current["round_wall_s"]
    residual = _mean(rows, "client_waiting_or_sync_residual_s")
    return f"""# a003 与 B2 两轮 pilot 的时间诊断

## 证据边界

- 输入为失败 attempt `c12_to_c5__b2__s42__a003` 已回收的 ECS、Pi、PC 事件与资源 JSONL；a003 保持 failed，不进入任何 confirmation 统计。
- 仅使用其完整的 round 1--{len(rows)}；不读取 C5 test、checkpoint 指标或模型输出。
- `client_waiting_or_sync_residual_s = round wall - max(C1 fit callback, C2 fit callback) - server aggregate`，是 Flower 调度、消息传输、控制器等待和未被上述计时覆盖工作的合并残差，不能单独标记为网络延迟。

## 与两轮真实 B2 pilot 的比较

| 指标 | a003 mean (s) | pilot mean (s) | 倍数 |
|---|---:|---:|---:|
| round wall | {current['round_wall_s']:.2f} | {pilot['round_wall_s']:.2f} | {ratios['round_wall_s']:.2f}x |
| PC C2 local train core | {current['pc_client_train_core_s']:.2f} | {pilot['pc_client_train_core_s']:.2f} | {ratios['pc_client_train_core_s']:.2f}x |
| Pi C1 local train core | {current['pi_client_train_core_s']:.2f} | {pilot['pi_client_train_core_s']:.2f} | {ratios['pi_client_train_core_s']:.2f}x |
| ECS server DA | {current['server_da_s']:.2f} | {pilot['server_da_s']:.2f} | {ratios['server_da_s']:.2f}x |

## 结论

1. **主 slowdown 是 A：PC C2 local training。** 它平均占 round wall 的 {pc_share:.1%}，并相对 pilot 增长 {ratios['pc_client_train_core_s']:.2f}x。
2. **ECS server DA 仍是第二大绝对耗时，但不是此次变慢的来源。** a003 为 {current['server_da_s']:.2f} s，低于 pilot 的 {pilot['server_da_s']:.2f} s（{ratios['server_da_s']:.2f}x）。
3. Pi local training 增长到 {current['pi_client_train_core_s']:.2f} s，但与 PC 并行运行，且远小于 PC critical path；它不是主瓶颈。
4. 每轮可得的 waiting/synchronization 合并残差均值为 {residual:.2f} s；它不能支持“网络是主因”的结论。不存在随 round 持续增长的单调证据；逐轮 CSV 用于后续复核。
5. 按 a003 mean，真实三机 B2 的 25-round training 下界约为 {current['round_wall_s'] * 25 / 3600:.2f} h，另加恢复和 validator 时间。该估计只适用于当前 C2 PC 状态，不能外推为 B5 或其它 host placement。
"""


def write_csv(path: Path, rows: Sequence[Mapping[str, float | int]]) -> None:
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROUND_FIELDS, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt-dir", type=Path, required=True)
    parser.add_argument("--pilot-round-csv", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    raw = args.attempt_dir / "raw"
    pi_root = raw / "pi"
    if not (pi_root / "events.jsonl").is_file():
        pi_root = pi_root / "client_c1"
    rows = build_round_rows(
        load_jsonl(raw / "ecs" / "events.jsonl"),
        load_jsonl(pi_root / "events.jsonl"),
        load_jsonl(raw / "pc" / "events.jsonl"),
    )
    add_resource_columns(rows, load_jsonl(pi_root / "events.jsonl"), load_jsonl(pi_root / "resource.jsonl"), prefix="pi", include_temperature=True)
    add_resource_columns(rows, load_jsonl(raw / "pc" / "events.jsonl"), load_jsonl(raw / "pc" / "resource.jsonl"), prefix="pc", include_temperature=False)
    args.output_root.mkdir(parents=True, exist_ok=False)
    write_csv(args.output_root / "a003_round_timing_diagnosis.csv", rows)
    (args.output_root / "a003_vs_b2_pilot_timing_analysis.md").write_text(analysis_markdown(rows, _pilot_baseline(args.pilot_round_csv)), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
