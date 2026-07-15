"""Sample a training process tree from an external observer process."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import psutil

from gaps_flower.observability import JsonlObserver, load_observer


DEFAULT_INTERVAL_SECONDS = 1.0
DEFAULT_SYSFS_TEMP_PATH = Path("/sys/class/thermal/thermal_zone0/temp")
_PROCESS_READ_ERRORS = (
    psutil.NoSuchProcess,
    psutil.AccessDenied,
    psutil.ZombieProcess,
)


class TargetProcessNotFound(RuntimeError):
    """Raised when the target root PID no longer exists."""


@dataclass(frozen=True)
class ProcessTreeSample:
    """A resource event payload plus state needed for the next CPU delta."""

    payload: dict[str, Any]
    cpu_times_seconds_by_pid: dict[int, float]
    monotonic_ns: int


@dataclass(frozen=True)
class SamplerResult:
    status: str
    shutdown_reason: str
    sample_count: int


def _record_process_error(errors: list[str], pid: int, metric: str, exc: Exception) -> None:
    errors.append(f"pid={pid} metric={metric}: {type(exc).__name__}: {exc}")


def collect_process_tree_sample(
    *,
    root_pid: int,
    sampler_pid: int,
    previous: ProcessTreeSample | None,
    now_ns: int,
) -> ProcessTreeSample:
    """Collect one de-duplicated process-tree sample.

    CPU percentages use deltas only for PIDs readable in both consecutive
    samples.  A first sample, a non-positive wall delta, or a newly observed
    PID therefore contributes zero rather than its lifetime CPU total.
    """

    errors: list[str] = []
    try:
        root = psutil.Process(root_pid)
    except psutil.NoSuchProcess as exc:
        raise TargetProcessNotFound(
            f"target PID {root_pid} does not exist"
        ) from exc

    try:
        descendants = root.children(recursive=True)
    except psutil.NoSuchProcess as exc:
        raise TargetProcessNotFound(
            f"target PID {root_pid} exited while enumerating its process tree"
        ) from exc
    except (psutil.AccessDenied, psutil.ZombieProcess) as exc:
        descendants = []
        _record_process_error(errors, root_pid, "children_recursive", exc)

    processes_by_pid: dict[int, psutil.Process] = {}
    for process in [root, *descendants]:
        try:
            pid = int(process.pid)
        except (AttributeError, TypeError, ValueError) as exc:
            errors.append(
                f"pid=unknown metric=pid: {type(exc).__name__}: {exc}"
            )
            continue
        if pid == sampler_pid:
            continue
        processes_by_pid.setdefault(pid, process)

    rss_tree_bytes = 0
    thread_count_tree = 0
    cpu_times_seconds_by_pid: dict[int, float] = {}
    readable_pids: list[int] = []

    for pid in sorted(processes_by_pid):
        process = processes_by_pid[pid]
        readable = False
        try:
            rss_tree_bytes += int(process.memory_info().rss)
            readable = True
        except _PROCESS_READ_ERRORS as exc:
            _record_process_error(errors, pid, "rss", exc)

        try:
            thread_count_tree += int(process.num_threads())
            readable = True
        except _PROCESS_READ_ERRORS as exc:
            _record_process_error(errors, pid, "num_threads", exc)

        try:
            cpu_times = process.cpu_times()
            cpu_times_seconds_by_pid[pid] = float(cpu_times.user) + float(
                cpu_times.system
            )
            readable = True
        except _PROCESS_READ_ERRORS as exc:
            _record_process_error(errors, pid, "cpu_times", exc)

        if readable:
            readable_pids.append(pid)

    logical_cpu_count_value = psutil.cpu_count(logical=True)
    if logical_cpu_count_value is None or logical_cpu_count_value <= 0:
        logical_cpu_count = 1
        errors.append(
            "host metric=logical_cpu_count: unavailable; using denominator 1"
        )
    else:
        logical_cpu_count = int(logical_cpu_count_value)

    wall_delta_ns = 0 if previous is None else now_ns - previous.monotonic_ns
    cpu_delta_seconds = 0.0
    if previous is not None and wall_delta_ns > 0:
        for pid, current_cpu_seconds in cpu_times_seconds_by_pid.items():
            prior_cpu_seconds = previous.cpu_times_seconds_by_pid.get(pid)
            if prior_cpu_seconds is not None:
                cpu_delta_seconds += max(
                    0.0, current_cpu_seconds - prior_cpu_seconds
                )
        one_core_percent = cpu_delta_seconds / (wall_delta_ns / 1e9) * 100.0
    else:
        one_core_percent = 0.0

    previous_peak = (
        0
        if previous is None
        else int(previous.payload.get("rss_tree_peak_bytes", 0))
    )
    payload = {
        "root_pid": root_pid,
        "sampler_pid_excluded": sampler_pid,
        "pids": readable_pids,
        "rss_tree_bytes": rss_tree_bytes,
        "rss_tree_peak_bytes": max(previous_peak, rss_tree_bytes),
        "process_count_tree": len(readable_pids),
        "thread_count_tree": thread_count_tree,
        "cpu_time_tree_seconds": sum(cpu_times_seconds_by_pid.values()),
        "cpu_time_tree_delta_seconds": cpu_delta_seconds,
        "cpu_percent_tree_one_core_scale": one_core_percent,
        "cpu_percent_tree_host_scale": one_core_percent / logical_cpu_count,
        "logical_cpu_count": logical_cpu_count,
        "sample_interval_start_monotonic_ns": (
            now_ns if previous is None else previous.monotonic_ns
        ),
        "sample_interval_end_monotonic_ns": now_ns,
        "sample_interval_wall_ns": max(0, wall_delta_ns),
        "sample_errors": errors,
    }
    return ProcessTreeSample(
        payload=payload,
        cpu_times_seconds_by_pid=cpu_times_seconds_by_pid,
        monotonic_ns=now_ns,
    )


def _run_vcgencmd(executable: str, argument: str) -> str:
    completed = subprocess.run(
        [executable, argument],
        check=True,
        capture_output=True,
        text=True,
        timeout=2.0,
    )
    return completed.stdout.strip()


def read_thermal_state(
    *,
    sysfs_temp_path: str | os.PathLike[str] = DEFAULT_SYSFS_TEMP_PATH,
    vcgencmd_resolver: Callable[[str], str | None] = shutil.which,
) -> dict[str, Any]:
    """Read Pi temperature and throttling without inventing unavailable values."""

    thermal_errors: list[str] = []
    result: dict[str, Any] = {
        "cpu_temperature_c": None,
        "cpu_temperature_available": False,
        "cpu_temperature_source": None,
        "vcgencmd_available": False,
        "throttled_raw": None,
        "throttled_bits": None,
        "throttled_available": False,
        "thermal_errors": thermal_errors,
    }

    thermal_path = Path(sysfs_temp_path)
    try:
        millidegrees = float(thermal_path.read_text(encoding="ascii").strip())
        result["cpu_temperature_c"] = millidegrees / 1000.0
        result["cpu_temperature_available"] = True
        result["cpu_temperature_source"] = "sysfs"
    except FileNotFoundError:
        pass
    except (OSError, UnicodeError, ValueError) as exc:
        thermal_errors.append(
            f"sysfs_temperature: {type(exc).__name__}: {exc}"
        )

    executable = vcgencmd_resolver("vcgencmd")
    if executable is None:
        return result
    result["vcgencmd_available"] = True

    if not result["cpu_temperature_available"]:
        try:
            output = _run_vcgencmd(executable, "measure_temp")
            match = re.search(r"(-?\d+(?:\.\d+)?)", output)
            if match is None:
                raise ValueError(f"unrecognized vcgencmd temperature: {output!r}")
            result["cpu_temperature_c"] = float(match.group(1))
            result["cpu_temperature_available"] = True
            result["cpu_temperature_source"] = "vcgencmd"
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            thermal_errors.append(
                f"vcgencmd_measure_temp: {type(exc).__name__}: {exc}"
            )

    try:
        output = _run_vcgencmd(executable, "get_throttled")
        raw_value = output.split("=", 1)[-1].strip()
        result["throttled_bits"] = int(raw_value, 0)
        result["throttled_raw"] = raw_value
        result["throttled_available"] = True
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        thermal_errors.append(
            f"vcgencmd_get_throttled: {type(exc).__name__}: {exc}"
        )

    return result


def _sampler_process_metrics(process: psutil.Process) -> tuple[float, float, int]:
    try:
        cpu_times = process.cpu_times()
        memory = process.memory_info()
    except _PROCESS_READ_ERRORS:
        return 0.0, 0.0, 0
    rss_values = [int(memory.rss)]
    peak_wset = getattr(memory, "peak_wset", None)
    if peak_wset is not None:
        rss_values.append(int(peak_wset))
    return float(cpu_times.user), float(cpu_times.system), max(rss_values)


def _observer_cost_payload(observer: JsonlObserver) -> dict[str, Any]:
    cost = observer._accumulated_cost
    close_summary_path = observer.events_path.with_suffix(".close.json")
    return {
        "observer_cost_values_scope": "before_resource_sampler_end_emit",
        "observer_event_encode_ns": cost.event_encode_ns,
        "observer_io_write_ns": cost.io_write_ns,
        "observer_fsync_ns": cost.fsync_ns,
        "observer_event_bytes_written": cost.event_bytes_written,
        "observer_event_count": cost.event_count,
        "observer_close_summary_path": str(close_summary_path),
        "observer_close_summary_is_authoritative": True,
    }


def run_sampler(
    *,
    root_pid: int,
    observer: JsonlObserver,
    interval_seconds: float = DEFAULT_INTERVAL_SECONDS,
    stop_file: str | os.PathLike[str] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> SamplerResult:
    """Run sampling until the target exits, a stop file appears, or an error occurs."""

    if interval_seconds <= 0:
        raise ValueError("interval_seconds must be greater than zero")

    sampler_pid = os.getpid()
    sampler_process = psutil.Process(sampler_pid)
    stop_path = None if stop_file is None else Path(stop_file)
    client_id = observer.identity.client_id
    previous: ProcessTreeSample | None = None
    sample_count = 0
    status = "succeeded"
    shutdown_reason = "stop_file"
    shutdown_error: str | None = None
    _, _, sampler_rss_peak_bytes = _sampler_process_metrics(sampler_process)

    try:
        while True:
            if stop_path is not None and stop_path.exists():
                shutdown_reason = "stop_file"
                break

            now_ns = time.perf_counter_ns()
            try:
                process_sample = collect_process_tree_sample(
                    root_pid=root_pid,
                    sampler_pid=sampler_pid,
                    previous=previous,
                    now_ns=now_ns,
                )
            except TargetProcessNotFound as exc:
                shutdown_error = str(exc)
                if sample_count == 0:
                    status = "failed"
                    shutdown_reason = "target_not_found_initial"
                else:
                    shutdown_reason = "target_exited"
                break

            payload = dict(process_sample.payload)
            payload.update(read_thermal_state())
            observer.emit(
                "resource_sample",
                round_idx=None,
                client_id=client_id,
                status="succeeded",
                payload=payload,
            )
            sample_count += 1
            previous = process_sample
            _, _, current_sampler_rss = _sampler_process_metrics(sampler_process)
            sampler_rss_peak_bytes = max(
                sampler_rss_peak_bytes, current_sampler_rss
            )
            sleep(interval_seconds)
    except KeyboardInterrupt:
        status = "aborted"
        shutdown_reason = "interrupted"
    except Exception as exc:  # ensure failures still leave a durable end record
        status = "failed"
        shutdown_reason = "sampler_error"
        shutdown_error = f"{type(exc).__name__}: {exc}"

    sampler_cpu_user, sampler_cpu_system, final_sampler_rss = (
        _sampler_process_metrics(sampler_process)
    )
    sampler_rss_peak_bytes = max(sampler_rss_peak_bytes, final_sampler_rss)
    end_payload = {
        "root_pid": root_pid,
        "sampler_pid": sampler_pid,
        "shutdown_reason": shutdown_reason,
        "shutdown_error": shutdown_error,
        "sample_count": sample_count,
        "sampler_cpu_user_seconds": sampler_cpu_user,
        "sampler_cpu_system_seconds": sampler_cpu_system,
        "sampler_rss_peak_bytes": sampler_rss_peak_bytes,
        **_observer_cost_payload(observer),
    }
    try:
        observer.emit(
            "resource_sampler_end",
            round_idx=None,
            client_id=client_id,
            status=status,
            payload=end_payload,
        )
    finally:
        observer.close()

    return SamplerResult(
        status=status,
        shutdown_reason=shutdown_reason,
        sample_count=sample_count,
    )


def _positive_float(raw_value: str) -> float:
    value = float(raw_value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sample an external confirmation training process tree."
    )
    parser.add_argument("--pid", type=int, required=True, help="target root PID")
    parser.add_argument("--observer-context", required=True)
    parser.add_argument("--observer-events", required=True)
    parser.add_argument(
        "--interval-seconds",
        type=_positive_float,
        default=DEFAULT_INTERVAL_SECONDS,
    )
    parser.add_argument("--stop-file")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    observer = load_observer(args.observer_context, args.observer_events)
    if not isinstance(observer, JsonlObserver):
        raise RuntimeError("resource sampler requires an enabled JSONL observer")
    result = run_sampler(
        root_pid=args.pid,
        observer=observer,
        interval_seconds=args.interval_seconds,
        stop_file=args.stop_file,
    )
    return 0 if result.status == "succeeded" else 2


if __name__ == "__main__":
    raise SystemExit(main())
