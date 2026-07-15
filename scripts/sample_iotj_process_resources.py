"""Sample a training process tree from an external observer process."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
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
ProcessIdentity = tuple[int, float]


class TargetProcessNotFound(RuntimeError):
    """Raised when the target root PID no longer exists."""


@dataclass(frozen=True)
class ProcessTreeSample:
    """A resource event payload plus state needed for the next CPU delta."""

    payload: dict[str, Any]
    cpu_times_seconds_by_process_identity: dict[ProcessIdentity, float]
    monotonic_ns: int

    @property
    def cpu_times_seconds_by_pid(self) -> dict[int, float]:
        """Return the latest per-PID totals for compatibility with early callers."""

        return {
            pid: cpu_seconds
            for (pid, _), cpu_seconds in self.cpu_times_seconds_by_process_identity.items()
        }


@dataclass(frozen=True)
class SamplerResult:
    status: str
    shutdown_reason: str
    sample_count: int


@dataclass(frozen=True)
class SamplerRssPeak:
    peak_rss_bytes: int | None
    available: bool
    method: str
    error: str | None


def _record_process_error(errors: list[str], pid: int, metric: str, exc: Exception) -> None:
    errors.append(f"pid={pid} metric={metric}: {type(exc).__name__}: {exc}")


def _read_process_identity(process: psutil.Process) -> ProcessIdentity:
    return int(process.pid), float(process.create_time())


def get_process_identity(pid: int) -> ProcessIdentity:
    """Read a stable process identity or report that the PID does not exist."""

    try:
        return _read_process_identity(psutil.Process(pid))
    except psutil.NoSuchProcess as exc:
        raise TargetProcessNotFound(f"target PID {pid} does not exist") from exc


def collect_process_tree_sample(
    *,
    root_pid: int,
    sampler_pid: int,
    previous: ProcessTreeSample | None,
    now_ns: int,
    expected_root_identity: ProcessIdentity | None = None,
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
        root_identity = _read_process_identity(root)
    except psutil.NoSuchProcess as exc:
        raise TargetProcessNotFound(
            f"target PID {root_pid} exited before identity validation"
        ) from exc
    if (
        expected_root_identity is not None
        and root_identity != expected_root_identity
    ):
        raise TargetProcessNotFound(
            f"target PID {root_pid} identity changed from "
            f"{expected_root_identity!r} to {root_identity!r}"
        )

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
    cpu_times_seconds_by_process_identity: dict[ProcessIdentity, float] = {}
    identities_by_pid: dict[int, ProcessIdentity | None] = {}
    readable_pids: list[int] = []

    for pid in sorted(processes_by_pid):
        process = processes_by_pid[pid]
        readable = False
        identity: ProcessIdentity | None
        try:
            identity = (
                root_identity
                if pid == root_pid
                else _read_process_identity(process)
            )
        except _PROCESS_READ_ERRORS as exc:
            _record_process_error(errors, pid, "create_time", exc)
            identity = None
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

        if identity is None:
            errors.append(
                f"pid={pid} metric=cpu_times: skipped because stable identity "
                "is unavailable"
            )
        else:
            try:
                cpu_times = process.cpu_times()
                cpu_times_seconds_by_process_identity[identity] = (
                    float(cpu_times.user) + float(cpu_times.system)
                )
                readable = True
            except _PROCESS_READ_ERRORS as exc:
                _record_process_error(errors, pid, "cpu_times", exc)

        if readable:
            readable_pids.append(pid)
            identities_by_pid[pid] = identity

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
        for identity, current_cpu_seconds in (
            cpu_times_seconds_by_process_identity.items()
        ):
            prior_cpu_seconds = (
                previous.cpu_times_seconds_by_process_identity.get(identity)
            )
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
        "process_identities": [
            {
                "pid": pid,
                "create_time": (
                    None
                    if identities_by_pid[pid] is None
                    else identities_by_pid[pid][1]
                ),
                "identity_available": identities_by_pid[pid] is not None,
            }
            for pid in readable_pids
        ],
        "rss_tree_bytes": rss_tree_bytes,
        "rss_tree_peak_bytes": max(previous_peak, rss_tree_bytes),
        "process_count_tree": len(readable_pids),
        "thread_count_tree": thread_count_tree,
        "cpu_time_tree_seconds": sum(
            cpu_times_seconds_by_process_identity.values()
        ),
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
        cpu_times_seconds_by_process_identity=(
            cpu_times_seconds_by_process_identity
        ),
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


def _sampler_cpu_times(process: psutil.Process) -> tuple[float, float]:
    try:
        cpu_times = process.cpu_times()
    except _PROCESS_READ_ERRORS:
        return 0.0, 0.0
    return float(cpu_times.user), float(cpu_times.system)


def read_sampler_rss_peak(
    process: psutil.Process,
    *,
    platform: str | None = None,
    proc_status_path: str | os.PathLike[str] | None = None,
) -> SamplerRssPeak:
    """Read an OS high-water RSS value, failing closed when unavailable."""

    current_platform = sys.platform if platform is None else platform
    if current_platform.startswith("win"):
        method = "psutil_peak_wset"
        try:
            raw_peak = getattr(process.memory_info(), "peak_wset")
            peak_rss_bytes = int(raw_peak)
            if peak_rss_bytes < 0:
                raise ValueError("peak_wset must not be negative")
        except (AttributeError, TypeError, ValueError, *_PROCESS_READ_ERRORS) as exc:
            return SamplerRssPeak(
                peak_rss_bytes=None,
                available=False,
                method=method,
                error=f"unavailable peak_wset: {type(exc).__name__}: {exc}",
            )
        return SamplerRssPeak(
            peak_rss_bytes=peak_rss_bytes,
            available=True,
            method=method,
            error=None,
        )

    if current_platform.startswith("linux"):
        method = "proc_status_vm_hwm"
        status_path = (
            Path(f"/proc/{process.pid}/status")
            if proc_status_path is None
            else Path(proc_status_path)
        )
        try:
            status_text = status_path.read_text(encoding="ascii")
        except (OSError, UnicodeError) as exc:
            return SamplerRssPeak(
                peak_rss_bytes=None,
                available=False,
                method=method,
                error=f"unavailable VmHWM: {type(exc).__name__}: {exc}",
            )
        vm_hwm_lines = [
            line for line in status_text.splitlines() if line.startswith("VmHWM:")
        ]
        if not vm_hwm_lines:
            return SamplerRssPeak(
                peak_rss_bytes=None,
                available=False,
                method=method,
                error="missing VmHWM in proc status",
            )
        match = re.fullmatch(r"VmHWM:\s*(\d+)\s+kB", vm_hwm_lines[0])
        if match is None:
            return SamplerRssPeak(
                peak_rss_bytes=None,
                available=False,
                method=method,
                error=f"malformed VmHWM line: {vm_hwm_lines[0]!r}",
            )
        return SamplerRssPeak(
            peak_rss_bytes=int(match.group(1)) * 1024,
            available=True,
            method=method,
            error=None,
        )

    return SamplerRssPeak(
        peak_rss_bytes=None,
        available=False,
        method="unavailable",
        error=f"no reliable RSS high-water metric on platform {current_platform!r}",
    )


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
    expected_root_identity: ProcessIdentity | None = None
    sample_count = 0
    status = "succeeded"
    shutdown_reason = "stop_file"
    shutdown_error: str | None = None
    sampling_enabled = True

    try:
        if stop_path is None or not stop_path.exists():
            try:
                expected_root_identity = get_process_identity(root_pid)
            except TargetProcessNotFound as exc:
                sampling_enabled = False
                status = "failed"
                shutdown_reason = "target_not_found_initial"
                shutdown_error = str(exc)
        while sampling_enabled:
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
                    expected_root_identity=expected_root_identity,
                )
            except TargetProcessNotFound as exc:
                shutdown_error = str(exc)
                if expected_root_identity is None:
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
            sleep(interval_seconds)
    except KeyboardInterrupt:
        status = "aborted"
        shutdown_reason = "interrupted"
    except Exception as exc:  # ensure failures still leave a durable end record
        status = "failed"
        shutdown_reason = "sampler_error"
        shutdown_error = f"{type(exc).__name__}: {exc}"

    sampler_cpu_user, sampler_cpu_system = _sampler_cpu_times(sampler_process)
    sampler_rss_peak = read_sampler_rss_peak(sampler_process)
    end_payload = {
        "root_pid": root_pid,
        "sampler_pid": sampler_pid,
        "shutdown_reason": shutdown_reason,
        "shutdown_error": shutdown_error,
        "sample_count": sample_count,
        "sampler_cpu_user_seconds": sampler_cpu_user,
        "sampler_cpu_system_seconds": sampler_cpu_system,
        "sampler_rss_peak_bytes": sampler_rss_peak.peak_rss_bytes,
        "sampler_rss_peak_available": sampler_rss_peak.available,
        "sampler_rss_peak_method": sampler_rss_peak.method,
        "sampler_rss_peak_error": sampler_rss_peak.error,
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
