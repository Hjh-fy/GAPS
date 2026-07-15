from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import psutil
import pytest

from gaps_flower.observability import JsonlObserver, ObserverIdentity
from scripts import sample_iotj_process_resources as sampler
from scripts.sample_iotj_process_resources import (
    TargetProcessNotFound,
    build_parser,
    collect_process_tree_sample,
    read_thermal_state,
    run_sampler,
)


class FakeProcess:
    def __init__(
        self,
        pid: int,
        *,
        rss: int,
        threads: int,
        cpu_user: float,
        cpu_system: float,
        children: list["FakeProcess"] | None = None,
    ) -> None:
        self.pid = pid
        self.rss = rss
        self.threads = threads
        self.cpu_user = cpu_user
        self.cpu_system = cpu_system
        self._children = children or []
        self.recursive_calls: list[bool] = []

    def children(self, *, recursive: bool) -> list["FakeProcess"]:
        self.recursive_calls.append(recursive)
        return self._children

    def memory_info(self) -> SimpleNamespace:
        return SimpleNamespace(rss=self.rss)

    def num_threads(self) -> int:
        return self.threads

    def cpu_times(self) -> SimpleNamespace:
        return SimpleNamespace(user=self.cpu_user, system=self.cpu_system)


def install_process_tree(
    monkeypatch: pytest.MonkeyPatch,
    root: FakeProcess,
    *,
    logical_cpu_count: int = 4,
) -> None:
    processes = {process.pid: process for process in [root, *root._children]}
    monkeypatch.setattr(sampler.psutil, "Process", processes.__getitem__)
    monkeypatch.setattr(
        sampler.psutil,
        "cpu_count",
        lambda *, logical: logical_cpu_count,
    )


def make_identity() -> ObserverIdentity:
    return ObserverIdentity(
        run_id="c12_to_c5__b2__s42",
        attempt_id="c12_to_c5__b2__s42__a001",
        group_id="B2",
        training_seed=42,
        client_id="C1",
        host_id="pi-c1",
        producer="resource_sampler",
        confirmation_commit="a" * 40,
        source_archive_sha256="b" * 64,
        dataset_manifest_sha256="c" * 64,
        algorithm_config_sha256="d" * 64,
    )


def read_rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_collect_process_tree_sample_deduplicates_and_sums_readable_processes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    child = FakeProcess(
        101, rss=200, threads=5, cpu_user=2.0, cpu_system=0.25
    )
    excluded_sampler = FakeProcess(
        999, rss=10_000, threads=99, cpu_user=20.0, cpu_system=3.0
    )
    root = FakeProcess(
        100,
        rss=100,
        threads=2,
        cpu_user=1.0,
        cpu_system=0.5,
        children=[child, child, excluded_sampler],
    )
    install_process_tree(monkeypatch, root)

    sample = collect_process_tree_sample(
        root_pid=root.pid,
        sampler_pid=excluded_sampler.pid,
        previous=None,
        now_ns=1_000_000_000,
    )

    assert root.recursive_calls == [True]
    assert sample.payload["pids"] == [100, 101]
    assert sample.payload["rss_tree_bytes"] == 300
    assert sample.payload["rss_tree_peak_bytes"] == 300
    assert sample.payload["process_count_tree"] == 2
    assert sample.payload["thread_count_tree"] == 7


def test_cpu_percent_uses_process_time_delta_and_both_scales(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    child = FakeProcess(
        201, rss=200, threads=1, cpu_user=1.0, cpu_system=0.0
    )
    root = FakeProcess(
        200,
        rss=100,
        threads=1,
        cpu_user=1.0,
        cpu_system=0.0,
        children=[child],
    )
    install_process_tree(monkeypatch, root, logical_cpu_count=4)
    previous = collect_process_tree_sample(
        root_pid=root.pid,
        sampler_pid=999,
        previous=None,
        now_ns=1_000_000_000,
    )
    root.cpu_user += 0.25
    root.cpu_system += 0.25
    child.cpu_user += 0.5

    current = collect_process_tree_sample(
        root_pid=root.pid,
        sampler_pid=999,
        previous=previous,
        now_ns=2_000_000_000,
    )

    assert current.payload["cpu_time_tree_delta_seconds"] == pytest.approx(1.0)
    assert current.payload["cpu_percent_tree_one_core_scale"] == pytest.approx(100.0)
    assert current.payload["cpu_percent_tree_host_scale"] == pytest.approx(25.0)
    assert current.payload["logical_cpu_count"] == 4


@pytest.mark.parametrize("now_ns", [1_000_000_000, 999_999_999])
def test_cpu_percent_is_zero_for_zero_or_negative_wall_delta(
    monkeypatch: pytest.MonkeyPatch, now_ns: int
) -> None:
    root = FakeProcess(
        300, rss=100, threads=1, cpu_user=1.0, cpu_system=0.0
    )
    install_process_tree(monkeypatch, root)
    previous = collect_process_tree_sample(
        root_pid=root.pid,
        sampler_pid=999,
        previous=None,
        now_ns=1_000_000_000,
    )
    root.cpu_user += 1.0

    current = collect_process_tree_sample(
        root_pid=root.pid,
        sampler_pid=999,
        previous=previous,
        now_ns=now_ns,
    )

    assert current.payload["cpu_percent_tree_one_core_scale"] == 0.0
    assert current.payload["cpu_percent_tree_host_scale"] == 0.0


def test_process_disappearing_during_metric_reads_does_not_break_sample(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    child = FakeProcess(
        401, rss=200, threads=1, cpu_user=1.0, cpu_system=0.0
    )
    root = FakeProcess(
        400,
        rss=100,
        threads=2,
        cpu_user=1.0,
        cpu_system=0.0,
        children=[child],
    )
    install_process_tree(monkeypatch, root)

    def vanished() -> SimpleNamespace:
        raise psutil.NoSuchProcess(child.pid)

    child.memory_info = vanished  # type: ignore[method-assign]
    child.num_threads = vanished  # type: ignore[method-assign]
    child.cpu_times = vanished  # type: ignore[method-assign]

    sample = collect_process_tree_sample(
        root_pid=root.pid,
        sampler_pid=999,
        previous=None,
        now_ns=1,
    )

    assert sample.payload["pids"] == [400]
    assert sample.payload["rss_tree_bytes"] == 100
    assert sample.payload["process_count_tree"] == 1
    assert sample.payload["sample_errors"]


def test_initial_missing_root_pid_is_reported_clearly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing(pid: int) -> FakeProcess:
        raise psutil.NoSuchProcess(pid)

    monkeypatch.setattr(sampler.psutil, "Process", missing)

    with pytest.raises(TargetProcessNotFound, match="target PID 54321"):
        collect_process_tree_sample(
            root_pid=54321,
            sampler_pid=999,
            previous=None,
            now_ns=1,
        )


def test_collect_process_tree_sample_excludes_sampler() -> None:
    sample = collect_process_tree_sample(
        root_pid=os.getpid(),
        sampler_pid=os.getpid() + 100000,
        previous=None,
        now_ns=time.perf_counter_ns(),
    )
    assert sample.payload["rss_tree_bytes"] > 0
    assert os.getpid() in sample.payload["pids"]
    assert sample.payload["cpu_percent_tree_one_core_scale"] >= 0.0
    assert sample.payload["cpu_percent_tree_host_scale"] >= 0.0


def test_linux_thermal_value_is_converted_from_millidegrees(tmp_path: Path) -> None:
    thermal = tmp_path / "temp"
    thermal.write_text("48750\n", encoding="ascii")

    reading = read_thermal_state(
        sysfs_temp_path=thermal,
        vcgencmd_resolver=lambda _: None,
    )

    assert reading["cpu_temperature_c"] == pytest.approx(48.75)
    assert reading["cpu_temperature_available"] is True
    assert reading["cpu_temperature_source"] == "sysfs"


def test_missing_vcgencmd_is_null_with_explicit_availability(tmp_path: Path) -> None:
    reading = read_thermal_state(
        sysfs_temp_path=tmp_path / "missing",
        vcgencmd_resolver=lambda _: None,
    )

    assert reading["vcgencmd_available"] is False
    assert reading["cpu_temperature_c"] is None
    assert reading["cpu_temperature_available"] is False
    assert reading["throttled_raw"] is None
    assert reading["throttled_available"] is False


def test_target_exit_emits_successful_durable_end_with_self_accounting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events = tmp_path / "resource.jsonl"
    observer = JsonlObserver(make_identity(), events)
    real_collect = sampler.collect_process_tree_sample
    collect_calls = 0

    def collect_once(**kwargs: object) -> sampler.ProcessTreeSample:
        nonlocal collect_calls
        collect_calls += 1
        if collect_calls == 1:
            return real_collect(**kwargs)  # type: ignore[arg-type]
        raise TargetProcessNotFound(f"target PID {os.getpid()} does not exist")

    monkeypatch.setattr(sampler, "collect_process_tree_sample", collect_once)
    monkeypatch.setattr(
        sampler,
        "read_thermal_state",
        lambda: {
            "cpu_temperature_c": None,
            "cpu_temperature_available": False,
            "cpu_temperature_source": None,
            "vcgencmd_available": False,
            "throttled_raw": None,
            "throttled_available": False,
            "thermal_errors": [],
        },
    )

    result = run_sampler(
        root_pid=os.getpid(),
        observer=observer,
        interval_seconds=0.001,
        stop_file=None,
        sleep=lambda _: None,
    )

    rows = read_rows(events)
    end = next(row for row in rows if row["event_type"] == "resource_sampler_end")
    payload = end["payload"]
    assert result.status == "succeeded"
    assert end["status"] == "succeeded"
    assert payload["shutdown_reason"] == "target_exited"
    assert payload["sample_count"] == 1
    assert payload["sampler_cpu_user_seconds"] >= 0.0
    assert payload["sampler_cpu_system_seconds"] >= 0.0
    assert payload["sampler_rss_peak_bytes"] > 0
    assert payload["observer_event_encode_ns"] >= 0
    assert payload["observer_io_write_ns"] >= 0
    assert payload["observer_fsync_ns"] >= 0
    assert payload["observer_event_bytes_written"] > 0
    assert payload["observer_close_summary_path"] == str(
        events.with_suffix(".close.json")
    )
    assert events.with_suffix(".close.json").is_file()


def test_cli_defaults_to_one_second_interval(tmp_path: Path) -> None:
    context = tmp_path / "context.json"
    context.write_text(json.dumps(asdict(make_identity())), encoding="utf-8")
    args = build_parser().parse_args(
        [
            "--pid",
            "12345",
            "--observer-context",
            str(context),
            "--observer-events",
            str(tmp_path / "events.jsonl"),
        ]
    )

    assert args.interval_seconds == 1.0
