from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path

import pytest

import gaps_flower.observability as observability
from gaps_flower.observability import (
    JsonlObserver,
    NullObserver,
    ObserverIdentity,
    canonical_json_bytes,
    load_observer,
)


COMMON_FIELDS = {
    "schema_version",
    "event_id",
    "event_type",
    "run_id",
    "attempt_id",
    "group_id",
    "training_seed",
    "round",
    "client_id",
    "host_id",
    "producer",
    "process_instance_id",
    "sequence",
    "wall_time_utc",
    "monotonic_ns",
    "confirmation_commit",
    "source_archive_sha256",
    "dataset_manifest_sha256",
    "algorithm_config_sha256",
    "status",
    "payload",
}

OVERHEAD_FIELDS = {
    "observer_flower_serialize_ns",
    "observer_event_encode_ns",
    "observer_io_write_ns",
    "observer_fsync_ns",
    "observer_total_ns",
    "observer_event_bytes_written",
    "observer_event_count",
}


def make_identity() -> ObserverIdentity:
    return ObserverIdentity(
        run_id="c12_to_c5__b2__s42",
        attempt_id="c12_to_c5__b2__s42__a001",
        group_id="B2",
        training_seed=42,
        client_id=None,
        host_id="ecs",
        producer="server",
        confirmation_commit="a" * 40,
        source_archive_sha256="b" * 64,
        dataset_manifest_sha256="c" * 64,
        algorithm_config_sha256="d" * 64,
    )


def read_rows(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


def test_jsonl_observer_emits_contract_and_delayed_cost(tmp_path: Path) -> None:
    identity = make_identity()
    events = tmp_path / "events.jsonl"
    observer = JsonlObserver(identity, events)

    first_id = observer.emit(
        "fit_round_start",
        round_idx=1,
        client_id=None,
        status="started",
        payload={"fit_round_wall_ns": 0},
    )
    assert len(read_rows(events)) == 1

    observer.emit(
        "flower_fitins_prepared",
        round_idx=1,
        client_id=None,
        status="succeeded",
        payload={"proxy_id": "proxy-1"},
        flower_serialize_ns=17,
    )
    rows_before_close = read_rows(events)
    assert rows_before_close[1]["event_type"] == "observer_overhead"
    assert rows_before_close[1]["payload"]["observed_event_id"] == first_id

    observer.close()
    rows = read_rows(events)

    assert rows[0]["schema_version"] == "iotj.confirmation.observability.v1"
    assert rows[0]["event_id"] == first_id
    assert all(set(row) == COMMON_FIELDS for row in rows)
    assert all(row["run_id"] == identity.run_id for row in rows)
    overhead_rows = [row for row in rows if row["event_type"] == "observer_overhead"]
    assert all(set(row["payload"]) == OVERHEAD_FIELDS | {"observed_event_id"} for row in overhead_rows)
    assert (tmp_path / "events.close.json").is_file()


def test_sequences_and_event_ids_are_process_local_and_contiguous(tmp_path: Path) -> None:
    identity = make_identity()
    events = tmp_path / "events.jsonl"
    observer = JsonlObserver(identity, events)
    observer.emit(
        "fit_round_start",
        round_idx=1,
        client_id=None,
        status="started",
        payload={},
    )
    observer.emit(
        "fit_round_end",
        round_idx=1,
        client_id=None,
        status="succeeded",
        payload={},
    )
    observer.close()

    rows = read_rows(events)
    process_ids = {row["process_instance_id"] for row in rows}
    assert len(process_ids) == 1
    assert [row["sequence"] for row in rows] == list(range(1, len(rows) + 1))
    for row in rows:
        assert row["event_id"] == (
            f"{identity.attempt_id}/{identity.host_id}/{identity.producer}/"
            f"{row['process_instance_id']}/{row['sequence']}"
        )


def test_event_timestamps_and_canonical_json_are_stable(tmp_path: Path) -> None:
    events = tmp_path / "events.jsonl"
    observer = JsonlObserver(make_identity(), events)
    observer.emit(
        "fit_round_start",
        round_idx=None,
        client_id=None,
        status="started",
        payload={"label": "传感器"},
    )
    observer.close()

    first = read_rows(events)[0]
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z", first["wall_time_utc"])
    assert isinstance(first["monotonic_ns"], int)
    assert canonical_json_bytes({"z": "传感器", "a": 1}) == (
        '{"a":1,"z":"传感器"}'.encode("utf-8")
    )


def test_fsync_is_selected_only_for_durable_event_types(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fsync_file_descriptors: list[int] = []
    monkeypatch.setattr(observability.os, "fsync", fsync_file_descriptors.append)
    observer = JsonlObserver(make_identity(), tmp_path / "events.jsonl")

    observer.emit(
        "fit_round_start",
        round_idx=1,
        client_id=None,
        status="started",
        payload={},
    )
    observer.emit(
        "fit_round_end",
        round_idx=1,
        client_id=None,
        status="succeeded",
        payload={},
    )
    observer.close()

    assert len(fsync_file_descriptors) == 1


def test_close_summary_accounts_for_all_written_events(tmp_path: Path) -> None:
    events = tmp_path / "events.jsonl"
    observer = JsonlObserver(make_identity(), events)
    observer.emit(
        "fit_round_start",
        round_idx=1,
        client_id=None,
        status="started",
        payload={},
    )
    observer.close()

    rows = read_rows(events)
    summary = json.loads((tmp_path / "events.close.json").read_text(encoding="utf-8"))
    final_reporting_bytes = len(canonical_json_bytes(rows[-1])) + 1
    assert summary["observer_event_count"] == len(rows)
    assert summary["observer_event_bytes_written"] == events.stat().st_size
    assert summary["observer_reporting_tail_bytes"] == final_reporting_bytes
    assert summary["observer_event_encode_ns"] >= 0
    assert summary["observer_io_write_ns"] >= 0
    assert summary["observer_fsync_ns"] >= 0
    assert summary["observer_total_ns"] >= 0


def test_load_observer_reads_valid_context(tmp_path: Path) -> None:
    identity = make_identity()
    context = tmp_path / "context.json"
    context.write_text(json.dumps(asdict(identity)), encoding="utf-8")
    events = tmp_path / "events.jsonl"

    observer = load_observer(str(context), str(events))
    assert isinstance(observer, JsonlObserver)
    observer.emit(
        "attempt_end",
        round_idx=None,
        client_id=None,
        status="succeeded",
        payload={},
    )
    observer.close()

    assert read_rows(events)[0]["attempt_id"] == identity.attempt_id


def test_null_observer_is_a_stateless_no_op(tmp_path: Path) -> None:
    observer = NullObserver()
    result = observer.emit(
        "fit_round_start",
        round_idx=1,
        client_id=None,
        status="started",
        payload={"path": str(tmp_path / "must-not-exist")},
    )
    observer.close()

    assert result is None
    assert not hasattr(observer, "__dict__")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    ("run_id", "attempt_id", "match"),
    [
        ("c5_to_c1__b2__s42", "c5_to_c1__b2__s42__a001", "run_id"),
        ("c12_to_c5__b2__s42", "c12_to_c5__b2__s42__a01", "attempt_id"),
    ],
)
def test_identity_rejects_non_confirmation_scope(
    run_id: str, attempt_id: str, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        ObserverIdentity(
            run_id=run_id,
            attempt_id=attempt_id,
            group_id="B2",
            training_seed=42,
            client_id=None,
            host_id="ecs",
            producer="server",
            confirmation_commit="a" * 40,
            source_archive_sha256="b" * 64,
            dataset_manifest_sha256="c" * 64,
            algorithm_config_sha256="d" * 64,
        )


def test_confirmation_requirements_are_frozen() -> None:
    requirements = Path(__file__).parents[1] / "requirements-confirmation.txt"
    assert requirements.read_text(encoding="utf-8") == (
        "-r requirements.txt\n"
        "flwr==1.23.0\n"
        "protobuf==4.25.8\n"
        "psutil==7.0.0\n"
    )
