"""Contract tests for sealed confirmation evaluation and summaries."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from scripts import summarize_iotj_confirmation_observability as summary
from scripts.freeze_iotj_confirmation_protocol import build_dataset_manifest


COMMIT = "a" * 40
SOURCE_SHA = "b" * 64


def _write_split(directory: Path, split: str, rows: int) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    np.save(
        directory / f"{split}_features.npy",
        np.arange(rows, dtype=np.int32).reshape(rows, 1),
    )
    np.save(
        directory / f"{split}_classification_labels.npy",
        np.zeros(rows, dtype=np.int64),
    )
    np.save(
        directory / f"{split}_regression_labels.npy",
        np.zeros((rows, 3), dtype=np.float32),
    )


def _write_dataset(root: Path, *, norm_payload: bytes = b"frozen-normalization") -> Path:
    root.mkdir(parents=True)
    (root / "split_info.json").write_text(
        json.dumps(
            {
                "protocol": "c12_to_c5",
                "source_clients": [1, 2, 3, 4],
                "target_clients": [5],
                "seed": 42,
                "target_split": {
                    "train_used": False,
                    "calibration": 0.2,
                    "test": 0.8,
                },
                "stratify_by": ["client", "class", "concentration"],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "norm_stats.npz").write_bytes(norm_payload)
    for client_id in (1, 2):
        for split in ("train", "calibration", "test"):
            _write_split(root / f"client_{client_id}", split, 3)
    _write_split(root / "client_5", "calibration", 320)
    _write_split(root / "client_5", "test", 1360)
    return root


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_bytes(value) + b"\n")


@pytest.fixture
def canonical_fixture(tmp_path: Path) -> dict[str, Any]:
    raw_root = tmp_path / "raw"
    data_root = _write_dataset(tmp_path / "sealed_dataset")
    dataset_manifest = build_dataset_manifest(data_root)
    dataset_sha = dataset_manifest["dataset_manifest_sha256"]
    rows: list[dict[str, Any]] = []
    schedule: list[dict[str, Any]] = []
    for group in ("B2", "B5"):
        for seed in range(42, 47):
            run_id = f"c12_to_c5__{group.lower()}__s{seed}"
            attempt_id = f"{run_id}__a001"
            attempt_dir = raw_root / run_id / attempt_id
            checkpoint = attempt_dir / "raw" / "ecs" / "training" / "server_latest_adapted.pth"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(f"round25:{group}:{seed}".encode("ascii"))
            algorithm_sha = hashlib.sha256(f"algorithm:{group}:{seed}".encode()).hexdigest()
            schedule.append(
                {
                    "run_id": run_id,
                    "group_id": group,
                    "seed": seed,
                    "algorithm_config_sha256": algorithm_sha,
                    "transport_status": "not_collected",
                }
            )
            rows.append(
                {
                    "run_id": run_id,
                    "attempt_id": attempt_id,
                    "attempt_dir": attempt_dir,
                    "group_id": group,
                    "seed": seed,
                    "state": "canonical",
                    "status_reason": "validator_accepted",
                    "direction": "C1/C2 -> C5",
                    "historical_seed42_included": False,
                    "confirmation_commit": COMMIT,
                    "source_archive_sha256": SOURCE_SHA,
                    "dataset_manifest_sha256": dataset_sha,
                    "algorithm_config_sha256": algorithm_sha,
                    "checkpoint_path": checkpoint,
                    "checkpoint_sha256": _sha_file(checkpoint),
                    "transport_status": "not_collected",
                }
            )

    protocol = {
        "schema_version": 1,
        "protocol_id": "iotj_main_direction_confirmation",
        "direction": "C1/C2 -> C5",
        "active_source_clients": ["C1", "C2"],
        "active_target_clients": ["C5"],
        "groups": ["B2", "B5"],
        "seeds": [42, 43, 44, 45, 46],
        "historical_seed42_included": False,
        "confirmation_commit": COMMIT,
        "source_archive_sha256": SOURCE_SHA,
        "dataset_manifest_sha256": dataset_sha,
        "schedule": schedule,
    }
    protocol["protocol_manifest_sha256"] = hashlib.sha256(
        _canonical_bytes(protocol)
    ).hexdigest()
    for row in rows:
        audit = {
            "schema_version": "iotj.confirmation.attempt_audit.v1",
            "run_id": row["run_id"],
            "attempt_id": row["attempt_id"],
            "status": "valid",
            "reasons": [],
            "counts": {"fitins": 50, "fitres": 50, "rounds": 25},
            "protocol_manifest_sha256": protocol["protocol_manifest_sha256"],
            "resource": {
                "C1": {"coverage": 1.0, "expected_sample_points": 25, "covered_sample_points": 25, "sample_count": 25},
                "C2": {"coverage": 1.0, "expected_sample_points": 25, "covered_sample_points": 25, "sample_count": 25},
            },
            "inputs": {},
        }
        audit_path = row["attempt_dir"] / "attempt_audit.json"
        _write_json(audit_path, audit)
        row["audit_path"] = audit_path
        row["audit_sha256"] = _sha_file(audit_path)
        row["protocol_manifest_sha256"] = protocol["protocol_manifest_sha256"]
        _write_json(
            row["attempt_dir"] / "attempt_provenance.json",
            {
                "run_id": row["run_id"],
                "attempt_id": row["attempt_id"],
                "confirmation_commit": row["confirmation_commit"],
                "source_archive_sha256": row["source_archive_sha256"],
                "dataset_manifest_sha256": row["dataset_manifest_sha256"],
                "algorithm_config_sha256": row["algorithm_config_sha256"],
                "controller_owner": {"pid": 123, "instance_id": "d" * 32},
            },
        )
        _write_json(
            row["attempt_dir"] / "attempt_status.json",
            {
                "run_id": row["run_id"],
                "attempt_id": row["attempt_id"],
                "state": "canonical",
                "reason": "validator_accepted",
                "audit_sha256": row["audit_sha256"],
                "confirmation_commit": row["confirmation_commit"],
                "source_archive_sha256": row["source_archive_sha256"],
                "dataset_manifest_sha256": row["dataset_manifest_sha256"],
                "algorithm_config_sha256": row["algorithm_config_sha256"],
            },
        )
    return {
        "raw_root": raw_root,
        "rows": rows,
        "protocol": protocol,
        "data_root": data_root,
        "dataset_manifest": dataset_manifest,
    }


def test_test_gate_requires_exact_matrix_and_common_frozen_revision(
    canonical_fixture: dict[str, Any],
) -> None:
    rows = canonical_fixture["rows"]
    assert len(summary.assert_test_gate(rows)) == 10
    with pytest.raises(RuntimeError, match="10 canonical"):
        summary.assert_test_gate(rows[:-1])

    mixed = copy.deepcopy(rows)
    mixed[-1]["confirmation_commit"] = "f" * 40
    with pytest.raises(RuntimeError, match="confirmation_commit"):
        summary.assert_test_gate(mixed)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda row: row.update(state="invalid"), "canonical"),
        (lambda row: row.update(direction="C2/C1 -> C5"), "direction"),
        (lambda row: row.update(historical_revision="feaa75b"), "historical"),
        (lambda row: row.update(rerun_reason="accuracy_low"), "metric-driven"),
        (lambda row: row.update(transport_status="collected"), "transport_status"),
    ],
)
def test_test_gate_rejects_nonconfirmation_or_metric_driven_inputs(
    canonical_fixture: dict[str, Any], mutation, message: str
) -> None:
    rows = copy.deepcopy(canonical_fixture["rows"])
    mutation(rows[-1])
    with pytest.raises(RuntimeError, match=message):
        summary.assert_test_gate(rows)


def test_test_gate_binds_protocol_audit_checkpoint_and_paths(
    canonical_fixture: dict[str, Any], tmp_path: Path
) -> None:
    rows = canonical_fixture["rows"]
    protocol = canonical_fixture["protocol"]
    raw_root = canonical_fixture["raw_root"]
    accepted = summary.assert_test_gate(rows, protocol=protocol, raw_root=raw_root)
    assert [row["run_id"] for row in accepted] == sorted(row["run_id"] for row in rows)

    wrong_audit = copy.deepcopy(rows)
    wrong_audit[0]["audit_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="audit.*SHA-256"):
        summary.assert_test_gate(wrong_audit, protocol=protocol, raw_root=raw_root)

    escaped = copy.deepcopy(rows)
    outside = tmp_path / "outside.pth"
    outside.write_bytes(b"outside")
    escaped[0]["checkpoint_path"] = outside
    escaped[0]["checkpoint_sha256"] = _sha_file(outside)
    with pytest.raises(RuntimeError, match="escapes attempt"):
        summary.assert_test_gate(escaped, protocol=protocol, raw_root=raw_root)


def test_discovery_uses_immutable_status_and_never_metrics(
    canonical_fixture: dict[str, Any]
) -> None:
    discovered = summary.discover_canonical_attempts(
        canonical_fixture["raw_root"], canonical_fixture["protocol"]
    )
    assert [row["run_id"] for row in discovered] == sorted(
        row["run_id"] for row in canonical_fixture["rows"]
    )

    status_path = canonical_fixture["rows"][0]["attempt_dir"] / "attempt_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["selection_reason"] = "accuracy_low"
    status_path.write_text(json.dumps(status), encoding="utf-8")
    with pytest.raises(RuntimeError, match="metric-driven"):
        summary.discover_canonical_attempts(
            canonical_fixture["raw_root"], canonical_fixture["protocol"]
        )


def test_test_gate_rejects_symlink_checkpoint(
    canonical_fixture: dict[str, Any], tmp_path: Path
) -> None:
    rows = copy.deepcopy(canonical_fixture["rows"])
    target = rows[0]["checkpoint_path"]
    link = target.with_name("linked.pth")
    try:
        os.symlink(target, link)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")
    rows[0]["checkpoint_path"] = link
    with pytest.raises(RuntimeError, match="symlink"):
        summary.assert_test_gate(
            rows,
            protocol=canonical_fixture["protocol"],
            raw_root=canonical_fixture["raw_root"],
        )


def _fake_metrics(group: str, seed: int) -> dict[str, Any]:
    group_offset = 0.02 if group == "B2" else 0.01
    seed_offset = (seed - 42) * 0.001
    accuracy = 0.90 + group_offset + seed_offset
    return {
        "N": 1360,
        "accuracy": accuracy,
        "macro_f1": accuracy - 0.01,
        "nll": 0.20 - group_offset + seed_offset,
        "ece": 0.04 - group_offset / 2 + seed_offset,
        "per_class_recall": {
            str(class_id): accuracy - class_id * 0.01 for class_id in range(4)
        },
    }


def _fake_stream() -> list[dict[str, Any]]:
    return [
        {
            "client": "C5",
            "split": "test",
            "sample_index": sample_index,
            "pred_class": 0,
            "true_class": 0,
        }
        for sample_index in range(1360)
    ]


def _successful_evaluator(canonical_fixture: dict[str, Any], calls: list[Path] | None = None):
    def evaluator(path: Path, **_kwargs):
        if calls is not None:
            calls.append(path)
        row = next(
            item
            for item in canonical_fixture["rows"]
            if item["checkpoint_path"] == path
        )
        return _fake_stream(), _fake_metrics(row["group_id"], row["seed"])

    return evaluator


def _stream_paths(canonical_fixture: dict[str, Any]) -> list[Path]:
    return [
        row["attempt_dir"]
        / "raw"
        / "ecs"
        / "evaluation"
        / "classification_test_stream.csv"
        for row in canonical_fixture["rows"]
    ]


def test_evaluator_is_zero_call_until_all_ten_pass_gate(
    canonical_fixture: dict[str, Any], tmp_path: Path
) -> None:
    calls: list[Path] = []

    def evaluator(path: Path, **_kwargs):
        calls.append(path)
        return [], _fake_metrics("B2", 42)

    with pytest.raises(RuntimeError, match="10 canonical"):
        summary.evaluate_canonical_attempts(
            canonical_fixture["rows"][:-1],
            data_root=canonical_fixture["data_root"],
            device="cpu",
            batch_size=32,
            evaluator=evaluator,
        )
    assert calls == []


def test_dataset_binding_rejects_wrong_root_before_evaluator(
    canonical_fixture: dict[str, Any], tmp_path: Path
) -> None:
    wrong_root = _write_dataset(
        tmp_path / "wrong_but_structurally_valid_dataset",
        norm_payload=b"different-normalization",
    )
    calls: list[Path] = []
    with pytest.raises(RuntimeError, match="dataset_manifest_sha256"):
        summary.evaluate_canonical_attempts(
            canonical_fixture["rows"],
            data_root=wrong_root,
            device="cpu",
            batch_size=32,
            evaluator=_successful_evaluator(canonical_fixture, calls),
            protocol=canonical_fixture["protocol"],
            raw_root=canonical_fixture["raw_root"],
        )
    assert calls == []
    assert not any(path.exists() for path in _stream_paths(canonical_fixture))


def test_dataset_is_rechecked_after_evaluation_before_stream_publish(
    canonical_fixture: dict[str, Any]
) -> None:
    calls: list[Path] = []
    delegate = _successful_evaluator(canonical_fixture, calls)

    def mutating_evaluator(path: Path, **kwargs):
        result = delegate(path, **kwargs)
        if len(calls) == 5:
            (canonical_fixture["data_root"] / "norm_stats.npz").write_bytes(
                b"changed-during-evaluation"
            )
        return result

    with pytest.raises(RuntimeError, match="changed during sealed evaluation"):
        summary.evaluate_canonical_attempts(
            canonical_fixture["rows"],
            data_root=canonical_fixture["data_root"],
            device="cpu",
            batch_size=32,
            evaluator=mutating_evaluator,
            protocol=canonical_fixture["protocol"],
            raw_root=canonical_fixture["raw_root"],
        )
    assert len(calls) == 10
    assert not any(path.exists() for path in _stream_paths(canonical_fixture))


def test_stream_transaction_evaluator_failure_publishes_none_and_is_retryable(
    canonical_fixture: dict[str, Any]
) -> None:
    calls: list[Path] = []
    delegate = _successful_evaluator(canonical_fixture, calls)

    def failing_evaluator(path: Path, **kwargs):
        if len(calls) == 4:
            calls.append(path)
            raise RuntimeError("synthetic evaluator failure")
        return delegate(path, **kwargs)

    with pytest.raises(RuntimeError, match="synthetic evaluator failure"):
        summary.evaluate_canonical_attempts(
            canonical_fixture["rows"],
            data_root=canonical_fixture["data_root"],
            device="cpu",
            batch_size=32,
            evaluator=failing_evaluator,
            protocol=canonical_fixture["protocol"],
            raw_root=canonical_fixture["raw_root"],
        )
    assert len(calls) == 5
    assert not any(path.exists() for path in _stream_paths(canonical_fixture))

    retried = summary.evaluate_canonical_attempts(
        canonical_fixture["rows"],
        data_root=canonical_fixture["data_root"],
        device="cpu",
        batch_size=32,
        evaluator=_successful_evaluator(canonical_fixture),
        protocol=canonical_fixture["protocol"],
        raw_root=canonical_fixture["raw_root"],
    )
    assert len(retried) == 10
    assert all(path.is_file() for path in _stream_paths(canonical_fixture))


def test_stream_transaction_preflights_all_final_paths_before_evaluator(
    canonical_fixture: dict[str, Any]
) -> None:
    existing = _stream_paths(canonical_fixture)[-1]
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"preexisting-owner-data")
    calls: list[Path] = []
    with pytest.raises(FileExistsError, match="overwrite"):
        summary.evaluate_canonical_attempts(
            canonical_fixture["rows"],
            data_root=canonical_fixture["data_root"],
            device="cpu",
            batch_size=32,
            evaluator=_successful_evaluator(canonical_fixture, calls),
            protocol=canonical_fixture["protocol"],
            raw_root=canonical_fixture["raw_root"],
        )
    assert calls == []
    assert existing.read_bytes() == b"preexisting-owner-data"


def test_stream_transaction_rolls_back_only_its_own_mid_publish_files(
    canonical_fixture: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    original_publish = summary._publish_stream
    publish_calls: list[Path] = []

    def failing_publish(path: Path, payload: bytes) -> None:
        publish_calls.append(path)
        if len(publish_calls) == 5:
            original_publish(path, payload)
            raise OSError("synthetic publish failure")
        original_publish(path, payload)

    monkeypatch.setattr(summary, "_publish_stream", failing_publish)
    with pytest.raises(OSError, match="synthetic publish failure"):
        summary.evaluate_canonical_attempts(
            canonical_fixture["rows"],
            data_root=canonical_fixture["data_root"],
            device="cpu",
            batch_size=32,
            evaluator=_successful_evaluator(canonical_fixture),
            protocol=canonical_fixture["protocol"],
            raw_root=canonical_fixture["raw_root"],
        )
    assert len(publish_calls) == 5
    assert not any(path.exists() for path in _stream_paths(canonical_fixture))
    assert not list(canonical_fixture["raw_root"].rglob("*.tmp"))


def test_link_receipt_failure_self_cleans_and_allows_retry(
    canonical_fixture: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    class ReceiptFailure(BaseException):
        pass

    original_record = summary._record_stream_receipt
    calls = 0

    def failing_record(receipts, receipt) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ReceiptFailure("synthetic link-to-receipt failure")
        original_record(receipts, receipt)

    monkeypatch.setattr(summary, "_record_stream_receipt", failing_record)
    with pytest.raises(ReceiptFailure, match="link-to-receipt"):
        summary.evaluate_canonical_attempts(
            canonical_fixture["rows"],
            data_root=canonical_fixture["data_root"],
            device="cpu",
            batch_size=32,
            evaluator=_successful_evaluator(canonical_fixture),
            protocol=canonical_fixture["protocol"],
            raw_root=canonical_fixture["raw_root"],
        )
    assert not any(path.exists() for path in _stream_paths(canonical_fixture))
    assert not list(canonical_fixture["raw_root"].rglob("*.tmp"))

    monkeypatch.setattr(summary, "_record_stream_receipt", original_record)
    assert len(
        summary.evaluate_canonical_attempts(
            canonical_fixture["rows"],
            data_root=canonical_fixture["data_root"],
            device="cpu",
            batch_size=32,
            evaluator=_successful_evaluator(canonical_fixture),
            protocol=canonical_fixture["protocol"],
            raw_root=canonical_fixture["raw_root"],
        )
    ) == 10


def test_later_checkpoint_replacement_stops_before_that_evaluator(
    canonical_fixture: dict[str, Any]
) -> None:
    ordered = sorted(canonical_fixture["rows"], key=lambda row: row["run_id"])
    calls: list[Path] = []
    delegate = _successful_evaluator(canonical_fixture, calls)

    def replacing_evaluator(path: Path, **kwargs):
        result = delegate(path, **kwargs)
        if len(calls) == 1:
            ordered[1]["checkpoint_path"].write_bytes(b"replaced-after-gate")
        return result

    with pytest.raises(RuntimeError, match="checkpoint.*SHA-256"):
        summary.evaluate_canonical_attempts(
            canonical_fixture["rows"],
            data_root=canonical_fixture["data_root"],
            device="cpu",
            batch_size=32,
            evaluator=replacing_evaluator,
            protocol=canonical_fixture["protocol"],
            raw_root=canonical_fixture["raw_root"],
        )
    assert len(calls) == 1
    assert not any(path.exists() for path in _stream_paths(canonical_fixture))


def test_classification_rows_stream_location_and_multiseed_statistics(
    canonical_fixture: dict[str, Any], tmp_path: Path
) -> None:
    calls: list[tuple[Path, dict[str, Any]]] = []

    def evaluator(path: Path, **kwargs):
        row = next(item for item in canonical_fixture["rows"] if item["checkpoint_path"] == path)
        calls.append((path, kwargs))
        stream = [
            {
                "client": "C5",
                "split": "test",
                "sample_index": sample_index,
                "pred_class": 0,
                "true_class": 0,
            }
            for sample_index in range(1360)
        ]
        return stream, _fake_metrics(row["group_id"], row["seed"])

    per_run = summary.evaluate_canonical_attempts(
        canonical_fixture["rows"],
        data_root=canonical_fixture["data_root"],
        device="cpu",
        batch_size=32,
        evaluator=evaluator,
    )
    assert len(calls) == 10
    assert all(call[1]["target_client"] == 5 and call[1]["split"] == "test" for call in calls)
    assert set(per_run[0]) == set(summary.CLASSIFICATION_PER_RUN_FIELDS)
    assert all(row["N"] == 1360 for row in per_run)
    assert {row["claim_status"] for row in per_run if row["group_id"] == "B2"} == {
        "post_screen_exploratory"
    }
    assert {row["claim_status"] for row in per_run if row["group_id"] == "B5"} == {
        "predeclared_full_method"
    }
    for row in canonical_fixture["rows"]:
        stream_path = (
            row["attempt_dir"]
            / "raw"
            / "ecs"
            / "evaluation"
            / "classification_test_stream.csv"
        )
        assert stream_path.is_file()
        assert not stream_path.is_relative_to(tmp_path / "summary")

    multiseed = summary.build_classification_multiseed_summary(per_run)
    assert set(multiseed[0]) == set(summary.CLASSIFICATION_MULTISEED_FIELDS)
    accuracy_rows = [row for row in multiseed if row["metric"] == "accuracy"]
    b2 = next(row for row in accuracy_rows if row["group_id"] == "B2")
    paired = next(row for row in accuracy_rows if row["group_id"] == "B2-B5")
    b2_values = np.asarray([_fake_metrics("B2", seed)["accuracy"] for seed in range(42, 47)])
    assert b2["mean"] == pytest.approx(float(b2_values.mean()))
    assert b2["sample_std_ddof1"] == pytest.approx(float(b2_values.std(ddof=1)))
    assert [b2[f"seed_{seed}"] for seed in range(42, 47)] == pytest.approx(b2_values)
    assert all(paired[f"seed_{seed}"] == pytest.approx(0.01) for seed in range(42, 47))
    required_paired = {
        "accuracy",
        "macro_f1",
        "nll",
        "ece",
        "recall_0",
        "recall_1",
        "recall_2",
        "recall_3",
    }
    assert required_paired <= {
        row["metric"] for row in multiseed if row["group_id"] == "B2-B5"
    }


def _logical(direction: str, value: int) -> dict[str, int]:
    if direction == "downlink":
        return {
            "logical_downlink_model_value_bytes": value + 9,
            "logical_downlink_parameter_blob_bytes": value,
            "logical_downlink_semantic_proto_utf8_bytes": 2,
            "logical_downlink_other_config_value_bytes": 3,
            "logical_downlink_total_bytes": value + 5,
        }
    return {
        "logical_uplink_model_value_bytes": value + 9,
        "logical_uplink_parameter_blob_bytes": value,
        "logical_uplink_prototype_utf8_bytes": 1,
        "logical_uplink_prototype_var_utf8_bytes": 2,
        "logical_uplink_statistics_utf8_bytes": 3,
        "logical_uplink_diagnostic_value_bytes": 4,
        "logical_uplink_total_bytes": value + 10,
    }


def _evidence_for(row: dict[str, Any]) -> dict[str, Any]:
    server: list[dict[str, Any]] = []
    c1: list[dict[str, Any]] = []
    c2: list[dict[str, Any]] = []
    resources: dict[str, list[dict[str, Any]]] = {"C1": [], "C2": []}
    for round_idx in range(1, 26):
        for client_idx, client_id in enumerate(("C1", "C2"), 1):
            server.append(
                {
                    "event_type": "flower_fitins_prepared",
                    "round": round_idx,
                    "client_id": client_id,
                    "payload": {
                        "downlink_audit": {
                            "logical": _logical("downlink", 100 + client_idx),
                            "application_message_bytes": 150 + client_idx,
                            "application_message_sha256": f"{round_idx + client_idx:064x}",
                        }
                    },
                }
            )
            server.append(
                {
                    "event_type": "flower_fitres_available",
                    "round": round_idx,
                    "client_id": client_id,
                    "payload": {
                        "uplink_audit": {
                            "logical": _logical("uplink", 200 + client_idx),
                            "application_message_bytes": 250 + client_idx,
                            "application_message_sha256": f"{round_idx + client_idx + 100:064x}",
                        }
                    },
                }
            )
            client_events = c1 if client_id == "C1" else c2
            client_events.extend(
                [
                    {
                        "event_type": "client_fit_start",
                        "round": round_idx,
                        "client_id": client_id,
                        "monotonic_ns": round_idx * 1_000_000_000,
                        "payload": {},
                    },
                    {
                        "event_type": "client_train_end",
                        "round": round_idx,
                        "client_id": client_id,
                        "monotonic_ns": round_idx * 1_000_000_000 + 300 + client_idx,
                        "payload": {"client_train_core_ns": 300 + client_idx},
                    },
                    {
                        "event_type": "client_fit_end",
                        "round": round_idx,
                        "client_id": client_id,
                        "monotonic_ns": round_idx * 1_000_000_000 + 500 + client_idx,
                        "payload": {"client_fit_callback_ns": 500 + client_idx},
                    },
                ]
            )
        server.extend(
            [
                {
                    "event_type": "server_aggregate_end",
                    "round": round_idx,
                    "client_id": None,
                    "payload": {
                        "server_aggregate_fit_total_ns": 100,
                        "server_da_total_ns": 40,
                        "server_aggregate_non_da_ns": 60,
                        "da_executed": True,
                    },
                },
                {
                    "event_type": "fit_round_end",
                    "round": round_idx,
                    "client_id": None,
                    "payload": {"fit_round_wall_ns": 1_000},
                },
            ]
        )
    for client_id, host in (("C1", "pi"), ("C2", "pc")):
        for sample_idx in range(25):
            resources[client_id].append(
                {
                    "event_type": "resource_sample",
                    "client_id": client_id,
                    "payload": {
                        "rss_tree_bytes": 1_000 + sample_idx,
                        "rss_tree_peak_bytes": 2_000 + sample_idx,
                        "cpu_percent_tree_one_core_scale": 50.0 + sample_idx,
                        "cpu_percent_tree_host_scale": 12.5 + sample_idx / 4,
                        "cpu_temperature_c": 60.0 if host == "pi" else None,
                        "cpu_temperature_available": host == "pi",
                        "throttled_bits": 0 if host == "pi" else None,
                        "throttled_available": host == "pi",
                        "sample_interval_start_monotonic_ns": (sample_idx + 1)
                        * 1_000_000_000,
                        "sample_interval_end_monotonic_ns": (sample_idx + 1)
                        * 1_000_000_000
                        + 400,
                    },
                }
            )
    close_summaries = []
    for host_id, producer in (
        ("ecs", "server"),
        ("pi-c1", "client"),
        ("pi-c1", "resource_sampler"),
        ("pc-c2", "client"),
        ("pc-c2", "resource_sampler"),
    ):
        close_summaries.append(
            {
                "run_id": row["run_id"],
                "attempt_id": row["attempt_id"],
                "host_id": host_id,
                "producer": producer,
                "observer_flower_serialize_ns": 10,
                "observer_event_encode_ns": 20,
                "observer_io_write_ns": 30,
                "observer_fsync_ns": 40,
                "observer_total_ns": 100,
                "observer_event_bytes_written": 500,
                "observer_event_count": 5,
                "observer_reporting_tail_bytes": 50,
            }
        )
    return {
        "server": server,
        "C1": c1,
        "C2": c2,
        "resource_C1": resources["C1"],
        "resource_C2": resources["C2"],
        "close_summaries": close_summaries,
    }


EVIDENCE_LAYOUT = {
    "server": "raw/ecs/events.jsonl",
    "C1": "raw/pi/events.jsonl",
    "resource_C1": "raw/pi/resource.jsonl",
    "C2": "raw/pc/events.jsonl",
    "resource_C2": "raw/pc/resource.jsonl",
}


def _rebind_audit(row: dict[str, Any], audit: dict[str, Any]) -> None:
    _write_json(row["audit_path"], audit)
    row["audit_sha256"] = _sha_file(row["audit_path"])
    status_path = row["attempt_dir"] / "attempt_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["audit_sha256"] = row["audit_sha256"]
    _write_json(status_path, status)


def _write_audit_approved_evidence(
    row: dict[str, Any], evidence: dict[str, Any]
) -> dict[str, str]:
    close_identity = {
        "server": ("ecs", "server"),
        "C1": ("pi-c1", "client"),
        "resource_C1": ("pi-c1", "resource_sampler"),
        "C2": ("pc-c2", "client"),
        "resource_C2": ("pc-c2", "resource_sampler"),
    }
    hashes: dict[str, str] = {}
    for key, relative in EVIDENCE_LAYOUT.items():
        path = row["attempt_dir"] / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = b"".join(_canonical_bytes(event) + b"\n" for event in evidence[key])
        path.write_bytes(payload)
        hashes[relative] = _sha_file(path)
        host_id, producer = close_identity[key]
        close = next(
            item
            for item in evidence["close_summaries"]
            if item["host_id"] == host_id and item["producer"] == producer
        )
        close_path = path.with_suffix(".close.json")
        _write_json(close_path, close)
        hashes[close_path.relative_to(row["attempt_dir"]).as_posix()] = _sha_file(
            close_path
        )
    audit = json.loads(row["audit_path"].read_text(encoding="utf-8"))
    audit["inputs"] = {key: hashes[key] for key in sorted(hashes)}
    _rebind_audit(row, audit)
    return hashes


def _install_all_audit_approved_evidence(canonical_fixture: dict[str, Any]) -> None:
    for row in canonical_fixture["rows"]:
        _write_audit_approved_evidence(row, _evidence_for(row))


def test_summarize_write_failure_rolls_back_streams_and_is_retryable(
    canonical_fixture: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_all_audit_approved_evidence(canonical_fixture)
    output = tmp_path / "summary"
    original_publish = summary._publish_summary_staging
    publish_calls = 0

    def failing_publish(staging: Path, destination: Path) -> None:
        nonlocal publish_calls
        publish_calls += 1
        if publish_calls == 1:
            raise OSError("synthetic summary rename failure")
        original_publish(staging, destination)

    monkeypatch.setattr(summary, "_publish_summary_staging", failing_publish)
    with pytest.raises(OSError, match="summary rename"):
        summary.summarize_confirmation(
            raw_root=canonical_fixture["raw_root"],
            protocol=canonical_fixture["protocol"],
            data_root=canonical_fixture["data_root"],
            output_root=output,
            device="cpu",
            batch_size=32,
            evaluator=_successful_evaluator(canonical_fixture),
        )
    assert not output.exists()
    assert not any(path.exists() for path in _stream_paths(canonical_fixture))

    summary.summarize_confirmation(
        raw_root=canonical_fixture["raw_root"],
        protocol=canonical_fixture["protocol"],
        data_root=canonical_fixture["data_root"],
        output_root=output,
        device="cpu",
        batch_size=32,
        evaluator=_successful_evaluator(canonical_fixture),
    )
    assert output.is_dir()
    assert all(path.is_file() for path in _stream_paths(canonical_fixture))


def test_evidence_change_after_consumption_rolls_back_all_streams(
    canonical_fixture: dict[str, Any], tmp_path: Path
) -> None:
    _install_all_audit_approved_evidence(canonical_fixture)
    output = tmp_path / "summary"
    calls: list[Path] = []
    delegate = _successful_evaluator(canonical_fixture, calls)
    first_evidence = (
        sorted(canonical_fixture["rows"], key=lambda row: row["run_id"])[0][
            "attempt_dir"
        ]
        / "raw"
        / "ecs"
        / "events.jsonl"
    )

    def mutating_evaluator(path: Path, **kwargs):
        result = delegate(path, **kwargs)
        if len(calls) == 10:
            first_evidence.write_bytes(first_evidence.read_bytes() + b"{}\n")
        return result

    with pytest.raises(RuntimeError, match="changed after summary consumption"):
        summary.summarize_confirmation(
            raw_root=canonical_fixture["raw_root"],
            protocol=canonical_fixture["protocol"],
            data_root=canonical_fixture["data_root"],
            output_root=output,
            device="cpu",
            batch_size=32,
            evaluator=mutating_evaluator,
        )
    assert not output.exists()
    assert not any(path.exists() for path in _stream_paths(canonical_fixture))


def test_post_publish_stream_rewrite_blocks_summary_and_preserves_external_owner(
    canonical_fixture: dict[str, Any], tmp_path: Path
) -> None:
    _install_all_audit_approved_evidence(canonical_fixture)
    rows = summary.assert_test_gate(
        canonical_fixture["rows"],
        protocol=canonical_fixture["protocol"],
        raw_root=canonical_fixture["raw_root"],
    )
    tables = summary.build_system_tables(
        rows,
        summary.load_attempt_evidence,
        protocol=canonical_fixture["protocol"],
        raw_root=canonical_fixture["raw_root"],
    )
    output = tmp_path / "summary"
    changed = _stream_paths(canonical_fixture)[0]
    with pytest.raises(RuntimeError, match="ownership changed"):
        with summary.StreamTransaction(rows) as transaction:
            per_run = summary.evaluate_canonical_attempts(
                rows,
                data_root=canonical_fixture["data_root"],
                device="cpu",
                batch_size=32,
                evaluator=_successful_evaluator(canonical_fixture),
                protocol=canonical_fixture["protocol"],
                raw_root=canonical_fixture["raw_root"],
                stream_transaction=transaction,
            )
            changed.write_bytes(b"external-replacement")
            summary.write_summary_bundle(
                output,
                rows,
                canonical_fixture["protocol"],
                per_run,
                tables,
                raw_root=canonical_fixture["raw_root"],
                stream_transaction=transaction,
            )
    assert not output.exists()
    assert changed.read_bytes() == b"external-replacement"
    assert not any(path.exists() for path in _stream_paths(canonical_fixture)[1:])


def test_post_publish_stream_disappearance_is_reported_and_rolls_back_owned_streams(
    canonical_fixture: dict[str, Any], tmp_path: Path
) -> None:
    _install_all_audit_approved_evidence(canonical_fixture)
    rows = summary.assert_test_gate(
        canonical_fixture["rows"],
        protocol=canonical_fixture["protocol"],
        raw_root=canonical_fixture["raw_root"],
    )
    tables = summary.build_system_tables(
        rows,
        summary.load_attempt_evidence,
        protocol=canonical_fixture["protocol"],
        raw_root=canonical_fixture["raw_root"],
    )
    output = tmp_path / "summary"
    disappeared = _stream_paths(canonical_fixture)[0]
    with pytest.raises(RuntimeError, match="ownership changed or disappeared"):
        with summary.StreamTransaction(rows) as transaction:
            per_run = summary.evaluate_canonical_attempts(
                rows,
                data_root=canonical_fixture["data_root"],
                device="cpu",
                batch_size=32,
                evaluator=_successful_evaluator(canonical_fixture),
                protocol=canonical_fixture["protocol"],
                raw_root=canonical_fixture["raw_root"],
                stream_transaction=transaction,
            )
            disappeared.unlink()
            summary.write_summary_bundle(
                output,
                rows,
                canonical_fixture["protocol"],
                per_run,
                tables,
                raw_root=canonical_fixture["raw_root"],
                stream_transaction=transaction,
            )
    assert not output.exists()
    assert not any(path.exists() for path in _stream_paths(canonical_fixture))


def test_checkpoint_change_before_final_summary_publish_rolls_back_streams(
    canonical_fixture: dict[str, Any], tmp_path: Path
) -> None:
    _install_all_audit_approved_evidence(canonical_fixture)
    calls: list[Path] = []
    delegate = _successful_evaluator(canonical_fixture, calls)
    first_checkpoint = sorted(
        canonical_fixture["rows"], key=lambda row: row["run_id"]
    )[0]["checkpoint_path"]

    def late_mutation(path: Path, **kwargs):
        result = delegate(path, **kwargs)
        if len(calls) == 10:
            first_checkpoint.write_bytes(b"late-checkpoint-replacement")
        return result

    with pytest.raises(RuntimeError, match="checkpoint.*SHA-256"):
        summary.summarize_confirmation(
            raw_root=canonical_fixture["raw_root"],
            protocol=canonical_fixture["protocol"],
            data_root=canonical_fixture["data_root"],
            output_root=tmp_path / "summary",
            device="cpu",
            batch_size=32,
            evaluator=late_mutation,
        )
    assert not any(path.exists() for path in _stream_paths(canonical_fixture))


def test_system_tables_are_exact_deterministic_and_use_parallel_critical_path(
    canonical_fixture: dict[str, Any]
) -> None:
    _install_all_audit_approved_evidence(canonical_fixture)
    tables = summary.build_system_tables(
        canonical_fixture["rows"],
        summary.load_attempt_evidence,
        protocol=canonical_fixture["protocol"],
        raw_root=canonical_fixture["raw_root"],
    )
    communication = tables["flower_communication_per_round.csv"]
    assert len(communication) == 10 * 25 * 2
    assert communication[0]["application_round_total_bytes"] == 151 + 251
    assert all(row["transport_status"] == "not_collected" for row in communication)
    communication_summary = tables["flower_communication_summary.csv"]
    assert len(communication_summary) == 10
    assert communication_summary[0]["application_25round_total_bytes"] == sum(
        row["application_round_total_bytes"]
        for row in communication
        if row["run_id"] == communication_summary[0]["run_id"]
    )

    timing = tables["flower_round_time_breakdown.csv"]
    raw = next(row for row in timing if row["statistic"] == "raw")
    assert raw["client_train_critical_path_ns"] == max(
        raw["c1_client_train_core_ns"], raw["c2_client_train_core_ns"]
    )
    assert raw["parallel_client_times_are_not_serially_additive"] is True
    assert raw["fit_round_wall_ns"] != (
        raw["c1_client_fit_callback_ns"]
        + raw["c2_client_fit_callback_ns"]
        + raw["server_aggregate_fit_total_ns"]
    )
    assert {row["statistic"] for row in timing} == {"raw", "mean", "p50", "p95", "total"}

    resource = tables["training_resource_summary.csv"]
    assert len(resource) == 20
    pi = next(row for row in resource if row["client_id"] == "C1")
    assert pi["host_role"] == "Raspberry Pi"
    assert pi["resource_coverage"] == 1.0
    assert pi["active_sample_count"] == 25
    assert pi["cpu_temperature_peak_c"] == 60.0
    assert pi["throttling_observed"] is False

    overhead = tables["observer_overhead_summary.csv"]
    total = next(
        row
        for row in overhead
        if row["attempt_id"] == canonical_fixture["rows"][0]["attempt_id"]
        and row["producer"] == "ALL"
    )
    assert total["close_observer_total_ns"] == 500
    assert total["observer_total_to_round_wall_ratio"] == pytest.approx(500 / 25_000)


def test_post_audit_evidence_rewrite_is_rejected_before_summary_consumes_it(
    canonical_fixture: dict[str, Any]
) -> None:
    _install_all_audit_approved_evidence(canonical_fixture)
    first = sorted(canonical_fixture["rows"], key=lambda row: row["run_id"])[0]
    tampered = first["attempt_dir"] / "raw" / "ecs" / "events.jsonl"
    tampered.write_bytes(tampered.read_bytes() + b"{}\n")
    consumed: list[str] = []

    def audited_loader(row: dict[str, Any]):
        evidence = summary.load_attempt_evidence(row)
        consumed.append(row["attempt_id"])
        return evidence

    with pytest.raises(RuntimeError, match="audit-approved.*SHA-256"):
        summary.build_system_tables(
            canonical_fixture["rows"],
            audited_loader,
            protocol=canonical_fixture["protocol"],
            raw_root=canonical_fixture["raw_root"],
        )
    assert consumed == []


@pytest.mark.parametrize("failure", ["missing", "extra_file", "extra_audit", "wrong_hash"])
def test_audit_input_set_and_hash_are_exact(
    canonical_fixture: dict[str, Any], failure: str
) -> None:
    row = canonical_fixture["rows"][0]
    _write_audit_approved_evidence(row, _evidence_for(row))
    event_path = row["attempt_dir"] / "raw" / "ecs" / "events.jsonl"
    if failure == "missing":
        event_path.unlink()
    elif failure == "extra_file":
        (row["attempt_dir"] / "raw" / "ecs" / "extra.jsonl").write_text(
            "{}\n", encoding="utf-8"
        )
    else:
        audit = json.loads(row["audit_path"].read_text(encoding="utf-8"))
        if failure == "extra_audit":
            audit["inputs"]["raw/ecs/extra.jsonl"] = "0" * 64
        else:
            audit["inputs"]["raw/ecs/events.jsonl"] = "0" * 64
        _rebind_audit(row, audit)

    with pytest.raises(RuntimeError, match="audit input|audit-approved"):
        summary.load_attempt_evidence(row)


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda ev: ev["server"].append(copy.deepcopy(ev["server"][0])),
            "duplicate communication",
        ),
        (
            lambda ev: ev["server"][0]["payload"]["downlink_audit"].update(
                application_message_bytes=-1
            ),
            "nonnegative",
        ),
        (
            lambda ev: ev["server"][0]["payload"]["downlink_audit"].update(
                application_message_bytes=math.nan
            ),
            "finite",
        ),
    ],
)
def test_system_tables_fail_closed_on_duplicate_or_bad_measurement(
    canonical_fixture: dict[str, Any], mutator, message: str
) -> None:
    evidence = {row["attempt_id"]: _evidence_for(row) for row in canonical_fixture["rows"]}
    mutator(evidence[canonical_fixture["rows"][0]["attempt_id"]])
    with pytest.raises(RuntimeError, match=message):
        summary.build_system_tables(
            canonical_fixture["rows"], lambda row: evidence[row["attempt_id"]]
        )


def test_csv_serialization_is_canonical_and_rejects_schema_drift() -> None:
    fields = ("run_id", "value")
    rows = [{"run_id": "a", "value": 1.25}, {"run_id": "b", "value": 2.5}]
    first = summary.csv_bytes(rows, fields)
    second = summary.csv_bytes(copy.deepcopy(rows), fields)
    assert first == second
    with pytest.raises(RuntimeError, match="CSV schema"):
        summary.csv_bytes([{"run_id": "a", "value": 1, "extra": 2}], fields)


def test_exclusive_summary_output_records_inputs_and_claim_boundaries(
    canonical_fixture: dict[str, Any], tmp_path: Path
) -> None:
    _install_all_audit_approved_evidence(canonical_fixture)
    rows = summary.assert_test_gate(
        canonical_fixture["rows"],
        protocol=canonical_fixture["protocol"],
        raw_root=canonical_fixture["raw_root"],
    )
    tables = summary.build_system_tables(
        rows,
        summary.load_attempt_evidence,
        protocol=canonical_fixture["protocol"],
        raw_root=canonical_fixture["raw_root"],
    )
    output = tmp_path / "summary"
    with summary.StreamTransaction(rows) as transaction:
        per_run = summary.evaluate_canonical_attempts(
            rows,
            data_root=canonical_fixture["data_root"],
            device="cpu",
            batch_size=32,
            evaluator=_successful_evaluator(canonical_fixture),
            protocol=canonical_fixture["protocol"],
            raw_root=canonical_fixture["raw_root"],
            stream_transaction=transaction,
        )
        summary.write_summary_bundle(
            output,
            rows,
            canonical_fixture["protocol"],
            per_run,
            tables,
            raw_root=canonical_fixture["raw_root"],
            stream_transaction=transaction,
        )
    expected = {
        "classification_per_run.csv",
        "classification_multiseed_summary.csv",
        "flower_communication_per_round.csv",
        "flower_communication_summary.csv",
        "flower_round_time_breakdown.csv",
        "training_resource_summary.csv",
        "observer_overhead_summary.csv",
        "attempt_registry.csv",
        "summary_input_manifest.json",
        "claim_boundary.md",
        "claim_to_evidence_map.md",
    }
    assert expected <= {path.name for path in output.iterdir()}
    boundary = (output / "claim_boundary.md").read_text(encoding="utf-8")
    assert "post-screen exploratory" in boundary
    assert "predeclared full method" in boundary
    assert "feaa75b" in boundary
    assert "not collected" in boundary.lower()
    input_manifest = json.loads(
        (output / "summary_input_manifest.json").read_text(encoding="utf-8")
    )
    assert input_manifest["protocol_manifest_sha256"] == canonical_fixture["protocol"][
        "protocol_manifest_sha256"
    ]
    assert len(input_manifest["attempts"]) == 10
    for attempt in input_manifest["attempts"]:
        approved = attempt["audit_approved_evidence_inputs"]
        consumed = attempt["consumed_evidence_inputs"]
        assert len(approved) == 10
        assert approved == consumed
    with pytest.raises(FileExistsError, match="overwrite"):
        summary.write_summary_bundle(
            output,
            canonical_fixture["rows"],
            canonical_fixture["protocol"],
            per_run,
            tables,
            raw_root=canonical_fixture["raw_root"],
            stream_transaction=transaction,
        )


def test_nonfinite_classification_metric_is_rejected(
    canonical_fixture: dict[str, Any]
) -> None:
    row = canonical_fixture["rows"][0]
    metrics = _fake_metrics(row["group_id"], row["seed"])
    metrics["ece"] = math.inf
    with pytest.raises(RuntimeError, match="finite"):
        summary.classification_row(row, metrics)
