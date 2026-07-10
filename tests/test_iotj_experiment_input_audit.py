import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.audit_iotj_experiment_inputs as audit


F2_RUN_NAME = "F2_C12_to_C5_fixed_da_strong_r25"


def _write_split(client_dir: Path, split: str, count: int) -> None:
    client_dir.mkdir(parents=True, exist_ok=True)
    labels = np.arange(count, dtype=np.int64) % 4
    regression = np.zeros((count, 4), dtype=np.float32)
    for index, class_id in enumerate(labels.tolist()):
        regression[index, class_id] = float(10 * (1 + ((index // 4) % 2)))
    np.save(client_dir / f"{split}_features.npy", np.zeros((count, 2), dtype=np.float32))
    np.save(client_dir / f"{split}_classification_labels.npy", labels)
    np.save(client_dir / f"{split}_regression_labels.npy", regression)


def _valid_split_info() -> dict[str, object]:
    return {
        "protocol": "stale_c12_source_c345_target_label",
        "source_clients": [1, 2, 3, 4],
        "target_clients": [5],
        "target_split": {"train_used": False, "calibration": 0.2, "test": 0.8},
        "stratify_by": ["client", "class", "concentration"],
        "seed": 42,
    }


def _write_complete_dataset(root: Path, include_inactive: bool = True) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "split_info.json").write_text(
        json.dumps(_valid_split_info(), indent=2),
        encoding="utf-8",
    )
    for client_id in (1, 2):
        for split in ("train", "calibration", "test"):
            _write_split(root / f"client_{client_id}", split, 8)
    _write_split(root / "client_5", "calibration", 320)
    _write_split(root / "client_5", "test", 1360)
    if include_inactive:
        for client_id in (3, 4):
            _write_split(root / f"client_{client_id}", "calibration", 4)


def _valid_run_config(data_root: Path) -> dict[str, object]:
    return {
        "args": {
            "run_name": F2_RUN_NAME,
            "rounds": 25,
            "strategy": "gaps",
            "profile": "strong_cls",
            "server_val_data": f"{data_root / 'client_1'},{data_root / 'client_2'}",
            "server_calib_data": str(data_root / "client_5"),
            "use_domain_adapt": True,
            "domain_adapt_steps": 100,
            "da_lambda_target_ce": 0.0,
            "use_adapted_as_global": True,
        }
    }


def _write_complete_run(run_dir: Path, data_root: Path, config: dict[str, object] | None = None) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = config if config is not None else _valid_run_config(data_root)
    (run_dir / "run_config.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (run_dir / "history.json").write_text('{"rounds": []}\n', encoding="utf-8")
    (run_dir / "server_latest_adapted.pth").write_bytes(b"adapted-checkpoint")


def test_audit_inputs_records_resolved_path_hash_and_role(tmp_path: Path) -> None:
    artifact = tmp_path / "results" / "h2_3_stream.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text('{"profile": "H2.3+"}\n', encoding="utf-8")

    manifest = audit.audit_inputs([artifact])

    entry = manifest["artifacts"][0]
    assert entry == {
        "resolved_path": str(artifact.resolve()),
        "role": "p4_h2_3_stream",
        "exists": True,
        "byte_size": artifact.stat().st_size,
        "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "status": "present",
    }


def test_explicit_arbitrary_run_dir_fails_when_required_file_is_missing(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    run_dir = tmp_path / "recovered" / "arbitrary-name"
    _write_complete_run(run_dir, data_root)
    (run_dir / "server_latest_adapted.pth").unlink()

    result = audit.audit_matrix_run(run_dir, audit.F2_RUN_EXPECTED)

    assert result["status"] == "incomplete"
    assert result["missing_required"] == ["server_latest_adapted.pth"]
    missing = next(
        entry for entry in result["artifacts"] if entry["resolved_path"].endswith("server_latest_adapted.pth")
    )
    assert missing["role"] == "matrix_checkpoint"
    assert missing["exists"] is False
    assert missing["sha256"] is None
    assert missing["status"] == "missing"


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    [
        (lambda info: info.update(target_clients=[3, 4, 5]), "target_clients must equal [5]"),
        (lambda info: info.update(seed=7), "seed must equal 42"),
        (lambda info: info["target_split"].update(train_used=True), "target_split.train_used must be false"),
        (lambda info: info["target_split"].update(calibration=0.25), "target_split.calibration must equal 0.2"),
        (lambda info: info["target_split"].update(test=0.75), "target_split.test must equal 0.8"),
        (lambda info: info.update(stratify_by=["client", "class"]), "stratify_by must contain"),
        (lambda info: info.update(source_clients=[1, 3, 4]), "source_clients must contain [1, 2]"),
    ],
)
def test_wrong_structured_dataset_metadata_fails(
    tmp_path: Path,
    mutation,
    expected_error: str,
) -> None:
    data_root = tmp_path / "dataset"
    _write_complete_dataset(data_root)
    info = _valid_split_info()
    mutation(info)
    (data_root / "split_info.json").write_text(json.dumps(info), encoding="utf-8")

    summary = audit.summarize_primary_dataset(data_root)

    assert summary["status"] == "incomplete"
    assert any(expected_error in error for error in summary["validation_errors"])


def test_missing_dataset_metadata_fails(tmp_path: Path) -> None:
    data_root = tmp_path / "dataset"
    _write_complete_dataset(data_root)
    (data_root / "split_info.json").unlink()

    summary = audit.summarize_primary_dataset(data_root)

    assert summary["status"] == "incomplete"
    assert summary["validation_errors"] == ["missing metadata: split_info.json"]


def test_missing_active_split_and_wrong_c5_count_fail(tmp_path: Path) -> None:
    data_root = tmp_path / "dataset"
    _write_complete_dataset(data_root)
    (data_root / "client_2" / "train_features.npy").unlink()
    _write_split(data_root / "client_5", "test", 1359)

    summary = audit.summarize_primary_dataset(data_root)

    assert summary["status"] == "incomplete"
    assert any("C2/train missing train_features.npy" in error for error in summary["validation_errors"])
    assert any("C5/test sample_count must equal 1360" in error for error in summary["validation_errors"])


def test_correct_c12_to_c5_dataset_audits_only_active_clients(tmp_path: Path) -> None:
    data_root = tmp_path / "dataset"
    _write_complete_dataset(data_root, include_inactive=True)

    summary = audit.summarize_primary_dataset(data_root)

    assert summary["status"] == "complete"
    assert summary["active_source_clients"] == ["C1", "C2"]
    assert summary["active_target_clients"] == ["C5"]
    assert list(summary["clients"]) == ["C1", "C2", "C5"]
    assert summary["clients"]["C5"]["calibration"]["sample_count"] == 320
    assert summary["clients"]["C5"]["test"]["sample_count"] == 1360
    assert summary["clients"]["C5"]["calibration"]["class_counts"] == {
        "0": 80,
        "1": 80,
        "2": 80,
        "3": 80,
    }
    assert summary["validation_errors"] == []
    assert any("stale" in warning for warning in summary["validation_warnings"])


@pytest.mark.parametrize(
    ("key", "value", "expected_error"),
    [
        ("run_name", "F5_C1_to_C2345_fixed_da_strong_r25", "run_name must equal"),
        ("rounds", 24, "rounds must equal 25"),
        ("strategy", "fedavg", "strategy must equal gaps"),
        ("profile", "smoke", "profile must equal strong_cls"),
        ("server_val_data", "x/client_1,x/client_3", "server_val_data clients must equal [1, 2]"),
        ("server_calib_data", "x/client_3,x/client_4", "server_calib_data clients must equal [5]"),
        ("use_domain_adapt", False, "use_domain_adapt must equal true"),
        ("domain_adapt_steps", 99, "domain_adapt_steps must equal 100"),
        ("da_lambda_target_ce", 0.1, "da_lambda_target_ce must equal 0.0"),
        ("use_adapted_as_global", False, "use_adapted_as_global must equal true"),
    ],
)
def test_wrong_f2_run_config_fails(
    tmp_path: Path,
    key: str,
    value: object,
    expected_error: str,
) -> None:
    data_root = tmp_path / "dataset"
    run_dir = tmp_path / F2_RUN_NAME
    config = _valid_run_config(data_root)
    config["args"][key] = value
    _write_complete_run(run_dir, data_root, config)

    result = audit.audit_matrix_run(run_dir, audit.F2_RUN_EXPECTED)

    assert result["status"] == "incomplete"
    assert any(expected_error in error for error in result["validation_errors"])


def test_extra_run_directories_do_not_affect_required_f2_completion(tmp_path: Path) -> None:
    data_root = tmp_path / "dataset"
    matrix_root = tmp_path / "matrix"
    run_dir = matrix_root / F2_RUN_NAME
    _write_complete_dataset(data_root)
    _write_complete_run(run_dir, data_root)
    (matrix_root / "F5_bad_incomplete").mkdir(parents=True)

    manifest = audit.build_manifest(data_root, run_dir)

    assert manifest["status"] == "complete"
    assert [run["run_id"] for run in manifest["required_runs"]] == [F2_RUN_NAME]
    assert manifest["required_runs"][0]["status"] == "complete"
    assert manifest["extra_run_inventory"] == [
        {
            "run_id": "F5_bad_incomplete",
            "resolved_path": str((matrix_root / "F5_bad_incomplete").resolve()),
            "role": "inventory_only",
            "file_count": 0,
            "status": "inventory_only",
        }
    ]


def test_complete_manifest_excludes_inactive_client_artifacts(tmp_path: Path) -> None:
    data_root = tmp_path / "dataset"
    run_dir = tmp_path / "matrix" / F2_RUN_NAME
    _write_complete_dataset(data_root, include_inactive=True)
    _write_complete_run(run_dir, data_root)

    manifest = audit.build_manifest(data_root, run_dir)

    assert manifest["status"] == "complete"
    assert manifest["protocol"]["active_clients"] == ["C1", "C2", "C5"]
    assert manifest["validation_errors"] == []
    assert len(manifest["required_runs"]) == 1
    paths = [entry["resolved_path"].replace("\\", "/") for entry in manifest["artifacts"]]
    assert not any("/client_3/" in path or "/client_4/" in path for path in paths)


@pytest.mark.parametrize("collision_kind", ["artifact", "dataset_child", "run_child"])
def test_output_collision_or_containment_is_rejected_before_writing(
    tmp_path: Path,
    collision_kind: str,
) -> None:
    data_root = tmp_path / "dataset"
    run_dir = tmp_path / "run"
    data_root.mkdir()
    run_dir.mkdir()
    artifact = tmp_path / "input.json"
    artifact.write_text("original\n", encoding="utf-8")
    output = {
        "artifact": artifact,
        "dataset_child": data_root / "manifest.json",
        "run_child": run_dir / "manifest.json",
    }[collision_kind]

    with pytest.raises(ValueError, match="output path overlaps protected input"):
        audit.write_manifest(
            {"status": "complete"},
            output,
            protected_inputs=[artifact, data_root, run_dir],
        )

    assert artifact.read_text(encoding="utf-8") == "original\n"
    if output != artifact:
        assert not output.exists()


def test_stable_manifest_payload_excludes_provenance_timestamp(tmp_path: Path) -> None:
    data_root = tmp_path / "dataset"
    run_dir = tmp_path / "matrix" / F2_RUN_NAME
    _write_complete_dataset(data_root)
    _write_complete_run(run_dir, data_root)

    first = audit.build_manifest(data_root, run_dir)
    second = audit.build_manifest(data_root, run_dir)
    with_time_a = audit.with_provenance(first, generated_at_utc="2026-07-11T00:00:00+00:00")
    with_time_b = audit.with_provenance(first, generated_at_utc="2026-07-11T01:00:00+00:00")

    assert first == second
    assert "provenance" not in first
    assert with_time_a["provenance"] == {"generated_at_utc": "2026-07-11T00:00:00+00:00"}
    assert with_time_b["provenance"] == {"generated_at_utc": "2026-07-11T01:00:00+00:00"}
    stable_a = copy.deepcopy(with_time_a)
    stable_b = copy.deepcopy(with_time_b)
    stable_a.pop("provenance")
    stable_b.pop("provenance")
    assert stable_a == stable_b == first
