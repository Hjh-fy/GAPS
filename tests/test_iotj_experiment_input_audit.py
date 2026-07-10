import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.audit_iotj_experiment_inputs import audit_inputs, summarize_primary_dataset


def test_audit_inputs_records_resolved_path_hash_and_role(tmp_path: Path) -> None:
    artifact = tmp_path / "results" / "h2_3_stream.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text('{"profile": "H2.3+"}\n', encoding="utf-8")

    manifest = audit_inputs([artifact])

    entry = manifest["artifacts"][0]
    assert entry["resolved_path"] == str(artifact.resolve())
    assert entry["role"] == "p4_h2_3_stream"
    assert entry["exists"] is True
    assert entry["byte_size"] == artifact.stat().st_size
    assert entry["sha256"] == hashlib.sha256(artifact.read_bytes()).hexdigest()
    assert entry["status"] == "present"


def test_audit_inputs_marks_missing_matrix_requirements(tmp_path: Path) -> None:
    matrix_root = tmp_path / "source_target_classification_matrix_20260708_clean"
    run_dir = matrix_root / "F4"
    run_dir.mkdir(parents=True)
    (run_dir / "run_config.json").write_text("{}\n", encoding="utf-8")

    manifest = audit_inputs([matrix_root])

    run = manifest["matrix_runs"][0]
    assert run["resolved_path"] == str(run_dir.resolve())
    assert run["status"] == "missing"
    assert run["missing_required"] == ["server_latest_adapted.pth"]
    checkpoint = next(
        entry
        for entry in manifest["artifacts"]
        if entry["resolved_path"] == str((run_dir / "server_latest_adapted.pth").resolve())
    )
    assert checkpoint["role"] == "matrix_checkpoint"
    assert checkpoint["exists"] is False
    assert checkpoint["byte_size"] == 0
    assert checkpoint["sha256"] is None
    assert checkpoint["status"] == "missing"


def test_audit_inputs_uses_full_priority_matrix_directory_names_without_duplicates(
    tmp_path: Path,
) -> None:
    matrix_root = tmp_path / "source_target_classification_matrix_20260708_clean"
    full_run_names = [
        "F4_C1234_to_C5_fixed_da_strong_r25",
        "F5_C1_to_C2345_fixed_da_strong_r25",
        "R1_C5_to_C1_fixed_da_strong_r25",
        "R2_C45_to_C1_fixed_da_strong_r25",
        "R3_C345_to_C1_fixed_da_strong_r25",
        "R4_C2345_to_C1_fixed_da_strong_r25",
    ]
    for run_name in full_run_names:
        run_dir = matrix_root / run_name
        run_dir.mkdir(parents=True)
        (run_dir / "run_config.json").write_text("{}\n", encoding="utf-8")
        (run_dir / "server_latest_adapted.pth").write_bytes(b"checkpoint")
    h23_stream = tmp_path / "h2_3_stream.json"
    h23_stream.write_text("{}\n", encoding="utf-8")

    manifest = audit_inputs([h23_stream, matrix_root])

    assert manifest["status"] == "complete"
    assert [run["run_id"] for run in manifest["matrix_runs"]] == full_run_names
    assert all(run["status"] == "complete" for run in manifest["matrix_runs"])
    assert not {"F4", "F5", "R1", "R2", "R3", "R4"}.intersection(
        run["run_id"] for run in manifest["matrix_runs"]
    )


def test_summarize_primary_dataset_counts_target_calibration_by_class_and_concentration(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "dataset"
    client_dir = data_root / "client_3"
    client_dir.mkdir(parents=True)
    np.save(client_dir / "calibration_features.npy", np.zeros((3, 2), dtype=np.float32))
    np.save(client_dir / "calibration_classification_labels.npy", np.asarray([0, 0, 1]))
    np.save(
        client_dir / "calibration_regression_labels.npy",
        np.asarray([[10.0, 0.0], [20.0, 0.0], [0.0, 50.0]], dtype=np.float32),
    )
    (data_root / "split_protocol_manifest.json").write_text(
        json.dumps({"seed": 42, "target_ratios": {"calibration": 0.2, "test": 0.8}}),
        encoding="utf-8",
    )

    summary = summarize_primary_dataset(data_root)

    split = summary["clients"]["C3"]["calibration"]
    assert summary["split_seed"] == 42
    assert summary["calibration_fit_ratio"] == 0.75
    assert summary["calibration_validation_ratio"] == 0.25
    assert split["role"] == "target_calibration"
    assert split["sample_count"] == 3
    assert split["class_counts"] == {"0": 2, "1": 1}
    assert split["concentration_counts"] == {"0": {"10.0": 1, "20.0": 1}, "1": {"50.0": 1}}
