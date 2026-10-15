from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.run_iotj_a0t_vs_a4_regression import (
    EXPECTED_DATASET_SHA256,
    _test_manifest_hashes,
    EndpointSpec,
    audit_checkpoint,
    audit_endpoint_pair,
    audit_inputs,
    build_four_scopes,
    endpoint_specs,
    execute_study,
    fit_fixed_alpha_models,
    frozen_alphas,
    orchestrate_sealed_run,
    special_slice_rows,
    summarize_scope,
)


def test_target_test_manifest_hashes_exact_canonical_files() -> None:
    hashes = _test_manifest_hashes("C3")
    assert set(hashes) == {
        "test_features.npy",
        "test_classification_labels.npy",
        "test_regression_labels.npy",
        "test_phase_labels.npy",
        "test_experiment_info.json",
    }


def test_exact_six_endpoints_and_checkpoint_is_only_method_factor() -> None:
    specs = endpoint_specs()
    assert [(spec.method, spec.target) for spec in specs] == [
        ("A0T", "C3"),
        ("A0T", "C4"),
        ("A0T", "C5"),
        ("A4", "C3"),
        ("A4", "C4"),
        ("A4", "C5"),
    ]
    for target in ("C3", "C4", "C5"):
        a0t = next(spec for spec in specs if spec.method == "A0T" and spec.target == target)
        a4 = next(spec for spec in specs if spec.method == "A4" and spec.target == target)
        assert audit_endpoint_pair(a0t, a4) == {
            "status": "PASS",
            "target": target,
            "varying_fields": ["checkpoint", "checkpoint_sha256", "classification_manifest", "completion_marker", "experiment_id", "method"],
        }


def test_pair_audit_rejects_non_checkpoint_protocol_drift() -> None:
    a0t, a4 = [spec for spec in endpoint_specs() if spec.target == "C5"]
    drifted = replace(a4, split_protocol="changed")
    with pytest.raises(RuntimeError, match="held-constant drift"):
        audit_endpoint_pair(a0t, drifted)


def test_frozen_alpha_table_is_exact_and_method_independent() -> None:
    assert frozen_alphas() == {
        "C3": {0: 100.0, 1: 0.0, 2: 0.1, 3: 0.1},
        "C4": {0: 1.0, 1: 10.0, 2: 0.1, 3: 10.0},
        "C5": {0: 1.0, 1: 0.01, 2: 10.0, 3: 0.1},
    }
    assert EXPECTED_DATASET_SHA256 == "2f810d7e93cae5f361923184e9dc87d5ae59e0f59be9f52aff7e14f9f33e94f6"


def test_endpoint_paths_resolve_to_existing_frozen_assets() -> None:
    for spec in endpoint_specs():
        assert isinstance(spec, EndpointSpec)
        assert Path(spec.checkpoint).is_file()
        assert Path(spec.classification_manifest).is_file()
        assert Path(spec.completion_marker).is_file()


def test_audit_inputs_writes_sealed_freeze_and_registry(tmp_path: Path) -> None:
    result = audit_inputs(tmp_path / "study")
    assert result["status"] == "PASS"
    assert result["endpoint_count"] == 6
    assert result["target_test_state"] == "SEALED"
    assert result["alpha_selection_performed"] is False
    assert (tmp_path / "study" / "PRE_RUN_FREEZE.json").is_file()
    assert (tmp_path / "study" / "experiment_registry.csv").is_file()


def test_checkpoint_audit_rejects_wrong_registered_hash() -> None:
    spec = endpoint_specs()[0]
    tampered = replace(spec, checkpoint_sha256="0" * 64)
    with pytest.raises(RuntimeError, match="checkpoint SHA256 differs"):
        audit_checkpoint(tampered)


def test_fixed_alpha_fit_uses_exact_target_values() -> None:
    rows = []
    for class_id in range(4):
        for index, ppm in enumerate((10.0, 20.0, 30.0)):
            rows.append(
                {
                    "sample_index": class_id * 10 + index,
                    "true_class": class_id,
                    "true_ppm": ppm,
                    "feature_dict": {"sensor": ppm, "srcpred_H1_federated_source_ridge_ppm": ppm + 1.0},
                }
            )
    models = fit_fixed_alpha_models("C5", rows)
    assert {class_id: model.alpha for class_id, model in models.items()} == {
        0: 1.0,
        1: 0.01,
        2: 10.0,
        3: 0.1,
    }


class _EchoModel:
    def predict(self, rows, clip=True):
        return [float(row["feature_dict"]["value"]) for row in rows]


def test_oracle_cc_uses_exactly_the_s_cc_sample_indices() -> None:
    deployment = [
        {"sample_index": 0, "true_class": 0, "pred_class": 0, "true_ppm": 10.0, "feature_dict": {"value": 9.0}},
        {"sample_index": 1, "true_class": 1, "pred_class": 2, "true_ppm": 20.0, "feature_dict": {"value": 18.0}},
        {"sample_index": 2, "true_class": 2, "pred_class": 2, "true_ppm": 30.0, "feature_dict": {"value": 31.0}},
    ]
    oracle = [
        {**row, "pred_class": row["true_class"], "feature_dict": {"value": row["true_ppm"]}}
        for row in deployment
    ]
    scopes = build_four_scopes(deployment, oracle, {index: _EchoModel() for index in range(4)})
    assert [row["sample_index"] for row in scopes["S_ALL"]] == [0, 1, 2]
    assert [row["sample_index"] for row in scopes["S_CC"]] == [0, 2]
    assert [row["sample_index"] for row in scopes["Oracle_ALL"]] == [0, 1, 2]
    assert [row["sample_index"] for row in scopes["Oracle_CC"]] == [0, 2]


def test_scope_metrics_include_bias_and_range_nrmse() -> None:
    rows = [
        {"true_class": 0, "true_ppm": 10.0, "pred_ppm": 9.0},
        {"true_class": 0, "true_ppm": 20.0, "pred_ppm": 21.0},
    ]
    metrics = summarize_scope(rows)
    assert metrics["N"] == 2
    assert metrics["RMSE"] == pytest.approx(1.0)
    assert metrics["MAE"] == pytest.approx(1.0)
    assert metrics["NRMSE_range"] == pytest.approx(1.0 / 112.5)
    assert metrics["R2"] == pytest.approx(0.96)
    assert metrics["Bias"] == pytest.approx(0.0)


def test_all_calibration_locks_precede_first_test_access(tmp_path: Path) -> None:
    events = []

    def fit_stage(spec, endpoint_dir):
        events.append(("fit", spec.experiment_id))
        endpoint_dir.mkdir(parents=True)
        lock = endpoint_dir / "calibration_lock.json"
        lock.write_text('{"status":"SEALED_BEFORE_TARGET_TEST"}', encoding="utf-8")
        return lock

    def test_stage(spec, endpoint_dir):
        assert len(list((tmp_path / "endpoints").glob("*/calibration_lock.json"))) == 6
        events.append(("test", spec.experiment_id))

    orchestrate_sealed_run(endpoint_specs(), tmp_path, fit_stage, test_stage)
    assert [phase for phase, _ in events] == ["fit"] * 6 + ["test"] * 6


def test_c5_methane_225_repeat1_slice_is_exact() -> None:
    rows = [
        {"gas": "methane", "true_ppm": 225.0, "repeat_id": 1, "sample_index": 10},
        {"gas": "methane", "true_ppm": 225.0, "repeat_id": 2, "sample_index": 11},
        {"gas": "carbon_monoxide", "true_ppm": 225.0, "repeat_id": 1, "sample_index": 12},
    ]
    selected = special_slice_rows("C5", rows)
    assert [row["sample_index"] for row in selected] == [10]


def test_execute_refuses_without_pre_run_freeze(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="PRE_RUN_FREEZE"):
        execute_study(tmp_path / "missing", "cpu", 32)


def test_direct_audit_only_entry_creates_no_endpoint(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "audit"
    result = subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "run_iotj_a0t_vs_a4_regression.py"),
            "--audit-only",
            "--output",
            str(output),
        ],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert (output / "PRE_RUN_FREEZE.json").is_file()
    assert not (output / "endpoints").exists()
