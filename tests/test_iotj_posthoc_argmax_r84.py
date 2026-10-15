"""Protocol tests for Phase-3 post-hoc argmax R84 baseline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_phase3_registered_selection_uses_simplest_effective_b20_identity() -> None:
    from scripts.run_iotj_posthoc_argmax_r84 import select_phase3_identity

    metrics = {"I0": 0.9765440505346443, "I1": 0.966918332676123, "I2": 0.9838205427751326}
    selected = select_phase3_identity("DG_TO_COMMISSIONING_NOT_SUPPORTED", metrics)
    assert selected["identity"] == "I0"
    assert selected["budget"] == 20
    assert selected["selection_rule"] == "simplest_effective_within_0.01_of_best_B20"
    assert selected["target_test_checkpoint_selection"] is False


def test_phase3_supported_dg_decision_would_select_i2_without_metric_search() -> None:
    from scripts.run_iotj_posthoc_argmax_r84 import select_phase3_identity

    selected = select_phase3_identity(
        "DG_TO_COMMISSIONING_SUPPORTED", {"I0": 0.99, "I1": 0.98, "I2": 0.90}
    )
    assert selected["identity"] == "I2"
    assert selected["selection_rule"] == "registered_dg_commissioning_decision"


def test_phase3_classifier_audit_resolves_exact_i0_b20_g1_endpoint() -> None:
    from scripts.run_iotj_posthoc_argmax_r84 import audit_selected_classifier

    audit = audit_selected_classifier("I0")
    assert audit["status"] == "PASS"
    assert audit["identity"] == "I0"
    assert audit["budget"] == 20
    assert audit["step"] == 100
    assert audit["target_test_opened_before_lock"] is False
    assert audit["classifier_training_performed"] is False


def test_phase3_h1_pool_and_c5_alphas_are_frozen() -> None:
    from scripts.run_iotj_posthoc_argmax_r84 import audit_r84_inputs

    audit = audit_r84_inputs()
    assert audit["status"] == "PASS"
    assert audit["h1_sha256"] == "d32217a30f491ba46be436f3baf469b764b54a08d4d542b4eb71dbc007338ecc"
    assert audit["alphas"] == {"0": 1.0, "1": 0.01, "2": 10.0, "3": 0.1}
    assert audit["alpha_selection_performed"] is False


def test_phase3_calibration_lock_rejects_test_open(tmp_path: Path) -> None:
    from scripts.run_iotj_posthoc_argmax_r84 import verify_calibration_lock

    model = tmp_path / "r84_models.json"
    model.write_text("{}\n", encoding="utf-8")
    import hashlib
    lock = tmp_path / "calibration_lock.json"
    lock.write_text(
        json.dumps(
            {
                "status": "SEALED_BEFORE_TARGET_TEST",
                "target_test_opened": False,
                "alpha_selection_performed": False,
                "fixed_alphas": {"0": 1.0, "1": 0.01, "2": 10.0, "3": 0.1},
                "r84_models_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    assert verify_calibration_lock(lock, model)["status"] == "PASS"
    payload = json.loads(lock.read_text(encoding="utf-8"))
    payload["target_test_opened"] = True
    lock.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="calibration lock"):
        verify_calibration_lock(lock, model)


def test_route_rows_forwards_explicit_posthoc_endpoint(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from scripts import run_iotj_canonical_v1_r84 as module

    observed = {}

    def fake_evaluate(_checkpoint, **kwargs):
        observed.update(kwargs)
        return ([{"sample_index": 0}], {"accuracy": 1.0})

    monkeypatch.setattr(module, "evaluate_checkpoint_stream", fake_evaluate)
    rows, _metrics = module.route_rows(
        tmp_path / "posthoc.pth",
        "C5",
        "calibration",
        __import__("torch").device("cpu"),
        32,
        expected_endpoint=("step", 100),
    )
    assert rows == [{"sample_index": 0}]
    assert observed["expected_endpoint"] == ("step", 100)


def test_phase3_finalizer_uses_scope_column_without_reopening_test(tmp_path: Path) -> None:
    """A report-only retry must consume completed predictions, not evaluate again."""
    import csv
    import hashlib

    from scripts.run_iotj_posthoc_argmax_r84 import finalize_existing_evaluation

    endpoint = tmp_path / "endpoint"
    endpoint.mkdir()
    model = endpoint / "r84_models.json"
    model.write_text("{}\n", encoding="utf-8")
    model_sha = hashlib.sha256(model.read_bytes()).hexdigest()
    (endpoint / "calibration_lock.json").write_text(
        json.dumps(
            {
                "status": "SEALED_BEFORE_TARGET_TEST",
                "target_test_opened": False,
                "alpha_selection_performed": False,
                "fixed_alphas": {"0": 1.0, "1": 0.01, "2": 10.0, "3": 0.1},
                "r84_models_sha256": model_sha,
            }
        ),
        encoding="utf-8",
    )
    classification = {"accuracy": 0.98, "macro_f1": 0.97}
    (endpoint / "endpoint_manifest.json").write_text(
        json.dumps(
            {
                "status": "COMPLETE",
                "experiment_id": "CAN-V1-MB-P3-POSTHOC-R84-S42",
                "checkpoint_sha256": "classifier-sha",
                "test_classification": classification,
                "target_test_used_for_selection": False,
            }
        ),
        encoding="utf-8",
    )
    rows = [
        {
            "experiment_id": "CAN-V1-MB-P3-POSTHOC-R84-S42",
            "method": "Posthoc-A0T",
            "target": "C5",
            "scope": scope,
            "seed": 42,
            "N": 100,
            "RMSE": 1.0,
            "MAE": 0.8,
            "NRMSE_range": 0.01,
            "R2": 0.9,
            "Bias": 0.1,
        }
        for scope in ("S_ALL", "S_CC", "Oracle_ALL", "Oracle_CC")
    ]
    with (tmp_path / "POSTHOC_ARGMAX_BASELINE.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for name in ("POSTHOC_ARGMAX_PER_GAS.csv", "POSTHOC_ARGMAX_PER_CONCENTRATION.csv"):
        (tmp_path / name).write_text("scope,value\nS_ALL,1\n", encoding="utf-8")
    (tmp_path / "SEALED_TEST_OPEN.json").write_text(
        json.dumps({"status": "OPENED_AFTER_PHASE3_CALIBRATION_LOCK"}), encoding="utf-8"
    )
    freeze = {
        "selection": {"identity": "I0", "selection_rule": "registered"},
        "classifier": {"checkpoint_sha256": "classifier-sha"},
        "r84": {"alphas": {"0": 1.0, "1": 0.01, "2": 10.0, "3": 0.1}},
    }

    result = finalize_existing_evaluation(tmp_path, freeze)

    assert result["status"] == "PASS"
    assert (endpoint / "fixed_endpoint_complete.json").is_file()
    assert "| S_ALL | 100 |" in (tmp_path / "POSTHOC_ARGMAX_BASELINE_REPORT.md").read_text(encoding="utf-8")
