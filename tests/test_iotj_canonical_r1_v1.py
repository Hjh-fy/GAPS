import numpy as np
import pytest
import json
from pathlib import Path
import subprocess, sys

from gaps_flower.canonical_r1_v1 import (
    assign_balanced_group_folds,
    bootstrap_paired_group_deltas,
    compute_route_metrics,
    decide_r84,
    select_grouped_cv_alpha,
    validate_r0_prerequisite,
    validate_classifier_registry,
    assert_test_access_released,
    validate_evidence_bundle,
    predicted_classes_from_rows,
)


def test_group_folds_are_stable_balanced_and_disjoint():
    groups = np.array(["z", "a", "a", "b", "c", "d", "e", "f"])
    first = assign_balanced_group_folds(groups, n_folds=5)
    second = assign_balanced_group_folds(groups, n_folds=5)
    assert np.array_equal(first, second)
    for group in np.unique(groups):
        assert len(set(first[groups == group])) == 1
    counts = np.bincount(first, minlength=5)
    assert counts.max() - counts.min() <= 1


def test_target_alpha_selector_has_no_test_api_and_first_tie():
    x = np.arange(24, dtype=float).reshape(8, 3)
    y = np.ones(8)
    groups = np.array(list("aabbccdd"))
    result = select_grouped_cv_alpha(x, y, groups, [0.0, 0.01], n_folds=2)
    assert result.alpha == 0.0
    assert set(result.fold_by_group) == set(groups)
    with pytest.raises(TypeError):
        select_grouped_cv_alpha(x, y, groups, [0.0], test_x=x)  # type: ignore


def test_route_scopes_are_exact():
    truth = np.array([10.0, 20.0, 30.0])
    true_class = np.array([0, 1, 0])
    predicted_class = np.array([0, 0, 0])
    predictions = np.array([[10.0, 100.0], [19.0, 20.0], [31.0, 300.0]])
    metrics = compute_route_metrics(truth, true_class, predicted_class, predictions, {0: 100.0, 1: 100.0})
    assert metrics["S_ALL"].n == 3
    assert metrics["S_CC"].n == 2
    assert metrics["Oracle_ALL"].n == 3
    assert metrics["Oracle_CC"].n == 2
    assert metrics["S_CC"].rmse == metrics["Oracle_CC"].rmse


def test_bootstrap_resamples_whole_paired_groups():
    groups = np.array(["C3|a", "C3|a", "C3|b", "C3|b"])
    truth = np.array([0.0, 1.0, 2.0, 3.0])
    p83 = np.array([0.0, 2.0, 2.0, 4.0])
    p84 = np.array([0.0, 1.0, 2.0, 3.0])
    out = bootstrap_paired_group_deltas(truth, p83, p84, groups, np.ones(4), 25, 42)
    assert len(out["rmse_delta"]) == 25
    assert np.all(np.asarray(out["rmse_delta"]) <= 0)


def test_registered_decision_rule_and_severe_collapse():
    assert decide_r84(-1.0, (-2.0, -0.1), {"C3": -0.1}, {"CO": -0.1}, {}) ["decision"] == "CANONICAL_R84_SUPPORTED"
    d = decide_r84(-1.0, (-2.0, 0.1), {"C3": 0.2}, {"CO": -0.1}, {"C3": (10.0, 10.6)})
    assert d["decision"] == "CANONICAL_R84_DEVICE_DEPENDENT"
    assert d["severe_collapse"] == ["C3"]
    assert decide_r84(0.0, (-1.0, 1.0), {}, {}, {})["decision"] == "CANONICAL_R84_NOT_SUPPORTED"


def test_r0_v2_prerequisite_requires_pass_decision_and_index(tmp_path):
    decision = tmp_path / "R0_V2_DECISION.json"
    index = tmp_path / "sha256_index.json"
    decision.write_text(json.dumps({"study_id":"CAN-V1-FEDRIDGE-R0V2-20260812","decision":"R0_V2_FAILED"}))
    index.write_text("{}")
    with pytest.raises(RuntimeError, match="prerequisite"):
        validate_r0_prerequisite(decision, index, expected_index_sha256="bad")


def test_classifier_registry_rejects_wrong_hash(tmp_path):
    ckpt = tmp_path / "c3.pth"
    ckpt.write_bytes(b"checkpoint")
    with pytest.raises(RuntimeError, match="classifier"):
        validate_classifier_registry({"C3":{"path":str(ckpt),"sha256":"0"*64}})


def test_target_test_release_requires_all_target_locks():
    access = []
    locks = {"C3":{"alpha":True,"models":True,"classifier":True,"cache":True,"bootstrap":True}}
    assert_test_access_released("C3", locks, access)
    assert access[-1]["operation"] == "target_test_released"
    locks["C3"]["bootstrap"] = False
    with pytest.raises(RuntimeError, match="locked"):
        assert_test_access_released("C3", locks, [])


def test_audit_rejects_coordinated_rehash_semantic_tamper(tmp_path):
    (tmp_path / "R1_DECISION.json").write_text(json.dumps({"decision":"CANONICAL_R84_SUPPORTED"}))
    (tmp_path / "target_model_lock.json").write_text(json.dumps({"all_targets_locked":False}))
    import hashlib
    index = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.iterdir()}
    (tmp_path / "sha256_index.json").write_text(json.dumps(index))
    with pytest.raises(RuntimeError, match="semantic"):
        validate_evidence_bundle(tmp_path)


def test_runner_inspect_does_not_create_formal_root():
    root = Path("results/iotj_canonical_v1_final/canonical_r1_83d_vs_r84_20260812")
    assert not root.exists()
    proc = subprocess.run([sys.executable, "scripts/run_iotj_canonical_r1_v1.py", "inspect"], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["formal_execution_started"] is False
    assert payload["r0_v2_prerequisite"] == "PASS"
    assert not root.exists()


def test_classifier_stream_uses_registered_pred_class_field():
    assert predicted_classes_from_rows([{"pred_class": 2}, {"pred_class": 1}]).tolist() == [2, 1]
    with pytest.raises(RuntimeError):
        predicted_classes_from_rows([{"predicted_class": 2}])
