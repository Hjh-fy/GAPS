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
    validate_r0_bundle,
    metric_rows_for_slices,
    summarize_bootstrap,
)
from gaps_flower.canonical_quantitative_features import extract_canonical_features, build_feature_cache, load_feature_cache


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


def test_r0_bundle_rejects_indexed_model_semantic_tamper(tmp_path):
    import shutil, hashlib
    source = Path("results/iotj_canonical_v1_final/canonical_fedridge_r0_v2_20260812")
    shutil.copytree(source, tmp_path / "r0")
    root = tmp_path / "r0"
    lock = json.loads((root / "model_lock.json").read_text())
    lock["models"]["0"]["federated"]["alpha"] = 1000
    (root / "model_lock.json").write_text(json.dumps(lock))
    index = json.loads((root / "sha256_index.json").read_text())
    index["model_lock.json"] = hashlib.sha256((root / "model_lock.json").read_bytes()).hexdigest()
    (root / "sha256_index.json").write_text(json.dumps(index))
    with pytest.raises(RuntimeError, match="R0-v2"):
        validate_r0_bundle(root, expected_index_sha256=hashlib.sha256((root / "sha256_index.json").read_bytes()).hexdigest())


def test_slice_metrics_include_four_scopes_and_joint_gas_concentration():
    rows = metric_rows_for_slices("C5", "R84_CONCAT", np.array([225.,225.,50.,50.]),
        np.array([3,3,0,0]), np.array([3,0,0,0]),
        np.array([[20,20,20,24],[20,20,20,26],[49,80,80,80],[51,80,80,80.]]),
        {0:100.,3:225.}, np.array([1,1,2,2]))
    assert {r["scope"] for r in rows["overall"]} == {"S_ALL","S_CC","Oracle_ALL","Oracle_CC"}
    assert all("gas_id" in r and "concentration" in r for r in rows["concentration"])
    special = rows["special"]
    assert special and all(r["target"] == "C5" and r["gas_id"] == 3 and r["concentration"] == 225 for r in special)


def test_bootstrap_summary_records_exact_design_and_groups():
    out={"rmse_delta":[-2,-1],"mae_delta":[-1,-.5],"nrmse_range_delta":[-.2,-.1]}
    rows=summarize_bootstrap("C3",out,point={"RMSE":-1.5,"MAE":-.75,"NRMSE_range":-.15},n_groups=7,n_replicates=2)
    assert {r["metric"] for r in rows} == {"RMSE","MAE","NRMSE_range"}
    assert all(r["n_groups"] == 7 and r["n_replicates"] == 2 for r in rows)


def test_r1_identity_retains_registered_slice_fields():
    rec=extract_canonical_features(np.zeros((50,8)),phase=2,metadata={"filename":"f.txt","repeat_id":1,"gas_code":"Me","class_id":3,"concentration":225},client="C5",split="test",sample_index=0)
    assert {k:rec.identity[k] for k in ("repeat_id","gas_code","class_id","concentration")} == {"repeat_id":1,"gas_code":"Me","class_id":3,"concentration":225.0}


def test_r1_cache_binds_wrapper_and_extractor_hashes(tmp_path):
    canonical_dataset=tmp_path/"dataset"; client=canonical_dataset/"client_3"; client.mkdir(parents=True)
    np.save(client/"calibration_features.npy",np.zeros((2,50,8)))
    np.save(client/"calibration_phase_labels.npy",np.array([1,2]))
    (client/"calibration_experiment_info.json").write_text(json.dumps([{"filename":"a","repeat_id":1,"gas_code":"CO","class_id":1,"concentration":25},{"filename":"b","repeat_id":2,"gas_code":"CO","class_id":1,"concentration":25}]))
    wrapper=Path("gaps_flower/canonical_quantitative_features.py")
    extractor=Path("run_regression_head_ablation.py")
    manifest=build_feature_cache(canonical_dataset,tmp_path/"cache",client="C3",split="calibration",dataset_aggregate_sha256="a"*64,extractor_path=extractor,study_id="R1",wrapper_path=wrapper)
    assert len(manifest["wrapper_file_sha256"])==64 and len(manifest["extractor_file_sha256"])==64
    with pytest.raises(RuntimeError):
        load_feature_cache(tmp_path/"cache"/"C3"/"calibration",expected_dataset_sha256="a"*64,expected_study_id="R1",expected_wrapper_sha256="0"*64,expected_extractor_sha256=manifest["extractor_file_sha256"])


def test_access_receipt_must_precede_first_test_event():
    from gaps_flower.canonical_r1_v1 import validate_access_chronology
    locks={name:"h"*64 for name in ("target_alpha_lock.json","target_model_lock.json","classifier_lock.json","bootstrap_design_lock.json")}
    events=[{"sequence":0,"target":"C3","split":"test","artifact":"test_features.npy","operation":"load"}]
    with pytest.raises(RuntimeError,match="receipt"):
        validate_access_chronology(events,locks,{"published_before_sequence":1,"lock_sha256":locks})
    assert validate_access_chronology(events,locks,{"published_before_sequence":0,"lock_sha256":locks})


def test_semantic_audit_recomputes_prediction_metrics_after_coordinated_rehash(tmp_path):
    from gaps_flower.canonical_r1_v1 import write_synthetic_semantic_bundle
    root=tmp_path/"bundle"; write_synthetic_semantic_bundle(root)
    assert validate_evidence_bundle(root)
    rows=list(__import__("csv").DictReader((root/"predictions.csv").open()))
    rows[0]["prediction"]="999"; rows[0]["route_0"]="999"
    with (root/"predictions.csv").open("w",newline="") as f:
        w=__import__("csv").DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    import hashlib
    index=json.loads((root/"sha256_index.json").read_text()); index["predictions.csv"]=hashlib.sha256((root/"predictions.csv").read_bytes()).hexdigest(); (root/"sha256_index.json").write_text(json.dumps(index))
    with pytest.raises(RuntimeError,match="semantic"):
        validate_evidence_bundle(root)
