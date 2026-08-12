"""Versioned fail-closed entry point for canonical-v1 R1 (no implicit execution)."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gaps_flower.canonical_r1_v1 import (assert_test_access_released, bootstrap_paired_group_deltas,
    compute_route_metrics, decide_r84, fit_ridge_model, predict_ridge_model,
    predicted_classes_from_rows,
    select_grouped_cv_alpha, validate_classifier_registry, validate_evidence_bundle,
    validate_r0_prerequisite)
from gaps_flower.canonical_quantitative_features import build_feature_cache, load_feature_cache

STUDY_ID = "CAN-V1-CRRQ-R1-CANONICAL-83D-R84-20260812"
FORMAL_ROOT = ROOT / "results/iotj_canonical_v1_final/canonical_r1_83d_vs_r84_20260812"
R0_ROOT = ROOT / "results/iotj_canonical_v1_final/canonical_fedridge_r0_v2_20260812"
R0_INDEX_SHA256 = "04e62b0a2b363fcb79cbeb60ad6c2c6c9a3dedef380b2b0f7c2aa808732d4d82"
CLASSIFIERS = {
    "C3": {"path": str(ROOT / "results/iotj_canonical_v1_final_20260808/classification/CANONICAL-V1-A4-C3/remote_server/server_latest_adapted.pth"), "sha256": "e2364290ffc7fd9748fe86edb3745dca0eac692165f6c8aba1825728ddcd4414"},
    "C4": {"path": str(ROOT / "results/iotj_canonical_v1_final_20260808/classification/CANONICAL-V1-A4-C4/remote_server/server_latest_adapted.pth"), "sha256": "422a49f28331e5486d215a8d34bc9a972dc8fc1992f8b5bf27428329143599c3"},
    "C5": {"path": str(ROOT / "results/iotj_canonical_v1_final_20260808/classification/CANONICAL-V1-A4-C5/remote_server/server_latest_adapted.pth"), "sha256": "3965ec8618a2d496804bbc141f49e00b451fce05e9edbefde721f0dd4f912b93"},
}
DATA_ROOT = ROOT / "dataset/iotj_canonical_v1"
PROTOCOL = ROOT / "docs/experiments/iotj_canonical_v1_final/canonical_r1_83d_vs_r84_20260812/protocol_manifest.json"
ALPHAS = [0.0, .01, .1, 1.0, 10.0, 100.0, 1000.0]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, payload):
    path = Path(path)
    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-")
    try:
        import os
        os.write(fd, data); os.close(fd); fd = -1
        try: os.link(temp, path)
        except FileExistsError:
            if path.read_bytes() != data: raise
    finally:
        if fd >= 0:
            import os; os.close(fd)
        Path(temp).unlink(missing_ok=True)


def _head():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def _critical_tree_clean():
    paths = ["gaps_flower/canonical_r1_v1.py", "scripts/run_iotj_canonical_r1_v1.py",
             "docs/experiments/iotj_canonical_v1_final/canonical_r1_83d_vs_r84_20260812"]
    return subprocess.run(["git", "diff", "--quiet", "HEAD", "--", *paths], cwd=ROOT).returncode == 0


def formal_preflight(authorized_head):
    if authorized_head != _head():
        raise RuntimeError("authorized freeze HEAD mismatch")
    if not _critical_tree_clean():
        raise RuntimeError("critical R1 bytes differ from authorized freeze HEAD")
    if FORMAL_ROOT.exists():
        raise FileExistsError("immutable formal result root exists")
    inspect()
    dataset_index = json.loads((DATA_ROOT / "dataset_sha256.json").read_text(encoding="utf-8"))
    calibration = {}
    deferred_test = {}
    for client in (3, 4, 5):
        for split, sink in (("calibration", calibration), ("test", deferred_test)):
            for suffix in ("features.npy", "phase_labels.npy", "experiment_info.json", "classification_labels.npy", "regression_labels.npy"):
                rel = f"client_{client}/{split}_{suffix}"
                path = DATA_ROOT / rel
                expected = dataset_index["files"].get(rel)
                if not path.is_file() or not expected:
                    raise RuntimeError(f"canonical registered path missing: {rel}")
                # Hashing authenticates bytes; tensor/label parsing remains forbidden here.
                if sha256(path) != expected:
                    raise RuntimeError(f"canonical artifact hash mismatch: {rel}")
                sink[rel] = expected
    return {"schema_version":"iotj.canonical_v1.r1.preflight.v1", "study_id":STUDY_ID,
            "authorized_head":authorized_head, "formal_execution_started":False,
            "calibration_artifacts":calibration, "deferred_test_artifacts":deferred_test,
            "test_tensor_or_label_parsed":False, "protocol_sha256":sha256(PROTOCOL)}


def _source_models():
    lock = json.loads((R0_ROOT / "model_lock.json").read_text(encoding="utf-8"))
    return {int(k): v["federated"] for k, v in lock["models"].items()}


def _source_matrix(models, h1):
    return np.column_stack([predict_ridge_model(models[g], h1) for g in range(4)])


def _write_csv(path, rows):
    if not rows: raise RuntimeError(f"empty evidence: {path}")
    with Path(path).open("x", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def formal_run(authorized_head):
    receipt = formal_preflight(authorized_head)
    FORMAL_ROOT.mkdir(parents=True)
    atomic_json(FORMAL_ROOT / "preflight_receipt.json", receipt)
    access = []
    models_source = _source_models()
    target_models, alpha_rows, locks = {}, [], {}
    calibration_payload = {}
    for target in ("C3", "C4", "C5"):
        client = int(target[1:])
        cache_root = FORMAL_ROOT / "canonical_feature_caches"
        build_feature_cache(DATA_ROOT, cache_root, client=target, split="calibration",
            dataset_aggregate_sha256="2f810d7e93cae5f361923184e9dc87d5ae59e0f59be9f52aff7e14f9f33e94f6",
            extractor_path=ROOT / "run_regression_head_ablation.py", study_id=STUDY_ID)
        sensor, h1, ids, _ = load_feature_cache(cache_root / target / "calibration",
            expected_dataset_sha256="2f810d7e93cae5f361923184e9dc87d5ae59e0f59be9f52aff7e14f9f33e94f6", expected_study_id=STUDY_ID)
        y4 = np.load(DATA_ROOT / f"client_{client}/calibration_regression_labels.npy", allow_pickle=False)
        cls = np.load(DATA_ROOT / f"client_{client}/calibration_classification_labels.npy", allow_pickle=False).astype(int)
        groups = np.asarray([d["filename"] for d in ids])
        prior = _source_matrix(models_source, h1)
        target_models[target] = {}
        for method in ("TARGET_ONLY_83D_RIDGE", "R84_CONCAT"):
            target_models[target][method] = {}
            for gas in range(4):
                mask = cls == gas
                x = sensor[mask] if method.startswith("TARGET") else np.column_stack([sensor[mask], prior[mask, gas]])
                cv = select_grouped_cv_alpha(x, y4[mask, gas], groups[mask], ALPHAS)
                model = fit_ridge_model(x, y4[mask, gas], cv.alpha, y4[mask, gas].min(), y4[mask, gas].max())
                target_models[target][method][gas] = model
                alpha_rows.append({"target":target,"method":method,"gas_id":gas,"alpha":cv.alpha,
                                   "fold_by_group":json.dumps(cv.fold_by_group,sort_keys=True),"pooled_rmse":json.dumps(cv.pooled_rmse,sort_keys=True)})
        calibration_payload[target] = {"row_count":len(ids), "cache_locked":True}
        locks[target] = {"alpha":True,"models":True,"classifier":True,"cache":True,"bootstrap":True}
    _write_csv(FORMAL_ROOT / "target_alpha_audit.csv", alpha_rows)
    atomic_json(FORMAL_ROOT / "target_alpha_lock.json", {"study_id":STUDY_ID,"all_targets_locked":True,"rows":alpha_rows})
    atomic_json(FORMAL_ROOT / "target_model_lock.json", {"study_id":STUDY_ID,"all_targets_locked":True,"models":target_models})
    atomic_json(FORMAL_ROOT / "classifier_lock.json", {"study_id":STUDY_ID,"classifiers":CLASSIFIERS})
    atomic_json(FORMAL_ROOT / "bootstrap_design_lock.json", {"replicates":5000,"seed":42,"unit":"target|raw_filename","paired":True,"pooled_stratified_by_target":True})
    # Import only after all selection locks are immutable.
    import torch
    from scripts.summarize_iotj_classification_ablation import evaluate_checkpoint_stream
    metric_rows, gas_rows, concentration_rows, prediction_rows = [], [], [], []
    boot_inputs = {}
    for target in ("C3", "C4", "C5"):
        assert_test_access_released(target, locks, access)
        client = int(target[1:]); cache_root = FORMAL_ROOT / "canonical_feature_caches"
        build_feature_cache(DATA_ROOT, cache_root, client=target, split="test",
            dataset_aggregate_sha256="2f810d7e93cae5f361923184e9dc87d5ae59e0f59be9f52aff7e14f9f33e94f6",
            extractor_path=ROOT / "run_regression_head_ablation.py", study_id=STUDY_ID)
        sensor, h1, ids, _ = load_feature_cache(cache_root / target / "test",
            expected_dataset_sha256="2f810d7e93cae5f361923184e9dc87d5ae59e0f59be9f52aff7e14f9f33e94f6", expected_study_id=STUDY_ID)
        y4 = np.load(DATA_ROOT / f"client_{client}/test_regression_labels.npy", allow_pickle=False)
        tc = np.load(DATA_ROOT / f"client_{client}/test_classification_labels.npy", allow_pickle=False).astype(int)
        cls_rows, _ = evaluate_checkpoint_stream(Path(CLASSIFIERS[target]["path"]), data_root=DATA_ROOT,
                                                  target_client=client, split="test", device=torch.device("cpu"), batch_size=32)
        pc = predicted_classes_from_rows(cls_rows); truth = y4[np.arange(len(tc)), tc]
        prior = _source_matrix(models_source, h1)
        method_matrices = {"SOURCE_ONLY_FEDRIDGE":prior}
        for method in ("TARGET_ONLY_83D_RIDGE","R84_CONCAT"):
            matrix=[]
            for gas in range(4):
                x = sensor if method.startswith("TARGET") else np.column_stack([sensor, prior[:,gas]])
                matrix.append(predict_ridge_model(target_models[target][method][gas], x))
            method_matrices[method] = np.column_stack(matrix)
        ranges={g:float(models_source[g]["clip_max"]-models_source[g]["clip_min"]) for g in range(4)}
        for method,matrix in method_matrices.items():
            scopes=compute_route_metrics(truth,tc,pc,matrix,ranges)
            for scope,m in scopes.items(): metric_rows.append({"target":target,"method":method,"scope":scope,**m.__dict__})
            routed=matrix[np.arange(len(tc)),pc]
            for gas in range(4):
                mask=tc==gas; m=compute_route_metrics(truth[mask],tc[mask],pc[mask],matrix[mask],ranges)["S_ALL"]
                gas_rows.append({"target":target,"method":method,"gas_id":gas,**m.__dict__})
            for conc in sorted(set(truth.tolist())):
                mask=truth==conc; m=compute_route_metrics(truth[mask],tc[mask],pc[mask],matrix[mask],ranges)["S_ALL"]
                concentration_rows.append({"target":target,"method":method,"concentration":conc,**m.__dict__})
            for i,d in enumerate(ids): prediction_rows.append({"target":target,"method":method,"sample_index":i,"physical_identity":d["physical_identity"],"filename":d["filename"],"true_class":int(tc[i]),"predicted_class":int(pc[i]),"truth":truth[i],"prediction":routed[i]})
        groups=np.asarray([f"{target}|{d['filename']}" for d in ids]); ranges_arr=np.asarray([ranges[int(g)] for g in tc])
        boot_inputs[target]=(truth,method_matrices["TARGET_ONLY_83D_RIDGE"][np.arange(len(tc)),pc],method_matrices["R84_CONCAT"][np.arange(len(tc)),pc],groups,ranges_arr,tc)
    _write_csv(FORMAL_ROOT / "canonical_regression_comparison.csv",metric_rows)
    _write_csv(FORMAL_ROOT / "canonical_regression_per_gas.csv",gas_rows)
    _write_csv(FORMAL_ROOT / "canonical_regression_per_concentration.csv",concentration_rows)
    _write_csv(FORMAL_ROOT / "predictions.csv",prediction_rows)
    bootstrap_rows=[]
    target_delta={}; gas_delta={}; paired={}
    for target,(y,a,b,g,r,tc) in boot_inputs.items():
        out=bootstrap_paired_group_deltas(y,a,b,g,r,5000,42); vals=np.asarray(out["rmse_delta"])
        point=float(np.sqrt(np.mean((b-y)**2))-np.sqrt(np.mean((a-y)**2))); target_delta[target]=point
        paired[target]=(float(np.sqrt(np.mean((a-y)**2))),float(np.sqrt(np.mean((b-y)**2))))
        for metric,key in (("RMSE","rmse_delta"),("MAE","mae_delta"),("NRMSE_range","nrmse_range_delta")):
            metric_vals=np.asarray(out[key]); bootstrap_rows.append({"scope":target,"metric":metric,"delta":float(np.mean(metric_vals)),"ci_low":np.percentile(metric_vals,2.5),"ci_high":np.percentile(metric_vals,97.5),"replicates":5000})
        for gas in range(4):
            m=tc==gas; gas_delta[f"{target}|{gas}"]=float(np.sqrt(np.mean((b[m]-y[m])**2))-np.sqrt(np.mean((a[m]-y[m])**2)))
    y=np.concatenate([v[0] for v in boot_inputs.values()]); a=np.concatenate([v[1] for v in boot_inputs.values()]); b=np.concatenate([v[2] for v in boot_inputs.values()]); g=np.concatenate([v[3] for v in boot_inputs.values()]); r=np.concatenate([v[4] for v in boot_inputs.values()])
    out=bootstrap_paired_group_deltas(y,a,b,g,r,5000,42); vals=np.asarray(out["rmse_delta"]); point=float(np.sqrt(np.mean((b-y)**2))-np.sqrt(np.mean((a-y)**2))); ci=(float(np.percentile(vals,2.5)),float(np.percentile(vals,97.5)))
    for metric,key in (("RMSE","rmse_delta"),("MAE","mae_delta"),("NRMSE_range","nrmse_range_delta")):
        metric_vals=np.asarray(out[key]); bootstrap_rows.append({"scope":"POOLED","metric":metric,"delta":point if metric=="RMSE" else float(np.mean(metric_vals)),"ci_low":np.percentile(metric_vals,2.5),"ci_high":np.percentile(metric_vals,97.5),"replicates":5000})
    _write_csv(FORMAL_ROOT / "canonical_regression_bootstrap.csv",bootstrap_rows)
    decision=decide_r84(point,ci,target_delta,gas_delta,paired); atomic_json(FORMAL_ROOT / "R1_DECISION.json",decision)
    atomic_json(FORMAL_ROOT / "DATA_ACCESS_AUDIT.json", {"events":access,"target_test_opened_after_all_locks":True})
    report="# Canonical 83D versus R84 report\n\nDecision: `%s`. Known limitation: calibration/test windows may share raw-file/time neighborhoods. Dedicated C5 Methane 225 ppm repeat1 rows are retained in concentration and prediction evidence.\n" % decision["decision"]
    (FORMAL_ROOT / "CANONICAL_83D_VS_R84_REPORT.md").write_text(report,encoding="utf-8")
    index={p.relative_to(FORMAL_ROOT).as_posix():sha256(p) for p in FORMAL_ROOT.rglob("*") if p.is_file() and p.name not in {"sha256_index.json","COMPLETE.json"}}
    atomic_json(FORMAL_ROOT / "sha256_index.json",index)
    validate_evidence_bundle(FORMAL_ROOT)
    atomic_json(FORMAL_ROOT / "COMPLETE.json", {"study_id":STUDY_ID,"status":"PASS","decision":decision["decision"]})
    return decision


def inspect():
    validate_r0_prerequisite(R0_ROOT / "R0_V2_DECISION.json", R0_ROOT / "sha256_index.json", expected_index_sha256=R0_INDEX_SHA256)
    validate_classifier_registry(CLASSIFIERS)
    if FORMAL_ROOT.exists():
        raise RuntimeError("formal R1 result root already exists; immutable-root gate closed")
    return {"study_id": STUDY_ID, "formal_execution_started": False, "formal_root_exists": False,
            "r0_v2_prerequisite": "PASS", "classifier_registry": "PASS",
            "next_action": "freeze_and_authorize_before_formal_preflight"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["inspect", "preflight", "run", "audit"])
    parser.add_argument("--authorized-freeze-commit", default="")
    args = parser.parse_args()
    if args.command == "inspect": result=inspect()
    elif not args.authorized_freeze_commit: raise SystemExit("--authorized-freeze-commit required")
    elif args.command == "preflight": result=formal_preflight(args.authorized_freeze_commit)
    elif args.command == "run": result=formal_run(args.authorized_freeze_commit)
    else:
        if args.authorized_freeze_commit != _head(): raise RuntimeError("authorized freeze HEAD mismatch")
        result={"status":"PASS" if validate_evidence_bundle(FORMAL_ROOT) else "FAIL"}
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
