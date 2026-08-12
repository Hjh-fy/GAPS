"""Canonical-v1 R2 transfer-safe regression gate."""
from __future__ import annotations

import argparse, csv, hashlib, json, subprocess, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from gaps_flower.canonical_r1_v1 import (bootstrap_paired_group_deltas, fit_ridge_model,
    predict_ridge_model, metric_rows_for_slices)
from gaps_flower.canonical_r2_transfer import (BETAS, decide_transfer_candidate,
    residual_transfer_prediction, select_grouped_residual_alpha,
    grouped_shrinkage_oof_predictions, select_grouped_shrinkage_beta,
    shrinkage_transfer_prediction)
from gaps_flower.canonical_quantitative_features import load_feature_cache

STUDY_ID = "CAN-V1-CRRQ-R2-TRANSFER-SAFE-20260812"
R1_ROOT = ROOT / "results/iotj_canonical_v1_final/canonical_r1_83d_vs_r84_20260812"
R0_ROOT = ROOT / "results/iotj_canonical_v1_final/canonical_fedridge_r0_v2_20260812"
FORMAL_ROOT = ROOT / "results/iotj_canonical_v1_final/canonical_r2_transfer_safe_20260812"
DOC_ROOT = ROOT / "docs/experiments/iotj_canonical_v1_final/canonical_r2_transfer_safe_20260812"
DATA_ROOT = ROOT / "dataset/iotj_canonical_v1"
ALPHAS = (0.0, .01, .1, 1.0, 10.0, 100.0, 1000.0)
GAS_NAMES = {0:"Ethanol",1:"CO",2:"Ethylene",3:"Methane"}
R0_MODEL_SHA256 = "40c06848f19d211920b200946328e6e95cae9656a17ae0d18036951b7c5d67f6"


def sha256(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def head(): return subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
def read_json(path): return json.loads(Path(path).read_text(encoding="utf-8-sig"))
def write_json(path,payload): Path(path).write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
def read_csv(path):
    with Path(path).open(encoding="utf-8-sig",newline="") as f: return list(csv.DictReader(f))
def write_csv(path,rows):
    if not rows: raise RuntimeError(f"empty output: {path}")
    with Path(path).open("x",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def validate_trigger():
    if read_json(R1_ROOT/"R1_DECISION.json").get("decision") != "CANONICAL_R84_DEVICE_DEPENDENT":
        raise RuntimeError("R2 trigger is not DEVICE_DEPENDENT")
    if read_json(R1_ROOT/"COMPLETE.json").get("status") != "PASS": raise RuntimeError("R1 incomplete")
    index=read_json(R1_ROOT/"sha256_index.json")
    for name,digest in index.items():
        if not (R1_ROOT/name).is_file() or sha256(R1_ROOT/name)!=digest: raise RuntimeError(f"R1 hash mismatch: {name}")
    if sha256(R0_ROOT/"model_lock.json") != R0_MODEL_SHA256: raise RuntimeError("R0 source model lock mismatch")
    return index


def inspect():
    validate_trigger()
    if FORMAL_ROOT.exists(): raise FileExistsError("immutable R2 result root exists")
    return {"study_id":STUDY_ID,"trigger":"CANONICAL_R84_DEVICE_DEPENDENT","formal_root_exists":False}


def preflight(authorized_head):
    if authorized_head != head(): raise RuntimeError("authorized freeze HEAD mismatch")
    inspect()
    return {"study_id":STUDY_ID,"authorized_head":authorized_head,"r1_index_sha256":sha256(R1_ROOT/"sha256_index.json"),
            "protocol_sha256":sha256(DOC_ROOT/"protocol_manifest.json"),"test_opened":False}


def _source_models():
    lock=read_json(R0_ROOT/"model_lock.json")
    return {int(k):v["federated"] for k,v in lock["models"].items()}


def _predict_source(models,h1): return np.column_stack([predict_ridge_model(models[g],h1) for g in range(4)])


def _r1_test_matrices():
    rows=read_csv(R1_ROOT/"predictions.csv"); out={}
    for target in ("C3","C4","C5"):
        methods={m:[r for r in rows if r["target"]==target and r["method"]==m] for m in ("SOURCE_ONLY_FEDRIDGE","TARGET_ONLY_83D_RIDGE","R84_CONCAT")}
        ids=[[r["physical_identity"] for r in methods[m]] for m in methods]
        if not ids[0] or not all(v==ids[0] for v in ids): raise RuntimeError(f"R1 row alignment mismatch: {target}")
        base=methods["R84_CONCAT"]
        out[target]={"rows":base,"source":np.asarray([[float(r[f"route_{g}"]) for g in range(4)] for r in methods["SOURCE_ONLY_FEDRIDGE"]]),
                     "target83":np.asarray([[float(r[f"route_{g}"]) for g in range(4)] for r in methods["TARGET_ONLY_83D_RIDGE"]]),
                     "r84":np.asarray([[float(r[f"route_{g}"]) for g in range(4)] for r in base])}
    return out


def run(authorized_head):
    receipt=preflight(authorized_head); FORMAL_ROOT.mkdir(parents=True); write_json(FORMAL_ROOT/"preflight_receipt.json",receipt)
    source_models=_source_models(); r1_models=read_json(R1_ROOT/"target_model_lock.json")["models"]
    models={}; selections=[]
    for target in ("C3","C4","C5"):
        client=int(target[1:]); cache=R1_ROOT/f"canonical_feature_caches/{target}/calibration"
        sensor,h1,ids,_=load_feature_cache(cache,expected_dataset_sha256="2f810d7e93cae5f361923184e9dc87d5ae59e0f59be9f52aff7e14f9f33e94f6",expected_study_id="CAN-V1-CRRQ-R1-CANONICAL-83D-R84-20260812")
        y4=np.load(DATA_ROOT/f"client_{client}/calibration_regression_labels.npy",allow_pickle=False)
        cls=np.load(DATA_ROOT/f"client_{client}/calibration_classification_labels.npy",allow_pickle=False).astype(int)
        groups=np.asarray([d["filename"] for d in ids]); source=_predict_source(source_models,h1)
        target83=np.column_stack([predict_ridge_model(r1_models[target]["TARGET_ONLY_83D_RIDGE"][str(g)],sensor) for g in range(4)])
        models[target]={"residual":{},"beta":{}}
        for gas in range(4):
            mask=cls==gas; cv=select_grouped_residual_alpha(sensor[mask],y4[mask,gas],source[mask,gas],groups[mask],ALPHAS)
            residual=y4[mask,gas]-source[mask,gas]
            model=fit_ridge_model(sensor[mask],residual,cv.alpha,residual.min(),residual.max()); models[target]["residual"][str(gas)]=model
            oof83,_=grouped_shrinkage_oof_predictions(sensor[mask],y4[mask,gas],source[mask,gas],groups[mask],ALPHAS,5)
            b=select_grouped_shrinkage_beta(y4[mask,gas],oof83,source[mask,gas],groups[mask],BETAS)
            models[target]["beta"][str(gas)]=b["selected_beta"]
            selections += [{"target":target,"candidate":"RESIDUAL_TRANSFER","gas_id":gas,"selected":cv.alpha,"parameter":"alpha","scores":json.dumps(cv.pooled_rmse,sort_keys=True),"group_folds":json.dumps(cv.fold_by_group,sort_keys=True)},
                           {"target":target,"candidate":"SHRINKAGE_TRANSFER","gas_id":gas,"selected":b["selected_beta"],"parameter":"beta","scores":json.dumps(b["pooled_rmse"],sort_keys=True),"group_folds":json.dumps(b["fold_by_group"],sort_keys=True)}]
    write_json(FORMAL_ROOT/"selection_lock.json",{"study_id":STUDY_ID,"models":models,"selections":selections,"test_opened":False})
    write_csv(FORMAL_ROOT/"selection_audit.csv",selections)
    lock_sha=sha256(FORMAL_ROOT/"selection_lock.json"); write_json(FORMAL_ROOT/"target_test_release_receipt.json",{"selection_lock_sha256":lock_sha,"r1_predictions_sha256":sha256(R1_ROOT/"predictions.csv")})
    data=_r1_test_matrices(); prediction_rows=[]; comparison=[]; gas_rows=[]; boot_rows=[]; decisions={}
    routed_by_method={m:[] for m in ("R84_CONCAT","RESIDUAL_TRANSFER","SHRINKAGE_TRANSFER")}; truth_all=[]; groups_all=[]; tc_all=[]
    for target,payload in data.items():
        rows=payload["rows"]; sensor,_,ids,_=load_feature_cache(R1_ROOT/f"canonical_feature_caches/{target}/test",expected_dataset_sha256="2f810d7e93cae5f361923184e9dc87d5ae59e0f59be9f52aff7e14f9f33e94f6",expected_study_id="CAN-V1-CRRQ-R1-CANONICAL-83D-R84-20260812")
        residual=np.column_stack([residual_transfer_prediction(payload["source"][:,g],predict_ridge_model(models[target]["residual"][str(g)],sensor)) for g in range(4)])
        shrink=np.column_stack([shrinkage_transfer_prediction(payload["target83"][:,g],payload["source"][:,g],models[target]["beta"][str(g)]) for g in range(4)])
        matrices={"R84_CONCAT":payload["r84"],"RESIDUAL_TRANSFER":residual,"SHRINKAGE_TRANSFER":shrink}
        truth=np.asarray([float(r["truth"]) for r in rows]); tc=np.asarray([int(r["true_class"]) for r in rows]); pc=np.asarray([int(r["predicted_class"]) for r in rows]); repeats=np.asarray([int(r["repeat_id"]) for r in rows]); ranges={g:float(source_models[g]["clip_max"]-source_models[g]["clip_min"]) for g in range(4)}
        for method,matrix in matrices.items():
            slices=metric_rows_for_slices(target,method,truth,tc,pc,matrix,ranges,repeats); comparison+=slices["overall"]; gas_rows+=slices["gas"]
            routed=matrix[np.arange(len(tc)),pc]; routed_by_method[method].append(routed)
            for i,r in enumerate(rows): prediction_rows.append({"target":target,"method":method,"physical_identity":r["physical_identity"],"filename":r["filename"],"true_class":tc[i],"predicted_class":pc[i],"truth":truth[i],"prediction":routed[i],**{f"route_{g}":matrix[i,g] for g in range(4)}})
        truth_all.append(truth); groups_all.append(np.asarray([f"{target}|{r['filename']}" for r in rows])); tc_all.append(tc)
    y=np.concatenate(truth_all); groups=np.concatenate(groups_all); tc=np.concatenate(tc_all); r84=np.concatenate(routed_by_method["R84_CONCAT"])
    for candidate in ("RESIDUAL_TRANSFER","SHRINKAGE_TRANSFER"):
        cand=np.concatenate(routed_by_method[candidate]); out=bootstrap_paired_group_deltas(y,r84,cand,groups,np.ones(len(y)),5000,42)["rmse_delta"]
        r84rmse=float(np.sqrt(np.mean((r84-y)**2))); crmse=float(np.sqrt(np.mean((cand-y)**2))); degrad={}
        for gas in range(4):
            m=tc==gas; a=float(np.sqrt(np.mean((r84[m]-y[m])**2))); b=float(np.sqrt(np.mean((cand[m]-y[m])**2))); degrad[GAS_NAMES[gas]]=(b-a)/a
        ci=(float(np.percentile(out,2.5)),float(np.percentile(out,97.5))); decision=decide_transfer_candidate(r84rmse,crmse,ci[1],degrad); decisions[candidate]={**decision,"r84_rmse":r84rmse,"candidate_rmse":crmse,"ci_low":ci[0],"ci_high":ci[1],"gas_relative_degradation":degrad}
        boot_rows.append({"candidate":candidate,"metric":"RMSE","point_delta":crmse-r84rmse,"ci_low":ci[0],"ci_high":ci[1],"n_replicates":5000,"seed":42,"n_groups":len(np.unique(groups))})
    retained=[k for k,v in decisions.items() if v["retained"]]
    final="RETAIN_R84_DEVICE_DEPENDENT" if not retained else "RETAIN_"+min(retained,key=lambda k:decisions[k]["candidate_rmse"])
    write_csv(FORMAL_ROOT/"transfer_regression_comparison.csv",comparison); write_csv(FORMAL_ROOT/"transfer_regression_per_gas.csv",gas_rows); write_csv(FORMAL_ROOT/"transfer_regression_bootstrap.csv",boot_rows); write_csv(FORMAL_ROOT/"predictions.csv",prediction_rows)
    write_json(FORMAL_ROOT/"R2_DECISION.json",{"study_id":STUDY_ID,"decision":final,"candidates":decisions})
    (FORMAL_ROOT/"TRANSFER_SAFE_REGRESSION_REPORT.md").write_text(f"# Transfer-safe regression report\n\nDecision: `{final}`. Selection used target calibration only; sealed test was opened after `selection_lock.json`. No additional candidate or search was used.\n",encoding="utf-8")
    index={p.relative_to(FORMAL_ROOT).as_posix():sha256(p) for p in FORMAL_ROOT.rglob("*") if p.is_file() and p.name not in {"sha256_index.json","COMPLETE.json"}}; write_json(FORMAL_ROOT/"sha256_index.json",index); write_json(FORMAL_ROOT/"COMPLETE.json",{"study_id":STUDY_ID,"status":"PASS","decision":final})
    return read_json(FORMAL_ROOT/"R2_DECISION.json")


def audit():
    index=read_json(FORMAL_ROOT/"sha256_index.json")
    for name,digest in index.items():
        if sha256(FORMAL_ROOT/name)!=digest: raise RuntimeError(f"R2 hash mismatch: {name}")
    if read_json(FORMAL_ROOT/"target_test_release_receipt.json")["selection_lock_sha256"]!=sha256(FORMAL_ROOT/"selection_lock.json"): raise RuntimeError("selection lock mismatch")
    if read_json(FORMAL_ROOT/"COMPLETE.json").get("status")!="PASS": raise RuntimeError("R2 incomplete")
    return {"status":"PASS","decision":read_json(FORMAL_ROOT/"R2_DECISION.json")["decision"]}


def main():
    p=argparse.ArgumentParser(); p.add_argument("command",choices=["inspect","preflight","run","audit"]); p.add_argument("--authorized-freeze-commit",default=""); a=p.parse_args()
    if a.command=="inspect": result=inspect()
    elif a.command=="audit": result=audit()
    elif not a.authorized_freeze_commit: raise SystemExit("--authorized-freeze-commit required")
    elif a.command=="preflight": result=preflight(a.authorized_freeze_commit)
    else: result=run(a.authorized_freeze_commit)
    print(json.dumps(result,sort_keys=True))
if __name__=="__main__": main()
