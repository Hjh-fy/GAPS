"""Frozen numerical primitives for canonical-v1 R1 target regression."""
from dataclasses import dataclass
from typing import Mapping, Sequence
from pathlib import Path
import hashlib
import json
import math
import csv

import numpy as np


@dataclass(frozen=True)
class CVResult:
    alpha: float
    pooled_rmse: Mapping[float, float]
    fold_by_group: Mapping[str, int]


@dataclass(frozen=True)
class RegressionMetrics:
    n: int
    rmse: float
    mae: float
    nrmse_range: float
    r2: float
    bias: float


def assign_balanced_group_folds(groups: Sequence[str], n_folds: int = 5) -> np.ndarray:
    values = np.asarray(groups, dtype=str)
    unique, counts = np.unique(values, return_counts=True)
    loads = [0] * n_folds
    mapping = {}
    for group, count in sorted(zip(unique.tolist(), counts.tolist()), key=lambda x: (-x[1], x[0])):
        fold = min(range(n_folds), key=lambda f: (loads[f], f))
        mapping[group] = fold
        loads[fold] += count
    return np.asarray([mapping[g] for g in values], dtype=int)


def _ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float):
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-9] = 1.0
    z = (x - mean) / scale
    design = np.column_stack([np.ones(len(z)), z])
    penalty = np.diag([0.0] + [alpha] * x.shape[1])
    beta = np.linalg.pinv(design.T @ design + penalty) @ design.T @ y
    return mean, scale, beta


def _predict(model, x):
    mean, scale, beta = model
    return np.column_stack([np.ones(len(x)), (x - mean) / scale]) @ beta


def fit_ridge_model(x, y, alpha, clip_min, clip_max):
    mean, scale, beta = _ridge_fit(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64), float(alpha))
    return {"alpha": float(alpha), "mean": mean.tolist(), "scale": scale.tolist(), "coef": beta.tolist(),
            "clip_min": float(clip_min), "clip_max": float(clip_max), "solver": "numpy.linalg.pinv",
            "intercept_regularized": False}


def predict_ridge_model(model, x):
    pred = _predict((np.asarray(model["mean"]), np.asarray(model["scale"]), np.asarray(model["coef"])), np.asarray(x, dtype=np.float64))
    return np.clip(pred, float(model["clip_min"]), float(model["clip_max"]))


def select_grouped_cv_alpha(x, y, groups, alphas, n_folds=5) -> CVResult:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    groups = np.asarray(groups, dtype=str)
    folds = assign_balanced_group_folds(groups, n_folds)
    scores = {}
    for alpha in alphas:
        sse = 0.0
        n = 0
        for fold in range(n_folds):
            valid = folds == fold
            if not np.any(valid) or not np.any(~valid):
                continue
            pred = _predict(_ridge_fit(x[~valid], y[~valid], float(alpha)), x[valid])
            sse += float(np.sum((pred - y[valid]) ** 2))
            n += int(valid.sum())
        if not n:
            raise ValueError("insufficient raw-filename groups for grouped CV")
        scores[float(alpha)] = float(np.sqrt(sse / n))
    first = min(range(len(alphas)), key=lambda i: (scores[float(alphas[i])], i))
    return CVResult(float(alphas[first]), scores, {g: int(folds[np.flatnonzero(groups == g)[0]]) for g in np.unique(groups)})


def _metrics(y, p, ranges) -> RegressionMetrics:
    err = np.asarray(p) - np.asarray(y)
    sst = float(np.sum((y - np.mean(y)) ** 2))
    return RegressionMetrics(len(y), float(np.sqrt(np.mean(err**2))), float(np.mean(abs(err))),
                             float(np.sqrt(np.mean((err / ranges) ** 2))),
                             float(1.0 - np.sum(err**2) / sst) if sst else float("nan"), float(np.mean(err)))


def compute_route_metrics(truth, true_class, predicted_class, predictions_by_route, gas_ranges):
    y = np.asarray(truth, dtype=float)
    tc = np.asarray(true_class, dtype=int)
    pc = np.asarray(predicted_class, dtype=int)
    matrix = np.asarray(predictions_by_route, dtype=float)
    correct = tc == pc
    s = matrix[np.arange(len(y)), pc]
    oracle = matrix[np.arange(len(y)), tc]
    ranges = np.asarray([gas_ranges[int(v)] for v in tc])
    return {"S_ALL": _metrics(y, s, ranges), "S_CC": _metrics(y[correct], s[correct], ranges[correct]),
            "Oracle_ALL": _metrics(y, oracle, ranges), "Oracle_CC": _metrics(y[correct], oracle[correct], ranges[correct])}


def metric_rows_for_slices(target, method, truth, true_class, predicted_class, matrix, gas_ranges, repeat_ids):
    truth=np.asarray(truth); tc=np.asarray(true_class); pc=np.asarray(predicted_class); repeats=np.asarray(repeat_ids); matrix=np.asarray(matrix)
    def rows(mask, **keys):
        result=[]
        for scope,m in compute_route_metrics(truth[mask],tc[mask],pc[mask],matrix[mask],gas_ranges).items():
            result.append({"target":target,"method":method,"scope":scope,**keys,**m.__dict__})
        return result
    overall=rows(np.ones(len(truth),dtype=bool))
    gas=[]; concentration=[]
    for gas_id in sorted(set(tc.tolist())):
        mask=tc==gas_id; gas += rows(mask,gas_id=int(gas_id))
        for conc in sorted(set(truth[mask].tolist())):
            cmask=mask & (truth==conc); concentration += rows(cmask,gas_id=int(gas_id),concentration=float(conc))
    special=rows((tc==3)&(truth==225)&(repeats==1),gas_id=3,concentration=225.0,repeat_id=1) if target=="C5" and np.any((tc==3)&(truth==225)&(repeats==1)) else []
    return {"overall":overall,"gas":gas,"concentration":concentration,"special":special}


def bootstrap_paired_group_deltas(truth, pred83, pred84, groups, gas_ranges, replicates=5000, seed=42):
    y, a, b, groups, ranges = map(np.asarray, (truth, pred83, pred84, groups, gas_ranges))
    unique = np.unique(groups.astype(str))
    indices = {g: np.flatnonzero(groups.astype(str) == g) for g in unique}
    strata = {}
    for group in unique:
        strata.setdefault(str(group).split("|", 1)[0], []).append(group)
    rng = np.random.default_rng(seed)
    out = {"rmse_delta": [], "mae_delta": [], "nrmse_range_delta": []}
    for _ in range(replicates):
        chosen = np.concatenate([rng.choice(values, len(values), replace=True) for _, values in sorted(strata.items())])
        idx = np.concatenate([indices[g] for g in chosen])
        ea, eb = a[idx] - y[idx], b[idx] - y[idx]
        out["rmse_delta"].append(float(np.sqrt(np.mean(eb**2)) - np.sqrt(np.mean(ea**2))))
        out["mae_delta"].append(float(np.mean(abs(eb)) - np.mean(abs(ea))))
        out["nrmse_range_delta"].append(float(np.sqrt(np.mean((eb/ranges[idx])**2)) - np.sqrt(np.mean((ea/ranges[idx])**2))))
    return out


def summarize_bootstrap(scope, output, *, point, n_groups, n_replicates):
    mapping={"RMSE":"rmse_delta","MAE":"mae_delta","NRMSE_range":"nrmse_range_delta"}
    return [{"scope":scope,"metric":metric,"point_delta":float(point[metric]),"ci_low":float(np.percentile(output[key],2.5)),
             "ci_high":float(np.percentile(output[key],97.5)),"n_groups":int(n_groups),"n_replicates":int(n_replicates),"seed":42}
            for metric,key in mapping.items()]


def decide_r84(pooled_rmse_delta, pooled_ci, target_deltas, gas_deltas, paired_rmse):
    severe = sorted(k for k, (old, new) in paired_rmse.items() if old > 0 and new / old > 1.05)
    if pooled_rmse_delta >= 0:
        decision = "CANONICAL_R84_NOT_SUPPORTED"
    elif pooled_ci[1] < 0 and all(v <= 0 for v in target_deltas.values()) and all(v <= 0 for v in gas_deltas.values()):
        decision = "CANONICAL_R84_SUPPORTED"
    else:
        decision = "CANONICAL_R84_DEVICE_DEPENDENT"
    return {"decision": decision, "severe_collapse": severe}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_r0_prerequisite(decision_path, index_path, *, expected_index_sha256):
    decision_path, index_path = Path(decision_path), Path(index_path)
    try:
        decision = json.loads(decision_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError("R0-v2 prerequisite invalid") from exc
    if (_sha256(index_path) != expected_index_sha256 or
        decision.get("study_id") != "CAN-V1-FEDRIDGE-R0V2-20260812" or
        decision.get("decision") != "FEDRIDGE_ALGEBRAIC_EXACT_NUMERICAL_EQUIVALENCE_ESTABLISHED" or
        not decision.get("evidence_complete")):
        raise RuntimeError("R0-v2 prerequisite not established")
    return decision


def _safe_regular_files(root):
    result={}
    for path in root.rglob("*"):
        if path.is_symlink(): raise RuntimeError("R0-v2 linked evidence forbidden")
        if path.is_file(): result[path.relative_to(root).as_posix()]=path
    return result


def validate_r0_bundle(root, *, expected_index_sha256):
    root=Path(root); index_path=root/"sha256_index.json"
    index_bytes=index_path.read_bytes()
    if hashlib.sha256(index_bytes).hexdigest()!=expected_index_sha256: raise RuntimeError("R0-v2 index anchor mismatch")
    try: index=json.loads(index_bytes)
    except Exception as exc: raise RuntimeError("R0-v2 index invalid") from exc
    files=_safe_regular_files(root); expected=set(files)-{"sha256_index.json","fixed_endpoint_complete.json"}; evidence_bytes={k:p.read_bytes() for k,p in files.items()}
    if set(index)!=expected or any(hashlib.sha256(evidence_bytes[k]).hexdigest()!=v for k,v in index.items()): raise RuntimeError("R0-v2 index coverage/hash mismatch")
    required={"model_lock.json","source_alpha_lock.json","R0_V2_DECISION.json","R0_V2_EXPERIMENT_AUDIT.md","fixed_endpoint_complete.json"}
    if not required <= set(files): raise RuntimeError("R0-v2 required evidence missing")
    model_bytes=evidence_bytes["model_lock.json"]; model=json.loads(model_bytes); alpha=json.loads(evidence_bytes["source_alpha_lock.json"]); decision=json.loads(evidence_bytes["R0_V2_DECISION.json"]); completion=json.loads(evidence_bytes["fixed_endpoint_complete.json"])
    canonical=json.dumps(model.get("models"),sort_keys=True,separators=(",",":"),allow_nan=False).encode(); models_hash=hashlib.sha256(canonical).hexdigest()
    if model.get("schema_version")!="iotj.canonical_v1.fedridge_r0_v2.execution.v1.model_lock" or model.get("study_id")!="CAN-V1-FEDRIDGE-R0V2-20260812" or model.get("models_sha256")!=models_hash: raise RuntimeError("R0-v2 model lock semantic mismatch")
    if set(model.get("models",{}))!={"0","1","2","3"}: raise RuntimeError("R0-v2 gas model coverage mismatch")
    for key,pair in model["models"].items():
        if set(pair)!={"federated","pooled"}: raise RuntimeError("R0-v2 route model mismatch")
        fed=pair["federated"]
        if fed.get("gas_id")!=int(key) or len(fed.get("feature_names",[]))!=104 or len(fed.get("mean",[]))!=104 or len(fed.get("scale",[]))!=104 or len(fed.get("coef",[]))!=105: raise RuntimeError("R0-v2 model dimensional mismatch")
        numeric=fed["mean"]+fed["scale"]+fed["coef"]+[fed.get("alpha"),fed.get("clip_min"),fed.get("clip_max")]
        if not all(math.isfinite(float(v)) for v in numeric) or any(float(v)<=0 for v in fed["scale"]): raise RuntimeError("R0-v2 model nonfinite/scale mismatch")
        if fed["alpha"]!=alpha.get("selected_alpha",{}).get(key,{}).get("federated"): raise RuntimeError("R0-v2 alpha lock mismatch")
    if decision.get("decision")!="FEDRIDGE_ALGEBRAIC_EXACT_NUMERICAL_EQUIVALENCE_ESTABLISHED" or not decision.get("evidence_complete") or completion.get("status")!="COMPLETE" or not completion.get("R1_released") or "PASS; Evidence eligible" not in evidence_bytes["R0_V2_EXPERIMENT_AUDIT.md"].decode("utf-8"): raise RuntimeError("R0-v2 decision/audit prerequisite mismatch")
    return {int(k):v["federated"] for k,v in model["models"].items()}


def validate_classifier_registry(registry):
    if set(registry) != {"C3", "C4", "C5"}:
        raise RuntimeError("classifier registry target mismatch")
    for target, record in registry.items():
        path = Path(record["path"])
        if not path.is_file() or _sha256(path) != record["sha256"]:
            raise RuntimeError(f"classifier checkpoint mismatch: {target}")
    return True


def predicted_classes_from_rows(rows):
    try:
        return np.asarray([int(row["pred_class"]) for row in rows], dtype=int)
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("classifier prediction schema mismatch") from exc


def assert_test_access_released(target, lock_state, access_events):
    required = {"alpha", "models", "classifier", "cache", "bootstrap"}
    state = lock_state.get(target, {})
    if set(state) != required or not all(state.values()):
        raise RuntimeError(f"target test remains locked: {target}")
    access_events.append({"target": target, "operation": "target_test_released", "sequence": len(access_events)})


def validate_access_chronology(events, lock_sha256, receipt):
    if receipt.get("lock_sha256") != lock_sha256:
        raise RuntimeError("target test release receipt lock mismatch")
    test_sequences=[int(e["sequence"]) for e in events if e.get("split")=="test"]
    if test_sequences and int(receipt.get("published_before_sequence",-1)) > min(test_sequences):
        raise RuntimeError("target test release receipt published after test access")
    if any(int(e.get("sequence",-1))!=i for i,e in enumerate(events)):
        raise RuntimeError("target access chronology sequence mismatch")
    return True


def validate_evidence_bundle(root):
    root = Path(root)
    index = json.loads((root / "sha256_index.json").read_text(encoding="utf-8"))
    files = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and p.name not in {"sha256_index.json", "COMPLETE.json"}}
    if set(index) != files or any(_sha256(root / name) != digest for name, digest in index.items()):
        raise RuntimeError("evidence hash coverage mismatch")
    decision = json.loads((root / "R1_DECISION.json").read_text(encoding="utf-8"))
    lock = json.loads((root / "target_model_lock.json").read_text(encoding="utf-8"))
    if decision.get("decision") not in {"CANONICAL_R84_SUPPORTED", "CANONICAL_R84_DEVICE_DEPENDENT", "CANONICAL_R84_NOT_SUPPORTED"}:
        raise RuntimeError("evidence semantic decision mismatch")
    if not lock.get("all_targets_locked"):
        raise RuntimeError("evidence semantic lock mismatch")
    for name in ("target_alpha_lock.json","classifier_lock.json","bootstrap_design_lock.json","target_test_release_receipt.json","DATA_ACCESS_AUDIT.json"):
        if not (root/name).is_file(): raise RuntimeError(f"evidence semantic required lock missing: {name}")
    bootstrap_lock=json.loads((root/"bootstrap_design_lock.json").read_text(encoding="utf-8"))
    if bootstrap_lock.get("replicates")!=5000 or bootstrap_lock.get("seed")!=42 or not bootstrap_lock.get("paired"): raise RuntimeError("evidence semantic bootstrap lock mismatch")
    for row in _csv_rows(root/"canonical_regression_bootstrap.csv"):
        values=[float(row[k]) for k in ("point_delta","ci_low","ci_high")]
        if int(row["n_replicates"])!=5000 or int(row["seed"])!=42 or int(row["n_groups"])<1 or not all(math.isfinite(v) for v in values) or values[1]>values[2]: raise RuntimeError("evidence semantic bootstrap summary mismatch")
    access=json.loads((root/"DATA_ACCESS_AUDIT.json").read_text(encoding="utf-8")); receipt=json.loads((root/"target_test_release_receipt.json").read_text(encoding="utf-8"))
    if not access.get("target_test_opened_after_all_locks") or not validate_access_chronology(access.get("events",[]),receipt.get("lock_sha256",{}),receipt): raise RuntimeError("evidence semantic access chronology mismatch")
    _validate_metric_csv_semantics(root)
    return True


def _csv_rows(path):
    with Path(path).open(encoding="utf-8",newline="") as handle: return list(csv.DictReader(handle))


def _close(a,b,tol=1e-10):
    a=float(a); b=float(b)
    return (math.isnan(a) and math.isnan(b)) or abs(a-b)<=tol


def _validate_metric_csv_semantics(root):
    root=Path(root); predictions=_csv_rows(root/"predictions.csv")
    if not predictions: raise RuntimeError("evidence semantic predictions empty")
    expected={"overall":[],"gas":[],"concentration":[],"special":[]}
    for method in sorted(set(r["method"] for r in predictions)):
        for target in sorted(set(r["target"] for r in predictions)) + ["POOLED"]:
            subset=[r for r in predictions if r["method"]==method and (target=="POOLED" or r["target"]==target)]
            if not subset: continue
            y=np.asarray([float(r["truth"]) for r in subset]); tc=np.asarray([int(r["true_class"]) for r in subset]); pc=np.asarray([int(r["predicted_class"]) for r in subset]); matrix=np.asarray([[float(r[f"route_{g}"]) for g in range(4)] for r in subset]); repeats=np.asarray([int(r["repeat_id"]) for r in subset]); ranges={g:float(next(r["gas_range"] for r in subset if int(r["true_class"])==g)) for g in sorted(set(tc.tolist()))}
            slices=metric_rows_for_slices(target,method,y,tc,pc,matrix,ranges,repeats)
            for name in expected: expected[name]+=slices[name]
    files={"overall":"canonical_regression_comparison.csv","gas":"canonical_regression_per_gas.csv","concentration":"canonical_regression_per_concentration.csv","special":"c5_methane_225_repeat1.csv"}
    for family,filename in files.items():
        observed=_csv_rows(root/filename); keys=["target","method","scope"] + (["gas_id"] if family in {"gas","concentration","special"} else []) + (["concentration"] if family in {"concentration","special"} else []) + (["repeat_id"] if family=="special" else [])
        def key(row): return tuple(str(row.get(k,"")) for k in keys)
        emap={key(r):r for r in expected[family]}; omap={key(r):r for r in observed}
        if set(emap)!=set(omap): raise RuntimeError(f"evidence semantic {family} key mismatch")
        for k,e in emap.items():
            o=omap[k]
            for field in ("n","rmse","mae","nrmse_range","r2","bias"):
                if not _close(e[field],o[field]): raise RuntimeError(f"evidence semantic {family} metric mismatch")
    return True


def _write_csv_rows(path, rows, fields=None):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    fields=fields or list(rows[0])
    with path.open("w",encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def write_synthetic_semantic_bundle(root):
    """Small functional audit fixture; never accepted as formal evidence."""
    root=Path(root); root.mkdir(parents=True)
    predictions=[]
    for i,(truth,tc,pc,repeat) in enumerate([(25.,0,0,1),(50.,0,0,2),(225.,3,3,1),(225.,3,0,2)]):
        routes=[truth+1,truth+2,truth+3,truth+4]
        predictions.append({"target":"C5","method":"R84_CONCAT","sample_index":i,"physical_identity":f"id{i}","filename":f"f{i//2}","repeat_id":repeat,"gas_code":"Me" if tc==3 else "Et","true_class":tc,"predicted_class":pc,"concentration":truth,"truth":truth,"prediction":routes[pc],"gas_range":225 if tc==3 else 100,**{f"route_{g}":routes[g] for g in range(4)}})
    _write_csv_rows(root/"predictions.csv",predictions)
    y=np.asarray([r["truth"] for r in predictions]); tc=np.asarray([r["true_class"] for r in predictions]); pc=np.asarray([r["predicted_class"] for r in predictions]); mx=np.asarray([[r[f"route_{g}"] for g in range(4)] for r in predictions]); repeats=np.asarray([r["repeat_id"] for r in predictions]); ranges={0:100.,3:225.}
    target=metric_rows_for_slices("C5","R84_CONCAT",y,tc,pc,mx,ranges,repeats); pooled=metric_rows_for_slices("POOLED","R84_CONCAT",y,tc,pc,mx,ranges,repeats)
    _write_csv_rows(root/"canonical_regression_comparison.csv",target["overall"]+pooled["overall"])
    _write_csv_rows(root/"canonical_regression_per_gas.csv",target["gas"]+pooled["gas"])
    _write_csv_rows(root/"canonical_regression_per_concentration.csv",target["concentration"]+pooled["concentration"])
    _write_csv_rows(root/"c5_methane_225_repeat1.csv",target["special"])
    boot=[]
    for scope in ("C5","POOLED","POOLED_GAS_0","POOLED_GAS_3"):
        for metric in ("RMSE","MAE","NRMSE_range"): boot.append({"scope":scope,"metric":metric,"point_delta":-1,"ci_low":-2,"ci_high":0,"n_groups":2,"n_replicates":5000,"seed":42})
    _write_csv_rows(root/"canonical_regression_bootstrap.csv",boot)
    locks={n:"a"*64 for n in ("target_alpha_lock.json","target_model_lock.json","classifier_lock.json","bootstrap_design_lock.json")}
    payloads={"target_alpha_lock.json":{"study_id":"synthetic","all_targets_locked":True},"target_model_lock.json":{"study_id":"synthetic","all_targets_locked":True},"classifier_lock.json":{"study_id":"synthetic"},"bootstrap_design_lock.json":{"replicates":5000,"seed":42,"paired":True}}
    for name,payload in payloads.items(): (root/name).write_text(json.dumps(payload),encoding="utf-8")
    receipt={"lock_sha256":locks,"published_before_sequence":0}; (root/"target_test_release_receipt.json").write_text(json.dumps(receipt),encoding="utf-8"); (root/"DATA_ACCESS_AUDIT.json").write_text(json.dumps({"events":[{"sequence":0,"target":"C5","split":"test","artifact":"features","operation":"load"}],"target_test_opened_after_all_locks":True}),encoding="utf-8")
    (root/"R1_DECISION.json").write_text(json.dumps({"decision":"CANONICAL_R84_DEVICE_DEPENDENT","severe_collapse":[]}),encoding="utf-8")
    index={p.relative_to(root).as_posix():_sha256(p) for p in root.rglob("*") if p.is_file() and p.name not in {"sha256_index.json","COMPLETE.json"}}; (root/"sha256_index.json").write_text(json.dumps(index),encoding="utf-8")
    return root
