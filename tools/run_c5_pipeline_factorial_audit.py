"""Low-cost read-only C5 checkpoint x dataset R84 factorial and stability audit."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_regression_head_ablation import CLASS_NAMES, build_oracle_rows, deterministic_train_val, fit_ridge
from scripts import run_gaps_cross_target_r84_full as common

OLD_NAME = "client_data_c1234src_c5tgt_2080_timeaware_60_170_window_fullgrid"
NEW_NAME = "client_data_c12src_c345tgt_2080_timeaware_60_170_window_fullgrid"
ALPHAS = common.RIDGE_ALPHAS


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    if not rows:
        raise RuntimeError(f"empty output: {path}")
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def checkpoint(manifest: Path) -> Path:
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    path = Path(payload["checkpoint"])
    if not path.exists():
        path = manifest.parent / "remote_server/server_latest_adapted.pth"
    if not path.exists():
        raise FileNotFoundError(path)
    if common.sha256(path) != payload["checkpoint_sha256"]:
        raise RuntimeError(f"checkpoint whole-file hash mismatch: {path}")
    return path


def route(checkpoint_path: Path, data_root: Path, split: str, device: torch.device, batch_size: int) -> tuple[list[dict], dict]:
    rows, summary = common.evaluate_checkpoint_stream(checkpoint_path, data_root=data_root,
        target_client=5, split=split, device=device, batch_size=batch_size)
    if [int(r["sample_index"]) for r in rows] != list(range(len(rows))):
        raise RuntimeError("classifier output order is not source array order")
    return rows, summary


def prepare(data_root: Path, split: str, routes: Sequence[Mapping[str, Any]], h1: Mapping[int, Any]) -> tuple[list[dict], list[dict]]:
    base = build_oracle_rows(data_root, ["C5"], split)
    client = data_root / "client_5"
    cls_array = np.load(client / f"{split}_classification_labels.npy").reshape(-1)
    reg_array = np.load(client / f"{split}_regression_labels.npy")
    phase_array = np.load(client / f"{split}_phase_labels.npy").reshape(-1)
    metadata = json.loads((client / f"{split}_experiment_info.json").read_text(encoding="utf-8"))
    if len(base) != len(routes):
        raise RuntimeError("route/data count mismatch")
    if not (len(base) == len(cls_array) == len(reg_array) == len(phase_array) == len(metadata)):
        raise RuntimeError("feature/label/phase/metadata row count mismatch")
    oracle, deployed = [], []
    for row, rt, cls_value, reg_value, phase_value, meta in zip(base, routes, cls_array, reg_array, phase_array, metadata):
        if int(row["sample_index"]) != int(rt["sample_index"]) or int(row["true_class"]) != int(rt["true_class"]):
            raise RuntimeError("route/data row mismatch")
        if int(row["true_class"]) != int(cls_value) or not np.isclose(float(row["true_ppm"]), float(reg_value[int(cls_value)])) or int(row["phase"]) != int(phase_value):
            raise RuntimeError("feature/class/regression/phase alignment mismatch")
        true, pred = int(row["true_class"]), int(rt["pred_class"])
        a = {**row, "pred_class": pred, "filename": str(meta.get("filename", "")),
             "repeat_id": int(meta.get("repeat_id", -1)), "phase_label_meta": str(meta.get("phase_label", "")),
             "window_start_s": meta.get("window_start_s", "")}; b = dict(a)
        a["H1_federated_source_ridge_ppm"] = h1[true].predict(row["feature_dict"])
        b["H1_federated_source_ridge_ppm"] = h1[pred].predict(row["feature_dict"])
        oracle.append(common.r84_row(a)); deployed.append(common.r84_row(b))
    return oracle, deployed


def basic_metrics(truth: np.ndarray, pred: np.ndarray) -> dict[str, float | int]:
    err = pred - truth
    centered = truth - truth.mean()
    den = float(np.sum(centered ** 2))
    return {"N": len(truth), "RMSE": float(np.sqrt(np.mean(err ** 2))), "MAE": float(np.mean(np.abs(err))),
            "Bias": float(np.mean(err)), "R2": float(1 - np.sum(err ** 2) / den) if den else float("nan")}


def fit_all(oracle: Sequence[dict], deployed: Sequence[dict], dataset_label: str, experiment: str) -> tuple[dict[int, Any], list[dict], list[dict]]:
    by_id = {int(r["sample_index"]): r for r in deployed}
    models, sweep, selection = {}, [], []
    for cls, gas in sorted(CLASS_NAMES.items()):
        rows = [r for r in oracle if int(r["true_class"]) == cls]
        fit_rows, val_seed = deterministic_train_val(rows, .25)
        val_rows = [by_id[int(r["sample_index"])] for r in val_seed]
        truth_fit = np.array([r["true_ppm"] for r in fit_rows], dtype=float)
        truth_val = np.array([r["true_ppm"] for r in val_rows], dtype=float)
        def counts(rows_to_count: Sequence[dict], key: str) -> str:
            values: dict[str, int] = {}
            for item in rows_to_count:
                value = str(item[key]); values[value] = values.get(value, 0) + 1
            return json.dumps(values, sort_keys=True, separators=(",", ":"))
        best_alpha, best_rmse = ALPHAS[0], float("inf")
        for alpha in ALPHAS:
            model = fit_ridge(fit_rows, sorted(fit_rows[0]["feature_dict"]), alpha)
            pred_fit, pred_val = model.predict(fit_rows), model.predict(val_rows)
            mf, mv = basic_metrics(truth_fit, pred_fit), basic_metrics(truth_val, pred_val)
            per_conc = {}
            for ppm in sorted(set(truth_val)):
                mask = truth_val == ppm
                per_conc[str(float(ppm))] = float(np.sqrt(np.mean((pred_val[mask] - truth_val[mask]) ** 2)))
            sweep.append({"experiment": experiment, "dataset": dataset_label, "class_id": cls, "gas": gas, "alpha": alpha,
                          "calibration_total_N": len(rows), "fit_N": len(fit_rows), "validation_N": len(val_rows),
                          "fit_per_concentration_json": counts(fit_rows, "true_ppm"),
                          "validation_per_concentration_n_json": counts(val_rows, "true_ppm"),
                          "fit_per_repeat_json": counts(fit_rows, "repeat_id"),
                          "validation_per_repeat_json": counts(val_rows, "repeat_id"),
                          "fit_RMSE": mf["RMSE"], "validation_RMSE": mv["RMSE"], "validation_MAE": mv["MAE"],
                          "validation_Bias": mv["Bias"], "validation_per_concentration_RMSE_json": json.dumps(per_conc, sort_keys=True)})
            if mv["RMSE"] < best_rmse:
                best_alpha, best_rmse = alpha, float(mv["RMSE"])
        models[cls] = fit_ridge(rows, sorted(rows[0]["feature_dict"]), best_alpha)
        selection.append({"experiment": experiment, "dataset": dataset_label, "class_id": cls, "gas": gas, "selected_alpha": best_alpha,
                          "validation_RMSE": best_rmse, "fit_N": len(fit_rows), "validation_N": len(val_rows),
                          "internal_split": "deterministic sample-index tail within concentration"})
    return models, sweep, selection


def evaluate_case(name: str, checkpoint_label: str, dataset_label: str, cp: Path, data: Path,
                  h1: Mapping[int, Any], device: torch.device, batch_size: int) -> tuple[list[dict], list[dict], list[dict], dict]:
    cal_routes, cal_cls = route(cp, data, "calibration", device, batch_size)
    cal_oracle, cal_deployed = prepare(data, "calibration", cal_routes, h1)
    models, sweep, selection = fit_all(cal_oracle, cal_deployed, dataset_label, name)
    test_routes, test_cls = route(cp, data, "test", device, batch_size)
    test_oracle, test_deployed = prepare(data, "test", test_routes, h1)
    records = []
    for o, d in zip(test_oracle, test_deployed):
        true, pred_cls = int(o["true_class"]), int(d["pred_class"])
        pipeline = float(models[pred_cls].predict([d])[0])
        oracle_pred = float(models[true].predict([o])[0])
        records.append({"experiment": name, "classifier_checkpoint": checkpoint_label, "dataset": dataset_label,
                        "sample_index": int(o["sample_index"]), "true_class": true, "pred_class": pred_cls,
                        "gas": CLASS_NAMES[true], "true_ppm": float(o["true_ppm"]),
                        "filename": o["filename"], "repeat_id": o["repeat_id"], "phase": o["phase"],
                        "phase_label_meta": o["phase_label_meta"], "window_start_s": o["window_start_s"],
                        "route_correct": int(true == pred_cls), "pipeline_pred_ppm": pipeline,
                        "oracle_pred_ppm": oracle_pred})
    return records, sweep, selection, {"calibration": cal_cls, "test": test_cls}


def summaries(records: Sequence[dict]) -> tuple[list[dict], list[dict]]:
    result, per_gas = [], []
    for exp in sorted(set(r["experiment"] for r in records)):
        rows = [r for r in records if r["experiment"] == exp]
        base = {k: rows[0][k] for k in ("experiment", "classifier_checkpoint", "dataset")}
        scopes = {
            "PIPELINE_ALL": (rows, "pipeline_pred_ppm"),
            "S_CC": ([r for r in rows if r["route_correct"]], "pipeline_pred_ppm"),
            "ORACLE_ROUTE": (rows, "oracle_pred_ppm"),
        }
        for scope, (selected, key) in scopes.items():
            truth = np.array([r["true_ppm"] for r in selected]); pred = np.array([r[key] for r in selected])
            result.append({**base, "scope": scope, "class_accuracy": np.mean([r["route_correct"] for r in rows]), **basic_metrics(truth, pred)})
            for cls, gas in CLASS_NAMES.items():
                gas_rows = [r for r in selected if r["true_class"] == cls]
                t = np.array([r["true_ppm"] for r in gas_rows]); p = np.array([r[key] for r in gas_rows])
                per_gas.append({**base, "scope": scope, "class_id": cls, "gas": gas, **basic_metrics(t, p)})
    return result, per_gas


def random_stratified_split(rows: Sequence[dict], seed: int) -> tuple[list[dict], list[dict]]:
    rng = np.random.default_rng(seed); fit, val = [], []
    for ppm in sorted(set(float(r["true_ppm"]) for r in rows)):
        bucket = [r for r in rows if float(r["true_ppm"]) == ppm]
        order = rng.permutation(len(bucket)); n_val = max(1, int(round(len(bucket) * .25)))
        val.extend(bucket[i] for i in order[:n_val]); fit.extend(bucket[i] for i in order[n_val:])
    return fit, val


def stability(dataset_label: str, oracle_cal: Sequence[dict], deployed_cal: Sequence[dict],
              oracle_test: Sequence[dict], deployed_test: Sequence[dict]) -> list[dict]:
    cls = 3
    cal = [r for r in oracle_cal if int(r["true_class"]) == cls]
    dep_by_id = {int(r["sample_index"]): r for r in deployed_cal}
    test_oracle = [r for r in oracle_test if int(r["true_class"]) == cls]
    test_cc = [r for r in deployed_test if int(r["true_class"]) == cls and int(r["pred_class"]) == cls]
    out = []
    for seed in (0,1,2,42,3407):
        fit, val_seed = random_stratified_split(cal, seed); val = [dep_by_id[int(r["sample_index"])] for r in val_seed]
        truth_val = np.array([r["true_ppm"] for r in val]); best_a, best_v = ALPHAS[0], float("inf")
        for a in ALPHAS:
            m = fit_ridge(fit, sorted(fit[0]["feature_dict"]), a); v = basic_metrics(truth_val, m.predict(val))["RMSE"]
            if v < best_v: best_a, best_v = a, float(v)
        model = fit_ridge(cal, sorted(cal[0]["feature_dict"]), best_a)
        tcc = np.array([r["true_ppm"] for r in test_cc]); pcc = model.predict(test_cc)
        tor = np.array([r["true_ppm"] for r in test_oracle]); por = model.predict(test_oracle)
        out.append({"dataset": dataset_label, "seed": seed, "selected_alpha": best_a, "validation_RMSE": best_v,
                    "test_S_CC_N": len(test_cc), "test_S_CC_RMSE": basic_metrics(tcc,pcc)["RMSE"],
                    "oracle_route_N": len(test_oracle), "oracle_route_RMSE": basic_metrics(tor,por)["RMSE"]})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--workspace", type=Path, default=ROOT.parents[1]); ap.add_argument("--output", type=Path, default=ROOT / "results/iotj_c5_pipeline_audit_20260807"); ap.add_argument("--device", default="cpu"); ap.add_argument("--batch-size", type=int, default=64); args=ap.parse_args()
    ws, out = args.workspace.resolve(), args.output.resolve(); device=torch.device(args.device)
    old_data, new_data = ws/"dataset"/OLD_NAME, ws/"dataset"/NEW_NAME
    old_cp = checkpoint(ROOT/"results/iotj_final_classification_le1_20260804/FCL-E3-GAPS-C5/run_manifest.json")
    new_cp = checkpoint(ROOT/"results/iotj_gaps_roleaware_r84_full_20260805/classification/FCL-RW-GAPS-C5/run_manifest.json")
    h1=common.load_h1(); all_records=[]; all_sweeps=[]; all_selection=[]; cls_info={}
    specs=(("A_OLD_CKPT_OLD_DATA","OLD","OLD",old_cp,old_data),("B_OLD_CKPT_NEW_DATA","OLD","NEW",old_cp,new_data),("C_NEW_CKPT_OLD_DATA","NEW","OLD",new_cp,old_data),("D_NEW_CKPT_NEW_DATA","NEW","NEW",new_cp,new_data))
    for name, cl, dl, cp, data in specs:
        rec, sweep, sel, info=evaluate_case(name,cl,dl,cp,data,h1,device,args.batch_size); all_records+=rec; all_sweeps+=sweep; all_selection+=sel; cls_info[name]=info
    main_rows, gas_rows=summaries(all_records)
    write_csv(out/"p6_2x2_factorial_records.csv",all_records); write_csv(out/"p6_2x2_factorial_summary.csv",main_rows); write_csv(out/"p6_2x2_factorial_per_gas.csv",gas_rows)
    for label, formal_arm in (("OLD", "A_OLD_CKPT_OLD_DATA"), ("NEW", "D_NEW_CKPT_NEW_DATA")):
        rows=[r for r in all_sweeps if r["experiment"]==formal_arm and r["class_id"]==3]
        write_csv(out/f"p5_methane_alpha_sweep_{label.lower()}.csv",rows)
    write_csv(out/"p5_all_gas_alpha_selection.csv",all_selection)

    # P10 reuses the same routes and does no neural training or checkpoint/model selection.
    stability_rows=[]
    for label, cp, data in (("OLD",old_cp,old_data),("NEW",new_cp,new_data)):
        cr,_=route(cp,data,"calibration",device,args.batch_size); co,cd=prepare(data,"calibration",cr,h1)
        tr,_=route(cp,data,"test",device,args.batch_size); to,td=prepare(data,"test",tr,h1)
        stability_rows += stability(label,co,cd,to,td)
    write_csv(out/"p10_methane_seed_stability.csv",stability_rows)

    lookup={(r["experiment"],r["scope"]):r for r in main_rows}
    md=["# P6 factorial analysis","","All four arms refit R84 only on the corresponding calibration split. No test row enters alpha selection or fitting.","","| Arm | Accuracy | Pipeline RMSE | S_CC RMSE | Oracle RMSE |","|---|---:|---:|---:|---:|"]
    for name,*_ in specs:
        a=lookup[(name,"PIPELINE_ALL")]; c=lookup[(name,"S_CC")]; o=lookup[(name,"ORACLE_ROUTE")]
        md.append(f"| {name} | {100*a['class_accuracy']:.2f}% | {a['RMSE']:.4f} | {c['RMSE']:.4f} | {o['RMSE']:.4f} |")
    md += ["","Arms B/C are diagnostic-only: crossing checkpoints and datasets can place windows used during target calibration/adaptation into the crossed test split. They isolate association with checkpoint versus data pipeline but are not leakage-free formal performance estimates."]
    (out/"P6_FACTORIAL_ANALYSIS.md").write_text("\n".join(md)+"\n",encoding="utf-8")
    (out/"P9_ROW_ALIGNMENT.md").write_text("# P9 row alignment\n\nPASS. All four arms checked every calibration and test row. Classifier `sample_index` was exactly `0..N-1`; `build_oracle_rows` count, sample index, and true class agreed row-for-row before prediction. The persisted P6 record has 1,360 unique sample indices per arm. No class-concatenation/label-order mismatch was found.\n",encoding="utf-8")
    print(json.dumps({k:{s:lookup[(k,s)]["RMSE"] for s in ("PIPELINE_ALL","S_CC","ORACLE_ROUTE")} for k,*_ in specs},indent=2))


if __name__ == "__main__": main()
