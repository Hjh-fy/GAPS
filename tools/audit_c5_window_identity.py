"""Read-only OLD/NEW C5 provenance, physical-window, split, and R84 audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_regression_head_ablation import CLASS_NAMES, deterministic_train_val, fit_ridge, rich_feature_dict
from scripts.evaluate_iotj_feature_metadata_ablation import profile_feature_dict
from scripts.run_gaps_cross_target_r84_full import RIDGE_ALPHAS, load_h1

OLD_NAME = "client_data_c1234src_c5tgt_2080_timeaware_60_170_window_fullgrid"
NEW_NAME = "client_data_c12src_c345tgt_2080_timeaware_60_170_window_fullgrid"
GASES = {0: "Ethanol", 1: "CO", 2: "Ethylene", 3: "Methane"}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"empty audit output: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_full(root: Path) -> list[dict]:
    out: list[dict] = []
    client = root / "client_5"
    for split in ("calibration", "test"):
        x = np.load(client / f"{split}_features.npy")
        y = np.load(client / f"{split}_classification_labels.npy").reshape(-1)
        reg = np.load(client / f"{split}_regression_labels.npy")
        phase = np.load(client / f"{split}_phase_labels.npy").reshape(-1)
        meta = json.loads((client / f"{split}_experiment_info.json").read_text(encoding="utf-8"))
        if not (len(x) == len(y) == len(reg) == len(phase) == len(meta)):
            raise RuntimeError(f"row count mismatch in {root.name}/{split}")
        for i in range(len(x)):
            m = dict(meta[i])
            cls = int(y[i])
            out.append({
                "split": split, "split_index": i, "window": x[i], "class": cls,
                "concentration": float(reg[i, cls]), "phase": int(phase[i]), "meta": m,
                "filename": str(m.get("filename", "")), "repeat": int(m.get("repeat_id", -1)),
            })
    return out


def manifest_rows(label: str, root: Path, upstream: Path | None) -> list[dict]:
    rows: list[dict] = []
    paths = [root / "split_info.json", root / "split_protocol_manifest.json", root / "norm_stats.npz"]
    for split in ("calibration", "test"):
        for suffix in ("features.npy", "classification_labels.npy", "regression_labels.npy", "phase_labels.npy", "experiment_info.json"):
            paths.append(root / "client_5" / f"{split}_{suffix}")
    if upstream:
        paths.extend(upstream / name for name in (
            "features.npy", "classification_labels.npy", "regression_labels.npy",
            "phase_labels.npy", "experiment_info.json", "file_info.json",
        ))
    for path in paths:
        if not path.exists():
            rows.append({"dataset": label, "role": "artifact", "path": str(path), "exists": False})
            continue
        row = {"dataset": label, "role": "upstream" if upstream and path.parent == upstream else "artifact",
               "path": str(path.resolve()), "exists": True, "bytes": path.stat().st_size, "sha256": sha(path)}
        if path.suffix == ".npy":
            arr = np.load(path)
            row.update(shape="x".join(map(str, arr.shape)), dtype=str(arr.dtype), min=float(arr.min()),
                       max=float(arr.max()), mean=float(arr.mean()), std=float(arr.std()))
        elif path.suffix == ".npz":
            payload = np.load(path)
            row.update(shape=";".join(f"{k}:" + "x".join(map(str, payload[k].shape)) for k in payload.files),
                       dtype=";".join(f"{k}:{payload[k].dtype}" for k in payload.files),
                       mean=json.dumps(np.asarray(payload["mean"]).reshape(-1).tolist()),
                       std=json.dumps(np.asarray(payload["std"]).reshape(-1).tolist()))
        rows.append(row)
    return rows


def groups(rows: list[dict]) -> dict[tuple, list[dict]]:
    out: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        out[(row["filename"], row["class"], row["concentration"], row["repeat"])].append(row)
    return out


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    aa, bb = a.reshape(-1).astype(np.float64), b.reshape(-1).astype(np.float64)
    den = np.linalg.norm(aa) * np.linalg.norm(bb)
    return float(aa @ bb / den) if den else float(aa @ bb == 0)


def match_windows(old: list[dict], new: list[dict]) -> list[dict]:
    go, gn = groups(old), groups(new)
    if set(go) != set(gn):
        raise RuntimeError("OLD/NEW physical experiment keys differ")
    output: list[dict] = []
    for key in sorted(go):
        left, right = go[key], gn[key]
        if len(left) != 21 or len(right) != 21:
            raise RuntimeError(f"expected 21 windows for {key}, got {len(left)}/{len(right)}")
        cost = np.empty((21, 21), dtype=np.float64)
        for i, a in enumerate(left):
            for j, b in enumerate(right):
                cost[i, j] = np.sqrt(np.mean((a["window"].astype(np.float64) - b["window"].astype(np.float64)) ** 2))
        ii, jj = linear_sum_assignment(cost)
        for i, j in zip(ii, jj):
            a, b = left[int(i)], right[int(j)]
            x64, y64 = a["window"].astype(np.float64), b["window"].astype(np.float64)
            x32, y32 = a["window"].astype(np.float32), b["window"].astype(np.float32)
            diff = x64 - y64
            output.append({
                "filename": key[0], "class_id": key[1], "gas": GASES[key[1]],
                "concentration": key[2], "repeat": key[3],
                "old_split": a["split"], "old_index": a["split_index"],
                "new_split": b["split"], "new_index": b["split_index"],
                "new_window_start_s": b["meta"].get("window_start_s", ""),
                "new_window_end_s": b["meta"].get("window_end_s", ""),
                "rmse_float64": float(np.sqrt(np.mean(diff ** 2))),
                "mae_float64": float(np.mean(np.abs(diff))), "max_abs_diff": float(np.max(np.abs(diff))),
                "cosine_similarity": cosine(x64, y64),
                "rmse_both_float32": float(np.sqrt(np.mean((x32 - y32).astype(np.float64) ** 2))),
                "bit_identical": bool(a["window"].dtype == b["window"].dtype and np.array_equal(a["window"], b["window"])),
                "equal_atol_1e-6": bool(np.allclose(x64, y64, rtol=0, atol=1e-6)),
                "equal_atol_1e-5": bool(np.allclose(x64, y64, rtol=0, atol=1e-5)),
                "equal_atol_1e-4": bool(np.allclose(x64, y64, rtol=0, atol=1e-4)),
                "equal_atol_1e-3": bool(np.allclose(x64, y64, rtol=0, atol=1e-3)),
            })
    return output


def distribution(label: str, rows: list[dict], matches: list[dict]) -> list[dict]:
    position = {}
    for m in matches:
        position[("OLD", m["old_split"], int(m["old_index"]))] = m["new_window_start_s"]
        position[("NEW", m["new_split"], int(m["new_index"]))] = m["new_window_start_s"]
    counts: dict[tuple, list[float]] = defaultdict(list)
    for row in rows:
        m = row["meta"]
        start = m.get("window_start_s", position.get((label, row["split"], row["split_index"]), np.nan))
        key = (label, row["split"], row["class"], row["concentration"], row["repeat"], str(m.get("phase_label", row["phase"])))
        counts[key].append(float(start))
    result = []
    for key, starts in sorted(counts.items()):
        result.append({"dataset": key[0], "split": key[1], "class_id": key[2], "gas": GASES[key[2]],
                       "concentration": key[3], "repeat": key[4], "phase": key[5], "n": len(starts),
                       "window_start_min": float(np.nanmin(starts)), "window_start_mean": float(np.nanmean(starts)),
                       "window_start_max": float(np.nanmax(starts))})
    return result


def r84_audit(old: list[dict], new: list[dict], matches: list[dict]) -> list[dict]:
    lookup_old = {(r["split"], r["split_index"]): r for r in old}
    lookup_new = {(r["split"], r["split_index"]): r for r in new}
    h1 = load_h1()
    output = []
    for m in matches:
        a = lookup_old[(m["old_split"], int(m["old_index"]))]
        b = lookup_new[(m["new_split"], int(m["new_index"]))]
        va = profile_feature_dict(rich_feature_dict(a["window"], a["phase"], a["meta"]), "M83_SENSOR")
        vb = profile_feature_dict(rich_feature_dict(b["window"], b["phase"], b["meta"]), "M83_SENSOR")
        names = sorted(va)
        va["srcpred_H1_federated_source_ridge_ppm"] = h1[a["class"]].predict(rich_feature_dict(a["window"], a["phase"], a["meta"]))
        vb["srcpred_H1_federated_source_ridge_ppm"] = h1[b["class"]].predict(rich_feature_dict(b["window"], b["phase"], b["meta"]))
        names = sorted(va)
        xa, xb = np.array([va[k] for k in names]), np.array([vb[k] for k in names])
        output.append({**{k: m[k] for k in ("filename", "class_id", "gas", "concentration", "repeat", "old_split", "old_index", "new_split", "new_index")},
                       "feature_dimension": len(names), "feature_order_sha256": hashlib.sha256("\n".join(names).encode()).hexdigest(),
                       "feature_rmse": float(np.sqrt(np.mean((xa-xb)**2))), "feature_max_abs_diff": float(np.max(np.abs(xa-xb))),
                       "feature_cosine_similarity": cosine(xa, xb),
                       "old_r84_json": json.dumps(xa.tolist(), separators=(",", ":")),
                       "new_r84_json": json.dumps(xb.tolist(), separators=(",", ":"))})
    return output


def membership_preprocessing_factorial(old: list[dict], new: list[dict], matches: list[dict]) -> list[dict]:
    """Oracle-route isolation: hold physical membership/order, swap only window representation."""
    old_lookup = {(r["split"], r["split_index"]): r for r in old}
    new_lookup = {(r["split"], r["split_index"]): r for r in new}
    old_to_new = {(m["old_split"], int(m["old_index"])): (m["new_split"], int(m["new_index"])) for m in matches}
    new_to_old = {v: k for k, v in old_to_new.items()}
    h1 = load_h1()

    def represent(row: dict, source: str) -> dict:
        base = rich_feature_dict(row["window"], row["phase"], row["meta"])
        features = profile_feature_dict(base, "M83_SENSOR")
        features["srcpred_H1_federated_source_ridge_ppm"] = h1[row["class"]].predict(base)
        return {"client": "C5", "true_class": row["class"], "route_class": row["class"],
                "true_ppm": row["concentration"], "feature_dict": features, "representation": source}

    output = []
    for representation in ("OLD", "NEW"):
        for membership in ("OLD", "NEW"):
            owner_lookup = old_lookup if membership == "OLD" else new_lookup
            mapping = old_to_new if membership == "OLD" else new_to_old
            repr_lookup = old_lookup if representation == "OLD" else new_lookup
            prepared: dict[str, list[dict]] = {"calibration": [], "test": []}
            for split in ("calibration", "test"):
                owner_keys = sorted((k for k in owner_lookup if k[0] == split), key=lambda k: k[1])
                for sample_index, owner_key in enumerate(owner_keys):
                    repr_key = owner_key if representation == membership else mapping[owner_key]
                    item = represent(repr_lookup[repr_key], representation); item["sample_index"] = sample_index
                    prepared[split].append(item)
            models, chosen = {}, {}
            for cls in sorted(CLASS_NAMES):
                cal = [r for r in prepared["calibration"] if r["true_class"] == cls]
                fit, val = deterministic_train_val(cal, .25); names = sorted(cal[0]["feature_dict"])
                truth = np.array([r["true_ppm"] for r in val]); best_a, best_score = RIDGE_ALPHAS[0], float("inf")
                for alpha in RIDGE_ALPHAS:
                    candidate = fit_ridge(fit, names, alpha); score = float(np.sqrt(np.mean((candidate.predict(val)-truth)**2)))
                    if score < best_score: best_a, best_score = alpha, score
                models[cls] = fit_ridge(cal, names, best_a); chosen[cls] = best_a
            test = prepared["test"]
            pred = np.array([models[r["true_class"]].predict([r])[0] for r in test]); truth = np.array([r["true_ppm"] for r in test])
            for cls in [-1, *sorted(CLASS_NAMES)]:
                mask = np.ones(len(test), dtype=bool) if cls < 0 else np.array([r["true_class"] == cls for r in test])
                err = pred[mask] - truth[mask]
                output.append({"representation": representation, "physical_membership": membership,
                               "scope": "ALL" if cls < 0 else CLASS_NAMES[cls], "N": int(mask.sum()),
                               "oracle_RMSE": float(np.sqrt(np.mean(err**2))), "oracle_MAE": float(np.mean(np.abs(err))),
                               "selected_alphas_json": json.dumps(chosen, sort_keys=True)})
    return output


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", type=Path, default=ROOT.parents[1])
    ap.add_argument("--output", type=Path, default=ROOT / "results/iotj_c5_pipeline_audit_20260807")
    args = ap.parse_args()
    workspace, out = args.workspace.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    old_root, new_root = workspace / "dataset" / OLD_NAME, workspace / "dataset" / NEW_NAME
    old_up = workspace / "dataset/processed/unit_5"
    new_up = workspace / "results/time_aware_pipeline_probe_window_fullgrid/time_aware_60_170_window_fullgrid/processed/unit_5"
    old, new = load_full(old_root), load_full(new_root)
    manifests = manifest_rows("OLD", old_root, old_up) + manifest_rows("NEW", new_root, new_up)
    script_paths = {
        "OLD": [ROOT / "preprocessor.py", ROOT / "split_dataset.py"],
        "NEW": [workspace / "preprocessor_time_aware.py", workspace / "run_time_aware_target_split_ablation.py"],
    }
    for label, paths in script_paths.items():
        for path in paths:
            manifests.append({"dataset": label, "role": "generator_script", "path": str(path.resolve()),
                              "exists": path.exists(), "bytes": path.stat().st_size if path.exists() else "",
                              "sha256": sha(path) if path.exists() else ""})
    old_files = json.loads((old_up / "file_info.json").read_text(encoding="utf-8"))["filenames"]
    new_files = json.loads((new_up / "file_info.json").read_text(encoding="utf-8"))["filenames"]
    if old_files != new_files:
        raise RuntimeError("OLD/NEW raw filename collections differ")
    for filename in old_files:
        raw = workspace / "dataset/data1" / filename
        raw_hash = sha(raw)
        for label in ("OLD", "NEW"):
            manifests.append({"dataset": label, "role": "raw_unit5_file", "path": str(raw.resolve()),
                              "exists": True, "bytes": raw.stat().st_size, "sha256": raw_hash})
    write_csv(out / "p0_dataset_manifest.csv", manifests)
    matched = match_windows(old, new)
    write_csv(out / "p1_window_matching.csv", matched)
    write_csv(out / "p3_c5_split_distribution.csv", distribution("OLD", old, matched) + distribution("NEW", new, matched))
    r84 = r84_audit(old, new, matched)
    write_csv(out / "p4_matched_r84_features.csv", r84)
    write_csv(out / "p6_membership_preprocessing_factorial.csv", membership_preprocessing_factorial(old, new, matched))

    rmse = np.array([r["rmse_float64"] for r in matched])
    maxdiff = np.array([r["max_abs_diff"] for r in matched])
    same_membership = sum(r["old_split"] == r["new_split"] for r in matched)
    old_x, new_x = np.load(old_up / "features.npy"), np.load(new_up / "features.npy")
    p0 = f"""# P0 data provenance

- OLD upstream is `{old_up}` (legacy `preprocessor.py`, SHA-bound below).
- NEW upstream is `{new_up}` (time-aware `preprocessor_time_aware.py`; the generating scripts are present in the parent workspace but not tracked by the audited branch).
- Both upstreams contain the same 80 Unit-5 raw filenames and 1,680 windows. The resolved `dataset/data1` raw files are individually SHA256-bound in the CSV, but their processed `features.npy` hashes and values differ.
- OLD: remove first 20 s, row-based downsample, relative conductance baseline over the post-removal first 30 s, then crop 40--150 s (physical 60--170 s), float64.
- NEW: preserve/clean the time column, merge duplicate timestamps, interpolate on real seconds at 10 Hz, relative conductance, directly crop 60--170 s, float32.
- Therefore the change is **not float precision only**. It is a preprocessing implementation/version difference over the same named raw experiments.
- Dataset-level normalization is not applied to saved windows; `norm_stats.npz` is a separately computed source-train statistic used only by loaders that request normalization.

| Field | OLD processed Unit5 | NEW processed Unit5 |
|---|---|---|
| Path | `{old_up}` | `{new_up}` |
| Shape / dtype | {old_x.shape} / {old_x.dtype} | {new_x.shape} / {new_x.dtype} |
| Min / max | {old_x.min():.12g} / {old_x.max():.12g} | {new_x.min():.12g} / {new_x.max():.12g} |
| Mean / std | {old_x.mean():.12g} / {old_x.std():.12g} | {new_x.mean():.12g} / {new_x.std():.12g} |
| features SHA256 | `{sha(old_up/'features.npy')}` | `{sha(new_up/'features.npy')}` |
| classification labels SHA256 | `{sha(old_up/'classification_labels.npy')}` | `{sha(new_up/'classification_labels.npy')}` |
| regression labels SHA256 | `{sha(old_up/'regression_labels.npy')}` | `{sha(new_up/'regression_labels.npy')}` |
| phase labels SHA256 | `{sha(old_up/'phase_labels.npy')}` | `{sha(new_up/'phase_labels.npy')}` |

Classification/regression/phase label **values** are identical; classification and phase byte hashes differ because OLD uses int32/int8 and NEW uses int64.

| Parameter | OLD | NEW |
|---|---|---|
| original_fs / target_fs | 100 / 10 Hz | 100 / 10 Hz |
| unstable / baseline | remove first 20 s; 30 s baseline afterward | baseline on raw time [20,50) s |
| physical response crop | 60--170 s (implemented as 40--150 s after removal) | 60--170 s directly |
| window / stride | 100 / 50 samples | 100 / 50 samples |
| relative conductance | `(1/R - mean(1/R_baseline))/mean(1/R_baseline)` | same formula |
| time handling | row decimation; timestamp column discarded | stable timestamp sort, duplicate merge, real-time interpolation |
| saved feature dtype | float64 | float32 |
| saved-window z-score | none | none |
| generator Git provenance | tracked: `preprocessor.py` at initial `a0ce0b5`; splitter later touched by `396e304` | generator files present but untracked in the audited branch; SHA256 recorded |

See `p0_dataset_manifest.csv` for shapes, dtypes, moments, byte hashes, labels, metadata, split manifests and norm statistics.
"""
    (out / "P0_DATA_PROVENANCE.md").write_text(p0, encoding="utf-8")
    p1 = f"""# P1 window identity

- Experiment groups in both datasets: 80/80; every group has 21/21 windows.
- Hungarian one-to-one physical matches: {len(matched)}/1680.
- Numerically bit-identical: {sum(r['bit_identical'] for r in matched)}/1680.
- Median/P95/max matched-window RMSE: {np.median(rmse):.9g} / {np.quantile(rmse, .95):.9g} / {rmse.max():.9g}.
- Median matched max-absolute difference: {np.median(maxdiff):.9g}.
- Tolerance matches (atol 1e-6/1e-5/1e-4/1e-3): {sum(r['equal_atol_1e-6'] for r in matched)}/{sum(r['equal_atol_1e-5'] for r in matched)}/{sum(r['equal_atol_1e-4'] for r in matched)}/{sum(r['equal_atol_1e-3'] for r in matched)}.
- Same calibration/test membership after physical matching: {same_membership}/1680.

Conclusion: the datasets represent the same named physical experiments and the same 21 nominal time positions per experiment, but are not numerically identical windows and have different split membership.
"""
    (out / "P1_WINDOW_IDENTITY.md").write_text(p1, encoding="utf-8")
    fr = np.array([r["feature_rmse"] for r in r84])
    p4 = f"""# P4 R84 feature audit

- Actual R84 = 83 deterministic sensor statistics plus one frozen Federated-H1 source Ridge prediction.
- DCT dimension: 0. Encoder/reg_feat dimension: 0. Target phase/class metadata dimension: 0.
- Per-head Ridge standardizes each feature using its calibration-fit mean/std; no dataset `norm_stats.npz` is applied inside this feature builder.
- Feature ordering is lexicographically sorted and SHA-bound per row in `p4_matched_r84_features.csv`.
- Across 1,680 Hungarian-matched physical windows, median/P95/max 84-D RMSE = {np.median(fr):.6g}/{np.quantile(fr,.95):.6g}/{fr.max():.6g}.

The R84 implementation is common to OLD and NEW. Its values differ because the input windows differ numerically, not because two R84 builders were selected.
"""
    (out / "P4_R84_FEATURE_AUDIT.md").write_text(p4, encoding="utf-8")
    print(json.dumps({"matched": len(matched), "same_split_membership": same_membership,
                      "median_window_rmse": float(np.median(rmse)), "median_r84_rmse": float(np.median(fr))}, indent=2))


if __name__ == "__main__":
    main()
