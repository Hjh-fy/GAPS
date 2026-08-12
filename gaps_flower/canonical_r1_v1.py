"""Frozen numerical primitives for canonical-v1 R1 target regression."""
from dataclasses import dataclass
from typing import Mapping, Sequence
from pathlib import Path
import hashlib
import json

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
    return True
