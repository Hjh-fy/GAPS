"""Fail-closed calibration-only QC primitives for the Runtime v5 candidate."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = "iotj.runtime_v5_qc_policy.v1"
COMPONENTS = (
    "entropy",
    "inverse_margin",
    "prototype_distance",
    "support_distance",
    "normalized_regression_disagreement",
)
WORKPOINTS = ("HC95", "HC90")


class QCCandidate(str, Enum):
    QC1 = "QC1"
    QC2 = "QC2"
    QC3 = "QC3"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def descriptor(path: Path) -> dict[str, Any]:
    target = Path(path).resolve()
    if not target.is_file():
        raise RuntimeError(f"asset is missing: {target}")
    return {"path": str(target), "bytes": target.stat().st_size, "sha256": sha256_file(target)}


def _finite_vector(value: Any, label: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.ndim != 1 or vector.size == 0 or not np.isfinite(vector).all():
        raise ValueError(f"{label} contains NaN/Inf or has invalid shape")
    return vector


def assign_group_folds(
    metadata: Sequence[Mapping[str, Any]], *, n_splits: int = 5, seed: int = 20260725
) -> tuple[list[int], dict[str, Any]]:
    """Assign four-window filename groups without crossing folds."""
    if n_splits != 5 or len(metadata) != 320:
        raise ValueError("group folds require exactly 320 metadata rows and five folds")
    groups: dict[str, list[int]] = {}
    for index, row in enumerate(metadata):
        filename = row.get("filename")
        if not isinstance(filename, str) or not filename.strip():
            raise ValueError(f"filename is missing at row {index}")
        groups.setdefault(filename, []).append(index)
    if len(groups) != 80 or any(not 1 <= len(indices) <= 7 for indices in groups.values()):
        raise ValueError("filename groups must be exactly 80 variable-size groups of 1..7 calibration rows")

    records: list[dict[str, Any]] = []
    for filename, indices in groups.items():
        rows = [metadata[index] for index in indices]
        labels = {int(row.get("classification_label", -1)) for row in rows}
        gases = {str(row.get("gas_code", "")) for row in rows}
        concentrations = {str(row.get("concentration_code", "")) for row in rows}
        repeats = {str(row.get("repeat_id", "")) for row in rows}
        if len(labels) != 1 or min(labels) < 0 or len(gases) != 1 or "" in gases or len(concentrations) != 1 or "" in concentrations or len(repeats) != 1:
            raise ValueError(f"filename group metadata is inconsistent: {filename}")
        gas = next(iter(labels))
        concentration = next(iter(concentrations))
        tie = hashlib.sha256(f"{seed}:{filename}".encode("utf-8")).hexdigest()
        records.append({"filename": filename, "indices": indices, "size": len(indices), "gas": gas, "concentration": concentration, "tie": tie})

    stratum_frequency: dict[tuple[int, str], int] = {}
    for record in records:
        key = (record["gas"], record["concentration"])
        stratum_frequency[key] = stratum_frequency.get(key, 0) + 1
    records.sort(key=lambda record: (-record["size"], -stratum_frequency[(record["gas"], record["concentration"])], record["tie"]))

    fold_groups: list[list[dict[str, Any]]] = [[] for _ in range(n_splits)]
    gas_counts = [dict() for _ in range(n_splits)]
    stratum_counts = [dict() for _ in range(n_splits)]
    row_counts = [0 for _ in range(n_splits)]
    for record in records:
        stratum = (record["gas"], record["concentration"])
        candidates = sorted(
            range(n_splits),
            key=lambda fold: (
                row_counts[fold],
                gas_counts[fold].get(record["gas"], 0),
                stratum_counts[fold].get(stratum, 0),
                len(fold_groups[fold]),
                fold,
            ),
        )
        fold = candidates[0]
        fold_groups[fold].append(record)
        row_counts[fold] += record["size"]
        gas_counts[fold][record["gas"]] = gas_counts[fold].get(record["gas"], 0) + record["size"]
        stratum_counts[fold][stratum] = stratum_counts[fold].get(stratum, 0) + record["size"]

    assignments = [-1] * len(metadata)
    for fold, fold_records in enumerate(fold_groups):
        for record in fold_records:
            for index in record["indices"]:
                if assignments[index] != -1:
                    raise ValueError("filename group row was assigned more than once")
                assignments[index] = fold
    if any(fold < 0 for fold in assignments):
        raise ValueError("filename group assignment is incomplete")
    cross = sum(
        len({assignments[index] for index in indices}) != 1
        for indices in groups.values()
    )
    fold_rows = [assignments.count(fold) for fold in range(n_splits)]
    if cross or max(fold_rows) - min(fold_rows) > max(len(indices) for indices in groups.values()):
        raise ValueError("filename group/fold isolation or balance failed")
    size_distribution = {
        str(size): sum(len(indices) == size for indices in groups.values())
        for size in range(1, 8)
    }
    audit = {
        "group_key": "filename",
        "row_count": len(metadata),
        "group_count": len(groups),
        "group_size_semantics": "all calibration rows sharing filename remain together",
        "group_size_min": min(len(indices) for indices in groups.values()),
        "group_size_max": max(len(indices) for indices in groups.values()),
        "group_size_distribution": size_distribution,
        "fold_count": n_splits,
        "seed": seed,
        "fold_row_counts": fold_rows,
        "fold_group_counts": [len(items) for items in fold_groups],
        "fold_gas_row_counts": [{str(k): int(v) for k, v in sorted(counts.items())} for counts in gas_counts],
        "fold_gas_concentration_row_counts": [
            {f"{gas}:{concentration}": int(value) for (gas, concentration), value in sorted(counts.items())}
            for counts in stratum_counts
        ],
        "group_cross_fold_count": cross,
    }
    return assignments, audit


def fit_feature_reference(rows: Sequence[Mapping[str, Any]], *, epsilon: float) -> dict[str, Any]:
    if not math.isfinite(epsilon) or epsilon <= 0 or not rows:
        raise ValueError("feature reference requires rows and a positive finite epsilon")
    vectors = [_finite_vector(row.get("representation"), "representation") for row in rows]
    dimension = vectors[0].size
    if any(vector.size != dimension for vector in vectors):
        raise ValueError("representation dimensions differ")
    matrix = np.stack(vectors)
    global_var = np.var(matrix, axis=0)
    by_class: dict[int, list[int]] = {}
    for index, row in enumerate(rows):
        class_id = int(row.get("true_class", -1))
        if class_id not in (0, 1, 2, 3):
            raise ValueError("feature reference true class is invalid")
        by_class.setdefault(class_id, []).append(index)
    if not by_class:
        raise ValueError("feature reference has no classes")
    classes: dict[str, Any] = {}
    for class_id, indices in sorted(by_class.items()):
        selected = matrix[np.asarray(indices, dtype=np.int64)]
        scale = np.sqrt(0.5 * np.var(selected, axis=0) + 0.5 * global_var + epsilon)
        classes[str(class_id)] = {
            "mean": selected.mean(axis=0).tolist(),
            "scale": np.maximum(scale, epsilon).tolist(),
            "support": selected.tolist(),
            "n": len(indices),
        }
    return {"feature_dimension": int(dimension), "epsilon": float(epsilon), "classes": classes}


def fit_regression_consistency_scales(rows: Sequence[Mapping[str, Any]], *, epsilon: float) -> dict[str, Any]:
    if not rows or not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("regression consistency scales require rows and positive epsilon")
    by_gas: dict[int, list[float]] = {}
    for row in rows:
        gas = int(row.get("pred_class", -1))
        h1 = float(row.get("source_h1_ppm", np.nan))
        target = float(row.get("prediction_ppm", np.nan))
        if gas not in (0, 1, 2, 3) or not math.isfinite(h1) or not math.isfinite(target):
            raise ValueError("regression consistency row contains NaN/Inf or invalid gas")
        by_gas.setdefault(gas, []).append(target - h1)
    records: dict[str, Any] = {}
    for gas, values in sorted(by_gas.items()):
        array = np.asarray(values, dtype=np.float64)
        median = float(np.median(array))
        mad = float(np.median(np.abs(array - median)))
        records[str(gas)] = {
            "n": len(array),
            "signed_delta_median": median,
            "mad": mad,
            "mad_multiplier": 1.4826,
            "scale": float(max(1.4826 * mad, epsilon)),
        }
    return {"method": "1.4826_times_MAD_of_signed_target_minus_H1", "epsilon": float(epsilon), "per_predicted_gas": records}


def empirical_percentile(value: float, distribution: Sequence[float]) -> float:
    values = np.asarray(distribution, dtype=np.float64)
    if not math.isfinite(float(value)) or values.ndim != 1 or values.size == 0 or not np.isfinite(values).all() or np.any(values[1:] < values[:-1]):
        raise ValueError("ECDF value/distribution contains NaN/Inf or is invalid")
    return float(np.searchsorted(values, float(value), side="right") / len(values))


@dataclass(frozen=True)
class RuntimeV5QCPolicy:
    payload: Mapping[str, Any]

    @classmethod
    def from_path(cls, path: Path) -> "RuntimeV5QCPolicy":
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError("QC policy is not valid JSON") from error
        return cls.from_payload(payload)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "RuntimeV5QCPolicy":
        required = {
            "schema_version", "status", "selected_candidate", "epsilon",
            "feature_reference", "regression_consistency_scale",
            "component_distributions", "workpoints", "decision_semantics",
        }
        if set(payload) != required or payload.get("schema_version") != SCHEMA_VERSION or payload.get("status") != "locked":
            raise ValueError("QC policy schema/status differs")
        if payload.get("selected_candidate") not in {item.value for item in QCCandidate}:
            raise ValueError("QC policy candidate is invalid")
        epsilon = float(payload.get("epsilon", np.nan))
        if not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("QC policy epsilon is invalid")
        distributions = payload.get("component_distributions")
        if not isinstance(distributions, Mapping) or set(distributions) != set(COMPONENTS):
            raise ValueError("QC policy component distributions differ")
        for values in distributions.values():
            empirical_percentile(float(values[0]), values)
        workpoints = payload.get("workpoints")
        if not isinstance(workpoints, Mapping) or set(workpoints) != set(WORKPOINTS):
            raise ValueError("QC policy workpoints differ")
        for workpoint in WORKPOINTS:
            item = workpoints[workpoint]
            if not isinstance(item, Mapping) or set(item) != {"accept_threshold", "reject_threshold"}:
                raise ValueError("QC policy workpoint schema differs")
            accept, reject = float(item["accept_threshold"]), float(item["reject_threshold"])
            if not (math.isfinite(accept) and math.isfinite(reject) and 0 <= accept <= reject <= 1):
                raise ValueError("QC policy thresholds are invalid")
        semantics = payload.get("decision_semantics")
        if semantics != {"auto_output_only_for_accept": True}:
            raise ValueError("QC decision semantics differ")
        reference = payload.get("feature_reference")
        scales = payload.get("regression_consistency_scale")
        if not isinstance(reference, Mapping) or not isinstance(reference.get("classes"), Mapping):
            raise ValueError("QC feature reference differs")
        if not isinstance(scales, Mapping) or not isinstance(scales.get("per_predicted_gas"), Mapping):
            raise ValueError("QC regression scales differ")
        return cls(dict(payload))

    def raw_components(
        self,
        *,
        probabilities: Sequence[float],
        representation: Sequence[float],
        pred_class: int,
        source_h1_ppm: float,
        prediction_ppm: float,
    ) -> dict[str, float]:
        probabilities_array = _finite_vector(probabilities, "probabilities")
        if probabilities_array.size != 4 or np.any(probabilities_array < 0) or not np.isclose(probabilities_array.sum(), 1.0, atol=1e-6):
            raise ValueError("probabilities are invalid")
        if pred_class not in (0, 1, 2, 3) or int(np.argmax(probabilities_array)) != pred_class:
            raise ValueError("predicted class/probability route differs")
        feature = _finite_vector(representation, "representation")
        reference = self.payload["feature_reference"]
        if feature.size != int(reference["feature_dimension"]):
            raise ValueError("representation dimension differs")
        cell = reference["classes"].get(str(pred_class))
        if not isinstance(cell, Mapping):
            raise ValueError("predicted class feature reference is missing")
        mean = _finite_vector(cell.get("mean"), "prototype mean")
        scale = np.maximum(_finite_vector(cell.get("scale"), "prototype scale"), float(self.payload["epsilon"]))
        support = np.asarray(cell.get("support"), dtype=np.float64)
        if mean.size != feature.size or scale.size != feature.size or support.ndim != 2 or support.shape[1] != feature.size or not np.isfinite(support).all():
            raise ValueError("feature reference contains NaN/Inf or invalid shape")
        ordered = np.sort(probabilities_array)
        entropy = float(-np.sum(probabilities_array * np.log(np.maximum(probabilities_array, 1e-12))) / math.log(4.0))
        inverse_margin = float(1.0 - (ordered[-1] - ordered[-2]))
        prototype = float(np.sqrt(np.mean(((feature - mean) / scale) ** 2)))
        support_distance = float(np.min(np.sqrt(np.mean(((support - feature) / scale) ** 2, axis=1))))
        scale_record = self.payload["regression_consistency_scale"]["per_predicted_gas"].get(str(pred_class))
        if not isinstance(scale_record, Mapping):
            raise ValueError("predicted gas regression consistency scale is missing")
        regression_scale = float(scale_record.get("scale", np.nan))
        h1, target = float(source_h1_ppm), float(prediction_ppm)
        if not all(math.isfinite(value) for value in (regression_scale, h1, target)) or regression_scale <= 0:
            raise ValueError("regression consistency contains NaN/Inf or invalid scale")
        output = {
            "entropy": entropy,
            "inverse_margin": inverse_margin,
            "prototype_distance": prototype,
            "support_distance": support_distance,
            "normalized_regression_disagreement": abs(target - h1) / regression_scale,
        }
        if not all(math.isfinite(value) for value in output.values()):
            raise ValueError("raw QC component contains NaN/Inf")
        return output

    def aggregate_percentiles(self, percentiles: Mapping[str, float]) -> dict[str, float]:
        if set(percentiles) != set(COMPONENTS) or not all(math.isfinite(float(value)) and 0 <= float(value) <= 1 for value in percentiles.values()):
            raise ValueError("normalized QC components differ or contain NaN/Inf")
        confidence = float(np.mean([percentiles["entropy"], percentiles["inverse_margin"]]))
        distance = float(np.mean([percentiles["prototype_distance"], percentiles["support_distance"]]))
        regression = float(percentiles["normalized_regression_disagreement"])
        candidate = self.payload["selected_candidate"]
        groups = [confidence]
        if candidate in (QCCandidate.QC2.value, QCCandidate.QC3.value):
            groups.append(distance)
        if candidate == QCCandidate.QC3.value:
            groups.append(regression)
        return {
            "confidence_group": confidence,
            "distance_group": distance,
            "regression_consistency_group": regression,
            "deployment_risk": float(np.mean(groups)),
        }

    def score(self, **kwargs: Any) -> dict[str, float]:
        raw = self.raw_components(**kwargs)
        normalized = {
            key: empirical_percentile(raw[key], self.payload["component_distributions"][key])
            for key in COMPONENTS
        }
        aggregates = self.aggregate_percentiles(normalized)
        return {
            **{f"raw_{key}": value for key, value in raw.items()},
            **{f"percentile_{key}": value for key, value in normalized.items()},
            **aggregates,
        }

    @staticmethod
    def decision(risk: float, prediction_ppm: float, thresholds: Mapping[str, Any]) -> tuple[str, float | None]:
        risk_value, prediction = float(risk), float(prediction_ppm)
        accept = float(thresholds.get("accept_threshold", np.nan))
        reject = float(thresholds.get("reject_threshold", np.nan))
        if not all(math.isfinite(value) for value in (risk_value, prediction, accept, reject)):
            raise ValueError("QC decision contains NaN/Inf")
        if not 0 <= accept <= reject <= 1:
            raise ValueError("QC decision thresholds are invalid")
        if risk_value <= accept:
            return "accept", prediction
        if risk_value > reject:
            return "reject", None
        return "review", None


def make_selection_lock(
    *,
    selected_candidate: str,
    selection_reason: str,
    policy_path: Path,
    bound_assets: Mapping[str, Path],
    build_commit: str,
) -> dict[str, Any]:
    if selected_candidate not in {item.value for item in QCCandidate} or len(build_commit) != 40 or any(char not in "0123456789abcdef" for char in build_commit):
        raise ValueError("selection lock candidate/build commit is invalid")
    records = {name: descriptor(path) for name, path in sorted(bound_assets.items())}
    policy_record = descriptor(policy_path)
    if "qc_policy" not in records or records["qc_policy"] != policy_record:
        raise ValueError("selection lock must bind qc_policy")
    return {
        "schema_version": "iotj.runtime_v5_qc_selection_lock.v1",
        "status": "locked_before_test",
        "build_commit": build_commit,
        "selected_candidate": selected_candidate,
        "selection_reason": str(selection_reason),
        "qc_policy": policy_record,
        "bound_assets": records,
        "test_opened_after_lock": False,
        "test_used_for_fit_select_refit_or_thresholds": False,
    }


def require_selection_lock(lock_path: Path, bound_assets: Mapping[str, Path]) -> dict[str, Any]:
    try:
        lock = json.loads(Path(lock_path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError("QC selection lock is invalid") from error
    required = {
        "schema_version", "status", "build_commit", "selected_candidate",
        "selection_reason", "qc_policy", "bound_assets",
        "test_opened_after_lock", "test_used_for_fit_select_refit_or_thresholds",
    }
    if set(lock) != required or lock.get("schema_version") != "iotj.runtime_v5_qc_selection_lock.v1" or lock.get("status") != "locked_before_test":
        raise RuntimeError("QC selection lock schema/status differs")
    if lock.get("test_opened_after_lock") is not False or lock.get("test_used_for_fit_select_refit_or_thresholds") is not False:
        raise RuntimeError("QC selection lock permits test leakage")
    expected = lock.get("bound_assets")
    if not isinstance(expected, Mapping) or set(expected) != set(bound_assets):
        raise RuntimeError("QC selection lock asset set differs")
    for name, path in bound_assets.items():
        observed = descriptor(path)
        if expected[name].get("path") != observed["path"]:
            raise RuntimeError(f"QC selection lock {name} path differs")
        if expected[name].get("sha256") != observed["sha256"]:
            raise RuntimeError(f"QC selection lock {name} hash differs")
        if expected[name].get("bytes") != observed["bytes"]:
            raise RuntimeError(f"QC selection lock {name} size differs")
    return lock
