"""Materialize formal IoT-J H1 through real C1/C2 sufficient-statistics flow.

Client modes are the only modes that accept ``--data-root``.  Server modes
accept JSON statistics exclusively and reject sample-level/raw fields.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from run_regression_head_ablation import (
    CLASS_NAMES,
    build_oracle_rows,
    matrix_from_rows,
)


SCHEMA = "iotj.federated_h1.real_topology.v1"
ALPHAS = (0.0, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
FORBIDDEN_SERVER_KEYS = {
    "raw_rows",
    "rows",
    "raw_x",
    "x",
    "raw_y",
    "y",
    "labels",
    "predictions",
    "sample_predictions",
    "sample_labels",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def write_new(path: Path, payload: Any) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def reject_forbidden_payload(value: Any, path: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).lower()
            if normalized in FORBIDDEN_SERVER_KEYS:
                raise ValueError(f"forbidden server payload field: {path}.{key}")
            reject_forbidden_payload(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            reject_forbidden_payload(item, f"{path}[{index}]")


def client_rows(data_root: Path, client: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if client not in {"C1", "C2"}:
        raise ValueError("client must be C1 or C2")
    client_number = int(client[1:])
    allowed = (data_root / f"client_{client_number}").resolve()
    other = (data_root / f"client_{3 - client_number}").resolve()
    if not allowed.is_dir():
        raise FileNotFoundError(allowed)
    train = build_oracle_rows(data_root, [client], "train")
    calibration = build_oracle_rows(data_root, [client], "calibration")
    if len(train) != 2360 or len(calibration) != 320:
        raise ValueError(f"{client} cardinality differs")
    return train, calibration


def selected(rows: Sequence[Mapping[str, Any]], gas_id: int) -> list[Mapping[str, Any]]:
    result = [row for row in rows if int(row["true_class"]) == gas_id]
    if not result:
        raise ValueError(f"gas {gas_id} has no rows")
    return result


def finite_xy(
    rows: Sequence[Mapping[str, Any]], names: Sequence[str]
) -> tuple[np.ndarray, np.ndarray]:
    x = matrix_from_rows(rows, names)
    y = np.asarray([float(row["true_ppm"]) for row in rows], dtype=np.float64)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("client source rows contain NaN/Inf")
    return x, y


def row_provenance(
    client: str, phase: str, gas_id: int, rows: Sequence[Mapping[str, Any]]
) -> str:
    identity = [
        [
            client,
            phase,
            gas_id,
            int(row["sample_index"]),
            int(row["true_class"]),
            float(row["true_ppm"]),
        ]
        for row in rows
    ]
    return json_hash(identity)


def moments_record(
    client: str,
    phase: str,
    gas_id: int,
    rows: Sequence[Mapping[str, Any]],
    names: Sequence[str],
) -> dict[str, Any]:
    gas_rows = selected(rows, gas_id)
    x, _y = finite_xy(gas_rows, names)
    return {
        "client_id": client,
        "phase": phase,
        "gas_id": gas_id,
        "gas": CLASS_NAMES[gas_id],
        "n": len(gas_rows),
        "sum_x": np.sum(x, axis=0, dtype=np.float64).tolist(),
        "sum_x2": np.sum(x * x, axis=0, dtype=np.float64).tolist(),
        "row_provenance_sha256": row_provenance(
            client, phase, gas_id, gas_rows
        ),
    }


def client_moments(args: argparse.Namespace) -> None:
    data_root = Path(args.data_root)
    train, calibration = client_rows(data_root, args.client)
    names = sorted(train[0]["feature_dict"])
    if len(names) != 104:
        raise ValueError("H1 feature schema must be 104D")
    combined = [*train, *calibration]
    client_dir = data_root / f"client_{int(args.client[1:])}"
    files = [
        client_dir / f"{split}_{suffix}"
        for split in ("train", "calibration")
        for suffix in (
            "features.npy",
            "classification_labels.npy",
            "regression_labels.npy",
            "phase_labels.npy",
            "experiment_info.json",
        )
    ]
    if not all(path.is_file() for path in files):
        raise FileNotFoundError("client source split asset missing")
    payload = {
        "schema_version": SCHEMA,
        "record_type": "client_feature_moments",
        "created_at": now(),
        "host": socket.gethostname(),
        "client_id": args.client,
        "allowed_dataset_directory": str(client_dir.resolve()),
        "other_source_client_opened": False,
        "feature_names": names,
        "feature_schema_sha256": json_hash(names),
        "dataset_asset_sha256": {
            path.name: sha256_file(path) for path in files
        },
        "records": [
            *[
                moments_record(
                    args.client,
                    "source_train_selection",
                    gas_id,
                    train,
                    names,
                )
                for gas_id in sorted(CLASS_NAMES)
            ],
            *[
                moments_record(
                    args.client,
                    "source_train_plus_calibration_refit",
                    gas_id,
                    combined,
                    names,
                )
                for gas_id in sorted(CLASS_NAMES)
            ],
        ],
        "raw_rows_transmitted": False,
        "raw_X_y_transmitted": False,
    }
    write_new(Path(args.output), payload)


def validate_pair(
    payloads: Sequence[Mapping[str, Any]], record_type: str
) -> tuple[list[str], str]:
    if len(payloads) != 2:
        raise ValueError("server requires exactly two client payloads")
    reject_forbidden_payload(payloads)
    if {item.get("client_id") for item in payloads} != {"C1", "C2"}:
        raise ValueError("server requires independent C1/C2 provenance")
    if any(
        item.get("schema_version") != SCHEMA
        or item.get("record_type") != record_type
        for item in payloads
    ):
        raise ValueError("client payload schema/type differs")
    schemas = {str(item["feature_schema_sha256"]) for item in payloads}
    if len(schemas) != 1:
        raise ValueError("C1/C2 feature schemas differ")
    names = payloads[0]["feature_names"]
    if len(names) != 104 or payloads[1]["feature_names"] != names:
        raise ValueError("C1/C2 104D feature names differ")
    return list(names), next(iter(schemas))


def record_map(payload: Mapping[str, Any]) -> dict[tuple[str, int], Mapping[str, Any]]:
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError("payload records missing")
    result = {
        (str(record["phase"]), int(record["gas_id"])): record
        for record in records
    }
    if len(result) != len(records):
        raise ValueError("duplicate phase/gas record")
    return result


def server_scalers(args: argparse.Namespace) -> None:
    payloads = [load_json(Path(path)) for path in args.inputs]
    names, schema_hash = validate_pair(
        payloads, "client_feature_moments"
    )
    maps = [record_map(payload) for payload in payloads]
    scalers: list[dict[str, Any]] = []
    for phase in (
        "source_train_selection",
        "source_train_plus_calibration_refit",
    ):
        for gas_id in sorted(CLASS_NAMES):
            records = [mapping[(phase, gas_id)] for mapping in maps]
            n = sum(int(item["n"]) for item in records)
            sum_x = np.sum(
                [np.asarray(item["sum_x"], dtype=np.float64) for item in records],
                axis=0,
            )
            sum_x2 = np.sum(
                [np.asarray(item["sum_x2"], dtype=np.float64) for item in records],
                axis=0,
            )
            mean = sum_x / n
            variance = np.maximum(sum_x2 / n - mean * mean, 0.0)
            scale = np.sqrt(variance)
            scale = np.where(np.abs(scale) < 1e-9, 1.0, scale)
            if not np.isfinite(mean).all() or not np.isfinite(scale).all():
                raise ValueError("aggregated scaler is non-finite")
            scalers.append(
                {
                    "phase": phase,
                    "gas_id": gas_id,
                    "gas": CLASS_NAMES[gas_id],
                    "n": n,
                    "mean": mean.tolist(),
                    "scale": scale.tolist(),
                    "client_provenance": {
                        str(item["client_id"]): item[
                            "row_provenance_sha256"
                        ]
                        for item in records
                    },
                }
            )
    payload = {
        "schema_version": SCHEMA,
        "record_type": "server_global_scalers",
        "created_at": now(),
        "host": socket.gethostname(),
        "feature_names": names,
        "feature_schema_sha256": schema_hash,
        "input_sha256": {
            Path(path).name: sha256_file(Path(path)) for path in args.inputs
        },
        "scalers": scalers,
        "raw_or_sample_level_input_accepted": False,
    }
    write_new(Path(args.output), payload)


def design_stats(
    client: str,
    phase: str,
    gas_id: int,
    rows: Sequence[Mapping[str, Any]],
    names: Sequence[str],
    scaler: Mapping[str, Any],
) -> dict[str, Any]:
    gas_rows = selected(rows, gas_id)
    x, y = finite_xy(gas_rows, names)
    mean = np.asarray(scaler["mean"], dtype=np.float64)
    scale = np.asarray(scaler["scale"], dtype=np.float64)
    z = (x - mean) / scale
    design = np.concatenate([np.ones((len(z), 1)), z], axis=1)
    return {
        "client_id": client,
        "phase": phase,
        "gas_id": gas_id,
        "gas": CLASS_NAMES[gas_id],
        "n": len(gas_rows),
        "A": (design.T @ design).tolist(),
        "b": (design.T @ y).tolist(),
        "yTy": float(y @ y),
        "y_min": float(y.min()),
        "y_max": float(y.max()),
        "row_provenance_sha256": row_provenance(
            client, phase, gas_id, gas_rows
        ),
    }


def client_equations(args: argparse.Namespace) -> None:
    data_root = Path(args.data_root)
    train, calibration = client_rows(data_root, args.client)
    names = sorted(train[0]["feature_dict"])
    scalers_payload = load_json(Path(args.scalers))
    reject_forbidden_payload(scalers_payload)
    if (
        scalers_payload.get("record_type") != "server_global_scalers"
        or scalers_payload.get("feature_schema_sha256") != json_hash(names)
    ):
        raise ValueError("server scaler contract differs")
    scalers = {
        (str(item["phase"]), int(item["gas_id"])): item
        for item in scalers_payload["scalers"]
    }
    combined = [*train, *calibration]
    records: list[dict[str, Any]] = []
    for gas_id in sorted(CLASS_NAMES):
        records.extend(
            [
                design_stats(
                    args.client,
                    "source_train_selection",
                    gas_id,
                    train,
                    names,
                    scalers[("source_train_selection", gas_id)],
                ),
                design_stats(
                    args.client,
                    "source_calibration_validation",
                    gas_id,
                    calibration,
                    names,
                    scalers[("source_train_selection", gas_id)],
                ),
                design_stats(
                    args.client,
                    "source_train_plus_calibration_refit",
                    gas_id,
                    combined,
                    names,
                    scalers[
                        ("source_train_plus_calibration_refit", gas_id)
                    ],
                ),
            ]
        )
    payload = {
        "schema_version": SCHEMA,
        "record_type": "client_normal_equations",
        "created_at": now(),
        "host": socket.gethostname(),
        "client_id": args.client,
        "feature_names": names,
        "feature_schema_sha256": json_hash(names),
        "server_scaler_sha256": sha256_file(Path(args.scalers)),
        "records": records,
        "raw_rows_transmitted": False,
        "raw_X_y_transmitted": False,
        "sample_predictions_transmitted": False,
        "sample_labels_transmitted": False,
    }
    write_new(Path(args.output), payload)


def sum_stats(
    records: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, np.ndarray, float, int, float, float]:
    a = np.sum(
        [np.asarray(item["A"], dtype=np.float64) for item in records], axis=0
    )
    b = np.sum(
        [np.asarray(item["b"], dtype=np.float64) for item in records], axis=0
    )
    yty = sum(float(item["yTy"]) for item in records)
    n = sum(int(item["n"]) for item in records)
    return (
        a,
        b,
        yty,
        n,
        min(float(item["y_min"]) for item in records),
        max(float(item["y_max"]) for item in records),
    )


def ridge_coef(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    regularizer = np.eye(a.shape[0]) * float(alpha)
    regularizer[0, 0] = 0.0
    return np.linalg.pinv(a + regularizer) @ b


def server_candidates(args: argparse.Namespace) -> None:
    equations = [load_json(Path(path)) for path in args.inputs]
    names, schema_hash = validate_pair(
        equations, "client_normal_equations"
    )
    scalers_payload = load_json(Path(args.scalers))
    reject_forbidden_payload(scalers_payload)
    scaler_hash = sha256_file(Path(args.scalers))
    if any(
        payload.get("server_scaler_sha256") != scaler_hash
        for payload in equations
    ):
        raise ValueError("client equations used a different scaler")
    maps = [record_map(payload) for payload in equations]
    scalers = {
        (str(item["phase"]), int(item["gas_id"])): item
        for item in scalers_payload["scalers"]
    }
    candidates: list[dict[str, Any]] = []
    for gas_id in sorted(CLASS_NAMES):
        train = [
            mapping[("source_train_selection", gas_id)] for mapping in maps
        ]
        train_a, train_b, _yty, _n, clip_min, clip_max = sum_stats(train)
        scaler = scalers[("source_train_selection", gas_id)]
        for alpha in ALPHAS:
            candidates.append(
                {
                    "gas_id": gas_id,
                    "gas": CLASS_NAMES[gas_id],
                    "alpha": alpha,
                    "mean": scaler["mean"],
                    "scale": scaler["scale"],
                    "coef": ridge_coef(train_a, train_b, alpha).tolist(),
                    "clip_min": clip_min,
                    "clip_max": clip_max,
                }
            )
    output = {
        "schema_version": SCHEMA,
        "record_type": "server_h1_candidates",
        "created_at": now(),
        "host": socket.gethostname(),
        "feature_names": names,
        "feature_schema_sha256": schema_hash,
        "server_scaler_sha256": scaler_hash,
        "candidates": candidates,
        "sample_level_content": False,
    }
    write_new(Path(args.output), output)


def client_scores(args: argparse.Namespace) -> None:
    data_root = Path(args.data_root)
    _train, calibration = client_rows(data_root, args.client)
    names = sorted(calibration[0]["feature_dict"])
    candidates = load_json(Path(args.candidates))
    reject_forbidden_payload(candidates)
    if (
        candidates.get("record_type") != "server_h1_candidates"
        or candidates.get("feature_schema_sha256") != json_hash(names)
    ):
        raise ValueError("candidate model contract differs")
    scores: list[dict[str, Any]] = []
    for candidate in candidates["candidates"]:
        gas_id = int(candidate["gas_id"])
        gas_rows = selected(calibration, gas_id)
        x, y = finite_xy(gas_rows, names)
        mean = np.asarray(candidate["mean"], dtype=np.float64)
        scale = np.asarray(candidate["scale"], dtype=np.float64)
        design = np.concatenate(
            [np.ones((len(x), 1)), (x - mean) / scale], axis=1
        )
        coef = np.asarray(candidate["coef"], dtype=np.float64)
        prediction = np.clip(
            design @ coef,
            float(candidate["clip_min"]),
            float(candidate["clip_max"]),
        )
        error = prediction - y
        scores.append(
            {
                "client_id": args.client,
                "gas_id": gas_id,
                "gas": CLASS_NAMES[gas_id],
                "alpha": float(candidate["alpha"]),
                "n": len(gas_rows),
                "clipped_sse": float(error @ error),
                "row_provenance_sha256": row_provenance(
                    args.client,
                    "source_calibration_validation",
                    gas_id,
                    gas_rows,
                ),
            }
        )
    payload = {
        "schema_version": SCHEMA,
        "record_type": "client_validation_scores",
        "created_at": now(),
        "host": socket.gethostname(),
        "client_id": args.client,
        "feature_names": names,
        "feature_schema_sha256": json_hash(names),
        "server_candidates_sha256": sha256_file(Path(args.candidates)),
        "records": scores,
        "server_received_only_clipped_SSE_and_count": True,
        "sample_predictions_transmitted": False,
        "sample_labels_transmitted": False,
    }
    write_new(Path(args.output), payload)


def server_model(args: argparse.Namespace) -> None:
    equations = [load_json(Path(path)) for path in args.inputs]
    names, schema_hash = validate_pair(
        equations, "client_normal_equations"
    )
    scalers_payload = load_json(Path(args.scalers))
    reject_forbidden_payload(scalers_payload)
    scaler_hash = sha256_file(Path(args.scalers))
    if any(
        payload.get("server_scaler_sha256") != scaler_hash
        for payload in equations
    ):
        raise ValueError("client equations used a different scaler")
    maps = [record_map(payload) for payload in equations]
    scores_payload = [load_json(Path(path)) for path in args.scores]
    validate_pair(scores_payload, "client_validation_scores")
    score_maps = [
        {
            (int(record["gas_id"]), float(record["alpha"])): record
            for record in payload["records"]
        }
        for payload in scores_payload
    ]
    scalers = {
        (str(item["phase"]), int(item["gas_id"])): item
        for item in scalers_payload["scalers"]
    }
    models: dict[str, Any] = {}
    alpha_rows: list[dict[str, Any]] = []
    for gas_id in sorted(CLASS_NAMES):
        train = [
            mapping[("source_train_selection", gas_id)] for mapping in maps
        ]
        train_a, train_b, _yty, _n, train_min, train_max = sum_stats(train)
        candidates: list[tuple[float, float, np.ndarray]] = []
        for alpha in ALPHAS:
            coef = ridge_coef(train_a, train_b, alpha)
            score_records = [
                mapping[(gas_id, float(alpha))] for mapping in score_maps
            ]
            sse = sum(float(record["clipped_sse"]) for record in score_records)
            val_n = sum(int(record["n"]) for record in score_records)
            score = math.sqrt(sse / val_n)
            candidates.append((score, alpha, coef))
            alpha_rows.append(
                {
                    "gas_id": gas_id,
                    "gas": CLASS_NAMES[gas_id],
                    "alpha": alpha,
                    "validation_RMSE_clipped_from_client_SSE": score,
                    "validation_N": val_n,
                }
            )
        best_score, best_alpha, _coef = min(
            candidates, key=lambda item: (item[0], ALPHAS.index(item[1]))
        )
        refit = [
            mapping[("source_train_plus_calibration_refit", gas_id)]
            for mapping in maps
        ]
        refit_a, refit_b, _rty, _rn, clip_min, clip_max = sum_stats(refit)
        final_coef = ridge_coef(refit_a, refit_b, best_alpha)
        scaler = scalers[("source_train_plus_calibration_refit", gas_id)]
        models[str(gas_id)] = {
            "gas": CLASS_NAMES[gas_id],
            "alpha": best_alpha,
            "feature_names": names,
            "mean": scaler["mean"],
            "scale": scaler["scale"],
            "coef": final_coef.tolist(),
            "clip_min": clip_min,
            "clip_max": clip_max,
            "selection_validation_RMSE_clipped": best_score,
        }
    output = {
        "schema_version": SCHEMA,
        "record_type": "global_h1_model",
        "created_at": now(),
        "host": socket.gethostname(),
        "source": "real_topology_C1_C2_sufficient_statistics",
        "protocol": "two_phase_global_scaler_then_normal_equations",
        "feature_schema_sha256": schema_hash,
        "alpha_grid": list(ALPHAS),
        "models": models,
        "alpha_audit": alpha_rows,
        "server_api_received": [
            "n",
            "sum_x",
            "sum_x2",
            "A",
            "b",
            "yTy",
            "y_min",
            "y_max",
            "feature_schema_sha256",
            "row_provenance_sha256",
            "clipped_validation_SSE",
            "validation_count",
        ],
        "server_received_raw_rows": False,
        "server_received_raw_X_y": False,
        "server_received_sample_predictions_or_labels": False,
        "privacy_claim_boundary": (
            "source raw samples remain local and only aggregated sufficient "
            "statistics are used to reconstruct the global source Ridge reference"
        ),
    }
    write_new(Path(args.output), output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    client_a = sub.add_parser("client-moments")
    client_a.add_argument("--client", choices=("C1", "C2"), required=True)
    client_a.add_argument("--data-root", required=True)
    client_a.add_argument("--output", required=True)
    client_a.set_defaults(func=client_moments)
    server_a = sub.add_parser("server-scalers")
    server_a.add_argument("--inputs", nargs=2, required=True)
    server_a.add_argument("--output", required=True)
    server_a.set_defaults(func=server_scalers)
    client_b = sub.add_parser("client-equations")
    client_b.add_argument("--client", choices=("C1", "C2"), required=True)
    client_b.add_argument("--data-root", required=True)
    client_b.add_argument("--scalers", required=True)
    client_b.add_argument("--output", required=True)
    client_b.set_defaults(func=client_equations)
    server_candidates_parser = sub.add_parser("server-candidates")
    server_candidates_parser.add_argument("--inputs", nargs=2, required=True)
    server_candidates_parser.add_argument("--scalers", required=True)
    server_candidates_parser.add_argument("--output", required=True)
    server_candidates_parser.set_defaults(func=server_candidates)
    client_scores_parser = sub.add_parser("client-scores")
    client_scores_parser.add_argument(
        "--client", choices=("C1", "C2"), required=True
    )
    client_scores_parser.add_argument("--data-root", required=True)
    client_scores_parser.add_argument("--candidates", required=True)
    client_scores_parser.add_argument("--output", required=True)
    client_scores_parser.set_defaults(func=client_scores)
    server_b = sub.add_parser("server-model")
    server_b.add_argument("--inputs", nargs=2, required=True)
    server_b.add_argument("--scalers", required=True)
    server_b.add_argument("--scores", nargs=2, required=True)
    server_b.add_argument("--output", required=True)
    server_b.set_defaults(func=server_model)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    arguments.func(arguments)
