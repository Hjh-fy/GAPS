"""Serialize the B5 C1/C2-to-C5 R4 source-augmented Ridge deployment policy."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from export_hybrid_mlp_ridge_deployment_candidate import serialize_mlp_head
from run_regression_head_ablation import (
    CLASS_NAMES,
    add_target_features,
    client_name,
    deterministic_train_val,
    fit_select_refit,
    inum,
    read_csv,
)
from run_source_augmented_target_ridge_eval import (
    add_pred_features,
    attach_source_predictions,
    fit_source_heads,
)
from export_h8_source_aug_deployment_candidate import serialize_source_heads


PRED_KEYS = ["H1_source_ridge_ppm", "H2_source_per_gas_mlp_ppm", "H3_source_shared_mlp_ppm"]
RIDGE_ALPHAS = [0.0, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
MLP_HIDDENS = [(16,)]
MLP_ALPHAS = [0.01, 0.1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_policy_payload(
    *,
    source_heads: dict[str, Any],
    target_models: list[dict[str, Any]],
    feature_names: list[str],
    classifier_sha256: str,
) -> dict[str, Any]:
    """Build an all-gas C5-only runtime policy without legacy fallback routes."""
    if len(classifier_sha256) != 64:
        raise ValueError("classifier SHA-256 must contain 64 hexadecimal characters")
    for model in target_models:
        if str(model.get("client")) != "C5":
            raise ValueError("R4 deployment policy must be C5-only")
        if int(model.get("class_id", -1)) not in CLASS_NAMES:
            raise ValueError("target model has invalid gas class")
    return {
        "schema_version": "iotj.b5_c5_r4_policy.v1",
        "direction": "C1_C2_to_C5",
        "classifier_sha256": classifier_sha256,
        "source_aug_target_ridge_policy": {
            "schema": "source_aug_target_ridge.v2",
            "switch_rule": {
                "type": "pred_class_in",
                "class_ids": [0, 1, 2, 3],
                "enabled_clients": ["C5"],
            },
            "source_prediction_keys": PRED_KEYS,
            "source_heads": source_heads,
            "feature_names": list(feature_names),
            "models": target_models,
        },
        "forbidden_runtime_dependencies": ["C3", "C4", "R3aK16", "H8+C4", "P4"],
    }


def _fit_target_models(
    calibration_rows: list[dict[str, Any]], feature_names: list[str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    models: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for cls_id in sorted(CLASS_NAMES):
        class_rows = [row for row in calibration_rows if inum(row["true_class"]) == cls_id]
        train_rows, validation_rows = deterministic_train_val(class_rows, val_ratio=0.25)
        model, audit = fit_select_refit(train_rows, validation_rows, feature_names, RIDGE_ALPHAS)
        payload = model.to_json()
        payload.update(
            {
                "client": "C5",
                "class_id": cls_id,
                "gas": CLASS_NAMES[cls_id],
                "enabled": True,
                "selected_mode": "source_aug_target_ridge",
                "n_fit": len(class_rows),
            }
        )
        models.append(payload)
        audit_rows.append(
            {
                "client": "C5",
                "class_id": cls_id,
                "gas": CLASS_NAMES[cls_id],
                "train_N": len(train_rows),
                "validation_N": len(validation_rows),
                "best_alpha": audit["best_alpha"],
                "best_validation_RMSE": audit["best_val_RMSE"],
            }
        )
    return models, audit_rows


def export_policy(
    *,
    data_root: Path,
    target_predictions: Path,
    classifier_checkpoint: Path,
    output: Path,
    seed: int,
) -> dict[str, Any]:
    if not classifier_checkpoint.is_file():
        raise FileNotFoundError(classifier_checkpoint)
    raw_rows = [
        row
        for row in read_csv(target_predictions)
        if client_name(row.get("client") or row.get("client_id")) == "C5"
        and str(row.get("split")) == "calibration"
    ]
    if len(raw_rows) != 320:
        raise ValueError(f"expected 320 C5 calibration rows; got {len(raw_rows)}")
    calibration = add_target_features(raw_rows, data_root)
    for row in calibration:
        row["route_class"] = row["true_class"]
    ridge_heads, mlp_heads, shared_head, _ = fit_source_heads(
        data_root, ["C1", "C2"], RIDGE_ALPHAS, MLP_HIDDENS, MLP_ALPHAS, seed
    )
    calibration_with_source = attach_source_predictions(
        calibration, ridge_heads, mlp_heads, shared_head
    )
    augmented = add_pred_features(calibration_with_source, PRED_KEYS)
    feature_names = sorted(augmented[0]["feature_dict"].keys())
    target_models, fit_audit = _fit_target_models(augmented, feature_names)
    policy = build_policy_payload(
        source_heads=serialize_source_heads(ridge_heads, mlp_heads, shared_head),
        target_models=target_models,
        feature_names=feature_names,
        classifier_sha256=_sha256(classifier_checkpoint),
    )
    policy["fit_protocol"] = {
        "source_clients": ["C1", "C2"],
        "target_client": "C5",
        "target_fit_split": "calibration_only",
        "target_test_used_for_fit": False,
        "seed": seed,
        "ridge_alphas": RIDGE_ALPHAS,
        "source_mlp_hidden_grid": [list(item) for item in MLP_HIDDENS],
        "source_mlp_alphas": MLP_ALPHAS,
        "target_fit_audit": fit_audit,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(policy, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    return policy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--target-predictions", type=Path, required=True)
    parser.add_argument("--classifier-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    policy = export_policy(
        data_root=args.data_root,
        target_predictions=args.target_predictions,
        classifier_checkpoint=args.classifier_checkpoint,
        output=args.output,
        seed=args.seed,
    )
    print(json.dumps({"output": str(args.output), "classifier_sha256": policy["classifier_sha256"]}))


if __name__ == "__main__":
    main()
