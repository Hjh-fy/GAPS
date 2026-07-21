"""Serialize the C5-only H23 reference heads required by the HC90 QC policy."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from export_hybrid_mlp_ridge_deployment_candidate import serialize_mlp_head
from run_formal_target_mlp_auto_v2_eval import fit_mlp
from run_h2_3_backbone_feature_ablation import load_feature_rows, merge_backbone_features
from run_h2_3_plus_fusion_profile import add_group_features
from run_regression_head_ablation import CLASS_NAMES, add_target_features, client_name, fnum, fit_ridge, inum, read_csv


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_hidden(text: str) -> tuple[int, ...]:
    return tuple(int(v.strip()) for v in str(text).strip().strip("()").split(",") if v.strip())


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def build_h23_payload(*, mlp_models: list[dict[str, Any]], ridge_models: list[dict[str, Any]], selected_weight: float, classifier_sha256: str) -> dict[str, Any]:
    if len(classifier_sha256) != 64:
        raise ValueError("classifier SHA-256 must contain 64 characters")
    if not 0.0 <= float(selected_weight) <= 1.0:
        raise ValueError("H23 blend weight must be in [0, 1]")
    for model in [*mlp_models, *ridge_models]:
        if str(model.get("client")) != "C5":
            raise ValueError("H23 reference policy must be C5-only")
    return {
        "schema_version": "iotj.b5_c5_h23_reference.v1",
        "classifier_sha256": classifier_sha256,
        "h23_reference_policy": {
            "target_client": "C5",
            "anchor": "per_gas_mlp",
            "secondary": "regfeat_ridge",
            "blend_weight": float(selected_weight),
            "mlp_models": mlp_models,
            "ridge_models": ridge_models,
        },
        "forbidden_runtime_dependencies": ["C3", "C4", "R3aK16", "H8+C4", "P4"],
    }


def export_h23_reference(*, data_root: Path, target_predictions: Path, backbone_calibration: Path, backbone_test: Path, fit_audit: Path, selection: Path, classifier_checkpoint: Path, output: Path, seed: int) -> dict[str, Any]:
    rows = [row for row in read_csv(target_predictions) if client_name(row.get("client") or row.get("client_id")) == "C5" and row.get("split") == "calibration"]
    if len(rows) != 320:
        raise ValueError(f"expected 320 C5 calibration rows; got {len(rows)}")
    base_rows = add_target_features(rows, data_root)
    for row in base_rows:
        row["route_class"] = row["true_class"]
    backbone = load_feature_rows(backbone_calibration, backbone_test)
    rich_rows = merge_backbone_features(base_rows, backbone)
    rich_names = sorted(rich_rows[0]["feature_dict"].keys())
    regfeat_rows = add_group_features(base_rows, "A3_rich_plus_reg_feat")
    regfeat_names = sorted(regfeat_rows[0]["feature_dict"].keys())
    audit_rows = _read_csv(fit_audit)
    audit = {(row["family"], int(row["class_id"])): row for row in audit_rows}
    selection_rows = _read_csv(selection)
    selected = [row for row in selection_rows if str(row.get("selected")) == "1" and row.get("client") == "C5"]
    if len(selected) != 1:
        raise ValueError("expected exactly one frozen C5 H23 blend weight")
    weight = fnum(selected[0]["weight"])
    mlp_models: list[dict[str, Any]] = []
    ridge_models: list[dict[str, Any]] = []
    for class_id in sorted(CLASS_NAMES):
        rich_class = [row for row in rich_rows if inum(row["true_class"]) == class_id]
        grid = audit[("h2_c5_grid_mlp", class_id)]
        mlp = serialize_mlp_head(fit_mlp(rich_class, rich_names, _parse_hidden(grid["best_hidden"]), fnum(grid["best_alpha"]), seed + class_id + 500))
        mlp.update({"client": "C5", "class_id": class_id, "gas": CLASS_NAMES[class_id], "n_fit": len(rich_class)})
        mlp_models.append(mlp)
        reg_class = [row for row in regfeat_rows if inum(row["true_class"]) == class_id]
        ridge = fit_ridge(reg_class, regfeat_names, fnum(audit[("regfeat_ridge", class_id)]["best_alpha"])).to_json()
        ridge.update({"client": "C5", "class_id": class_id, "gas": CLASS_NAMES[class_id], "n_fit": len(reg_class)})
        ridge_models.append(ridge)
    payload = build_h23_payload(mlp_models=mlp_models, ridge_models=ridge_models, selected_weight=weight, classifier_sha256=_sha256(classifier_checkpoint))
    payload["fit_protocol"] = {"target_fit_split": "calibration_only", "target_test_used_for_fit": False, "seed": seed, "fit_audit": str(fit_audit), "selection": str(selection)}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True); parser.add_argument("--target-predictions", type=Path, required=True)
    parser.add_argument("--backbone-calibration", type=Path, required=True); parser.add_argument("--backbone-test", type=Path, required=True)
    parser.add_argument("--fit-audit", type=Path, required=True); parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--classifier-checkpoint", type=Path, required=True); parser.add_argument("--output", type=Path, required=True); parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    payload = export_h23_reference(**vars(args))
    print(json.dumps({"output": str(args.output), "classifier_sha256": payload["classifier_sha256"]}))


if __name__ == "__main__":
    main()
