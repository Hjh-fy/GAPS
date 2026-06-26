"""Export a direction-specific target Ridge profile as a runtime artifact.

This is the generic version of the earlier C12->C345-only exporter.  It writes
the ``target_ridge_policy`` schema already consumed by ``gaps_deploy`` runtime,
and keeps route-rescue disabled unless a future direction explicitly adds a
calibration-selected rescue policy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from run_regression_head_ablation import (
    CLASS_NAMES,
    add_target_features,
    client_name,
    fit_ridge,
    fnum,
    inum,
    read_csv,
)


DEFAULT_DATA_ROOT = Path("dataset/client_data_c45src_c123tgt_2080_timeaware_60_170_window_fullgrid")
DEFAULT_TARGET_PREDICTIONS = Path(
    "results/timeaware_2080_c45src_c123tgt_flower_r3ak16_auto_v2_eval_after_fix/"
    "target_layer_predictions_full_calib_test.csv"
)
DEFAULT_FORMAL_DIR = Path("results/formal_target_ridge_auto_v2_c45_c123_20260625")
DEFAULT_OUTPUT = Path(
    "results/deployment_target_ridge_c45_c123_candidate_20260626/rich_residual_candidate.json"
)


def parse_clients(text: str | None, fallback: list[str]) -> list[str]:
    if not text:
        return fallback
    return [item.strip() for item in text.split(",") if item.strip()]


def selected_modes_from_manifest(manifest: dict[str, Any], target_clients: list[str]) -> dict[str, str]:
    for key in ["hybrid_selected_modes", "selected_modes", "coaware_selected_modes"]:
        modes = manifest.get(key)
        if isinstance(modes, dict) and modes:
            return {client: str(modes.get(client, "baseline_final")) for client in target_clients}
    return {client: "ridge_direct" for client in target_clients}


def load_alpha_by_key(formal_dir: Path) -> dict[tuple[str, int], float]:
    fit_audit = read_csv(formal_dir / "formal_target_ridge_fit_audit.csv")
    return {
        (str(row["client"]), int(float(row["class_id"]))): fnum(row["best_alpha"])
        for row in fit_audit
    }


def fit_export_models(
    data_root: Path,
    target_predictions: Path,
    target_clients: list[str],
    selected_modes: dict[str, str],
    alpha_by_key: dict[tuple[str, int], float],
) -> tuple[list[str], list[dict[str, Any]]]:
    target_set = set(target_clients)
    raw_rows = [
        row
        for row in read_csv(target_predictions)
        if client_name(row.get("client") or row.get("client_id")) in target_set
        and str(row.get("split")) == "calibration"
    ]
    if not raw_rows:
        raise RuntimeError(f"No calibration rows found in {target_predictions} for {target_clients}")
    rows = add_target_features(raw_rows, data_root)
    for row in rows:
        # The target direct head is fitted on the calibration true gas; runtime
        # routes to the same per-gas model using the predicted class.
        row["route_class"] = row["true_class"]
    feature_names = sorted(rows[0]["feature_dict"].keys())

    models: list[dict[str, Any]] = []
    for client in target_clients:
        for cls_id in sorted(CLASS_NAMES):
            cls_rows = [
                row
                for row in rows
                if row["client"] == client and inum(row["true_class"]) == cls_id
            ]
            if not cls_rows:
                raise RuntimeError(f"No calibration rows for {client} class {cls_id}")
            alpha = alpha_by_key[(client, cls_id)]
            model = fit_ridge(cls_rows, feature_names, alpha)
            payload = model.to_json()
            payload.update(
                {
                    "client": client,
                    "class_id": cls_id,
                    "gas": CLASS_NAMES[cls_id],
                    "enabled": selected_modes.get(client) == "ridge_direct",
                    "selected_mode": selected_modes.get(client, "baseline_final"),
                    "n_fit": len(cls_rows),
                    "best_alpha": alpha,
                }
            )
            models.append(payload)
    return feature_names, models


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direction", default="C45_to_C123")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--target-predictions", type=Path, default=DEFAULT_TARGET_PREDICTIONS)
    parser.add_argument("--formal-dir", type=Path, default=DEFAULT_FORMAL_DIR)
    parser.add_argument("--target-clients", default="")
    parser.add_argument("--candidate-name", default="c45_c123_target_ridge_direct_profile")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    manifest = json.loads((args.formal_dir / "manifest.json").read_text(encoding="utf-8"))
    target_clients = parse_clients(args.target_clients, [str(item) for item in manifest["target_clients"]])
    selected_modes = selected_modes_from_manifest(manifest, target_clients)
    alpha_by_key = load_alpha_by_key(args.formal_dir)
    feature_names, models = fit_export_models(
        data_root=args.data_root,
        target_predictions=args.target_predictions,
        target_clients=target_clients,
        selected_modes=selected_modes,
        alpha_by_key=alpha_by_key,
    )

    artifact = {
        "schema": "gaps_target_ridge_profile_artifact.v1",
        "candidate_name": args.candidate_name,
        "direction": args.direction,
        "target_ridge_policy": {
            "schema": "target_ridge_direct.v1",
            "selected_modes": selected_modes,
            "feature_names": feature_names,
            "models": models,
        },
        "route_rescue_policy": {
            "schema": "route_rescue_policy.v1",
            "enabled": False,
            "reason": "No calibration-selected route-rescue rule is enabled for this direction.",
        },
        "source_files": {
            "formal_manifest": str(args.formal_dir / "manifest.json"),
            "fit_audit": str(args.formal_dir / "formal_target_ridge_fit_audit.csv"),
            "selection_table": str(args.formal_dir / "formal_target_ridge_selection_table.csv"),
            "target_predictions": str(args.target_predictions),
            "data_root": str(args.data_root),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest_out = {
        "artifact": str(args.output),
        "direction": args.direction,
        "candidate_name": args.candidate_name,
        "target_clients": target_clients,
        "selected_modes": selected_modes,
        "model_count": len(models),
    }
    (args.output.parent / "manifest.json").write_text(
        json.dumps(manifest_out, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest_out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
