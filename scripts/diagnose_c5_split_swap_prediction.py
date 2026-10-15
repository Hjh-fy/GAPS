"""Cross-predict the legacy C5 test with frozen role-aware C5 assets.

This is a leakage-risk diagnostic only. It performs no fitting or selection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from federated_dataset import create_client_test_only_loader
from gaps_deploy.c5_h8_runtime import SerializedRidge
from gaps_flower.evaluate_checkpoint import load_checkpoint_model
from gaps_flower.state_fingerprint import checkpoint_provenance
from run_regression_head_ablation import build_oracle_rows
from scripts import run_gaps_cross_target_r84_full as common
from scripts.summarize_iotj_classification_ablation import classification_metrics


STUDY_ID = "iotj_c5_split_swap_diagnostic_20260807"
EXPERIMENT_ID = "DIAG-C5-SPLIT-SWAP-20260807"
LEGACY_DATA_ROOT = (
    ROOT.parents[1]
    / "dataset/client_data_c1234src_c5tgt_2080_timeaware_60_170_window_fullgrid"
)
ROLEAWARE_DATA_ROOT = (
    ROOT.parents[1]
    / "dataset/client_data_c12src_c345tgt_2080_timeaware_60_170_window_fullgrid"
)
ROLEAWARE_STUDY = ROOT / "results/iotj_gaps_roleaware_r84_full_20260805"
DEFAULT_OUTPUT = ROOT / "results" / STUDY_ID


def load_frozen_inputs() -> tuple[Path, dict[int, SerializedRidge], dict[str, Any]]:
    target_manifest_path = ROLEAWARE_STUDY / "regression/C5/target_manifest.json"
    target_manifest = json.loads(target_manifest_path.read_text(encoding="utf-8"))
    checkpoint = Path(target_manifest["classifier_checkpoint_provenance"]["path"])
    provenance = checkpoint_provenance(checkpoint)
    expected = target_manifest["classifier_checkpoint_provenance"]
    if provenance["ordered_state_content_fingerprint"] != expected[
        "ordered_state_content_fingerprint"
    ]:
        raise RuntimeError("FAIL_CLOSED role-aware checkpoint fingerprint differs")
    if provenance["formal_round"] != 25:
        raise RuntimeError("FAIL_CLOSED role-aware checkpoint is not round 25")
    model_path = ROLEAWARE_STUDY / "regression/C5/regression_models.json"
    payload = json.loads(model_path.read_text(encoding="utf-8"))
    models = {
        int(class_id): SerializedRidge.from_json(model)
        for class_id, model in payload.items()
    }
    if sorted(models) != [0, 1, 2, 3]:
        raise RuntimeError("FAIL_CLOSED expected four frozen R84 models")
    return checkpoint, models, {
        "checkpoint": provenance,
        "target_manifest_path": str(target_manifest_path),
        "target_manifest_sha256": common.sha256(target_manifest_path),
        "regression_models_path": str(model_path),
        "regression_models_sha256": common.sha256(model_path),
    }


def evaluate_routes(
    checkpoint: Path,
    *,
    mean_std: tuple[np.ndarray, np.ndarray] | None,
    device: torch.device,
    batch_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    model, config, checkpoint_payload = load_checkpoint_model(
        str(checkpoint), device, batch_size
    )
    if int(checkpoint_payload.get("round", -1)) != 25:
        raise RuntimeError("FAIL_CLOSED checkpoint round differs")
    loader = create_client_test_only_loader(
        LEGACY_DATA_ROOT / "client_5",
        batch_size=config.BATCH_SIZE,
        mean_std=mean_std,
    )
    rows: list[dict[str, Any]] = []
    labels: list[int] = []
    probabilities: list[np.ndarray] = []
    sample_index = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device)
            true = batch[1].long().to(device)
            logits, _, _ = model(x)
            probs = F.softmax(logits, dim=1)
            confidence, pred = probs.max(dim=1)
            true_np = true.cpu().numpy()
            pred_np = pred.cpu().numpy()
            probs_np = probs.cpu().numpy()
            confidence_np = confidence.cpu().numpy()
            for index in range(len(true_np)):
                row = {
                    "client": "C5",
                    "split": "legacy_test",
                    "sample_index": sample_index,
                    "true_class": int(true_np[index]),
                    "pred_class": int(pred_np[index]),
                    "confidence": float(confidence_np[index]),
                }
                for class_id in range(4):
                    row[f"prob_{class_id}"] = float(probs_np[index, class_id])
                rows.append(row)
                sample_index += 1
            labels.extend(true_np.tolist())
            probabilities.append(probs_np)
    probs_all = np.concatenate(probabilities, axis=0)
    if len(rows) != 1360:
        raise RuntimeError(f"FAIL_CLOSED legacy C5 test N={len(rows)}")
    return rows, classification_metrics(labels, probs_all)


def prepare_deployment_rows(
    routes: Sequence[Mapping[str, Any]],
    h1: Mapping[int, SerializedRidge],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    base = build_oracle_rows(LEGACY_DATA_ROOT, ["C5"], "test")
    if len(base) != len(routes):
        raise RuntimeError("FAIL_CLOSED route/data length differs")
    routed: list[dict[str, Any]] = []
    oracle: list[dict[str, Any]] = []
    for row, route in zip(base, routes):
        if (
            int(row["sample_index"]) != int(route["sample_index"])
            or int(row["true_class"]) != int(route["true_class"])
        ):
            raise RuntimeError("FAIL_CLOSED route/data identity differs")
        true_class = int(row["true_class"])
        pred_class = int(route["pred_class"])
        item = {**row, "pred_class": pred_class}
        item["H1_federated_source_ridge_ppm"] = h1[pred_class].predict(
            row["feature_dict"]
        )
        item["confidence"] = float(route["confidence"])
        for class_id in range(4):
            item[f"prob_class_{class_id}"] = float(route[f"prob_{class_id}"])
        routed.append(item)

        oracle_item = {**row, "pred_class": true_class}
        oracle_item["H1_federated_source_ridge_ppm"] = h1[true_class].predict(
            row["feature_dict"]
        )
        oracle.append(oracle_item)
    return routed, oracle


def summary_rows(
    records: Sequence[Mapping[str, Any]],
    *,
    normalization: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    main, per_gas, _, _ = common.summarize("C5", records)
    for rows in (main, per_gas):
        for row in rows:
            row["experiment_id"] = EXPERIMENT_ID
            row["normalization"] = normalization
            row["evidence_status"] = "LEAKAGE_RISK_DIAGNOSTIC_ONLY"
    return main, per_gas


def apply_serialized_models(
    rows: Sequence[Mapping[str, Any]],
    models: Mapping[int, SerializedRidge],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        item = common.r84_row(row)
        pred_class = int(item["pred_class"])
        prediction = float(models[pred_class].predict(item["feature_dict"]))
        truth = float(item["true_ppm"])
        output.append(
            {
                **{key: value for key, value in item.items() if key != "feature_dict"},
                "route_correct": int(int(item["true_class"]) == pred_class),
                "pred_84d_h1_ppm": prediction,
                "pred_ppm": prediction,
                "abs_error": abs(prediction - truth),
                "squared_error": (prediction - truth) ** 2,
            }
        )
    return output


def run(output: Path, device_text: str, batch_size: int) -> None:
    if output.exists():
        raise FileExistsError(f"Refusing existing output: {output}")
    output.mkdir(parents=True)
    checkpoint, models, provenance = load_frozen_inputs()
    h1 = common.load_h1()
    device = torch.device(device_text)
    role_norm = np.load(ROLEAWARE_DATA_ROOT / "norm_stats.npz")
    modes = {
        "legacy_native_norm": None,
        "roleaware_checkpoint_norm": (role_norm["mean"], role_norm["std"]),
    }

    all_main: list[dict[str, Any]] = []
    all_gas: list[dict[str, Any]] = []
    classifications: dict[str, Any] = {}
    oracle_written = False
    for name, mean_std in modes.items():
        routes, classification = evaluate_routes(
            checkpoint,
            mean_std=mean_std,
            device=device,
            batch_size=batch_size,
        )
        routed, oracle = prepare_deployment_rows(routes, h1)
        records = apply_serialized_models(routed, models)
        common.write_csv(output / f"{name}_test_records.csv", records)
        main, per_gas = summary_rows(records, normalization=name)
        all_main.extend(main)
        all_gas.extend(per_gas)
        classifications[name] = classification
        if not oracle_written:
            oracle_records = apply_serialized_models(oracle, models)
            common.write_csv(output / "oracle_route_test_records.csv", oracle_records)
            oracle_metric = common.metrics(oracle_records)
            all_main.append(
                {
                    "experiment_id": EXPERIMENT_ID,
                    "target": "C5",
                    "variant": "R84_FED_H1",
                    "input_dimension": 84,
                    "seed": 42,
                    "evaluation_scope": "S_ORACLE_ROUTE",
                    **oracle_metric,
                    "normalization": "not_applicable",
                    "evidence_status": "LEAKAGE_RISK_DIAGNOSTIC_ONLY",
                }
            )
            oracle_written = True

    common.write_csv(output / "split_swap_summary.csv", all_main)
    common.write_csv(output / "split_swap_per_gas.csv", all_gas)
    manifest = {
        "schema_version": "iotj.c5_split_swap_diagnostic.v1",
        "status": "COMPLETE_DIAGNOSTIC_ONLY",
        "experiment_id": EXPERIMENT_ID,
        "seed": 42,
        "checkpoint_and_models": provenance,
        "prediction_data_root": str(LEGACY_DATA_ROOT),
        "roleaware_normalization_root": str(ROLEAWARE_DATA_ROOT),
        "legacy_test_N": 1360,
        "training_or_refit_performed": False,
        "alpha_or_checkpoint_selection_performed": False,
        "target_test_used_for_selection": False,
        "classification": classifications,
        "interpretation_status": "LEAKAGE_RISK_DIAGNOSTIC_ONLY",
        "leakage_risk": (
            "The role-aware checkpoint used the role-aware C5 calibration split; "
            "legacy test membership is not disjoint from that calibration split."
        ),
    }
    common.write_json(output / "protocol_manifest.json", manifest)
    artifacts = []
    for path in sorted(output.glob("*")):
        if path.is_file() and path.name != "sha256_index.json":
            artifacts.append(
                {
                    "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                    "bytes": path.stat().st_size,
                    "sha256": common.sha256(path),
                }
            )
    common.write_json(
        output / "sha256_index.json",
        {
            "schema_version": "iotj.c5_split_swap_diagnostic.sha256.v1",
            "status": "PASS",
            "artifacts": artifacts,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    run(args.output.resolve(), args.device, args.batch_size)


if __name__ == "__main__":
    main()
