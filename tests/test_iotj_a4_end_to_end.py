from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from scripts.finalize_iotj_a4_end_to_end import (
    FINAL_VARIANTS,
    add_final_variant_features,
    apply_final_regressors,
    build_classifier_manifest,
    canonicalize_route_rows,
    checkpoint_identity,
    fit_final_regressors,
    load_calibration_lock,
    regression_metrics,
    prepare_output_root,
    prepare_output_subdir,
    seal_calibration,
    summarize_regression_records,
    validate_route_rows,
)


ROOT = Path(__file__).resolve().parents[1]
CLASSIFICATION_ROOT = ROOT / "results/iotj_final_classification_le1_20260804"


def test_cli_can_run_directly_from_repository_root() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/finalize_iotj_a4_end_to_end.py", "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--freeze-only" in completed.stdout
    assert "--data-root" in completed.stdout
    assert "--runtime-contract" in completed.stdout
    assert "--h1-manifest" in completed.stdout
    assert "--device" in completed.stdout


def test_checkpoint_identity_uses_ordered_state_content_not_container_bytes(
    tmp_path: Path,
) -> None:
    state = {
        "layer.weight": torch.tensor([[1.0, 2.0]], dtype=torch.float32),
        "layer.bias": torch.tensor([3.0], dtype=torch.float32),
    }
    first = tmp_path / "first.pth"
    second = tmp_path / "second.pth"
    torch.save({"round": 25, "model_state": state, "note": "first"}, first)
    torch.save({"round": 25, "model_state": state, "note": "second"}, second)

    first_id = checkpoint_identity(first)
    second_id = checkpoint_identity(second)

    assert first_id["ordered_state_content_fingerprint"] == second_id[
        "ordered_state_content_fingerprint"
    ]
    assert first_id["whole_file_sha256"] != second_id["whole_file_sha256"]
    assert first_id["equality_basis"] == "ordered_state_content_fingerprint"
    assert first_id["whole_file_sha256_role"] == "provenance_only"


def test_classifier_manifest_freezes_c5_and_blocks_unavailable_c3_c4() -> None:
    manifest = build_classifier_manifest(CLASSIFICATION_ROOT)

    assert manifest["protocol"] == {
        "method": "server-centric A4",
        "rounds": 25,
        "local_epochs": 1,
        "batch_size": 32,
        "seed": 42,
        "source_clients": ["C1", "C2"],
        "target_ce_weight": 0.0,
        "selective_aggregation": False,
        "fixed_endpoint_only": True,
    }
    assert manifest["targets"]["C5"]["status"] == "complete"
    assert manifest["targets"]["C5"]["accuracy"] == pytest.approx(
        0.9933823529411765
    )
    assert manifest["targets"]["C3"]["status"] == "blocked"
    assert manifest["targets"]["C4"]["status"] == "blocked"
    assert manifest["targets"]["C3"]["checkpoint"] is None
    assert manifest["targets"]["C4"]["checkpoint"] is None
    assert manifest["classification_retrained"] is False


def test_prepare_output_root_refuses_nonempty_destination(tmp_path: Path) -> None:
    destination = tmp_path / "final"
    destination.mkdir()
    (destination / "existing.txt").write_text("frozen", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        prepare_output_root(destination)


def _feature_row() -> dict:
    schema = json.loads(
        (
            ROOT
            / "results/iotj_feature_metadata_ablation_20260803_r2/feature_schema_lock.json"
        ).read_text(encoding="utf-8")
    )
    feature_dict = {
        key: float(index + 1)
        for index, key in enumerate(schema["sensor_keys"] + schema["metadata_keys"])
    }
    return {
        "feature_dict": feature_dict,
        "H1_federated_source_ridge_ppm": 10.0,
        "H2_source_per_gas_mlp_ppm": 11.0,
        "H3_source_shared_mlp_ppm": 12.0,
    }


def test_final_regression_variants_have_83_84_and_86_dimensions() -> None:
    row = _feature_row()
    expected = {
        "R83_TARGET_ONLY": 83,
        "R84_FED_H1": 84,
        "R86_ALL_PRIORS": 86,
    }
    assert set(FINAL_VARIANTS) == set(expected)
    for variant, dimension in expected.items():
        enriched = add_final_variant_features([row], variant)
        assert len(enriched[0]["feature_dict"]) == dimension
    assert "srcpred_H1_federated_source_ridge_ppm" in add_final_variant_features(
        [row], "R84_FED_H1"
    )[0]["feature_dict"]


def test_regression_metrics_separates_end_to_end_and_route_correct() -> None:
    rows = [
        {"true_class": 0, "pred_class": 0, "true_ppm": 0.0, "pred": 10.0},
        {"true_class": 1, "pred_class": 2, "true_ppm": 100.0, "pred": 200.0},
    ]
    all_metrics = regression_metrics(rows, "pred", [True, True])
    correct_metrics = regression_metrics(rows, "pred", [True, False])

    assert all_metrics["N"] == 2
    assert all_metrics["RMSE"] == pytest.approx((5050.0) ** 0.5)
    assert correct_metrics["N"] == 1
    assert correct_metrics["RMSE"] == pytest.approx(10.0)
    assert correct_metrics["NRMSE"] < all_metrics["NRMSE"]


def test_regression_metrics_rejects_nonfinite_predictions() -> None:
    rows = [
        {"true_class": 0, "pred_class": 0, "true_ppm": 1.0, "pred": float("nan")}
    ]
    with pytest.raises(RuntimeError, match="non-finite"):
        regression_metrics(rows, "pred", [True])


def test_validate_route_rows_rejects_misaligned_or_incomplete_probabilities() -> None:
    valid = [
        {
            "sample_index": 0,
            "true_class": 0,
            "pred_class": 0,
            "confidence": 0.7,
            "prob_class_0": 0.7,
            "prob_class_1": 0.1,
            "prob_class_2": 0.1,
            "prob_class_3": 0.1,
        }
    ]
    validate_route_rows(valid, expected_n=1)

    invalid = [dict(valid[0], sample_index=1)]
    with pytest.raises(RuntimeError, match="canonical"):
        validate_route_rows(invalid, expected_n=1)

    incomplete = [dict(valid[0])]
    incomplete[0].pop("prob_class_3")
    with pytest.raises(RuntimeError, match="probabilities"):
        validate_route_rows(incomplete, expected_n=1)


def test_canonicalize_route_rows_maps_frozen_evaluator_probability_schema() -> None:
    raw = [
        {
            "sample_index": 0,
            "true_class": 2,
            "pred_class": 2,
            "prob_0": 0.1,
            "prob_1": 0.1,
            "prob_2": 0.7,
            "prob_3": 0.1,
        }
    ]
    canonical = canonicalize_route_rows(raw)
    assert canonical[0]["prob_class_2"] == pytest.approx(0.7)
    assert all(f"prob_{class_id}" not in canonical[0] for class_id in range(4))
    validate_route_rows(canonical, expected_n=1)


def test_fit_and_apply_final_regressors_use_calibration_only() -> None:
    template = _feature_row()
    oracle = []
    deployment = []
    routes = []
    for class_id in range(4):
        for within_class in range(80):
            sample_index = class_id * 80 + within_class
            feature_dict = dict(template["feature_dict"])
            feature_dict["global_mean"] = float(within_class)
            common = {
                **template,
                "sample_index": sample_index,
                "client": "C5",
                "split": "calibration",
                "true_class": class_id,
                "pred_class": class_id,
                "true_ppm": float(within_class),
                "feature_dict": feature_dict,
            }
            oracle.append({**common, "route_class": class_id})
            deployment.append({**common, "route_class": class_id})
            routes.append(
                {
                    "sample_index": sample_index,
                    "true_class": class_id,
                    "pred_class": class_id,
                    "confidence": 0.7,
                    "prob_class_0": 0.7 if class_id == 0 else 0.1,
                    "prob_class_1": 0.7 if class_id == 1 else 0.1,
                    "prob_class_2": 0.7 if class_id == 2 else 0.1,
                    "prob_class_3": 0.7 if class_id == 3 else 0.1,
                }
            )

    models, selection = fit_final_regressors(oracle, deployment)
    records = apply_final_regressors(deployment, routes, models)

    assert len(selection) == 12
    assert {row["selection_split"] for row in selection} == {
        "C5_calibration_internal_60_fit_20_validation"
    }
    assert {row["target_input_dimension"] for row in selection} == {83, 84, 86}
    assert len(records) == 320
    assert set(FINAL_VARIANTS.values()) == {
        "pred_83d_ppm",
        "pred_84d_h1_ppm",
        "pred_86d_all_priors_ppm",
    }
    assert all(row["route_correct"] == 1 for row in records)
    assert all(0.0 <= row["class_entropy"] <= 1.0 for row in records)
    assert all(row["qc_risk_score"] >= 0.0 for row in records)


def test_regression_output_subdir_is_new_and_refuses_overwrite(tmp_path: Path) -> None:
    root = tmp_path / "final"
    root.mkdir()
    (root / "final_classifier_manifest.json").write_text("{}", encoding="utf-8")

    regression = prepare_output_subdir(root, "regression")
    assert regression.is_dir()
    (regression / "sealed.txt").write_text("fixed", encoding="utf-8")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        prepare_output_subdir(root, "regression")


def test_calibration_lock_must_be_persisted_and_read_back(tmp_path: Path) -> None:
    selection = [
        {
            "variant": "R84_FED_H1",
            "class_id": 0,
            "selected_alpha": 1.0,
            "target_test_used_for_selection": False,
        }
    ]
    models = {"R84_FED_H1": {0: type("M", (), {"to_json": lambda self: {"alpha": 1.0}})()}}
    lock_path = seal_calibration(tmp_path, selection, models)
    loaded = load_calibration_lock(lock_path)

    assert loaded["status"] == "SEALED_BEFORE_TARGET_TEST"
    assert loaded["target_test_opened"] is False
    assert loaded["selection"][0]["target_test_used_for_selection"] is False

    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    payload["target_test_opened"] = True
    lock_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="target test"):
        load_calibration_lock(lock_path)


def test_summarize_regression_records_separates_route_decomposition() -> None:
    rows = []
    for class_id in range(4):
        rows.append(
            {
                "true_class": class_id,
                "pred_class": class_id,
                "gas_true": ["CO", "H2", "CH4", "C2H4"][class_id],
                "true_ppm": 100.0,
                "pred_83d_ppm": 110.0,
                "pred_84d_h1_ppm": 105.0,
                "pred_86d_all_priors_ppm": 104.0,
                "route_correct": 1,
            }
        )
    rows.append(
        {
            "true_class": 0,
            "pred_class": 1,
            "gas_true": "CO",
            "true_ppm": 100.0,
            "pred_83d_ppm": 200.0,
            "pred_84d_h1_ppm": 190.0,
            "pred_86d_all_priors_ppm": 180.0,
            "route_correct": 0,
        }
    )

    main, per_gas, route = summarize_regression_records(rows)

    assert len(main) == 6
    assert {row["evaluation_scope"] for row in main} == {"S_ALL", "S_CC"}
    assert len(per_gas) == 24
    assert any(row["evaluation_scope"] == "MISROUTED" for row in route)
