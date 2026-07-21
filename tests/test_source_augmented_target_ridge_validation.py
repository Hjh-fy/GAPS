from run_source_augmented_target_ridge_eval import fit_target_ridge_holdout_predictions
from scripts.run_iotj_c5_regression_suite import build_suite_commands
from pathlib import Path
import pytest


def row(sample_index, true_ppm, *, pred_class=0):
    return {
        "client": "C3",
        "split": "calibration",
        "sample_index": sample_index,
        "true_class": 0,
        "pred_class": pred_class,
        "route_class": 0,
        "true_ppm": true_ppm,
        "final_ppm": true_ppm,
        "feature_dict": {"x": float(sample_index)},
    }


def test_fit_target_ridge_holdout_predictions_outputs_validation_rows_with_deployment_route():
    train_feature_rows = [
        row(0, 10.0),
        row(1, 10.0),
        row(2, 20.0),
        row(3, 20.0, pred_class=1),
    ]
    validation_feature_rows = [dict(item) for item in train_feature_rows]

    val_pred, audit = fit_target_ridge_holdout_predictions(
        train_feature_rows,
        validation_feature_rows,
        ["C3"],
        ["x"],
        [0.0, 1.0],
        0.5,
        "demo",
    )

    assert [item["sample_index"] for item in val_pred] == [1, 3]
    assert [item["route_class"] for item in val_pred] == [0, 1]
    assert "demo_ppm" in val_pred[0]
    assert audit[0]["family"] == "demo"


class _JsonModel:
    def to_json(self):
        return {
            "feature_names": ["x"],
            "mean": [0.0],
            "scale": [1.0],
            "coef": [0.0, 1.0],
            "clip_min": 0.0,
            "clip_max": 250.0,
        }


def test_r4_runtime_policy_requires_c5_models_for_all_gases():
    from run_source_augmented_target_ridge_eval import build_r4_runtime_policy_payload

    with pytest.raises(ValueError, match="C5-only"):
        build_r4_runtime_policy_payload(
            source_heads={"ridge_per_gas": [], "mlp_per_gas": [], "shared_mlp": {}},
            target_models={("C4", 0): _JsonModel()},
            feature_names=["x"],
            classifier_sha256="a" * 64,
        )


def test_r4_runtime_policy_serializes_all_c5_target_models():
    from run_source_augmented_target_ridge_eval import build_r4_runtime_policy_payload

    payload = build_r4_runtime_policy_payload(
        source_heads={"ridge_per_gas": [], "mlp_per_gas": [], "shared_mlp": {}},
        target_models={("C5", class_id): _JsonModel() for class_id in range(4)},
        feature_names=["x"],
        classifier_sha256="a" * 64,
    )

    policy = payload["source_aug_target_ridge_policy"]
    assert policy["switch_rule"]["class_ids"] == [0, 1, 2, 3]
    assert [item["class_id"] for item in policy["models"]] == [0, 1, 2, 3]
    assert payload["forbidden_runtime_dependencies"] == ["C3", "C4", "R3aK16", "H8+C4", "P4"]


def test_suite_requests_in_process_r4_and_h23_assets(tmp_path):
    commands = build_suite_commands(
        classifier_checkpoint=Path("results/B5canonical.pth"),
        regression_checkpoint=Path("results/reference.pt"),
        data_root=Path("dataset/c12_c5"),
        output_root=tmp_path / "suite",
        device="cpu",
        seed=42,
        n_random=10,
    )
    assert "--runtime-reference-output" in commands[1]
    assert "--runtime-policy-output" in commands[2]
