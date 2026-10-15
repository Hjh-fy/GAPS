import math

import numpy as np

from run_regression_head_ablation import rich_feature_dict
from scripts.run_gaps_cross_target_r84_full import (
    RIDGE_ALPHAS,
    apply_r84_models,
    checkpoint_for,
    fit_r84_models,
)


def _calibration_rows(per_class: int):
    oracle = []
    deployment = []
    for class_id in range(4):
        for offset in range(per_class):
            index = class_id * per_class + offset
            window = np.arange(800, dtype=np.float64).reshape(100, 8) * 0.001 + offset
            sensor = rich_feature_dict(window, phase=offset % 3, meta={})
            truth = float(offset)
            common = {
                "client": "C4",
                "sample_index": index,
                "true_class": class_id,
                "true_ppm": truth,
                "feature_dict": sensor,
            }
            oracle.append({**common, "pred_class": class_id, "H1_federated_source_ridge_ppm": truth + 1.0})
            deployment.append({**common, "pred_class": class_id, "H1_federated_source_ridge_ppm": truth + 1.0})
    return oracle, deployment


def test_c4_full_r84_uses_30_10_calibration_then_refits_40():
    oracle, deployment = _calibration_rows(40)
    models, selection = fit_r84_models("C4", oracle, deployment)

    assert set(models) == {0, 1, 2, 3}
    assert len(selection) == 4
    assert all(row["calibration_fit_N"] == 30 for row in selection)
    assert all(row["calibration_validation_N"] == 10 for row in selection)
    assert all(row["calibration_refit_N"] == 40 for row in selection)
    assert all(row["selected_alpha"] in RIDGE_ALPHAS for row in selection)
    assert all(row["target_test_used_for_selection"] is False for row in selection)
    assert all(len(model.feature_names) == 84 for model in models.values())


def test_apply_r84_models_uses_predicted_route_not_true_route():
    class ConstantModel:
        def __init__(self, value):
            self.value = value

        def predict(self, rows):
            return np.asarray([self.value] * len(rows), dtype=np.float64)

    rows = [{
        "sample_index": 0,
        "true_class": 0,
        "pred_class": 2,
        "true_ppm": 50.0,
        "feature_dict": rich_feature_dict(np.arange(800).reshape(100, 8), phase=0, meta={}),
        "H1_federated_source_ridge_ppm": 25.0,
    }]
    models = {index: ConstantModel(float(index * 10)) for index in range(4)}

    observed = apply_r84_models(rows, models)

    assert observed[0]["pred_84d_h1_ppm"] == 20.0
    assert observed[0]["route_correct"] == 0
    assert math.isclose(observed[0]["abs_error"], 30.0)


def test_checkpoint_identity_uses_ordered_state_content_fingerprint():
    checkpoint, run_manifest, provenance = checkpoint_for("C3")

    assert checkpoint.is_file()
    assert provenance["formal_round"] == 25
    assert provenance["whole_file_sha256"] == run_manifest["checkpoint_sha256"]
    assert len(provenance["ordered_state_content_fingerprint"]) == 64
    assert provenance["equality_basis"] == "ordered_state_content_fingerprint"
    assert provenance["whole_file_sha256_role"] == "provenance_only"
