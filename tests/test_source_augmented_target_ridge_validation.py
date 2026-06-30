from run_source_augmented_target_ridge_eval import fit_target_ridge_holdout_predictions


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
