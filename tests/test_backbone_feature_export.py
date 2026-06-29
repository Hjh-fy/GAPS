import math

import numpy as np

from export_backbone_features import (
    build_feature_row,
    class_probability_metrics,
    feature_column_names,
)


def test_class_probability_metrics_returns_confidence_margin_entropy_and_prediction():
    probs = np.asarray([0.1, 0.7, 0.15, 0.05], dtype=np.float64)

    metrics = class_probability_metrics(probs)

    assert metrics["pred_class"] == 1
    assert metrics["confidence"] == 0.7
    assert metrics["margin"] == 0.55
    expected_entropy = -sum(float(p) * math.log(float(p)) for p in probs)
    assert abs(metrics["entropy"] - expected_entropy) < 1e-12


def test_feature_column_names_are_stable_for_cls_and_reg_features():
    names = feature_column_names(num_classes=4, cls_dim=2, reg_dim=3)

    assert names == [
        "pred_class_f6_r25",
        "prob_0",
        "prob_1",
        "prob_2",
        "prob_3",
        "confidence",
        "margin",
        "entropy",
        "cls_feat_000",
        "cls_feat_001",
        "reg_feat_000",
        "reg_feat_001",
        "reg_feat_002",
    ]


def test_build_feature_row_uses_alignment_key_and_feature_values():
    probs = np.asarray([0.1, 0.7, 0.15, 0.05], dtype=np.float64)
    cls_feat = np.asarray([1.0, 2.0], dtype=np.float64)
    reg_feat = np.asarray([3.0, 4.0, 5.0], dtype=np.float64)

    row = build_feature_row(
        client="C4",
        split="test",
        sample_index=12,
        probs=probs,
        cls_feat=cls_feat,
        reg_feat=reg_feat,
        pred_prefix="f6_r25",
    )

    assert row["client"] == "C4"
    assert row["split"] == "test"
    assert row["sample_index"] == 12
    assert row["pred_class_f6_r25"] == 1
    assert row["prob_1"] == 0.7
    assert row["cls_feat_001"] == 2.0
    assert row["reg_feat_002"] == 5.0
