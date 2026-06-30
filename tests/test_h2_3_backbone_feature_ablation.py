from run_h2_3_backbone_feature_ablation import (
    build_feature_groups,
    c5_nonco_wrong_route_audit,
    merge_backbone_features,
)


def test_merge_backbone_features_matches_client_split_sample_index():
    rows = [{"client": "C3", "split": "test", "sample_index": "7", "feature_dict": {"rich": 1.0}}]
    features = [{"client": "C3", "split": "test", "sample_index": "7", "confidence": "0.8", "reg_feat_000": "1.5"}]

    merged = merge_backbone_features(rows, features)

    assert merged[0]["backbone_feature_dict"]["confidence"] == 0.8
    assert merged[0]["backbone_feature_dict"]["reg_feat_000"] == 1.5


def test_build_feature_groups_separates_embedding_and_b0_priors():
    row = {
        "feature_dict": {"rich": 1.0},
        "backbone_feature_dict": {
            "confidence": 0.8,
            "prob_0": 0.1,
            "cls_feat_000": 2.0,
            "reg_feat_000": 3.0,
        },
        "final_ppm": "42.0",
        "base_r3ak16_raw_ppm": "41.0",
        "routed_pred_ppm": "40.0",
    }

    groups = build_feature_groups(row)

    assert groups["A0_rich_only"] == {"rich": 1.0}
    assert "reg_feat_000" in groups["A3_rich_plus_reg_feat"]
    assert "final_ppm" in groups["A4_rich_plus_b0"]
    assert "routed_pred_ppm" in groups["A5_rich_plus_source_priors"]
    assert "reg_feat_000" in groups["A7_rich_plus_all_priors"]
    assert "final_ppm" in groups["A7_rich_plus_all_priors"]


def test_c5_nonco_wrong_route_audit_counts_nonco_as_co_routes():
    rows = [
        {"client": "C5", "true_class": "0", "pred_class": "1"},
        {"client": "C5", "true_class": "2", "pred_class": "1"},
        {"client": "C5", "true_class": "1", "pred_class": "1"},
        {"client": "C4", "true_class": "0", "pred_class": "1"},
    ]

    audit = c5_nonco_wrong_route_audit(rows)

    assert audit["C5_nonCO_N"] == 2
    assert audit["C5_nonCO_pred_CO_N"] == 2
    assert audit["C5_nonCO_pred_CO_rate"] == 1.0
