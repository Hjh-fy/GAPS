import scripts.analyze_iotj_canonical_v1_strict_nonoverlap as analysis

from scripts.analyze_iotj_canonical_v1_strict_nonoverlap import (
    apply_serialized_r84_models,
    collapse_flags,
)


def test_strict_collapse_rule_is_preregistered_and_symmetric():
    assert collapse_flags(0.99, 0.80, 10.0, 12.0) == {"classification": False, "regression": False}
    assert collapse_flags(0.99, 0.70, 10.0, 12.0)["classification"] is True
    assert collapse_flags(0.99, 0.98, 10.0, 20.0)["regression"] is True


def test_strict_oracle_uses_serialized_ridge_mapping_api(monkeypatch):
    class MappingOnlyModel:
        def predict(self, features):
            assert features == {"amp_mean": 1.5}
            return 7.0

    monkeypatch.setattr(analysis.common, "r84_row", lambda row: dict(row))
    rows = [{
        "sample_index": 0,
        "true_class": 1,
        "pred_class": 1,
        "true_ppm": 5.0,
        "feature_dict": {"amp_mean": 1.5},
    }]

    observed = apply_serialized_r84_models(rows, {1: MappingOnlyModel()})

    assert observed[0]["pred_84d_h1_ppm"] == 7.0
    assert observed[0]["squared_error"] == 4.0
