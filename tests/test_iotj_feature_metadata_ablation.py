import numpy as np

from run_regression_head_ablation import rich_feature_dict
from scripts.evaluate_iotj_feature_metadata_ablation import (
    EXPECTED_EXCLUDED_FROM_SAFE,
    ONLINE_SAFE_KEYS,
    PROFILE_DIMENSIONS,
    VARIANTS,
    add_variant_features,
    profile_feature_dict,
    validate_feature_schema,
)


def full_feature_dict() -> dict[str, float]:
    meta = {
        "window_start_s": 120.0,
        "window_end_s": 130.0,
        "window_center_s": 125.0,
        "t_onset": 55.0,
        "t_min": 75.0,
        "interpolated_ratio": 0.03,
        "max_gap_inside_window": 0.05,
        "response_phase": "recovery",
        "phase_label": "late",
    }
    return rich_feature_dict(np.zeros((100, 8)), phase=2, meta=meta)


def test_feature_schema_is_exact_83_plus_21_partition() -> None:
    schema = validate_feature_schema(full_feature_dict())

    assert len(schema["sensor_keys"]) == 83
    assert len(schema["metadata_keys"]) == 21
    assert set(schema["online_safe_keys"]) == ONLINE_SAFE_KEYS
    assert set(schema["excluded_from_safe"]) == EXPECTED_EXCLUDED_FROM_SAFE


def test_profile_dimensions_and_noncausal_exclusion() -> None:
    features = full_feature_dict()
    profiles = {
        name: profile_feature_dict(features, name) for name in PROFILE_DIMENSIONS
    }

    assert {name: len(values) for name, values in profiles.items()} == PROFILE_DIMENSIONS
    assert "t_min" not in profiles["M91_ONLINE_SAFE"]
    assert "center_minus_t_min" not in profiles["M91_ONLINE_SAFE"]
    assert not any(key.startswith("response_phase_") for key in profiles["M91_ONLINE_SAFE"])
    assert "t_min" in profiles["M104_FULL"]


def test_h1_adds_exactly_one_feature_to_each_profile() -> None:
    row = {
        "feature_dict": full_feature_dict(),
        "H1_federated_source_ridge_ppm": 50.0,
    }

    for variant, (profile, uses_h1) in VARIANTS.items():
        result = add_variant_features([row], variant)[0]["feature_dict"]
        assert len(result) == PROFILE_DIMENSIONS[profile] + int(uses_h1)
        assert ("srcpred_H1_federated_source_ridge_ppm" in result) is uses_h1

