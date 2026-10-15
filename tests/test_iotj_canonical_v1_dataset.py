from tools.build_iotj_canonical_v1 import (
    canonical_preprocessing_manifest,
    merge_role_maps,
    physical_window_key,
)


def test_canonical_manifest_freezes_preprocessing_and_forbids_checkpoint_reuse():
    manifest = canonical_preprocessing_manifest(code_commit="abc123")
    assert manifest["candidate_id"] == "HZ5_MEAN_W10S"
    assert manifest["sampling_rate_hz"] == 5
    assert manifest["points_per_window"] == 50
    assert manifest["baseline"] == "raw_observation_mean_G0_20_50s"
    assert manifest["long_gap_continuous_interpolation"] is False
    assert manifest["reuse_historical_checkpoint"] is False


def test_frozen_role_merge_is_client_order_invariant():
    maps = {
        3: {(3, "a.txt", 60.0): "calibration"},
        4: {(4, "b.txt", 60.0): "test"},
        5: {(5, "c.txt", 60.0): "calibration"},
    }
    first = merge_role_maps(maps, [3, 4, 5])
    second = merge_role_maps(maps, [5, 3, 4])
    assert first == second


def test_physical_window_key_ignores_array_index():
    meta = {"client_id": 5, "filename": "B5_GMe_F090_R1.txt", "window_start_s": 60.0}
    assert physical_window_key(meta) == (5, "B5_GMe_F090_R1.txt", 60.0)
