import pytest

from scripts.run_iotj_c5_h23_plus import build_c5_anchor_rows


def test_c5_anchor_maps_expanded_grid_mlp_without_c4_rescue() -> None:
    rows = [
        {
            "client": "C5",
            "split": "test",
            "sample_index": 0,
            "h2_c5_grid_mlp_ppm": 123.0,
            "feature_dict": {"x": 1.0},
        }
    ]

    result = build_c5_anchor_rows(rows)

    assert result[0]["h23_anchor_ppm"] == 123.0
    assert result[0]["h2_3_current_ppm"] == 123.0
    assert "feature_dict" not in result[0]


def test_c5_anchor_rejects_c4_rows() -> None:
    with pytest.raises(ValueError, match="non-C5"):
        build_c5_anchor_rows(
            [
                {
                    "client": "C4",
                    "split": "test",
                    "sample_index": 0,
                    "h2_c5_grid_mlp_ppm": 100.0,
                }
            ]
        )
