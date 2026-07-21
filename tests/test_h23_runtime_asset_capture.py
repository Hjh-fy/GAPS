from __future__ import annotations

import pytest

from run_h2_3_plus_fusion_profile import fit_ridge_family


def _row(sample_index: int, class_id: int, ppm: float) -> dict[str, object]:
    return {
        "client": "C5",
        "split": "calibration",
        "sample_index": sample_index,
        "true_class": class_id,
        "pred_class": class_id,
        "route_class": class_id,
        "true_ppm": ppm,
        "final_ppm": ppm,
        "feature_dict": {"x": float(sample_index), "class_offset": float(class_id)},
    }


def test_ridge_family_returns_exact_final_models_when_requested() -> None:
    calibration = [
        _row(class_id * 10 + offset, class_id, 10.0 * (class_id + 1) + offset)
        for class_id in range(4)
        for offset in range(4)
    ]
    test_rows = [
        {
            **_row(100 + class_id, class_id, 50.0 + class_id),
            "split": "test",
        }
        for class_id in range(4)
    ]

    _validation, predicted, _audit, final_models = fit_ridge_family(
        calibration,
        test_rows,
        ["C5"],
        ["x", "class_offset"],
        [0.0],
        0.5,
        "demo_ridge",
        return_final_models=True,
    )

    assert set(final_models) == {("C5", 0), ("C5", 1), ("C5", 2), ("C5", 3)}
    for row in predicted:
        key = ("C5", int(row["route_class"]))
        expected = final_models[key].predict([row], clip=True)[0]
        assert row["demo_ridge_ppm"] == pytest.approx(expected)


def test_h23_reference_builder_rejects_non_c5_models() -> None:
    from scripts.run_iotj_c5_h23_plus import build_h23_runtime_reference

    with pytest.raises(ValueError, match="C5-only"):
        build_h23_runtime_reference(
            mlp_models={("C4", 0): object()},
            ridge_models={},
            selected_weight=0.5,
            classifier_sha256="a" * 64,
        )
