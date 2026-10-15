import pytest


def test_h23_payload_is_c5_only_with_frozen_blend() -> None:
    from scripts.export_iotj_b5_c5_h23_reference import build_h23_payload

    payload = build_h23_payload(
        mlp_models=[{"client": "C5", "class_id": 0}],
        ridge_models=[{"client": "C5", "class_id": 0}],
        selected_weight=0.5,
        classifier_sha256="a" * 64,
    )

    assert payload["h23_reference_policy"]["blend_weight"] == 0.5
    assert payload["h23_reference_policy"]["target_client"] == "C5"


def test_h23_payload_rejects_non_c5_model() -> None:
    from scripts.export_iotj_b5_c5_h23_reference import build_h23_payload

    with pytest.raises(ValueError, match="C5-only"):
        build_h23_payload(
            mlp_models=[{"client": "C4", "class_id": 0}],
            ridge_models=[],
            selected_weight=0.5,
            classifier_sha256="a" * 64,
        )
