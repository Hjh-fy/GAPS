import pytest
import subprocess
import sys
from pathlib import Path


def test_policy_payload_is_c5_only_and_enables_all_predicted_gases() -> None:
    from scripts.export_iotj_b5_c5_r4_policy import build_policy_payload

    payload = build_policy_payload(
        source_heads={"ridge_per_gas": [], "mlp_per_gas": [], "shared_mlp": {}},
        target_models=[{"client": "C5", "class_id": 0}],
        feature_names=["ch0_mean"],
        classifier_sha256="a" * 64,
    )

    rule = payload["source_aug_target_ridge_policy"]["switch_rule"]
    assert rule["enabled_clients"] == ["C5"]
    assert rule["class_ids"] == [0, 1, 2, 3]
    assert "R3aK16" in payload["forbidden_runtime_dependencies"]


def test_policy_payload_rejects_non_c5_target_model() -> None:
    from scripts.export_iotj_b5_c5_r4_policy import build_policy_payload

    with pytest.raises(ValueError, match="C5-only"):
        build_policy_payload(
            source_heads={"ridge_per_gas": [], "mlp_per_gas": [], "shared_mlp": {}},
            target_models=[{"client": "C4", "class_id": 1}],
            feature_names=["ch0_mean"],
            classifier_sha256="a" * 64,
        )


def test_direct_cli_imports_from_repository_root() -> None:
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "scripts/export_iotj_b5_c5_r4_policy.py", "--help"],
        cwd=root,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
