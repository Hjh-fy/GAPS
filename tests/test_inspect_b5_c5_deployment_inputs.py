from pathlib import Path


def _complete_nonlegacy_paths(tmp_path: Path) -> dict[str, Path]:
    from scripts.inspect_b5_c5_deployment_inputs import REQUIRED_KEYS

    paths: dict[str, Path] = {}
    for key in REQUIRED_KEYS:
        path = tmp_path / f"{key}.bin"
        path.write_bytes(b"bound")
        paths[key] = path
    return paths


def test_input_audit_blocks_legacy_c3_c4_or_r3ak16_paths(tmp_path: Path) -> None:
    from scripts.inspect_b5_c5_deployment_inputs import audit_input_paths

    paths = _complete_nonlegacy_paths(tmp_path)
    paths["legacy_runtime"] = tmp_path / "C3_C4_r3ak16_runtime.pt"
    result = audit_input_paths(paths)

    assert result["status"] == "blocked"
    assert result["reasons"] == ["legacy_forbidden:legacy_runtime"]


def test_input_audit_reports_missing_required_bound_asset(tmp_path: Path) -> None:
    from scripts.inspect_b5_c5_deployment_inputs import audit_input_paths

    paths = _complete_nonlegacy_paths(tmp_path)
    paths["classifier"] = tmp_path / "missing_b5_classifier.pth"
    result = audit_input_paths(paths)

    assert result["status"] == "blocked"
    assert result["reasons"] == ["missing_required:classifier"]
