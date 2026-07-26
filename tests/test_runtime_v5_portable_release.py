from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from gaps_deploy.c5_federated_source_ridge_bundle import sha256_file
from gaps_deploy.runtime_v5_cli import main as cli_main
from gaps_deploy.runtime_v5_portable import (
    RuntimeV5PortableBindingError,
    describe_portable_binding,
    verify_portable_binding,
)


ROOT = Path(__file__).resolve().parents[1]
BUILDER_PATH = ROOT / "scripts/build_iotj_runtime_v5_portable_release.py"
SPEC = importlib.util.spec_from_file_location(
    "build_iotj_runtime_v5_portable_release", BUILDER_PATH
)
assert SPEC and SPEC.loader
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)


@pytest.fixture(scope="module")
def portable_release(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    root = tmp_path_factory.mktemp("portable_v5")
    release = root / BUILDER.RELEASE_ID
    archive = root / f"{BUILDER.RELEASE_ID}.zip"
    result = BUILDER.build_release(release, archive)
    assert result["status"] == "BUILT"
    return release, archive


def test_builder_emits_test_free_verified_archive(
    portable_release: tuple[Path, Path],
) -> None:
    release, archive = portable_release
    result = BUILDER.verify_release_directory(release)
    assert result["status"] == "PASS"
    assert result["formal_test_material"] is False
    assert archive.is_file()
    sidecar = Path(f"{archive}.sha256")
    assert sidecar.read_text(encoding="ascii").split()[0] == sha256_file(archive)
    names = [path.relative_to(release).as_posix().lower() for path in release.rglob("*")]
    assert not any(
        token in name
        for token in BUILDER.FORBIDDEN_RELEASE_TOKENS
        for name in names
    )


def test_binding_uses_relative_paths_and_exact_four_assets(
    portable_release: tuple[Path, Path],
) -> None:
    release, _ = portable_release
    binding = verify_portable_binding(release / "portable_binding.json")
    assert set(binding.asset_paths) == {
        "classifier",
        "federated_h1",
        "target_ridge",
        "calibration_lock",
    }
    description = describe_portable_binding(binding.path)
    assert description["formal_test_material"] is False
    for record in description["assets"].values():
        assert not Path(record["path"]).is_absolute()
        assert ".." not in Path(record["path"]).parts


def test_cli_verify_describe_and_synthetic_inference(
    portable_release: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    release, _ = portable_release
    binding = release / "portable_binding.json"
    assert cli_main(["--contract", str(binding), "--verify-only"]) == 0
    assert cli_main(["--contract", str(binding), "--describe-contract"]) == 0
    output = tmp_path / "output.json"
    assert (
        cli_main(
            [
                "--contract",
                str(binding),
                "--input",
                str(release / "synthetic/input.npy"),
                "--metadata",
                str(release / "synthetic/metadata.json"),
                "--phase-file",
                str(release / "synthetic/phase.npy"),
                "--output",
                str(output),
                "--device",
                "cpu",
            ]
        )
        == 0
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "gaps.runtime_v5.inference_output.v1"
    assert payload["row_count"] == 1
    assert set(payload["rows"][0]) == set(payload["output_fields"])
    assert payload["formal_test_material_declared"] is False
    assert cli_main(
        [
            "--contract",
            str(binding),
            "--input",
            str(release / "synthetic/input.npy"),
            "--metadata",
            str(release / "synthetic/metadata.json"),
            "--phase-file",
            str(release / "synthetic/phase.npy"),
            "--output",
            str(output),
        ]
    ) == 2


def test_binding_fails_closed_on_asset_tamper(
    portable_release: tuple[Path, Path], tmp_path: Path
) -> None:
    release, _ = portable_release
    damaged = tmp_path / "damaged"
    shutil.copytree(release, damaged)
    with (damaged / "assets/federated_h1.json").open("ab") as handle:
        handle.write(b"\n")
    with pytest.raises(RuntimeV5PortableBindingError):
        verify_portable_binding(damaged / "portable_binding.json")


def test_cli_rejects_nan_before_runtime_inference(
    portable_release: tuple[Path, Path], tmp_path: Path
) -> None:
    release, _ = portable_release
    values = np.load(release / "synthetic/input.npy", allow_pickle=False)
    values[0, 0, 0] = np.nan
    bad = tmp_path / "nan.npy"
    np.save(bad, values, allow_pickle=False)
    assert (
        cli_main(
            [
                "--contract",
                str(release / "portable_binding.json"),
                "--input",
                str(bad),
                "--metadata",
                str(release / "synthetic/metadata.json"),
                "--phase-file",
                str(release / "synthetic/phase.npy"),
                "--output",
                str(tmp_path / "unused.json"),
            ]
        )
        == 2
    )
    assert not (tmp_path / "unused.json").exists()
