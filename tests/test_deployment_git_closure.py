from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.fixture(scope="module")
def tracked_only_checkout(tmp_path_factory: pytest.TempPathFactory) -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    checkout = tmp_path_factory.mktemp("tracked_only_checkout")
    prefix = checkout.resolve().as_posix() + "/"
    result = subprocess.run(
        ["git", "checkout-index", "--all", "--force", f"--prefix={prefix}"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return checkout


@pytest.mark.parametrize(
    "command",
    [
        [
            "-c",
            "import gaps_deploy.inference; import gaps_deploy.r4a_residual",
        ],
        ["-m", "gaps_deploy.build_package", "--help"],
        ["-m", "gaps_deploy.validate_deployment_packages", "--help"],
        ["scripts/validate_final_deployment_bundle.py", "--help"],
    ],
)
def test_deployment_entrypoints_work_from_tracked_only_checkout(
    tracked_only_checkout: Path,
    command: list[str],
) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(tracked_only_checkout)
    result = subprocess.run(
        [sys.executable, *command],
        cwd=tracked_only_checkout,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, (
        f"tracked-only command failed: {command}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
