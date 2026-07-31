from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTROLLER = (
    PROJECT_ROOT
    / "scripts"
    / "lab_three_gas_3class"
    / "run_lab_three_node_fold.ps1"
)
QUEUE = (
    PROJECT_ROOT
    / "scripts"
    / "lab_three_gas_3class"
    / "run_a4_crossboard_queue.ps1"
)


@pytest.mark.parametrize(
    ("direction", "sources", "target"),
    [
        ("P2_to_P1", [2], 1),
        ("P1_to_P3", [1], 3),
        ("P12_to_P3", [1, 2], 3),
    ],
)
def test_controller_resolves_only_requested_roles(
    direction: str,
    sources: list[int],
    target: int,
) -> None:
    result = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-File",
            str(CONTROLLER),
            "-Direction",
            direction,
            "-ContractOnly",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    contract = json.loads(result.stdout)
    assert contract["source_clients"] == sources
    assert contract["target_client"] == target
    assert contract["launch_cloud_b"] is (2 in sources)
    assert contract["launch_pi"] is (1 in sources)


def test_crossboard_queue_is_seed42_and_sequential() -> None:
    text = QUEUE.read_text(encoding="utf-8")

    assert "[int]$Seed = 42" in text
    assert text.count("Direction =") == 3
    assert 'Direction = "P2_to_P1"' in text
    assert 'Direction = "P1_to_P3"' in text
    assert 'Direction = "P12_to_P3"' in text
    assert "foreach ($experiment in $experiments)" in text
