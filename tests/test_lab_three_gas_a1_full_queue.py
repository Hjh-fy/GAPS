from __future__ import annotations

import json
from pathlib import Path
import subprocess


PROJECT_ROOT = Path(__file__).resolve().parents[1]
QUEUE = (
    PROJECT_ROOT
    / "scripts"
    / "lab_three_gas_3class"
    / "run_a1_full_crossboard_queue.ps1"
)


def test_queue_contract_freezes_six_ordered_overnight_runs() -> None:
    result = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-File",
            str(QUEUE),
            "-SourceSha",
            "a" * 64,
            "-ContractOnly",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    contract = json.loads(result.stdout)
    assert contract["rounds"] == 25
    assert contract["local_epochs"] == 1
    assert contract["server_da_steps"] == 100
    assert contract["seed"] == 42
    assert contract["selection_policy"] == "last_round"
    assert [
        (row["protocol"], row["direction"], row["dataset"])
        for row in contract["experiments"]
    ] == [
        (
            "A1",
            "P2_to_P3",
            "client_data_lab_3gas_a1_full_crossboard_p2p3_v1",
        ),
        (
            "A4",
            "P2_to_P3",
            "client_data_lab_3gas_a4_crossboard_p2p3_eval_v1",
        ),
        (
            "A1",
            "P1_to_P3",
            "client_data_lab_3gas_a1_full_crossboard_p1p3_v1",
        ),
        (
            "A1",
            "P12_to_P3",
            "client_data_lab_3gas_a1_full_crossboard_p12p3_v1",
        ),
        (
            "A1",
            "P2_to_P1",
            "client_data_lab_3gas_a1_full_crossboard_p2p1_v1",
        ),
        (
            "A1",
            "P3_to_P1",
            "client_data_lab_3gas_a1_full_crossboard_p3p1_v1",
        ),
    ]
    assert len({row["experiment_id"] for row in contract["experiments"]}) == 6
    assert len({row["run_label"] for row in contract["experiments"]}) == 6
