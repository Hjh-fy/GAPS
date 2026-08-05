from pathlib import Path
import subprocess
import sys

from scripts.run_iotj_roleaware_gaps_classification import (
    build_roleaware_commands,
    execute_target,
    validate_local_split,
)
from scripts.run_gaps_roleaware_r84_full import expected_calibration_counts


def test_roleaware_c4_commands_use_formal_split_and_frozen_gaps_protocol():
    commands = build_roleaware_commands("C4")
    server = commands["server"]

    assert "client_data_c12src_c345tgt_2080_timeaware_60_170_window_fullgrid" in " ".join(server)
    assert server[server.index("--server-calib-data") + 1].endswith("/client_4")
    assert server[server.index("--rounds") + 1] == "25"
    assert server[server.index("--selective-warmup") + 1] == "5"
    assert server[server.index("--domain-adapt-steps") + 1] == "100"
    assert commands["protocol"]["local_epochs"] == 1
    assert commands["protocol"]["optimizer"] == "Adam"
    assert commands["protocol"]["optimizer_lr"] == 5e-4
    assert commands["protocol"]["target_test_selection"] is False


def test_roleaware_target_counts_are_the_registered_20_80_endpoints():
    assert expected_calibration_counts("C3") == {
        "calibration": 680,
        "test": 2680,
        "per_gas": 170,
        "fit_per_gas": 130,
        "validation_per_gas": 40,
    }
    for target in ("C4", "C5"):
        assert expected_calibration_counts(target) == {
            "calibration": 320,
            "test": 1360,
            "per_gas": 80,
            "fit_per_gas": 60,
            "validation_per_gas": 20,
        }


def test_local_roleaware_manifest_matches_registered_counts():
    observed = validate_local_split()

    assert observed["C3"]["calibration"] == 680
    assert observed["C3"]["test"] == 2680
    assert observed["C4"]["calibration"] == 320
    assert observed["C4"]["test"] == 1360
    assert observed["C5"]["calibration"] == 320
    assert observed["C5"]["test"] == 1360


def test_roleaware_controller_is_directly_executable():
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, "scripts/run_iotj_roleaware_gaps_classification.py", "--help"],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--output" in completed.stdout

    regression = subprocess.run(
        [sys.executable, "scripts/run_gaps_roleaware_r84_full.py", "--help"],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert regression.returncode == 0, regression.stderr
    assert "--study-root" in regression.stdout


def test_execute_target_can_build_commands_without_recursive_patch(tmp_path, monkeypatch):
    from scripts import run_iotj_final_classification_le1 as frozen

    observed = {}

    def fake_execute(experiment_id, **kwargs):
        commands = frozen.build_flower_commands(experiment_id)
        observed["experiment_id"] = experiment_id
        observed["server"] = commands["server"]

    monkeypatch.setattr(frozen, "execute_full_fl", fake_execute)
    execute_target(
        "C3",
        tmp_path,
        "a" * 40,
        "protocol-hash",
        "ecs",
        "pi",
        "c2",
        1.0,
    )

    assert observed["experiment_id"] == "FCL-RW-GAPS-C3"
    assert "client_data_c12src_c345tgt_2080_timeaware_60_170_window_fullgrid" in " ".join(observed["server"])
