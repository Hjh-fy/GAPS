import importlib
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "iotj_b2_b5_cross_direction_20260713.json"
RESULTS_ROOT = "results/iotj_b2_b5_cross_direction_20260713"


def _generator():
    return importlib.import_module("scripts.generate_iotj_cross_direction_commands")


def test_frozen_directions_counts_and_device_assignments() -> None:
    generator = _generator()

    specs = generator.load_direction_specs(CONFIG_PATH)

    assert [
        (spec.direction_id, spec.source_clients, spec.target_client)
        for spec in specs
    ] == [
        ("F1_C1_TO_C5", (1,), 5),
        ("R1_C5_TO_C1", (5,), 1),
        ("R2_C45_TO_C1", (4, 5), 1),
    ]
    assert specs[0].executors == {1: "pi"}
    assert specs[1].executors == {5: "pi"}
    assert specs[2].executors == {4: "pi", 5: "pc"}
    assert specs[0].expected_source_train == {1: 2360}
    assert specs[0].expected_source_calibration == {1: 320}
    assert specs[0].expected_target_counts == {"calibration": 320, "test": 1360}
    assert specs[1].expected_source_train == {5: 1200}
    assert specs[1].expected_source_calibration == {5: 160}
    assert specs[1].expected_target_counts == {"calibration": 680, "test": 2680}


@pytest.mark.parametrize("group_id", ["B2", "B5"])
def test_manifest_keeps_frozen_training_budget_and_semantic_core(group_id: str) -> None:
    generator = _generator()
    direction = generator.load_direction_specs(CONFIG_PATH)[0]

    manifest = generator.build_run_manifest(
        direction,
        group_id,
        42,
        repo_root=REPO_ROOT,
        results_root=RESULTS_ROOT,
    )

    assert manifest["training"] == {
        "rounds": 25,
        "local_epochs": 5,
        "batch_size": 32,
        "client_lr": 5e-4,
        "profile": "proto_replay",
        "strategy": "gaps",
        "use_selective_agg": True,
        "use_proto_mmd_diagnostics": False,
    }
    adaptation = manifest["server_adaptation"]
    assert adaptation["steps"] == 100
    assert adaptation["lr"] == 5e-4
    assert adaptation["lambda_proto_anchor"] == 0.3
    assert adaptation["lambda_proto"] == 0.05
    assert adaptation["lambda_consistency"] == 2.0
    assert adaptation["lambda_residual"] == 0.1
    assert adaptation["lambda_proto_mmd"] == 0.0
    assert adaptation["lambda_target_ce"] == 0.0
    assert adaptation["mmd_objective"] == "mmd2"


def test_b2_and_b5_isolate_only_the_extra_full_stack() -> None:
    generator = _generator()
    direction = generator.load_direction_specs(CONFIG_PATH)[1]
    b2 = generator.build_run_manifest(
        direction, "B2", 42, repo_root=REPO_ROOT, results_root=RESULTS_ROOT
    )
    b5 = generator.build_run_manifest(
        direction, "B5", 42, repo_root=REPO_ROOT, results_root=RESULTS_ROOT
    )

    b2_da = b2["server_adaptation"]
    b5_da = b5["server_adaptation"]
    assert b2_da["lambda_global_mmd"] == 0.5
    assert b2_da["lambda_class_mmd"] == 0.5
    assert b2_da["lambda_coral"] == 0.0
    assert b2_da["lambda_stage_mmd"] == 0.0
    assert b2_da["lambda_adv"] == 0.0
    assert b5_da["lambda_global_mmd"] == 0.5
    assert b5_da["lambda_class_mmd"] == 0.5
    assert b5_da["lambda_coral"] == 0.5
    assert b5_da["lambda_stage_mmd"] == 0.2
    assert b5_da["lambda_adv"] == 0.5
    assert b5_da["stage_alignment"] == "cross_domain_same_class_phase"
    assert b5_da["adv_feature_objective"] == "wasserstein_min"


def test_manifest_commands_match_protocol_and_executor() -> None:
    generator = _generator()
    direction = generator.load_direction_specs(CONFIG_PATH)[2]

    manifest = generator.build_run_manifest(
        direction, "B2", 42, repo_root=REPO_ROOT, results_root=RESULTS_ROOT
    )

    assert manifest["protocol"]["source_clients"] == [4, 5]
    assert manifest["protocol"]["target_clients"] == [1]
    assert manifest["protocol"]["expected_source_calibration"] == {
        "4": 160,
        "5": 160,
    }
    server = manifest["commands"]["server_ecs"]
    assert server[server.index("--min-clients") + 1] == "2"
    assert server[server.index("--server-calib-data") + 1].endswith("/client_1")
    clients = manifest["commands"]["clients"]
    assert [(row["client_id"], row["executor"]) for row in clients] == [
        (4, "pi"),
        (5, "pc"),
    ]
    for row in clients:
        command = row["command"]
        assert int(command[command.index("--client-id") + 1]) == row["client_id"]
        assert direction.data_root in command[command.index("--data-root") + 1]
    hashes = manifest["provenance"]["active_file_sha256"]
    assert "client_4/calibration_features.npy" in hashes
    assert "client_5/calibration_classification_labels.npy" in hashes


def test_generator_emits_exact_approved_seed42_order(tmp_path: Path) -> None:
    generator = _generator()

    manifests = generator.generate_manifests(
        CONFIG_PATH,
        tmp_path / "commands",
        repo_root=REPO_ROOT,
        results_root=RESULTS_ROOT,
        seeds=(42,),
    )

    assert [(row["direction_id"], row["group_id"]) for row in manifests] == [
        ("F1_C1_TO_C5", "B2"),
        ("F1_C1_TO_C5", "B5"),
        ("R1_C5_TO_C1", "B2"),
        ("R1_C5_TO_C1", "B5"),
        ("R2_C45_TO_C1", "B2"),
        ("R2_C45_TO_C1", "B5"),
    ]
    index = generator.load_json(tmp_path / "commands" / "command_index.json")
    assert index["training_runs"] == [row["run_name"] for row in manifests]
    assert len(list((tmp_path / "commands").glob("*/command_manifest.json"))) == 6


def test_generator_cli_runs_from_repo_root(tmp_path: Path) -> None:
    output_root = tmp_path / "cli_commands"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/generate_iotj_cross_direction_commands.py",
            "--config",
            str(CONFIG_PATH),
            "--seed",
            "42",
            "--output-root",
            str(output_root),
            "--results-root",
            RESULTS_ROOT,
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    assert (output_root / "command_index.json").is_file()


def test_manifest_rejects_unapproved_group() -> None:
    generator = _generator()
    direction = generator.load_direction_specs(CONFIG_PATH)[0]

    with pytest.raises(ValueError, match="only B2 and B5"):
        generator.build_run_manifest(
            direction, "B3", 42, repo_root=REPO_ROOT, results_root=RESULTS_ROOT
        )
