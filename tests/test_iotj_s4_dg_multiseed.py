"""Protocol tests for Phase-1 S4 DG multi-seed confirmation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def _option(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]


def test_multiseed_matrix_reuses_42_and_trains_only_41_43() -> None:
    from scripts.run_iotj_s4_dg_multiseed import phase1_run_specs

    specs = phase1_run_specs()
    assert {(row["seed"], row["method"]) for row in specs} == {
        (41, "fedavg"),
        (41, "gaps_dg_p"),
        (42, "fedavg"),
        (42, "gaps_dg_p"),
        (43, "fedavg"),
        (43, "gaps_dg_p"),
    }
    assert {row["seed"] for row in specs if row["execution"] == "train"} == {41, 43}
    assert {row["seed"] for row in specs if row["execution"] == "reuse"} == {42}


@pytest.mark.parametrize("seed", [41, 43])
@pytest.mark.parametrize("method", ["fedavg", "gaps_dg_p"])
def test_multiseed_commands_propagate_seed_and_exclude_c5(seed: int, method: str) -> None:
    from scripts.run_iotj_s4_dg_multiseed import build_multiseed_commands

    commands = build_multiseed_commands(method, seed)
    expected_id = f"CAN-V1-MB-P1-S4-{'FEDAVG' if method == 'fedavg' else 'DGP'}-S{seed}"
    assert commands["protocol"]["seed"] == seed
    assert commands["protocol"]["experiment_id"] == expected_id
    assert commands["protocol"]["target_access"] == "NONE"
    assert commands["protocol"]["source_clients"] == ["C1", "C2", "C3", "C4"]
    assert all(commands["protocol"][key] is False for key in ("target_x", "target_y", "target_phase", "target_concentration"))
    assert _option(commands["server"], "--seed") == str(seed)
    assert _option(commands["server"], "--run-name") == expected_id
    assert expected_id in _option(commands["server"], "--output-dir")
    assert "client_5" not in json.dumps(commands)
    for command in commands["clients"].values():
        assert _option(command, "--seed") == str(seed)


def test_multiseed_dg_changes_only_frozen_gate_a_mechanism() -> None:
    from scripts.run_iotj_s4_dg_multiseed import build_multiseed_commands

    fedavg = build_multiseed_commands("fedavg", 41)["protocol"]
    dg = build_multiseed_commands("gaps_dg_p", 41)["protocol"]
    held = (
        "dataset",
        "dataset_aggregate_sha256",
        "source_clients",
        "target_clients",
        "target_access",
        "rounds",
        "local_epochs",
        "batch_size",
        "seed",
        "optimizer",
        "optimizer_lr",
        "checkpoint_selection",
        "replay",
        "selective_aggregation",
        "server_domain_adaptation",
        "hyperparameter_search",
        "target_test_selection",
    )
    assert {key: fedavg[key] for key in held} == {key: dg[key] for key in held}
    assert fedavg["prototype_alignment"] is False
    assert fedavg["lambda_proto"] == 0.0
    assert dg["prototype_alignment"] is True
    assert dg["lambda_proto"] == 0.05


def test_multiseed_decision_supported_requires_all_positive_mean_and_sd() -> None:
    from scripts.run_iotj_s4_dg_multiseed import decide_s4_dg_multiseed

    result = decide_s4_dg_multiseed({41: 0.04, 42: 0.075, 43: 0.03})
    assert result["decision"] == "SOURCE_DG_SUPPORTED"
    assert result["all_seeds_positive"] is True
    assert result["mean_gain"] >= 0.03
    assert result["sample_sd"] <= 0.05


def test_multiseed_decision_marks_reversal_or_large_sd_unstable() -> None:
    from scripts.run_iotj_s4_dg_multiseed import decide_s4_dg_multiseed

    reversal = decide_s4_dg_multiseed({41: 0.08, 42: 0.075, 43: -0.01})
    large_sd = decide_s4_dg_multiseed({41: 0.12, 42: 0.03, 43: 0.001})
    assert reversal["decision"] == "SOURCE_DG_UNSTABLE"
    assert large_sd["decision"] == "SOURCE_DG_UNSTABLE"


def test_multiseed_decision_does_not_expand_seeds_when_not_confirmed() -> None:
    from scripts.run_iotj_s4_dg_multiseed import decide_s4_dg_multiseed

    result = decide_s4_dg_multiseed({41: 0.01, 42: 0.02, 43: 0.01})
    assert result["decision"] == "SOURCE_DG_NOT_CONFIRMED"
    assert result["evaluated_seeds"] == [41, 42, 43]
    assert result["next_action"] == "STOP_SEED_EXPANSION_AND_ENTER_PHASE2"


def test_multiseed_lock_gate_requires_exactly_four_new_round25_endpoints(tmp_path: Path) -> None:
    from scripts.run_iotj_s4_dg_multiseed import verify_new_endpoint_locks

    for seed in (41, 43):
        for method in ("fedavg", "gaps_dg_p"):
            directory = tmp_path / f"seed_{seed}" / method
            directory.mkdir(parents=True)
            checkpoint = directory / "server_latest.pth"
            checkpoint.write_bytes(f"{seed}-{method}".encode())
            (directory / "run_manifest.json").write_text(
                json.dumps(
                    {
                        "checkpoint": str(checkpoint),
                        "checkpoint_sha256": __import__("hashlib").sha256(checkpoint.read_bytes()).hexdigest(),
                        "target_test_opened": False,
                        "protocol": {"seed": seed, "target_access": "NONE", "checkpoint_selection": "fixed_round_25"},
                    }
                ),
                encoding="utf-8",
            )
            (directory / "fixed_endpoint_complete.json").write_text(
                json.dumps({"fixed_endpoint": {"round": 25}}), encoding="utf-8"
            )
    locks = verify_new_endpoint_locks(tmp_path)
    assert len(locks) == 4
    bad = tmp_path / "seed_43/gaps_dg_p/run_manifest.json"
    payload = json.loads(bad.read_text(encoding="utf-8"))
    payload["protocol"]["seed"] = 42
    bad.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="seed"):
        verify_new_endpoint_locks(tmp_path)


def test_multiseed_final_hash_index_excludes_mutable_wrapper_files(tmp_path: Path) -> None:
    from scripts.run_iotj_s4_dg_multiseed import write_final_hash_index

    (tmp_path / "immutable.csv").write_text("metric,value\nf1,0.5\n", encoding="utf-8")
    (tmp_path / "RUN_PROGRESS.json").write_text('{"status":"RUNNING"}\n', encoding="utf-8")
    (tmp_path / "runner.stdout.log").write_text("mutable\n", encoding="utf-8")
    payload = write_final_hash_index(tmp_path)
    assert payload["status"] == "PASS"
    assert "immutable.csv" in payload["files"]
    assert "RUN_PROGRESS.json" not in payload["files"]
    assert "runner.stdout.log" not in payload["files"]
    (tmp_path / "RUN_PROGRESS.json").write_text('{"status":"COMPLETE"}\n', encoding="utf-8")
    assert write_final_hash_index(tmp_path) == payload
