from scripts.run_iotj_canonical_v1_strict_nonoverlap import (
    TARGETS,
    build_strict_commands,
    strict_run_config,
)


def value(command: list[str], option: str) -> str:
    return command[command.index(option) + 1]


def test_strict_runner_changes_only_dataset_membership_and_identity():
    config = strict_run_config()
    assert config["dataset"] == "iotj_canonical_v1_strict_nonoverlap"
    assert config["rounds"] == 25
    assert config["local_epochs"] == 1
    assert config["batch_size"] == 32
    assert config["seed"] == 42
    assert config["classifier"] == "A4"
    assert config["regression"] == "R84_FED_H1"
    assert config["hyperparameter_search"] is False


def test_strict_commands_keep_frozen_a4_and_use_strict_target():
    for target in TARGETS:
        commands = build_strict_commands(target)
        server = commands["server"]
        assert value(server, "--profile") == "ce_stats"
        assert value(server, "--ablation-variant") == "A4"
        assert value(server, "--target-information-method") == "a4"
        assert value(server, "--da-lambda-target-ce") == "0.0"
        assert value(server, "--da-window-length") == "50"
        assert value(server, "--server-calib-data").endswith(f"client_{target[1:]}")
        assert "iotj_canonical_v1_strict_nonoverlap" in " ".join(server)
        assert value(commands["client_c1"], "--local-epochs") == "1"
        assert value(commands["client_c2"], "--local-epochs") == "1"
        assert commands["protocol"]["strict_raw_time_overlap_seconds"] == 0.0
