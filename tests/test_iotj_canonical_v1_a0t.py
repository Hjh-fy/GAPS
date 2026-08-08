from __future__ import annotations

from scripts.run_iotj_canonical_v1_a0t import (
    TARGETS,
    build_a0t_commands,
    canonical_a0t_config,
)


def _value(command: list[str], option: str) -> str:
    return command[command.index(option) + 1]


def test_a0t_is_equal_label_target_ce_only() -> None:
    cfg = canonical_a0t_config()
    assert cfg["rounds"] == 25
    assert cfg["local_epochs"] == 1
    assert cfg["batch_size"] == 32
    assert cfg["seed"] == 42
    assert cfg["target_ce_weight"] == 1.0
    assert cfg["target_label_budget"] == "same_canonical_calibration_as_A4"
    assert cfg["target_test_selection"] is False
    assert cfg["hyperparameter_search"] is False


def test_a0t_commands_disable_every_non_ce_adaptation_term() -> None:
    for target in TARGETS:
        commands = build_a0t_commands(target)
        server = commands["server"]
        assert _value(server, "--profile") == "ce_only"
        assert _value(server, "--ablation-variant") == "A0T"
        assert _value(server, "--use-selective-agg") == "false"
        assert _value(server, "--use-proto-mmd") == "false"
        assert _value(server, "--use-domain-adapt") == "true"
        assert _value(server, "--da-use-coral") == "false"
        assert _value(server, "--da-use-mmd") == "false"
        assert _value(server, "--da-use-adversarial") == "false"
        assert _value(server, "--da-lambda-target-ce") == "1.0"
        for option in (
            "--da-lambda-coral", "--da-lambda-global-mmd",
            "--da-lambda-class-mmd", "--da-lambda-proto-anchor",
            "--da-lambda-adv", "--da-lambda-proto",
            "--da-lambda-consistency", "--da-lambda-residual",
            "--da-lambda-proto-mmd", "--da-lambda-stage-mmd",
        ):
            assert float(_value(server, option)) == 0.0
        assert _value(server, "--server-calib-data").endswith(f"client_{target[1:]}")
        assert _value(server, "--da-window-length") == "50"
        assert _value(commands["client_c1"], "--local-epochs") == "1"
        assert _value(commands["client_c2"], "--local-epochs") == "1"


def test_a0t_training_commands_never_reference_target_test_arrays() -> None:
    joined = " ".join(
        value
        for target in TARGETS
        for role in ("server", "client_c1", "client_c2")
        for value in build_a0t_commands(target)[role]
    ).lower()
    assert "test_features" not in joined
    assert "test_labels" not in joined
    assert "test_classification" not in joined

