import scripts.run_iotj_canonical_v1_comparators as runner

from scripts.run_iotj_canonical_v1_comparators import (
    METHODS,
    build_source_fl_commands,
    canonical_comparator_config,
    canonical_mmd_spec,
)


def value(command: list[str], option: str) -> str:
    return command[command.index(option) + 1]


def test_canonical_source_fl_commands_share_frozen_data_and_budget():
    for method in METHODS:
        commands = build_source_fl_commands(method)
        assert commands["protocol"]["rounds"] == 25
        assert commands["protocol"]["local_epochs"] == 1
        assert commands["protocol"]["batch_size"] == 32
        assert commands["protocol"]["seed"] == 42
        assert commands["protocol"]["target_information_regime"] == "source_only"
        assert value(commands["client_c1"], "--local-epochs") == "1"
        assert value(commands["client_c2"], "--local-epochs") == "1"
        joined = " ".join(commands["server"] + commands["client_c1"] + commands["client_c2"])
        assert "iotj_canonical_v1" in joined
        assert "--server-calib-data" not in commands["server"]


def test_fedavg_and_fedprox_keep_adam_while_scaffold_is_canonical_sgd():
    fedavg = build_source_fl_commands("FedAvg")
    fedprox = build_source_fl_commands("FedProx")
    scaffold = build_source_fl_commands("SCAFFOLD")
    assert value(fedavg["server"], "--strategy") == "fedavg"
    assert "--proximal-mu" not in fedavg["client_c1"]
    assert value(fedprox["client_c1"], "--proximal-mu") == "0.01"
    assert fedavg["protocol"]["optimizer"] == "Adam"
    assert fedprox["protocol"]["optimizer"] == "Adam"
    assert value(scaffold["server"], "--strategy") == "scaffold"
    assert value(scaffold["client_c1"], "--optimizer") == "scaffold_sgd"
    assert scaffold["protocol"]["optimizer"] == "SGD"
    assert scaffold["protocol"]["optimizer_lr"] == 5e-4


def test_mmd_is_canonical_x_only_fixed_endpoint():
    for target in ("C3", "C4", "C5"):
        spec = canonical_mmd_spec(target)
        assert spec["target_fields"] == ["x"]
        assert spec["steps"] == 100
        assert spec["optimizer_lr"] == 5e-4
        assert spec["target_ce"] is False
        assert spec["conditional"] is False
        assert spec["pseudo_labels"] is False
        assert spec["checkpoint_selection"] == "fixed_step_100"
        assert spec["target_test_selection"] is False


def test_matrix_has_only_minimal_registered_comparators():
    config = canonical_comparator_config()
    assert config["source_fl_methods"] == list(METHODS)
    assert config["posthoc_da_methods"] == ["MMD"]
    assert config["hyperparameter_search"] is False


def test_matching_comparator_freeze_survives_later_analysis_commits(tmp_path, monkeypatch):
    path = tmp_path / "freeze.json"
    monkeypatch.setattr(runner, "git_head", lambda: "training-code-commit")
    first = runner.write_or_validate_freeze(path)
    monkeypatch.setattr(runner, "git_head", lambda: "later-analysis-commit")
    second = runner.write_or_validate_freeze(path)
    assert first == second
    assert second["freeze_commit"] == "training-code-commit"


def test_source_command_builder_survives_execution_monkeypatch(monkeypatch):
    monkeypatch.setattr(
        runner.frozen,
        "build_flower_commands",
        lambda _experiment_id: runner.build_source_fl_commands("FedAvg"),
    )
    commands = runner.build_source_fl_commands("FedAvg")
    assert commands["protocol"]["method"] == "FedAvg"
