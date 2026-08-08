from pathlib import Path

import pytest

from scripts.run_iotj_canonical_v1_classification import (
    TARGETS,
    audit_protocol,
    build_canonical_commands,
    canonical_classification_config,
)


def test_canonical_classification_is_from_scratch_le1_fixed_endpoint() -> None:
    config = canonical_classification_config()
    assert config["rounds"] == 25
    assert config["local_epochs"] == 1
    assert config["batch_size"] == 32
    assert config["seed"] == 42
    assert config["checkpoint_reuse"] is False
    assert config["checkpoint_selection"] == "fixed_round_25"
    assert config["hyperparameter_search"] is False


@pytest.mark.parametrize("target", TARGETS)
def test_canonical_commands_use_only_canonical_v1_and_calibration_for_adaptation(
    target: str,
) -> None:
    commands = build_canonical_commands(target)
    flattened = " ".join(
        value
        for role in ("server", "client_c1", "client_c2")
        for value in commands[role]
    )
    assert "iotj_canonical_v1" in flattened
    assert "client_data_c12src_c345tgt" not in flattened
    assert "client_data_c1234src_c5tgt" not in flattened
    assert "--local-epochs 1" in " ".join(commands["client_c1"])
    assert "--local-epochs 1" in " ".join(commands["client_c2"])
    assert "--da-window-length 50" in " ".join(commands["server"])
    assert commands["protocol"]["target"] == target
    assert commands["protocol"]["adaptation_target_split"] == "calibration"
    assert commands["protocol"]["target_test_selection"] is False
    assert commands["protocol"]["checkpoint_reuse"] is False
    assert "--checkpoint" not in flattened
    assert "--resume" not in flattened


def test_each_target_has_distinct_from_scratch_run_identity() -> None:
    identities = [build_canonical_commands(target)["protocol"]["experiment_id"] for target in TARGETS]
    assert len(set(identities)) == 3
    assert all("CANONICAL-V1" in identity for identity in identities)


@pytest.mark.parametrize("target", TARGETS)
def test_canonical_classifier_preserves_final_a4_router_settings(target: str) -> None:
    commands = build_canonical_commands(target)
    server = commands["server"]
    client_c1 = commands["client_c1"]
    assert server[server.index("--profile") + 1] == "ce_stats"
    assert client_c1[client_c1.index("--profile") + 1] == "ce_stats"
    assert server[server.index("--ablation-variant") + 1] == "A4"
    assert server[server.index("--target-information-method") + 1] == "a4"
    assert server[server.index("--use-selective-agg") + 1] == "false"
    assert server[server.index("--require-selective-after-warmup") + 1] == "false"
    assert server[server.index("--use-domain-adapt") + 1] == "true"
    assert server[server.index("--domain-adapt-steps") + 1] == "100"
    assert "proto_replay" not in " ".join(commands["client_c1"])
    assert commands["protocol"]["classifier_router"] == "A4"


def test_canonical_dataset_freeze_is_present() -> None:
    root = Path(__file__).resolve().parents[1]
    assert (root / "dataset/iotj_canonical_v1/dataset_sha256.json").is_file()
    assert (root / "results/iotj_canonical_v1/preflight.json").is_file()


def test_canonical_classification_protocol_audit_passes() -> None:
    audit = audit_protocol()
    assert audit["status"] == "PASS"
    assert audit["test_arrays_referenced_by_training_commands"] is False
    assert audit["checkpoint_reuse_tokens_present"] is False
    assert audit["all_targets_fixed"] is True
