from __future__ import annotations

import torch

from client import Client
from gaps_flower.task import make_config


def test_g2_profile_is_only_ce_plus_direct_prototype_alignment() -> None:
    config = make_config(
        device="cpu", local_epochs=1, batch_size=32, profile="dg_proto", seed=42
    )
    assert config.USE_ALIGN is True
    assert config.USE_CONTRASTIVE_ALIGN is False
    assert config.USE_REPLAY_DISTILL is False
    assert config.USE_PROTO_DECOUPLING is False
    assert config.UPLOAD_PROTO_STATS is True
    assert config.LAMBDA_ALIGN == 0.05


def test_g2_alignment_normalizes_feature_and_global_prototype() -> None:
    config = make_config(
        device="cpu", local_epochs=1, batch_size=32, profile="dg_proto", seed=42
    )
    client = Client(client_id=1, config=config)
    features = torch.tensor([[2.0, 0.0]])
    classes = torch.tensor([0])
    phases = torch.tensor([1])
    prototypes = {(0, 1): torch.tensor([0.0, 3.0])}
    loss = client._compute_align_loss(features, classes, phases, prototypes)
    assert torch.isclose(loss, torch.tensor(0.1), atol=1e-7)


def test_g2_commands_have_no_target_access_and_no_extra_mechanisms() -> None:
    from scripts.run_iotj_source_dg_g2 import build_g2_commands

    commands = build_g2_commands()
    joined = " ".join(commands["server"] + commands["client_c1"] + commands["client_c2"])
    assert "client_5" not in joined
    assert "server-calib-data" not in joined
    assert "server-val-data" not in joined
    assert "--strategy gaps" in " ".join(commands["server"])
    assert "--profile dg_proto" in " ".join(commands["server"])
    assert "--use-selective-agg false" in " ".join(commands["server"])
    assert "--use-domain-adapt false" in " ".join(commands["server"])
    assert "--use-proto-mmd false" in " ".join(commands["server"])
    assert commands["protocol"]["target_x"] is False
    assert commands["protocol"]["target_y"] is False
    assert commands["protocol"]["target_phase"] is False
    assert commands["protocol"]["lambda_proto"] == 0.05


def test_g2_round_one_has_no_prototype_broadcast() -> None:
    from scripts.run_iotj_source_dg_g2 import g2_round_contract

    assert g2_round_contract(1) == "CE_ONLY_UPLOAD_PROTOTYPES"
    assert g2_round_contract(2) == "CE_PLUS_GLOBAL_PROTOTYPE_ALIGNMENT"
    assert g2_round_contract(25) == "CE_PLUS_GLOBAL_PROTOTYPE_ALIGNMENT"

