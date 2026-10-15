from __future__ import annotations

import json
from collections import OrderedDict
from types import SimpleNamespace

import torch
import pytest


REQUIRED_COLUMNS = {
    "loss_name",
    "configured_weight",
    "input_available",
    "active_steps",
    "mean_raw_loss",
    "mean_weighted_loss",
    "inactive_reason",
}


def test_loss_activity_accumulator_separates_raw_and_weighted_means() -> None:
    from gaps_flower.loss_activity import LossActivityAccumulator

    activity = LossActivityAccumulator(scope="client", variant="A2")
    activity.record(
        loss_name="semantic_alignment",
        configured_weight=0.5,
        input_available=True,
        raw_loss=2.0,
        active=True,
        inactive_reason="",
    )
    activity.record(
        loss_name="semantic_alignment",
        configured_weight=0.5,
        input_available=True,
        raw_loss=4.0,
        active=True,
        inactive_reason="",
    )
    row = activity.rows()[0]

    assert REQUIRED_COLUMNS <= set(row)
    assert row["active_steps"] == 2
    assert row["mean_raw_loss"] == pytest.approx(3.0)
    assert row["mean_weighted_loss"] == pytest.approx(1.5)
    assert row["inactive_reason"] == ""


def test_zero_weight_and_missing_input_are_observed_as_inactive() -> None:
    from gaps_flower.loss_activity import LossActivityAccumulator

    activity = LossActivityAccumulator(scope="server", variant="A4")
    activity.record(
        loss_name="target_ce",
        configured_weight=0.0,
        input_available=True,
        raw_loss=1.2,
        active=False,
        inactive_reason="configured_weight_zero",
    )
    activity.record(
        loss_name="device_residual",
        configured_weight=0.1,
        input_available=False,
        raw_loss=None,
        active=False,
        inactive_reason="client_residual_unavailable",
    )
    rows = {row["loss_name"]: row for row in activity.rows()}

    assert rows["target_ce"]["active_steps"] == 0
    assert rows["target_ce"]["inactive_reason"] == "configured_weight_zero"
    assert rows["device_residual"]["input_available"] is False
    assert rows["device_residual"]["inactive_reason"] == "client_residual_unavailable"


def _step(raw: float = 1.0) -> dict[str, list[float]]:
    return {
        "val_loss": [raw, raw * 0.9],
        "coral_loss": [raw, raw],
        "mmd_global": [raw, raw],
        "mmd_class": [raw, raw],
        "adv_loss": [raw, raw],
        "proto_anchor": [raw, raw],
        "proto_loss": [raw, raw],
        "consist_loss": [raw, raw],
        "residual_loss": [raw, raw],
        "mmd_proto_loss": [raw, raw],
        "stage_mmd_loss": [raw, raw],
        "align_reg_legacy_loss": [raw, raw],
        "target_ce_loss": [raw, raw],
    }


def _weights() -> dict[str, float]:
    return {
        "source_ce": 1.0,
        "coral": 0.5,
        "global_mmd": 0.5,
        "class_mmd": 0.5,
        "adversarial": 0.5,
        "proto_anchor": 0.3,
        "proto_loss": 0.05,
        "consistency": 2.0,
        "device_residual": 0.1,
        "proto_mmd": 0.2,
        "stage_mmd": 0.2,
        "align_reg_legacy": 0.05,
        "target_ce": 0.0,
    }


def test_a4_server_da_reports_actual_activity_without_client_semantic_or_replay() -> None:
    from gaps_flower.loss_activity import server_da_activity_rows

    availability = {
        "source_x_class_phase": True,
        "target_x": True,
        "target_class": True,
        "target_phase": True,
        "semantic_prototypes": True,
        "client_prototypes": True,
        "two_client_prototypes": True,
        "client_residuals": False,
        "domain_discriminator": True,
        "align_reg_legacy_enabled": False,
    }
    rows = {
        row["loss_name"]: row
        for row in server_da_activity_rows(
            variant="A4",
            step_diagnostics=_step(),
            configured_weights=_weights(),
            availability=availability,
        )
    }

    for active in (
        "source_ce",
        "coral",
        "global_mmd",
        "class_mmd",
        "adversarial",
        "proto_anchor",
        "proto_loss",
        "consistency",
        "proto_mmd",
        "stage_mmd",
    ):
        assert rows[active]["active_steps"] == 2
    assert rows["device_residual"]["active_steps"] == 0
    assert rows["device_residual"]["inactive_reason"] == "client_residual_unavailable"
    assert rows["target_ce"]["active_steps"] == 0
    assert rows["target_ce"]["inactive_reason"] == "configured_weight_zero"
    assert rows["align_reg_legacy"]["active_steps"] == 0


def test_a5_server_da_activates_residual_when_uploaded_input_exists() -> None:
    from gaps_flower.loss_activity import server_da_activity_rows

    availability = {
        "source_x_class_phase": True,
        "target_x": True,
        "target_class": True,
        "target_phase": True,
        "semantic_prototypes": True,
        "client_prototypes": True,
        "two_client_prototypes": True,
        "client_residuals": True,
        "domain_discriminator": True,
        "align_reg_legacy_enabled": False,
    }
    rows = {
        row["loss_name"]: row
        for row in server_da_activity_rows(
            variant="A5",
            step_diagnostics=_step(raw=0.5),
            configured_weights=_weights(),
            availability=availability,
        )
    }

    assert rows["device_residual"]["active_steps"] == 2
    assert rows["device_residual"]["mean_raw_loss"] == pytest.approx(0.5)
    assert rows["device_residual"]["mean_weighted_loss"] == pytest.approx(0.05)


def test_flower_client_metrics_carry_loss_activity_and_variant() -> None:
    from gaps_flower.task import train_one_round

    class StubClient:
        client_id = 1
        config = SimpleNamespace(
            DEVICE="cpu",
            LOCAL_EPOCHS=1,
            USE_ALIGN=False,
            USE_REPLAY_DISTILL=False,
            USE_PROTO_DECOUPLING=False,
            UPLOAD_PROTO_STATS=False,
            FEDPROX_MU=0.0,
        )
        prev_model = None
        train_loader = SimpleNamespace(dataset=[0, 1])

        def train_one_round(self, **_kwargs):
            self.last_loss_activity = [
                {
                    "variant": self.loss_activity_variant,
                    "scope": "client",
                    "loss_name": "source_ce",
                    "configured_weight": 1.0,
                    "input_available": True,
                    "active_steps": 1,
                    "mean_raw_loss": 0.4,
                    "mean_weighted_loss": 0.4,
                    "inactive_reason": "",
                }
            ]
            return OrderedDict(w=torch.tensor([1.0])), {}, {}, torch.empty(0), None, {}

    _arrays, _n, metrics = train_one_round(
        StubClient(),
        round_idx=3,
        fit_config={"ablation_variant": "A4"},
    )
    rows = json.loads(metrics["client_loss_activity_json"])
    assert rows[0]["variant"] == "A4"
    assert rows[0]["loss_name"] == "source_ce"


def test_server_da_summary_builder_uses_actual_diagnostics_and_inputs() -> None:
    from gaps_flower.domain_adaptation import build_server_da_loss_activity

    rows = {
        row["loss_name"]: row
        for row in build_server_da_loss_activity(
            variant="A4",
            diagnostics=_step(raw=0.25),
            hyperparams={
                "LAMBDA_DEEP_CORAL": 0.5,
                "LAMBDA_GLOBAL_MMD": 0.5,
                "LAMBDA_CLASS_MMD": 0.5,
                "LAMBDA_ADV_DOMAIN": 0.5,
                "LAMBDA_PROTO_ANCHOR": 0.3,
                "LAMBDA_TARGET_CE": 0.0,
                "LAMBDA_PROTO": 0.05,
                "LAMBDA_CONSISTENCY": 2.0,
                "LAMBDA_RES": 0.1,
                "LAMBDA_PROTO_MMD": 0.2,
                "LAMBDA_STAGE_MMD": 0.2,
                "LAMBDA_ALIGN_REG_LEGACY": 0.05,
            },
            availability={
                "source_x_class_phase": True,
                "target_x": True,
                "target_class": True,
                "target_phase": True,
                "semantic_prototypes": True,
                "client_prototypes": True,
                "two_client_prototypes": True,
                "client_residuals": False,
                "domain_discriminator": True,
                "align_reg_legacy_enabled": False,
            },
        )
    }
    assert rows["coral"]["mean_weighted_loss"] == pytest.approx(0.125)
    assert rows["device_residual"]["active_steps"] == 0
    assert rows["target_ce"]["inactive_reason"] == "configured_weight_zero"
