from pathlib import Path
import inspect

import pytest
import torch

from scripts.evaluate_iotj_federated_source_regression_prior import (
    MODEL_SELECTION_SPLIT,
    FORMAL_OUTPUT_FILES,
    assert_identical_initialization,
    assert_selection_rows,
    feature_schema,
    fit_target_variant_calibration,
    freeze_decision_gate,
    require_new_empty_output,
    selection_sha256,
    state_sha256,
    training_strength,
    validate_state_contract,
)
from gaps_flower.regression_task import fedavg_regression_states


class TinyTopologyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = torch.nn.Linear(2, 2)
        self.reg_proj = torch.nn.Linear(2, 2)
        self.reg_heads = torch.nn.ModuleList([torch.nn.Linear(2, 1)])


def test_rs_feature_schemas_are_exact_and_versioned() -> None:
    rich = ("rich_a", "rich_b")
    assert feature_schema(rich, "RS1_local_experts") == (
        "rich_a", "rich_b", "srcpred_pred_C1", "srcpred_pred_C2"
    )
    assert feature_schema(rich, "RS2_fedavg_prior") == (
        "rich_a", "rich_b", "srcpred_pred_FedAvg"
    )
    assert feature_schema(rich, "RS3_local_plus_fedavg") == (
        "rich_a", "rich_b", "srcpred_pred_C1", "srcpred_pred_C2",
        "srcpred_pred_FedAvg",
    )
    assert "decision_gate.json" in FORMAL_OUTPUT_FILES


def test_source_steps_are_total_per_client_not_per_round() -> None:
    strength = training_strength(3, 100)
    assert strength["round_step_allocation_per_client"] == [34, 33, 33]
    assert strength["actual_optimizer_steps_by_client"] == {"C1": 100, "C2": 100}
    assert strength["actual_optimizer_steps_all_clients"] == 200
    assert strength["fedavg_execution_count"] == 3


def test_calibration_fitter_has_no_test_input() -> None:
    assert "test_rows" not in inspect.signature(
        fit_target_variant_calibration
    ).parameters


def test_selection_rejects_test_rows() -> None:
    with pytest.raises(ValueError, match="calibration-validation"):
        assert_selection_rows([{"selection_split": "test"}])


def test_test_labels_cannot_change_selection_signature() -> None:
    base = [{
        "selection_split": MODEL_SELECTION_SPLIT,
        "variant": "RS3_local_plus_fedavg",
        "best_alpha": 1.0,
        "test_label": 10.0,
    }]
    changed = [{**base[0], "test_label": 999999.0}]
    assert selection_sha256(base) == selection_sha256(changed)


def _gate_inputs(rs3_rmse: float):
    metrics = {
        "RS0_pooled_source": {"ALL_RMSE": 100.0},
        "RS1_local_experts": {"ALL_RMSE": 85.0},
        "RS2_fedavg_prior": {"ALL_RMSE": 90.0},
        "RS3_local_plus_fedavg": {"ALL_RMSE": rs3_rmse},
        "RS4_rich_only": {"ALL_RMSE": 110.0},
    }
    per_gas = {
        variant: [
            {"gas": gas, "RMSE": float(metrics[variant]["ALL_RMSE"])}
            for gas in ("Ethanol", "CO", "Ethylene", "Methane")
        ]
        for variant in metrics
    }
    return metrics, per_gas


@pytest.mark.parametrize(
    ("rs3_rmse", "status", "candidate", "recommendation"),
    [
        (99.0, "candidate", "RS3_local_plus_fedavg", "PROMOTE_FOR_CONFIRMATION"),
        (
            105.0,
            "paper_preferred_candidate",
            "RS3_local_plus_fedavg",
            "PROMOTE_FOR_CONFIRMATION",
        ),
        (107.0, "inconclusive", None, "INCONCLUSIVE"),
        (111.0, "no_promotion", "RS0_pooled_source", "STOP_FEDERATED_REGRESSION"),
    ],
)
def test_decision_gate_uses_calibration_thresholds_only(
    rs3_rmse, status, candidate, recommendation
) -> None:
    metrics, per_gas = _gate_inputs(rs3_rmse)
    gate = freeze_decision_gate(
        metrics, per_gas, formal_commit="a" * 40
    )
    assert gate["selection_status"] == status
    assert gate["selected_candidate"] == candidate
    assert gate["next_stage_recommendation"] == recommendation
    assert gate["test_opened_after_selection"] is False
    assert gate["final_test_evaluation_timestamp"] is None
    assert gate["test_metrics_used_for_selection"] is False


def test_regression_aggregation_scope_excludes_backbone() -> None:
    contract = validate_state_contract(TinyTopologyModel())
    assert contract["parameter_count"] == 9
    assert all(not key.startswith("backbone.") for key in contract["state_keys"])


def test_local_models_start_from_identical_regression_state() -> None:
    template = TinyTopologyModel()
    copies = {1: TinyTopologyModel(), 2: TinyTopologyModel()}
    copies[1].load_state_dict(template.state_dict())
    copies[2].load_state_dict(template.state_dict())
    expected = assert_identical_initialization(template, copies)
    assert expected == state_sha256(
        template.state_dict(),
        [key for key in template.state_dict() if key.startswith(("reg_proj.", "reg_heads."))],
    )
    with torch.no_grad():
        copies[2].reg_heads[0].bias.add_(1.0)
    with pytest.raises(RuntimeError, match="initialization mismatch"):
        assert_identical_initialization(template, copies)


def test_fedavg_before_after_parameters_are_traceable() -> None:
    model = TinyTopologyModel()
    keys = validate_state_contract(model)["state_keys"]
    initial = {key: value.detach().clone() for key, value in model.state_dict().items()}
    client1 = {key: value.detach().clone() for key, value in initial.items()}
    client2 = {key: value.detach().clone() for key, value in initial.items()}
    with torch.no_grad():
        client1["reg_proj.bias"].add_(1.0)
        client2["reg_proj.bias"].add_(3.0)
    averaged = fedavg_regression_states(
        {1: client1, 2: client2}, {1: 1, 2: 3}, keys, torch.device("cpu")
    )
    assert torch.allclose(
        averaged["reg_proj.bias"], initial["reg_proj.bias"] + 2.5
    )
    assert state_sha256(initial, keys) != state_sha256(averaged, keys)


def test_output_contract_refuses_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "formal"
    require_new_empty_output(output)
    (output / "existing.txt").write_text("evidence", encoding="utf-8")
    with pytest.raises(FileExistsError, match="overwrite"):
        require_new_empty_output(output)
