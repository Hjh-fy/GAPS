from __future__ import annotations

import copy
import inspect
from collections import OrderedDict
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class TinyClassifier(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.classifier = torch.nn.Linear(1, 2, bias=False)
        with torch.no_grad():
            self.classifier.weight.copy_(torch.tensor([[0.2], [-0.1]]))

    def forward(self, x: torch.Tensor):
        logits = self.classifier(x)
        return logits, x, x


def _loader() -> DataLoader:
    x = torch.tensor([[1.0]], dtype=torch.float32)
    y = torch.tensor([0], dtype=torch.long)
    return DataLoader(TensorDataset(x, y), batch_size=1, shuffle=False)


def _zeros(model: torch.nn.Module) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict(
        (name, torch.zeros_like(parameter.detach()))
        for name, parameter in model.named_parameters()
    )


def test_scaffold_uses_sgd() -> None:
    from gaps_flower.scaffold import scaffold_train_one_round

    model = TinyClassifier()
    result = scaffold_train_one_round(
        model,
        _loader(),
        server_control=_zeros(model),
        client_control=_zeros(model),
        lr=5e-4,
        local_epochs=1,
        device=torch.device("cpu"),
    )

    assert result.optimizer_name == "SGD"
    assert result.optimizer_lr == pytest.approx(5e-4)
    assert result.optimizer_momentum == 0.0


def test_scaffold_gradient_contains_control_variate_correction() -> None:
    from gaps_flower.scaffold import scaffold_train_one_round

    model = TinyClassifier()
    reference = copy.deepcopy(model)
    x = torch.tensor([[1.0]], dtype=torch.float32)
    y = torch.tensor([0], dtype=torch.long)
    logits, _, _ = reference(x)
    F.cross_entropy(logits, y).backward()
    raw_grad = reference.classifier.weight.grad.detach().clone()
    server_control = OrderedDict(
        [("classifier.weight", torch.tensor([[0.3], [-0.2]], dtype=torch.float32))]
    )
    client_control = OrderedDict(
        [("classifier.weight", torch.tensor([[0.1], [0.05]], dtype=torch.float32))]
    )
    expected = (
        reference.classifier.weight.detach()
        - 0.1 * (raw_grad + server_control["classifier.weight"] - client_control["classifier.weight"])
    )

    result = scaffold_train_one_round(
        model,
        _loader(),
        server_control=server_control,
        client_control=client_control,
        lr=0.1,
        local_epochs=1,
        device=torch.device("cpu"),
    )

    assert torch.allclose(result.model_state["classifier.weight"], expected, atol=1e-7)
    assert result.control_correction_applied_steps == 1


def test_scaffold_treats_unused_parameter_data_gradient_as_zero() -> None:
    from gaps_flower.scaffold import scaffold_train_one_round

    class BranchedClassifier(TinyClassifier):
        def __init__(self) -> None:
            super().__init__()
            self.unused = torch.nn.Parameter(torch.tensor([0.4]))

    model = BranchedClassifier()
    server_control = _zeros(model)
    client_control = _zeros(model)
    server_control["unused"] = torch.tensor([0.3])
    client_control["unused"] = torch.tensor([0.1])

    result = scaffold_train_one_round(
        model,
        _loader(),
        server_control=server_control,
        client_control=client_control,
        lr=0.1,
        local_epochs=1,
        device=torch.device("cpu"),
    )

    assert result.model_state["unused"] == pytest.approx(torch.tensor([0.38]))


def test_scaffold_client_control_variate_persists() -> None:
    from gaps_flower.scaffold import ScaffoldClientControlState

    model = TinyClassifier()
    state = ScaffoldClientControlState.from_model(model)
    first = state.train(
        model,
        _loader(),
        server_control=_zeros(model),
        lr=0.1,
        local_epochs=1,
        device=torch.device("cpu"),
    )
    persisted = first.client_control_after_fingerprint
    second_model = TinyClassifier()
    second = state.train(
        second_model,
        _loader(),
        server_control=_zeros(second_model),
        lr=0.1,
        local_epochs=1,
        device=torch.device("cpu"),
    )

    assert state.rounds_completed == 2
    assert second.client_control_before_fingerprint == persisted
    assert second.client_control_before_fingerprint != second.zero_control_fingerprint


def test_scaffold_server_control_variate_updates() -> None:
    from gaps_flower.scaffold import ScaffoldServerControlState

    model = TinyClassifier()
    state = ScaffoldServerControlState.from_model(model, total_clients=2)
    delta_1 = OrderedDict(
        [("classifier.weight", torch.tensor([[0.2], [0.4]], dtype=torch.float32))]
    )
    delta_2 = OrderedDict(
        [("classifier.weight", torch.tensor([[0.6], [-0.2]], dtype=torch.float32))]
    )

    state.update([delta_1, delta_2])

    assert torch.allclose(
        state.control["classifier.weight"],
        torch.tensor([[0.4], [0.1]], dtype=torch.float32),
    )
    assert state.rounds_completed == 1


def test_scaffold_no_adam_state_present() -> None:
    from gaps_flower.scaffold import scaffold_train_one_round

    model = TinyClassifier()
    result = scaffold_train_one_round(
        model,
        _loader(),
        server_control=_zeros(model),
        client_control=_zeros(model),
        lr=5e-4,
        local_epochs=1,
        device=torch.device("cpu"),
    )

    assert result.optimizer_state_entries == 0
    assert result.adam_state_present is False
    assert "exp_avg" not in result.optimizer_state_field_names
    assert "exp_avg_sq" not in result.optimizer_state_field_names


def test_scaffold_control_transport_round_trip() -> None:
    from gaps_flower.scaffold import pack_control_variates, unpack_control_variates

    model = TinyClassifier()
    control = OrderedDict(
        [("classifier.weight", torch.tensor([[0.3], [-0.2]], dtype=torch.float32))]
    )
    restored = unpack_control_variates(
        pack_control_variates(control), list(control), _zeros(model)
    )

    assert list(restored) == list(control)
    assert torch.equal(restored["classifier.weight"], control["classifier.weight"])


def test_flower_scaffold_contract_is_explicit(tmp_path: Path) -> None:
    from flwr.common import ndarrays_to_parameters

    from gaps_flower.client_app import GapsFlowerClient
    from gaps_flower.server_app import DEFAULT_STRATEGIES
    from gaps_flower.strategy import ScaffoldStrategy
    from gaps_flower.task import get_parameters

    model = TinyClassifier()
    arrays, keys = get_parameters(model)
    strategy = ScaffoldStrategy(
        model_template=model,
        total_clients=2,
        parameter_keys=keys,
        reference_state=model.state_dict(),
        output_dir=str(tmp_path),
        initial_parameters=ndarrays_to_parameters(arrays),
        min_fit_clients=2,
        min_available_clients=2,
    )

    assert "scaffold" in DEFAULT_STRATEGIES
    assert "optimizer" in inspect.signature(GapsFlowerClient.__init__).parameters
    payload = strategy.scaffold_fit_config(1)
    assert payload["optimizer"] == "scaffold_sgd"
    assert payload["scaffold_lr"] == pytest.approx(5e-4)
    assert isinstance(payload["scaffold_server_control"], bytes)


def _source_diag(
    ce=(1.6, 1.4, 1.0, 0.7),
    grad=(1.0, 0.8),
    parameter=(4.0, 4.1),
):
    return {
        "ce_trajectory": list(ce),
        "grad_norms": list(grad),
        "parameter_norms": list(parameter),
    }


def test_scaffold_source_gate_passes_finite_source_optimization() -> None:
    from gaps_flower.source_numerical_gate import evaluate_source_gate

    verdict = evaluate_source_gate(
        [_source_diag(), _source_diag(ce=(1.5, 1.3, 0.9, 0.6))],
        source_accuracy=0.55,
        source_class_counts={0: 25, 1: 25, 2: 25, 3: 25},
    )

    assert verdict.passed is True
    assert all(verdict.checks.values())
    assert verdict.lr_search_performed is False
    assert verdict.target_information_accessed is False


@pytest.mark.parametrize(
    ("diagnostics", "accuracy", "counts", "failed_check"),
    [
        ([_source_diag(ce=(1.0, float("nan"), 0.8, 0.7))], 0.6, {0: 1, 1: 1}, "all_finite"),
        ([_source_diag(ce=(0.5, 0.6, 0.7, 0.8))], 0.6, {0: 1, 1: 1}, "ce_decreased"),
        ([_source_diag(grad=(0.0, 0.0))], 0.6, {0: 1, 1: 1}, "gradient_norm_valid"),
        ([_source_diag(parameter=(4.0, 1e4))], 0.6, {0: 1, 1: 1}, "parameter_norm_valid"),
        ([_source_diag()], 0.5, {0: 50, 1: 50}, "source_discrimination"),
    ],
)
def test_scaffold_source_gate_fails_closed(
    diagnostics, accuracy, counts, failed_check
) -> None:
    from gaps_flower.source_numerical_gate import evaluate_source_gate

    verdict = evaluate_source_gate(
        diagnostics,
        source_accuracy=accuracy,
        source_class_counts=counts,
    )

    assert verdict.passed is False
    assert verdict.checks[failed_check] is False
    assert verdict.action == "FAIL_CLOSED_NO_LR_SEARCH"


def test_scaffold_source_gate_has_no_target_input() -> None:
    from gaps_flower.source_numerical_gate import evaluate_source_gate

    parameter_names = set(inspect.signature(evaluate_source_gate).parameters)
    assert not any("target" in name for name in parameter_names)
