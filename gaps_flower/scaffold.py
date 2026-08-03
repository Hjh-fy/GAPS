"""Canonical SGD-style SCAFFOLD control-variate primitives."""

from __future__ import annotations

import io
from collections import OrderedDict
from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np
import torch
import torch.nn.functional as F

from gaps_flower.state_fingerprint import ordered_state_content_fingerprint


TensorMap = OrderedDict[str, torch.Tensor]


def _trainable_state(model: torch.nn.Module) -> TensorMap:
    return OrderedDict(
        (name, parameter.detach().clone()) for name, parameter in model.named_parameters()
    )


def _zero_control(model: torch.nn.Module) -> TensorMap:
    return OrderedDict(
        (name, torch.zeros_like(parameter.detach()))
        for name, parameter in model.named_parameters()
    )


def _clone_control(control: Mapping[str, torch.Tensor]) -> TensorMap:
    return OrderedDict((key, value.detach().clone()) for key, value in control.items())


def _validate_control(
    control: Mapping[str, torch.Tensor], reference: Mapping[str, torch.Tensor], label: str
) -> None:
    if list(control) != list(reference):
        raise RuntimeError(f"FAIL_CLOSED {label} control keys/order mismatch")
    for key, ref in reference.items():
        value = control[key]
        if value.shape != ref.shape or value.dtype != ref.dtype:
            raise RuntimeError(f"FAIL_CLOSED {label} control tensor mismatch for {key}")
        if not torch.isfinite(value).all():
            raise RuntimeError(f"FAIL_CLOSED {label} control contains non-finite values")


def control_fingerprint(control: Mapping[str, torch.Tensor]) -> str:
    return ordered_state_content_fingerprint(control)


def pack_control_variates(control: Mapping[str, torch.Tensor]) -> bytes:
    buffer = io.BytesIO()
    arrays = {f"v{index:04d}": value.detach().cpu().numpy() for index, value in enumerate(control.values())}
    arrays["keys"] = np.asarray(list(control), dtype=np.str_)
    np.savez(buffer, **arrays)
    return buffer.getvalue()


def unpack_control_variates(
    payload: bytes,
    expected_keys: Iterable[str],
    reference: Mapping[str, torch.Tensor],
) -> TensorMap:
    keys = list(expected_keys)
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        stored_keys = [str(value) for value in archive["keys"].tolist()]
        if stored_keys != keys:
            raise RuntimeError("FAIL_CLOSED packed control keys/order mismatch")
        control = OrderedDict(
            (
                key,
                torch.from_numpy(np.asarray(archive[f"v{index:04d}"])).to(
                    dtype=reference[key].dtype
                ),
            )
            for index, key in enumerate(keys)
        )
    _validate_control(control, reference, "packed")
    return control


@dataclass
class ScaffoldLocalResult:
    model_state: TensorMap
    model_delta: TensorMap
    client_control: TensorMap
    control_delta: TensorMap
    steps: int
    ce_trajectory: list[float]
    train_accuracy: float
    grad_norms: list[float]
    parameter_norms: list[float]
    optimizer_name: str
    optimizer_lr: float
    optimizer_momentum: float
    optimizer_state_entries: int
    optimizer_state_field_names: tuple[str, ...]
    adam_state_present: bool
    control_correction_applied_steps: int
    client_control_before_fingerprint: str
    client_control_after_fingerprint: str
    zero_control_fingerprint: str


def scaffold_train_one_round(
    model: torch.nn.Module,
    train_loader,
    *,
    server_control: Mapping[str, torch.Tensor],
    client_control: Mapping[str, torch.Tensor] | None,
    lr: float,
    local_epochs: int,
    device: torch.device,
) -> ScaffoldLocalResult:
    """Run canonical local SGD updates ``grad + c - c_i`` and Option-II c_i update."""
    if lr <= 0 or local_epochs <= 0:
        raise ValueError("SCAFFOLD lr and local_epochs must be positive")
    model.to(device)
    model.train()
    global_params = _trainable_state(model)
    zeros = _zero_control(model)
    client_before = _clone_control(client_control or zeros)
    server = _clone_control(server_control)
    _validate_control(server, global_params, "server")
    _validate_control(client_before, global_params, "client")

    optimizer = torch.optim.SGD(model.parameters(), lr=float(lr), momentum=0.0)
    ce_trajectory: list[float] = []
    grad_norms: list[float] = []
    parameter_norms: list[float] = []
    correct = 0
    examples = 0
    steps = 0

    for _ in range(local_epochs):
        for batch in train_loader:
            x, y_cls = batch[0].to(device), batch[1].to(device).long()
            optimizer.zero_grad(set_to_none=True)
            output = model(x)
            logits = output[0] if isinstance(output, (tuple, list)) else output
            loss = F.cross_entropy(logits, y_cls)
            if not torch.isfinite(loss):
                raise RuntimeError("FAIL_CLOSED SCAFFOLD non-finite CE")
            loss.backward()
            squared_grad = torch.zeros((), device=device)
            for name, parameter in model.named_parameters():
                if parameter.grad is None:
                    raise RuntimeError(f"FAIL_CLOSED SCAFFOLD missing gradient for {name}")
                correction = server[name].to(device) - client_before[name].to(device)
                parameter.grad.add_(correction)
                if not torch.isfinite(parameter.grad).all():
                    raise RuntimeError("FAIL_CLOSED SCAFFOLD non-finite corrected gradient")
                squared_grad += parameter.grad.detach().pow(2).sum()
            grad_norms.append(float(squared_grad.sqrt().item()))
            optimizer.step()
            squared_param = torch.zeros((), device=device)
            for parameter in model.parameters():
                if not torch.isfinite(parameter).all():
                    raise RuntimeError("FAIL_CLOSED SCAFFOLD non-finite parameter")
                squared_param += parameter.detach().pow(2).sum()
            parameter_norms.append(float(squared_param.sqrt().item()))
            ce_trajectory.append(float(loss.detach().item()))
            correct += int((logits.detach().argmax(dim=1) == y_cls).sum().item())
            examples += int(y_cls.numel())
            steps += 1

    if steps == 0:
        raise RuntimeError("FAIL_CLOSED SCAFFOLD local loader produced zero steps")
    local_params = _trainable_state(model)
    client_after = OrderedDict(
        (
            name,
            client_before[name]
            - server[name]
            + (global_params[name] - local_params[name].cpu()) / (steps * float(lr)),
        )
        for name in global_params
    )
    control_delta = OrderedDict(
        (name, client_after[name] - client_before[name]) for name in global_params
    )
    model_delta = OrderedDict(
        (name, local_params[name].cpu() - global_params[name]) for name in global_params
    )
    state_fields = tuple(
        sorted(
            {
                field
                for value in optimizer.state_dict().get("state", {}).values()
                for field in value
            }
        )
    )
    adam_present = any(field in {"exp_avg", "exp_avg_sq", "max_exp_avg_sq"} for field in state_fields)
    return ScaffoldLocalResult(
        model_state=OrderedDict(
            (name, value.detach().cpu().clone()) for name, value in model.state_dict().items()
        ),
        model_delta=model_delta,
        client_control=client_after,
        control_delta=control_delta,
        steps=steps,
        ce_trajectory=ce_trajectory,
        train_accuracy=correct / max(examples, 1),
        grad_norms=grad_norms,
        parameter_norms=parameter_norms,
        optimizer_name="SGD",
        optimizer_lr=float(lr),
        optimizer_momentum=0.0,
        optimizer_state_entries=len(optimizer.state_dict().get("state", {})),
        optimizer_state_field_names=state_fields,
        adam_state_present=adam_present,
        control_correction_applied_steps=steps,
        client_control_before_fingerprint=control_fingerprint(client_before),
        client_control_after_fingerprint=control_fingerprint(client_after),
        zero_control_fingerprint=control_fingerprint(zeros),
    )


@dataclass
class ScaffoldClientControlState:
    control: TensorMap
    rounds_completed: int = 0

    @classmethod
    def from_model(cls, model: torch.nn.Module) -> "ScaffoldClientControlState":
        return cls(control=_zero_control(model))

    def train(self, model: torch.nn.Module, train_loader, **kwargs) -> ScaffoldLocalResult:
        result = scaffold_train_one_round(
            model, train_loader, client_control=self.control, **kwargs
        )
        self.control = _clone_control(result.client_control)
        self.rounds_completed += 1
        return result


@dataclass
class ScaffoldServerControlState:
    control: TensorMap
    total_clients: int
    rounds_completed: int = 0

    @classmethod
    def from_model(
        cls, model: torch.nn.Module, total_clients: int
    ) -> "ScaffoldServerControlState":
        if total_clients <= 0:
            raise ValueError("total_clients must be positive")
        return cls(control=_zero_control(model), total_clients=total_clients)

    def update(self, deltas: Iterable[Mapping[str, torch.Tensor]]) -> None:
        values = list(deltas)
        if len(values) != self.total_clients:
            raise RuntimeError("FAIL_CLOSED missing SCAFFOLD client control delta")
        for delta in values:
            _validate_control(delta, self.control, "client delta")
        self.control = OrderedDict(
            (
                name,
                self.control[name]
                + sum(delta[name] for delta in values) / float(self.total_clients),
            )
            for name in self.control
        )
        self.rounds_completed += 1
