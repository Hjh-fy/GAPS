"""Observed loss-activity accounting for hierarchical ablations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence


@dataclass
class _LossState:
    configured_weight: float
    input_available: bool
    active_steps: int = 0
    raw_sum: float = 0.0
    weighted_sum: float = 0.0
    inactive_reasons: set[str] = field(default_factory=set)


class LossActivityAccumulator:
    def __init__(self, *, scope: str, variant: str):
        self.scope = str(scope)
        self.variant = str(variant)
        self._states: dict[str, _LossState] = {}

    def record(
        self,
        *,
        loss_name: str,
        configured_weight: float,
        input_available: bool,
        raw_loss: float | None,
        active: bool,
        inactive_reason: str,
    ) -> None:
        name = str(loss_name)
        weight = float(configured_weight)
        if name not in self._states:
            self._states[name] = _LossState(weight, bool(input_available))
        state = self._states[name]
        if state.configured_weight != weight:
            raise RuntimeError(f"FAIL_CLOSED configured weight changed for {name}")
        if state.input_available != bool(input_available):
            raise RuntimeError(f"FAIL_CLOSED input availability changed for {name}")
        if active:
            if not input_available:
                raise RuntimeError(f"FAIL_CLOSED active loss lacks input: {name}")
            if weight == 0.0:
                raise RuntimeError(f"FAIL_CLOSED zero-weight loss marked active: {name}")
            if raw_loss is None:
                raise RuntimeError(f"FAIL_CLOSED active loss lacks raw value: {name}")
            raw = float(raw_loss)
            state.active_steps += 1
            state.raw_sum += raw
            state.weighted_sum += weight * raw
        else:
            reason = str(inactive_reason).strip()
            if not reason:
                raise RuntimeError(f"FAIL_CLOSED inactive loss lacks reason: {name}")
            state.inactive_reasons.add(reason)

    def rows(self) -> list[dict]:
        rows = []
        for name, state in sorted(self._states.items()):
            denominator = max(state.active_steps, 1)
            rows.append(
                {
                    "variant": self.variant,
                    "scope": self.scope,
                    "loss_name": name,
                    "configured_weight": state.configured_weight,
                    "input_available": state.input_available,
                    "active_steps": state.active_steps,
                    "mean_raw_loss": (
                        state.raw_sum / denominator if state.active_steps else 0.0
                    ),
                    "mean_weighted_loss": (
                        state.weighted_sum / denominator if state.active_steps else 0.0
                    ),
                    "inactive_reason": (
                        "" if state.active_steps else ";".join(sorted(state.inactive_reasons))
                    ),
                }
            )
        return rows


_SERVER_TERMS = {
    "source_ce": "val_loss",
    "coral": "coral_loss",
    "global_mmd": "mmd_global",
    "class_mmd": "mmd_class",
    "adversarial": "adv_loss",
    "proto_anchor": "proto_anchor",
    "proto_loss": "proto_loss",
    "consistency": "consist_loss",
    "device_residual": "residual_loss",
    "proto_mmd": "mmd_proto_loss",
    "stage_mmd": "stage_mmd_loss",
    "align_reg_legacy": "align_reg_legacy_loss",
    "target_ce": "target_ce_loss",
}


def _availability(loss_name: str, flags: Mapping[str, bool]) -> tuple[bool, str]:
    requirements = {
        "source_ce": (("source_x_class_phase",), "source_input_unavailable"),
        "coral": (("target_x",), "target_x_unavailable"),
        "global_mmd": (("target_x",), "target_x_unavailable"),
        "class_mmd": (("target_x", "target_class"), "target_class_unavailable"),
        "adversarial": (("target_x", "target_class", "domain_discriminator"), "adversarial_input_unavailable"),
        "proto_anchor": (("target_class", "semantic_prototypes"), "prototype_anchor_input_unavailable"),
        "proto_loss": (("semantic_prototypes", "client_prototypes"), "client_prototypes_unavailable"),
        "consistency": (("source_x_class_phase", "semantic_prototypes"), "semantic_prototypes_unavailable"),
        "device_residual": (("client_residuals",), "client_residual_unavailable"),
        "proto_mmd": (("two_client_prototypes",), "two_client_prototypes_unavailable"),
        "stage_mmd": (("target_x", "target_class", "target_phase"), "target_phase_unavailable"),
        "align_reg_legacy": (("align_reg_legacy_enabled", "semantic_prototypes", "client_prototypes"), "align_reg_legacy_disabled"),
        "target_ce": (("target_class",), "target_class_unavailable"),
    }
    needed, reason = requirements[loss_name]
    available = all(bool(flags.get(key, False)) for key in needed)
    return available, "" if available else reason


def server_da_activity_rows(
    *,
    variant: str,
    step_diagnostics: Mapping[str, Sequence[float]],
    configured_weights: Mapping[str, float],
    availability: Mapping[str, bool],
) -> list[dict]:
    activity = LossActivityAccumulator(scope="server_da", variant=variant)
    for loss_name, diagnostic_key in _SERVER_TERMS.items():
        values = [float(value) for value in step_diagnostics.get(diagnostic_key, ())]
        weight = float(configured_weights.get(loss_name, 0.0))
        input_available, missing_reason = _availability(loss_name, availability)
        active = weight != 0.0 and input_available and bool(values)
        if active:
            for value in values:
                activity.record(
                    loss_name=loss_name,
                    configured_weight=weight,
                    input_available=True,
                    raw_loss=value,
                    active=True,
                    inactive_reason="",
                )
        else:
            reason = (
                "configured_weight_zero"
                if weight == 0.0
                else missing_reason or "no_observed_steps"
            )
            activity.record(
                loss_name=loss_name,
                configured_weight=weight,
                input_available=input_available,
                raw_loss=None,
                active=False,
                inactive_reason=reason,
            )
    return activity.rows()
