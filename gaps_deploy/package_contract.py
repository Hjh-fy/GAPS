"""Strict deployment package and checkpoint validation helpers."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping

import torch
import torch.nn as nn


class DeploymentPackageError(ValueError):
    """Raised when a production deployment asset is missing or incompatible."""


ALLOWED_ROUTING_MODES = frozenset({
    "none",
    "bias_only",
    "affine_only",
    "phase_affine_only",
    "full",
    "specialist",
    "specialist_full",
})

REQUIRED_MODEL_CONFIG_FIELDS = frozenset({
    "num_classes",
    "num_sensors",
    "feat_dim",
    "encoder_type",
    "transformer_d_model",
    "transformer_nhead",
    "transformer_num_layers",
    "transformer_ff_dim",
    "reg_head_depth",
    "reg_output_mode",
    "reg_window_stats",
    "reg_window_stats_mode",
    "reg_window_stats_dim",
    "reg_response_branch",
    "reg_tcn_adapter",
    "reg_use_shared_trunk",
    "use_reg_ratio_branch",
})


def require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise DeploymentPackageError(f"{label} is missing: {path}")
    return path


def load_json_object(path: Path, label: str) -> Dict[str, Any]:
    require_file(path, label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DeploymentPackageError(f"Cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise DeploymentPackageError(f"{label} must contain a JSON object: {path}")
    return value


def extract_state_dict(checkpoint: Any, path: Path) -> Dict[str, torch.Tensor]:
    state = checkpoint
    if isinstance(checkpoint, dict):
        if "model_state" in checkpoint:
            state = checkpoint["model_state"]
        elif "state_dict" in checkpoint:
            state = checkpoint["state_dict"]
    if not isinstance(state, Mapping) or not state:
        raise DeploymentPackageError(f"Checkpoint has no non-empty state dict: {path}")
    output: Dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if not isinstance(key, str) or not isinstance(value, torch.Tensor):
            raise DeploymentPackageError(f"Checkpoint state is malformed: {path}")
        output[key] = value
    return output


def load_checkpoint_state(path: Path) -> tuple[Any, Dict[str, torch.Tensor]]:
    require_file(path, "checkpoint")
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise DeploymentPackageError(f"Cannot load checkpoint {path}: {exc}") from exc
    return checkpoint, extract_state_dict(checkpoint, path)


def load_state_dict_strict(
    model: nn.Module,
    state: Mapping[str, torch.Tensor],
    path: Path,
) -> None:
    expected = model.state_dict()
    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))
    bad_shapes = sorted(
        key
        for key in set(expected) & set(state)
        if tuple(expected[key].shape) != tuple(state[key].shape)
    )
    if missing or unexpected or bad_shapes:
        raise DeploymentPackageError(
            f"Checkpoint mismatch {path}: missing={missing}, "
            f"unexpected={unexpected}, shape_mismatch={bad_shapes}"
        )
    try:
        model.load_state_dict(dict(state), strict=True)
    except RuntimeError as exc:
        raise DeploymentPackageError(f"Cannot load checkpoint state {path}: {exc}") from exc


def _normalize_class_mapping(raw: Any, label: str) -> Dict[int, Any]:
    if not isinstance(raw, Mapping):
        raise DeploymentPackageError(f"routing_config {label} must be a mapping")
    normalized: Dict[int, Any] = {}
    for raw_key, value in raw.items():
        if isinstance(raw_key, bool):
            raise DeploymentPackageError(f"routing_config {label} has invalid class key")
        try:
            class_id = int(raw_key)
        except (TypeError, ValueError) as exc:
            raise DeploymentPackageError(
                f"routing_config {label} has invalid class key: {raw_key!r}"
            ) from exc
        if class_id in normalized:
            raise DeploymentPackageError(
                f"routing_config {label} has colliding class key: {raw_key!r}"
            )
        normalized[class_id] = value
    return normalized


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise DeploymentPackageError(f"{label} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise DeploymentPackageError(f"{label} must be numeric") from exc
    if not math.isfinite(number):
        raise DeploymentPackageError(f"{label} must be finite")
    return number


def _validate_affine_params(
    raw: Any,
    label: str,
    *,
    selected_mode: str | None = None,
) -> Dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise DeploymentPackageError(f"{label} must be a mapping")
    if "a" not in raw or "b" not in raw:
        raise DeploymentPackageError(f"{label} must contain finite a and b params")
    normalized = dict(raw)
    normalized["a"] = _finite_number(raw["a"], f"{label}.a")
    normalized["b"] = _finite_number(raw["b"], f"{label}.b")
    if selected_mode is not None:
        raw_mode = str(raw.get("mode", selected_mode)).strip().lower()
        if raw_mode != selected_mode:
            raise DeploymentPackageError(
                f"{label}.mode {raw_mode!r} disagrees with selected mode {selected_mode!r}"
            )
        normalized["mode"] = selected_mode
    for metric in ("calib_r2", "calib_mae"):
        if metric in raw:
            normalized[metric] = _finite_number(raw[metric], f"{label}.{metric}")
    return normalized


def _validate_phase_affine_params(
    raw: Any,
    label: str,
    num_phases: int,
) -> Dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise DeploymentPackageError(f"{label} must be a mapping")
    raw_num_phases = raw.get("num_phases")
    if (
        isinstance(raw_num_phases, bool)
        or not isinstance(raw_num_phases, int)
        or raw_num_phases != num_phases
    ):
        raise DeploymentPackageError(
            f"{label}.num_phases must equal {num_phases}"
        )
    phases = _normalize_class_mapping(
        raw.get("phase_calibrators"), f"{label}.phase_calibrators"
    )
    expected = set(range(num_phases))
    if set(phases) != expected:
        raise DeploymentPackageError(
            f"{label}.phase_calibrators must cover exactly phases "
            f"{sorted(expected)}, got {sorted(phases)}"
        )
    normalized = dict(raw)
    normalized["num_phases"] = num_phases
    normalized["phase_calibrators"] = {
        phase: _validate_affine_params(
            params,
            f"{label}.phase_calibrators[{phase}]",
        )
        for phase, params in phases.items()
    }
    return normalized


def normalize_and_validate_routing_config(
    raw: Mapping[str, Any],
    num_classes: int,
    num_phases: int = 3,
) -> Dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise DeploymentPackageError("routing_config must be a mapping")
    selected_raw = _normalize_class_mapping(raw.get("selected_modes"), "selected_modes")
    expected = set(range(int(num_classes)))
    if set(selected_raw) != expected:
        raise DeploymentPackageError(
            "routing_config selected_modes must cover exactly classes "
            f"{sorted(expected)}, got {sorted(selected_raw)}"
        )
    selected: Dict[int, str] = {}
    for class_id, raw_mode in selected_raw.items():
        mode = str(raw_mode).strip().lower()
        if mode not in ALLOWED_ROUTING_MODES:
            raise DeploymentPackageError(
                f"routing_config selected_modes class {class_id} has unknown mode {raw_mode!r}"
            )
        selected[class_id] = mode

    affine = _normalize_class_mapping(raw.get("affine_params", {}), "affine_params")
    phase_affine = _normalize_class_mapping(
        raw.get("phase_affine_params", {}), "phase_affine_params"
    )
    for class_id, mode in selected.items():
        if mode in {"bias_only", "affine_only"} and class_id not in affine:
            raise DeploymentPackageError(
                f"routing_config selected mode {mode} lacks affine_params for class {class_id}"
            )
        if mode == "phase_affine_only" and class_id not in phase_affine:
            raise DeploymentPackageError(
                "routing_config selected mode phase_affine_only lacks "
                f"phase_affine_params for class {class_id}"
            )

    unknown_affine_classes = sorted(set(affine) - expected)
    unknown_phase_classes = sorted(set(phase_affine) - expected)
    if unknown_affine_classes or unknown_phase_classes:
        raise DeploymentPackageError(
            "routing_config params contain out-of-range classes: "
            f"affine={unknown_affine_classes}, phase_affine={unknown_phase_classes}"
        )
    normalized_affine = {
        class_id: _validate_affine_params(
            params,
            f"routing_config affine_params[{class_id}]",
            selected_mode=(
                selected[class_id]
                if selected[class_id] in {"bias_only", "affine_only"}
                else None
            ),
        )
        for class_id, params in affine.items()
    }
    normalized_phase_affine = {
        class_id: _validate_phase_affine_params(
            params,
            f"routing_config phase_affine_params[{class_id}]",
            num_phases,
        )
        for class_id, params in phase_affine.items()
    }

    normalized = dict(raw)
    normalized["selected_modes"] = selected
    normalized["affine_params"] = normalized_affine
    normalized["phase_affine_params"] = normalized_phase_affine
    return normalized


def validate_model_config(model_config: Mapping[str, Any]) -> None:
    if not isinstance(model_config, Mapping):
        raise DeploymentPackageError("model_config must be a mapping")
    missing = sorted(REQUIRED_MODEL_CONFIG_FIELDS - set(model_config))
    if missing:
        raise DeploymentPackageError(f"model_config missing required fields: {missing}")
    for name in (
        "num_classes",
        "num_sensors",
        "feat_dim",
        "transformer_d_model",
        "transformer_nhead",
        "transformer_num_layers",
        "transformer_ff_dim",
        "reg_head_depth",
        "reg_window_stats_dim",
    ):
        value = model_config[name]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise DeploymentPackageError(
                f"model_config {name} must be a positive integer"
            )
    if str(model_config["encoder_type"]).lower() not in {"tcn", "transformer"}:
        raise DeploymentPackageError("model_config encoder_type is invalid")
    if str(model_config["reg_output_mode"]).lower() not in {"sigmoid", "linear"}:
        raise DeploymentPackageError("model_config reg_output_mode is invalid")
    if str(model_config["reg_window_stats_mode"]).lower() not in {
        "global",
        "per_channel",
    }:
        raise DeploymentPackageError("model_config reg_window_stats_mode is invalid")
    if str(model_config["reg_response_branch"]).lower() not in {
        "none",
        "dct",
        "msconv",
    }:
        raise DeploymentPackageError("model_config reg_response_branch is invalid")
    for name in (
        "reg_window_stats",
        "reg_tcn_adapter",
        "reg_use_shared_trunk",
        "use_reg_ratio_branch",
    ):
        if not isinstance(model_config[name], bool):
            raise DeploymentPackageError(f"model_config {name} must be Boolean")


def validate_checkpoint_model_config(
    checkpoint: Any,
    model_config: Mapping[str, Any],
    path: Path,
) -> None:
    if not isinstance(checkpoint, Mapping):
        return
    embedded = checkpoint.get("model_config")
    if embedded is None:
        return
    if not isinstance(embedded, Mapping):
        raise DeploymentPackageError(f"Checkpoint model_config is malformed: {path}")
    disagreements = sorted(
        key
        for key, value in embedded.items()
        if key in model_config and model_config[key] != value
    )
    if disagreements:
        details = {
            key: {"package": model_config[key], "checkpoint": embedded[key]}
            for key in disagreements
        }
        raise DeploymentPackageError(
            f"Checkpoint model_config disagrees with package {path}: {details}"
        )
