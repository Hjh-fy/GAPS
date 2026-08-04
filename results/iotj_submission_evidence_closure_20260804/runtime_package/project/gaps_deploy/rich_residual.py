"""Runtime support for rich residual deployment candidates.

The policy is intentionally a post-processing layer. It never changes the base
classifier/regressor outputs; it only produces a corrected ppm value that can be
reported alongside the frozen ``final_ppm``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np

from .inference import DeployResult


CO_CLASS = 1


def _fnum(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _channel_summary_features(window: np.ndarray) -> dict[str, float]:
    arr = np.asarray(window, dtype=np.float64)
    names: dict[str, float] = {}
    first = arr[0]
    last = arr[-1]
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    amin = arr.min(axis=0)
    amax = arr.max(axis=0)
    p10 = np.percentile(arr, 10, axis=0)
    p50 = np.percentile(arr, 50, axis=0)
    p90 = np.percentile(arr, 90, axis=0)
    argmin = arr.argmin(axis=0) / max(1, arr.shape[0] - 1)
    argmax = arr.argmax(axis=0) / max(1, arr.shape[0] - 1)
    for ch in range(arr.shape[1]):
        prefix = f"ch{ch}"
        names[f"{prefix}_first"] = float(first[ch])
        names[f"{prefix}_last"] = float(last[ch])
        names[f"{prefix}_mean"] = float(mean[ch])
        names[f"{prefix}_std"] = float(std[ch])
        names[f"{prefix}_min"] = float(amin[ch])
        names[f"{prefix}_max"] = float(amax[ch])
        names[f"{prefix}_range"] = float(amax[ch] - amin[ch])
        names[f"{prefix}_slope"] = float(last[ch] - first[ch])
        names[f"{prefix}_p10"] = float(p10[ch])
        names[f"{prefix}_p50"] = float(p50[ch])
        names[f"{prefix}_p90"] = float(p90[ch])
        names[f"{prefix}_argmin"] = float(argmin[ch])
        names[f"{prefix}_argmax"] = float(argmax[ch])
    return names


def _window_feature_stats(window: np.ndarray) -> dict[str, float]:
    arr = np.asarray(window, dtype=np.float64)
    channel_amp = arr.max(axis=0) - arr.min(axis=0)
    slope = arr[-1] - arr[0]
    centered = arr - arr.mean(axis=0, keepdims=True)
    fft = np.fft.rfft(centered, axis=0)
    low = np.abs(fft[1:6]) ** 2
    total = np.abs(fft[1:]) ** 2
    low_energy = float(np.sum(low))
    total_energy = float(np.sum(total))
    return {
        "response_amp_mean": float(np.mean(channel_amp)),
        "response_amp_max": float(np.max(channel_amp)),
        "response_slope_mean": float(np.mean(slope)),
        "response_slope_abs_mean": float(np.mean(np.abs(slope))),
        "dct_low_energy": low_energy,
        "dct_low_energy_ratio": low_energy / total_energy if total_energy > 0 else 0.0,
    }


def _metadata_features(meta: Mapping[str, Any] | None) -> dict[str, float]:
    meta = meta or {}
    values = {
        "window_start_s": _fnum(meta.get("window_start_s")),
        "window_end_s": _fnum(meta.get("window_end_s")),
        "window_center_s": _fnum(meta.get("window_center_s")),
        "t_onset": _fnum(meta.get("t_onset")),
        "t_min": _fnum(meta.get("t_min")),
        "interpolated_ratio": _fnum(meta.get("interpolated_ratio")),
        "max_gap_inside_window": _fnum(meta.get("max_gap_inside_window")),
    }
    response_phase = str(meta.get("response_phase", "unknown"))
    phase_label = str(meta.get("phase_label", "unknown"))
    for name in ["pre_response", "main_response", "recovery", "unknown"]:
        values[f"response_phase_{name}"] = 1.0 if response_phase == name else 0.0
    for name in ["early", "middle", "late", "unknown"]:
        values[f"phase_label_{name}"] = 1.0 if phase_label == name else 0.0
    return values


def _safe_ratio(a: float, b: float) -> float:
    return float(a / b) if abs(b) > 1e-9 else 0.0


def _target_ridge_features(window: np.ndarray, meta: Mapping[str, Any] | None = None) -> dict[str, float]:
    """Feature set matching run_regression_head_ablation.rich_feature_dict."""
    arr = np.asarray(window, dtype=np.float64)
    meta = meta or {}
    diff = np.diff(arr, axis=0)
    ch_mean = arr.mean(axis=0)
    ch_std = arr.std(axis=0)
    ch_min = arr.min(axis=0)
    ch_max = arr.max(axis=0)
    ch_amp = ch_max - ch_min
    ch_slope = (arr[-1] - arr[0]) / max(arr.shape[0] - 1, 1)
    ch_absdiff_mean = np.abs(diff).mean(axis=0)
    ch_absdiff_max = np.abs(diff).max(axis=0)

    values: dict[str, float] = {}
    for name, vector in [
        ("mean", ch_mean),
        ("std", ch_std),
        ("min", ch_min),
        ("max", ch_max),
        ("amp", ch_amp),
        ("slope", ch_slope),
        ("absdiff_mean", ch_absdiff_mean),
        ("absdiff_max", ch_absdiff_max),
    ]:
        for idx, value in enumerate(vector):
            values[f"ch{idx}_{name}"] = float(value)

    amp_order = np.argsort(-ch_amp)
    top_amp = ch_amp[amp_order]
    top_slope = ch_slope[amp_order]
    values.update(
        {
            "global_mean": float(arr.mean()),
            "global_std": float(arr.std()),
            "global_min": float(arr.min()),
            "global_max": float(arr.max()),
            "global_amp": float(arr.max() - arr.min()),
            "global_absdiff_mean": float(np.abs(diff).mean()),
            "global_absdiff_max": float(np.abs(diff).max()),
            "slope_mean": float(ch_slope.mean()),
            "slope_std": float(ch_slope.std()),
            "amp_mean": float(ch_amp.mean()),
            "amp_std": float(ch_amp.std()),
            "amp_top1": float(top_amp[0]),
            "amp_top2": float(top_amp[1]) if len(top_amp) > 1 else 0.0,
            "amp_top3": float(top_amp[2]) if len(top_amp) > 2 else 0.0,
            "amp_top4": float(top_amp[3]) if len(top_amp) > 3 else 0.0,
            "amp_top1_top2_ratio": _safe_ratio(float(top_amp[0]), float(top_amp[1]) if len(top_amp) > 1 else 0.0),
            "amp_top1_top3_ratio": _safe_ratio(float(top_amp[0]), float(top_amp[2]) if len(top_amp) > 2 else 0.0),
            "amp_top1_top4_ratio": _safe_ratio(float(top_amp[0]), float(top_amp[3]) if len(top_amp) > 3 else 0.0),
            "slope_top1_top2_ratio": _safe_ratio(float(top_slope[0]), float(top_slope[1]) if len(top_slope) > 1 else 0.0),
        }
    )
    window_start = _fnum(meta.get("window_start_s"))
    window_end = _fnum(meta.get("window_end_s"))
    window_center = _fnum(meta.get("window_center_s"), (window_start + window_end) / 2.0)
    onset = _fnum(meta.get("t_onset"))
    t_min = _fnum(meta.get("t_min"))
    values.update(
        {
            "window_start_s": window_start,
            "window_end_s": window_end,
            "window_center_s": window_center,
            "window_len_s": window_end - window_start,
            "t_onset": onset,
            "t_min": t_min,
            "center_minus_onset": window_center - onset,
            "center_minus_t_min": window_center - t_min,
            "interpolated_ratio": _fnum(meta.get("interpolated_ratio")),
            "max_gap_inside_window": _fnum(meta.get("max_gap_inside_window")),
        }
    )
    response_phase = str(meta.get("response_phase", "unknown"))
    for name in ["main_response", "recovery", "unknown"]:
        values[f"response_phase_{name}"] = 1.0 if response_phase == name else 0.0
    phase_label = str(meta.get("phase_label", "unknown"))
    for name in ["early", "middle", "late", "unknown"]:
        values[f"phase_label_{name}"] = 1.0 if phase_label == name else 0.0
    phase_int = int(_fnum(meta.get("phase"), -1.0))
    for name in [0, 1, 2]:
        values[f"phase_id_{name}"] = 1.0 if phase_int == name else 0.0
    values["phase_id_unknown"] = 1.0 if phase_int < 0 else 0.0
    return values


def target_ridge_features(window: np.ndarray, meta: Mapping[str, Any] | None = None) -> dict[str, float]:
    """Public, validated entry point for the frozen rich C5 feature schema."""
    values = np.asarray(window)
    if values.shape != (100, 8) or not np.isfinite(values).all():
        raise ValueError("target Ridge window must be finite with shape (100,8)")
    return _target_ridge_features(values, meta)


def _base_features(
    window: np.ndarray,
    result: DeployResult,
    mode: str,
    meta: Mapping[str, Any] | None = None,
) -> dict[str, float]:
    final = float(result.final_ppm)
    routed = float(result.routed_pred_ppm)
    raw = float(result.base_r3ak16_raw_ppm)
    values: dict[str, float] = {
        "final_ppm": final,
        "routed_pred_ppm": routed,
        "base_r3ak16_raw_ppm": raw,
        "final_minus_routed": final - routed,
        "routed_minus_raw": routed - raw,
        "final_hinge_100": max(0.0, final - 100.0),
        "final_hinge_150": max(0.0, final - 150.0),
        "final_hinge_200": max(0.0, final - 200.0),
    }
    if mode in {"ridge_basic", "ridge_phase", "piecewise_ridge"}:
        values.update(_window_feature_stats(window))
        ch_values = _channel_summary_features(window)
        for key, value in ch_values.items():
            if key.startswith("ch") and (
                key.endswith("_range") or key.endswith("_slope") or key.endswith("_std")
            ):
                values[key] = value
    if mode in {"ridge_phase", "piecewise_ridge"}:
        values.update(_metadata_features(meta))
    if mode == "piecewise_ridge":
        if final <= 100.0:
            bin_name = "low"
        elif final <= 175.0:
            bin_name = "mid"
        else:
            bin_name = "high"
        for name in ["low", "mid", "high"]:
            flag = 1.0 if bin_name == name else 0.0
            values[f"pred_bin_{name}"] = flag
            values[f"final_ppm_x_{name}"] = final * flag
            values[f"routed_pred_ppm_x_{name}"] = routed * flag
            values[f"response_amp_max_x_{name}"] = values.get("response_amp_max", 0.0) * flag
            values[f"dct_low_energy_ratio_x_{name}"] = values.get("dct_low_energy_ratio", 0.0) * flag
    return values


class RichResidualPolicy:
    """Apply exported rich residual and optional route-rescue policies."""

    def __init__(self, artifact: Mapping[str, Any] | None = None) -> None:
        self.artifact = dict(artifact or {})
        residual = self.artifact.get("residual_policy", {})
        self.selected_modes = dict(residual.get("selected_modes", {}))
        self.clip_ppm = residual.get("clip_ppm_for_CO", [25.0, 250.0])
        self.models = {
            str(item.get("client")): dict(item)
            for item in residual.get("models", [])
        }
        self.route_rescue = self.artifact.get("route_rescue_policy", {})
        target_ridge = self.artifact.get("target_ridge_policy", {})
        self.target_ridge_modes = dict(target_ridge.get("selected_modes", {}))
        self.target_ridge_models = {
            (str(item.get("client")), int(item.get("class_id"))): dict(item)
            for item in target_ridge.get("models", [])
        }
        target_mlp = self.artifact.get("target_mlp_policy", {})
        self.target_mlp_modes = dict(target_mlp.get("selected_modes", {}))
        self.target_mlp_models = {
            (str(item.get("client")), int(item.get("class_id"))): dict(item)
            for item in target_mlp.get("models", [])
        }
        source_aug = self.artifact.get("source_aug_target_ridge_policy", {})
        self.source_aug_policy = dict(source_aug)
        source_aug_switch = source_aug.get("switch_rule", {})
        self.source_aug_enabled_clients = set(source_aug_switch.get("enabled_clients", []))
        configured_class_ids = source_aug_switch.get("class_ids")
        if configured_class_ids is None:
            configured_class_ids = [source_aug_switch.get("class_id", CO_CLASS)]
        self.source_aug_class_ids = {int(class_id) for class_id in configured_class_ids}
        source_heads = source_aug.get("source_heads", {})
        self.source_ridge_heads = {
            int(item.get("class_id")): dict(item)
            for item in source_heads.get("ridge_per_gas", [])
        }
        self.source_mlp_heads = {
            int(item.get("class_id")): dict(item)
            for item in source_heads.get("mlp_per_gas", [])
        }
        self.source_shared_mlp = dict(source_heads.get("shared_mlp", {}))
        self.source_aug_models = {
            (str(item.get("client")), int(item.get("class_id"))): dict(item)
            for item in source_aug.get("models", [])
        }
        self.enabled = bool(self.artifact)

    @classmethod
    def from_json(cls, path: str | Path) -> "RichResidualPolicy":
        path = Path(path)
        if not path.exists():
            return cls()
        return cls(json.loads(path.read_text(encoding="utf-8")))

    def apply(
        self,
        window: np.ndarray,
        result: DeployResult,
        client_id: str,
        meta: Mapping[str, Any] | None = None,
    ) -> float:
        ppm = float(result.final_ppm)
        if not self.enabled:
            return ppm
        rescue = self._route_rescue_ppm(result, client_id, meta)
        if rescue is not None:
            return rescue
        source_aug = self._source_aug_target_ridge_ppm(window, result, client_id, meta)
        if source_aug is not None:
            return source_aug
        target_mlp = self._target_mlp_ppm(window, result, client_id, meta)
        if target_mlp is not None:
            return target_mlp
        target_ridge = self._target_ridge_ppm(window, result, client_id, meta)
        if target_ridge is not None:
            return target_ridge
        residual = self._residual_ppm(window, result, client_id, meta)
        return ppm if residual is None else residual

    def _target_ridge_ppm(
        self,
        window: np.ndarray,
        result: DeployResult,
        client_id: str,
        meta: Mapping[str, Any] | None,
    ) -> float | None:
        if self.target_ridge_modes.get(str(client_id)) != "ridge_direct":
            return None
        model = self.target_ridge_models.get((str(client_id), int(result.pred_class)))
        if not model or model.get("enabled", True) is False:
            return None
        names = list(model.get("feature_names") or self.artifact.get("target_ridge_policy", {}).get("feature_names", []))
        if not names:
            return None
        meta_with_phase = dict(meta or {})
        meta_with_phase["phase"] = getattr(result, "phase", -1)
        values = _target_ridge_features(window, meta_with_phase)
        mean = np.asarray(model.get("mean", []), dtype=np.float64)
        scale = np.asarray(model.get("scale", []), dtype=np.float64)
        coef = np.asarray(model.get("coef", []), dtype=np.float64)
        if len(mean) != len(names) or len(scale) != len(names) or len(coef) != len(names) + 1:
            return None
        x = np.asarray([_fnum(values.get(name), 0.0) for name in names], dtype=np.float64)
        x = np.where(np.isfinite(x), x, mean)
        scale = np.where(np.abs(scale) < 1e-9, 1.0, scale)
        design = np.concatenate([[1.0], (x - mean) / scale])
        pred = float(design @ coef)
        return float(np.clip(pred, float(model.get("clip_min", 0.0)), float(model.get("clip_max", 250.0))))

    def _target_mlp_ppm(
        self,
        window: np.ndarray,
        result: DeployResult,
        client_id: str,
        meta: Mapping[str, Any] | None,
    ) -> float | None:
        if self.target_mlp_modes.get(str(client_id)) != "mlp_direct":
            return None
        model = self.target_mlp_models.get((str(client_id), int(result.pred_class)))
        if not model or model.get("enabled", True) is False:
            return None
        names = list(model.get("feature_names") or self.artifact.get("target_mlp_policy", {}).get("feature_names", []))
        if not names:
            return None
        meta_with_phase = dict(meta or {})
        meta_with_phase["phase"] = getattr(result, "phase", -1)
        values = _target_ridge_features(window, meta_with_phase)
        mean = np.asarray(model.get("mean", []), dtype=np.float64)
        scale = np.asarray(model.get("scale", []), dtype=np.float64)
        coefs = [np.asarray(item, dtype=np.float64) for item in model.get("coefs", [])]
        intercepts = [np.asarray(item, dtype=np.float64) for item in model.get("intercepts", [])]
        if len(mean) != len(names) or len(scale) != len(names) or len(coefs) != len(intercepts):
            return None
        x = np.asarray([_fnum(values.get(name), 0.0) for name in names], dtype=np.float64)
        x = np.where(np.isfinite(x), x, mean)
        scale = np.where(np.abs(scale) < 1e-9, 1.0, scale)
        layer = (x - mean) / scale
        activation = str(model.get("activation", "relu"))
        for idx, (coef, intercept) in enumerate(zip(coefs, intercepts)):
            layer = layer @ coef + intercept
            is_last = idx == len(coefs) - 1
            if not is_last and activation == "relu":
                layer = np.maximum(layer, 0.0)
            elif not is_last and activation == "tanh":
                layer = np.tanh(layer)
        pred = float(np.asarray(layer).reshape(-1)[0])
        return float(np.clip(pred, float(model.get("clip_min", 0.0)), float(model.get("clip_max", 250.0))))

    def _apply_ridge_json(self, values: Mapping[str, float], model: Mapping[str, Any]) -> float | None:
        names = list(model.get("feature_names", []))
        mean = np.asarray(model.get("mean", []), dtype=np.float64)
        scale = np.asarray(model.get("scale", []), dtype=np.float64)
        coef = np.asarray(model.get("coef", []), dtype=np.float64)
        if not names or len(mean) != len(names) or len(scale) != len(names) or len(coef) != len(names) + 1:
            return None
        x = np.asarray([_fnum(values.get(name), 0.0) for name in names], dtype=np.float64)
        x = np.where(np.isfinite(x), x, mean)
        scale = np.where(np.abs(scale) < 1e-9, 1.0, scale)
        design = np.concatenate([[1.0], (x - mean) / scale])
        pred = float(design @ coef)
        return float(np.clip(pred, float(model.get("clip_min", 0.0)), float(model.get("clip_max", 250.0))))

    def _apply_mlp_json(
        self,
        values: Mapping[str, float],
        model: Mapping[str, Any],
        extra: np.ndarray | None = None,
    ) -> float | None:
        names = list(model.get("feature_names", []))
        mean = np.asarray(model.get("mean", []), dtype=np.float64)
        scale = np.asarray(model.get("scale", []), dtype=np.float64)
        coefs = [np.asarray(item, dtype=np.float64) for item in model.get("coefs", [])]
        intercepts = [np.asarray(item, dtype=np.float64) for item in model.get("intercepts", [])]
        if not names or len(coefs) != len(intercepts):
            return None
        base = np.asarray([_fnum(values.get(name), 0.0) for name in names], dtype=np.float64)
        if extra is not None:
            raw = np.concatenate([base, np.asarray(extra, dtype=np.float64)])
        else:
            raw = base
        if len(mean) != len(raw) or len(scale) != len(raw):
            return None
        raw = np.where(np.isfinite(raw), raw, mean)
        scale = np.where(np.abs(scale) < 1e-9, 1.0, scale)
        layer = (raw - mean) / scale
        activation = str(model.get("activation", "relu"))
        for idx, (coef, intercept) in enumerate(zip(coefs, intercepts)):
            layer = layer @ coef + intercept
            is_last = idx == len(coefs) - 1
            if not is_last and activation == "relu":
                layer = np.maximum(layer, 0.0)
            elif not is_last and activation == "tanh":
                layer = np.tanh(layer)
        pred = float(np.asarray(layer).reshape(-1)[0])
        return float(np.clip(pred, float(model.get("clip_min", 0.0)), float(model.get("clip_max", 250.0))))

    def _source_aug_target_ridge_ppm(
        self,
        window: np.ndarray,
        result: DeployResult,
        client_id: str,
        meta: Mapping[str, Any] | None,
    ) -> float | None:
        if not self.source_aug_policy:
            return None
        if str(client_id) not in self.source_aug_enabled_clients:
            return None
        route_class = int(result.pred_class)
        if route_class not in self.source_aug_class_ids:
            return None
        model = self.source_aug_models.get((str(client_id), route_class))
        if not model or model.get("enabled", True) is False:
            return None

        meta_with_phase = dict(meta or {})
        meta_with_phase["phase"] = getattr(result, "phase", -1)
        values = _target_ridge_features(window, meta_with_phase)
        source_ridge = self._apply_ridge_json(values, self.source_ridge_heads.get(route_class, {}))
        source_mlp = self._apply_mlp_json(values, self.source_mlp_heads.get(route_class, {}))
        num_classes = max(4, route_class + 1)
        gas_onehot = np.zeros(num_classes, dtype=np.float64)
        gas_onehot[route_class] = 1.0
        source_shared = self._apply_mlp_json(values, self.source_shared_mlp, extra=gas_onehot)
        if source_ridge is None or source_mlp is None or source_shared is None:
            return None
        values = dict(values)
        values["srcpred_H1_source_ridge_ppm"] = float(source_ridge)
        values["srcpred_H2_source_per_gas_mlp_ppm"] = float(source_mlp)
        values["srcpred_H3_source_shared_mlp_ppm"] = float(source_shared)
        return self._apply_ridge_json(values, model)

    def _residual_ppm(
        self,
        window: np.ndarray,
        result: DeployResult,
        client_id: str,
        meta: Mapping[str, Any] | None,
    ) -> float | None:
        if int(result.pred_class) != CO_CLASS:
            return None
        model = self.models.get(str(client_id))
        if not model or model.get("enabled", True) is False:
            return None
        mode = str(model.get("mode", self.selected_modes.get(str(client_id), "")))
        values = _base_features(window, result, mode, meta)
        names = list(model.get("feature_names", []))
        if not names:
            return None
        mean = np.asarray(model.get("mean", []), dtype=np.float64)
        scale = np.asarray(model.get("scale", []), dtype=np.float64)
        coef = np.asarray(model.get("coef", []), dtype=np.float64)
        if len(mean) != len(names) or len(scale) != len(names) or len(coef) != len(names) + 1:
            return None
        x = np.asarray([_fnum(values.get(name), 0.0) for name in names], dtype=np.float64)
        x = np.where(np.isfinite(x), x, mean)
        scale = np.where(np.abs(scale) < 1e-9, 1.0, scale)
        design = np.concatenate([[1.0], (x - mean) / scale])
        delta = float(design @ coef)
        corrected = float(result.final_ppm) + delta
        lo, hi = float(self.clip_ppm[0]), float(self.clip_ppm[1])
        return float(np.clip(corrected, lo, hi))

    def _route_rescue_ppm(
        self,
        result: DeployResult,
        client_id: str,
        meta: Mapping[str, Any] | None,
    ) -> float | None:
        gates = []
        if self.route_rescue.get("selected_gate"):
            gates.append(self.route_rescue.get("selected_gate"))
        gates.extend(self.route_rescue.get("additional_gates", []))
        for gate in gates:
            rescue = self._single_route_rescue_ppm(gate, result, client_id, meta)
            if rescue is not None:
                return rescue
        return None

    def _single_route_rescue_ppm(
        self,
        gate: Mapping[str, Any] | None,
        result: DeployResult,
        client_id: str,
        meta: Mapping[str, Any] | None,
    ) -> float | None:
        if not gate:
            return None
        if str(client_id) != "C4":
            return None
        phase = str(gate.get("phase", "any"))
        if phase != "any" and str((meta or {}).get("response_phase", "unknown")) != phase:
            return None
        pred_classes = {
            int(part) for part in str(gate.get("pred_classes", "")).split(",") if part.strip()
        }
        if pred_classes and int(result.pred_class) not in pred_classes:
            return None
        if float(result.final_ppm) >= float(gate.get("max_ppm", 50.0)):
            return None
        if float(result.risk_score) < float(gate.get("risk_threshold", 0.0)):
            return None
        return float(gate.get("rescue_ppm", result.final_ppm))
