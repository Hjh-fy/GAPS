"""Build a runnable GAPS deployment package.

The package is intentionally simple: it copies model checkpoints and calibration
artifacts into the directory layout consumed by `DeployPredictor.from_package`.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any, Dict

from .deploy_config import DeployConfig
from .package_contract import (
    DeploymentPackageError,
    load_checkpoint_state,
    load_json_object,
    normalize_and_validate_routing_config,
    require_file,
    validate_checkpoint_model_config,
    validate_model_config,
)
from .qc_policy import (
    RESPONSE_DEPENDENT_SCORES,
    TwoThresholdDecider,
    validate_calibration_refs,
)


def _copy_file(src: str, dst: Path, required: bool = True) -> bool:
    if not src:
        if required:
            raise ValueError(f"Missing required source for {dst.name}")
        return False
    src_path = Path(src)
    if not src_path.exists():
        if required:
            raise FileNotFoundError(src_path)
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path, dst)
    return True


def _assert_override_matches(
    model_config: Dict[str, Any],
    field: str,
    override: Any,
    *,
    normalize=lambda value: value,
) -> None:
    if override is None or override == "":
        return
    if field not in model_config:
        raise DeploymentPackageError(
            f"Explicit override {field} is absent from model_config"
        )
    expected = normalize(model_config[field])
    actual = normalize(override)
    if expected != actual:
        raise DeploymentPackageError(
            f"Explicit override {field}={actual!r} disagrees with "
            f"model_config value {expected!r}"
        )


def _response_refs_from_stats(raw: Dict[str, Any]) -> Dict[int, Any]:
    root = raw.get("response_refs", raw)
    if not isinstance(root, dict):
        return {}
    refs: Dict[int, Any] = {}
    for raw_key, value in root.items():
        try:
            class_id = int(raw_key)
        except (TypeError, ValueError):
            continue
        if isinstance(value, dict) and any(
            key in value for key in ("center", "scale", "z_sigs")
        ):
            refs[class_id] = value
    return refs


def build_package(
    output_dir: str,
    classifier_ckpt: str,
    regression_ckpt: str,
    model_config_path: str = "",
    calibration_dir: str = "",
    qc_policy: str = "",
    model_version: str = "",
    specialist_dir: str = "",
    full_model_ckpt: str = "",
    reg_head_depth: int | None = None,
    reg_output_mode: str = "",
    reg_window_stats: bool | None = None,
    reg_window_stats_mode: str = "",
    reg_window_stats_dim: int | None = None,
    reg_response_branch: str = "",
    reg_dct_k: int | None = None,
    reg_dct_gamma_init: float | None = None,
    reg_dct_dropout: float | None = None,
    reg_msconv_channels: int | None = None,
    reg_msconv_kernels: str = "",
    reg_msconv_gamma_init: float | None = None,
    reg_msconv_dropout: float | None = None,
    reg_tcn_adapter: bool | None = None,
    reg_tcn_adapter_kernel: int | None = None,
    reg_tcn_adapter_gamma_init: float | None = None,
    reg_tcn_adapter_dropout: float | None = None,
    use_reg_shared_trunk: bool | None = None,
    reg_shared_trunk_dim: int | None = None,
    reg_gas_emb_dim: int | None = None,
    reg_residual_head_depth: int | None = None,
    use_reg_ratio_branch: bool | None = None,
    reg_ratio_gamma_init: float | None = None,
    reg_ratio_dropout: float | None = None,
) -> Path:
    if not model_config_path:
        raise ValueError("model_config_path is required")
    if not qc_policy:
        raise ValueError("qc_policy is required")
    if not calibration_dir:
        raise ValueError("calibration_dir is required")

    model_config_src = require_file(Path(model_config_path), "model_config")
    classifier_src = require_file(Path(classifier_ckpt), "classifier checkpoint")
    regression_src = require_file(Path(regression_ckpt), "regression checkpoint")
    calibration_src = Path(calibration_dir)
    routing_src = require_file(
        calibration_src / "routing_config.json", "routing_config.json"
    )
    qc_src = require_file(Path(qc_policy), "qc_policy")

    model_config = load_json_object(model_config_src, "model_config")
    validate_model_config(model_config)
    classifier_payload, _ = load_checkpoint_state(classifier_src)
    regression_payload, _ = load_checkpoint_state(regression_src)
    validate_checkpoint_model_config(classifier_payload, model_config, classifier_src)
    validate_checkpoint_model_config(regression_payload, model_config, regression_src)

    override_values = {
        "reg_head_depth": reg_head_depth,
        "reg_output_mode": reg_output_mode,
        "reg_window_stats": reg_window_stats,
        "reg_window_stats_mode": reg_window_stats_mode,
        "reg_window_stats_dim": reg_window_stats_dim,
        "reg_response_branch": reg_response_branch,
        "reg_dct_k": reg_dct_k,
        "reg_dct_gamma_init": reg_dct_gamma_init,
        "reg_dct_dropout": reg_dct_dropout,
        "reg_msconv_channels": reg_msconv_channels,
        "reg_msconv_kernels": reg_msconv_kernels,
        "reg_msconv_gamma_init": reg_msconv_gamma_init,
        "reg_msconv_dropout": reg_msconv_dropout,
        "reg_tcn_adapter": reg_tcn_adapter,
        "reg_tcn_adapter_kernel": reg_tcn_adapter_kernel,
        "reg_tcn_adapter_gamma_init": reg_tcn_adapter_gamma_init,
        "reg_tcn_adapter_dropout": reg_tcn_adapter_dropout,
        "reg_use_shared_trunk": use_reg_shared_trunk,
        "reg_shared_trunk_dim": reg_shared_trunk_dim,
        "reg_gas_emb_dim": reg_gas_emb_dim,
        "reg_residual_head_depth": reg_residual_head_depth,
        "use_reg_ratio_branch": use_reg_ratio_branch,
        "reg_ratio_gamma_init": reg_ratio_gamma_init,
        "reg_ratio_dropout": reg_ratio_dropout,
    }
    lower_fields = {
        "reg_output_mode",
        "reg_window_stats_mode",
        "reg_response_branch",
    }
    for field, override in override_values.items():
        normalizer = (lambda value: str(value).lower()) if field in lower_fields else (lambda value: value)
        _assert_override_matches(
            model_config, field, override, normalize=normalizer
        )

    routing_raw = load_json_object(routing_src, "routing_config.json")
    routing = normalize_and_validate_routing_config(
        routing_raw, int(model_config["num_classes"])
    )
    selected_modes = routing["selected_modes"]
    full_required = any(mode == "full" for mode in selected_modes.values())
    full_src = Path(full_model_ckpt) if full_model_ckpt else None
    if full_required and full_src is None:
        raise DeploymentPackageError("routing_config selects full but full_model_ckpt is missing")
    if full_src is not None:
        if full_src.is_file():
            full_payload, _ = load_checkpoint_state(full_src)
            validate_checkpoint_model_config(full_payload, model_config, full_src)
        elif full_required:
            require_file(full_src, "full_model checkpoint")
        else:
            full_src = None

    specialist_classes = sorted(
        class_id
        for class_id, mode in selected_modes.items()
        if mode in {"specialist", "specialist_full"}
    )
    specialist_sources: Dict[int, Path] = {}
    if specialist_classes:
        if not specialist_dir:
            raise DeploymentPackageError(
                "routing_config selects specialist models but specialist_dir is missing"
            )
        specialist_root = Path(specialist_dir)
        for class_id in specialist_classes:
            specialist_path = require_file(
                specialist_root / f"class_{class_id}.pth",
                f"specialist checkpoint class {class_id}",
            )
            specialist_payload, _ = load_checkpoint_state(specialist_path)
            validate_checkpoint_model_config(
                specialist_payload, model_config, specialist_path
            )
            specialist_sources[class_id] = specialist_path

    qc_decider = TwoThresholdDecider()
    qc_decider.load_policies_json(str(qc_src))
    if not qc_decider.policies:
        raise DeploymentPackageError("qc_policy contains no policies")
    response_qc_required = any(
        set(policy.scores) & RESPONSE_DEPENDENT_SCORES
        for policy in qc_decider.policies.values()
    )
    calibration_stats_src = calibration_src / "calibration_stats.json"
    if response_qc_required:
        require_file(calibration_stats_src, "calibration_stats.json")
        stats = load_json_object(calibration_stats_src, "calibration_stats.json")
        validate_calibration_refs(
            _response_refs_from_stats(stats), int(model_config["num_classes"])
        )

    out = Path(output_dir)
    (out / "models").mkdir(parents=True, exist_ok=True)
    (out / "calibration").mkdir(parents=True, exist_ok=True)
    (out / "qc").mkdir(parents=True, exist_ok=True)
    (out / "config").mkdir(parents=True, exist_ok=True)
    _copy_file(str(classifier_src), out / "models" / "classification_model.pth")
    _copy_file(str(regression_src), out / "models" / "regression_model.pth")
    _copy_file(str(model_config_src), out / "models" / "model_config.json")
    _copy_file(str(routing_src), out / "calibration" / "routing_config.json")
    _copy_file(str(qc_src), out / "qc" / "selected_policy.json")
    if calibration_stats_src.is_file():
        _copy_file(
            str(calibration_stats_src),
            out / "calibration" / "calibration_stats.json",
        )
    if full_src is not None:
        _copy_file(str(full_src), out / "models" / "full_model.pth")
    for class_id, specialist_path in specialist_sources.items():
        _copy_file(
            str(specialist_path),
            out / "models" / "specialists" / f"class_{class_id}.pth",
        )

    deploy_config = DeployConfig()
    deploy_config.model_config = model_config
    deploy_config.classifier_checkpoint = "models/classification_model.pth"
    deploy_config.regression_checkpoint = "models/regression_model.pth"
    if full_src is not None:
        deploy_config.full_model_checkpoint = "models/full_model.pth"
    deploy_config.routing_config_path = "calibration/routing_config.json"
    deploy_config.qc_policy_path = "qc/selected_policy.json"
    deploy_config.save_json(str(out / "config" / "deploy_config.json"))

    readme = [
        "# GAPS Deployment Package",
        "",
        f"- model_version: `{model_version or 'unknown'}`",
        "- models/classification_model.pth",
        "- models/regression_model.pth",
        "- models/full_model.pth (optional)",
        "- models/specialists/*.pth (optional)",
        "- calibration/routing_config.json",
        "- qc/selected_policy.json",
        "",
        "Run a client file:",
        "",
        "```bash",
        "python -m gaps_deploy.predict_client_file --deploy-package <this_dir> --input <client/test_features.npy> --client-id C5",
        "```",
    ]
    (out / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a runnable GAPS deployment package.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--classifier-ckpt", required=True)
    parser.add_argument("--regression-ckpt", required=True)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--calibration-dir", required=True)
    parser.add_argument("--qc-policy", required=True)
    parser.add_argument("--model-version", default="")
    parser.add_argument("--specialist-dir", default="")
    parser.add_argument("--full-model-ckpt", default="")
    parser.add_argument("--reg-head-depth", type=int, default=None)
    parser.add_argument("--reg-output-mode", default="", choices=["", "sigmoid", "linear"])
    parser.add_argument("--reg-window-stats", action="store_true")
    parser.add_argument("--reg-window-stats-mode", default="", choices=["", "global", "per_channel"])
    parser.add_argument("--reg-window-stats-dim", type=int, default=None)
    parser.add_argument("--reg-response-branch", default="", choices=["", "none", "dct", "msconv"])
    parser.add_argument("--reg-dct-k", type=int, default=None)
    parser.add_argument("--reg-dct-gamma-init", type=float, default=None)
    parser.add_argument("--reg-dct-dropout", type=float, default=None)
    parser.add_argument("--reg-msconv-channels", type=int, default=None)
    parser.add_argument("--reg-msconv-kernels", default="")
    parser.add_argument("--reg-msconv-gamma-init", type=float, default=None)
    parser.add_argument("--reg-msconv-dropout", type=float, default=None)
    parser.add_argument("--reg-tcn-adapter", action="store_true")
    parser.add_argument("--reg-tcn-adapter-kernel", type=int, default=None)
    parser.add_argument("--reg-tcn-adapter-gamma-init", type=float, default=None)
    parser.add_argument("--reg-tcn-adapter-dropout", type=float, default=None)
    parser.add_argument("--reg-use-shared-trunk", action="store_true")
    parser.add_argument("--reg-shared-trunk-dim", type=int, default=None)
    parser.add_argument("--reg-gas-emb-dim", type=int, default=None)
    parser.add_argument("--reg-residual-head-depth", type=int, default=None)
    parser.add_argument("--use-reg-ratio-branch", action="store_true")
    parser.add_argument("--reg-ratio-gamma-init", type=float, default=None)
    parser.add_argument("--reg-ratio-dropout", type=float, default=None)
    args = parser.parse_args()

    out = build_package(
        output_dir=args.output_dir,
        classifier_ckpt=args.classifier_ckpt,
        regression_ckpt=args.regression_ckpt,
        model_config_path=args.model_config,
        calibration_dir=args.calibration_dir,
        qc_policy=args.qc_policy,
        model_version=args.model_version,
        specialist_dir=args.specialist_dir,
        full_model_ckpt=args.full_model_ckpt,
        reg_head_depth=args.reg_head_depth,
        reg_output_mode=args.reg_output_mode,
        reg_window_stats=args.reg_window_stats if args.reg_window_stats else None,
        reg_window_stats_mode=args.reg_window_stats_mode,
        reg_window_stats_dim=args.reg_window_stats_dim,
        reg_response_branch=args.reg_response_branch,
        reg_dct_k=args.reg_dct_k,
        reg_dct_gamma_init=args.reg_dct_gamma_init,
        reg_dct_dropout=args.reg_dct_dropout,
        reg_msconv_channels=args.reg_msconv_channels,
        reg_msconv_kernels=args.reg_msconv_kernels,
        reg_msconv_gamma_init=args.reg_msconv_gamma_init,
        reg_msconv_dropout=args.reg_msconv_dropout,
        reg_tcn_adapter=args.reg_tcn_adapter if args.reg_tcn_adapter else None,
        reg_tcn_adapter_kernel=args.reg_tcn_adapter_kernel,
        reg_tcn_adapter_gamma_init=args.reg_tcn_adapter_gamma_init,
        reg_tcn_adapter_dropout=args.reg_tcn_adapter_dropout,
        use_reg_shared_trunk=args.reg_use_shared_trunk if args.reg_use_shared_trunk else None,
        reg_shared_trunk_dim=args.reg_shared_trunk_dim,
        reg_gas_emb_dim=args.reg_gas_emb_dim,
        reg_residual_head_depth=args.reg_residual_head_depth,
        use_reg_ratio_branch=args.use_reg_ratio_branch if args.use_reg_ratio_branch else None,
        reg_ratio_gamma_init=args.reg_ratio_gamma_init,
        reg_ratio_dropout=args.reg_ratio_dropout,
    )
    print(f"Deployment package written to: {out}")


if __name__ == "__main__":
    main()
