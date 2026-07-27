"""Build target-specific deployment packages for several clients.

This is a small orchestration layer around:

1. ``gaps_flower.specialist_calibration_fit``
2. ``gaps_deploy.build_package``
3. ``gaps_deploy.validate_deployment_packages``

It is meant for formal deployment experiments where the classifier/regression
checkpoints are shared, but calibration/routing/full/specialist/response-ref
artifacts are generated separately for each target client.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence


def _client_ids(raw_clients: Sequence[str]) -> List[str]:
    client_ids: List[str] = []
    for raw in raw_clients:
        value = str(raw).strip()
        if not value:
            continue
        if value.lower().startswith("client_"):
            value = value.split("_", 1)[1]
        if value.lower().startswith("client"):
            value = value[len("client"):]
        if value.upper().startswith("C"):
            value = value[1:]
        client_ids.append(f"C{int(value)}")
    if not client_ids:
        raise ValueError("At least one client is required")
    return client_ids


def _client_dir_name(client_id: str) -> str:
    return f"client_{int(client_id[1:])}"


def _command_to_text(cmd: Sequence[str]) -> str:
    return " ".join(f'"{part}"' if " " in part else part for part in cmd)


def _run_or_print(cmd: Sequence[str], dry_run: bool) -> None:
    print(_command_to_text(cmd))
    if not dry_run:
        subprocess.run(list(cmd), check=True)


def _calibration_command(args: argparse.Namespace, client_id: str, calib_dir: Path) -> List[str]:
    data_dir = Path(args.data_root) / _client_dir_name(client_id)
    cmd = [
        sys.executable,
        "-m",
        "gaps_flower.specialist_calibration_fit",
        "--classifier-ckpt",
        args.classifier_ckpt,
        "--regression-ckpt",
        args.regression_ckpt,
        "--calib-data-dir",
        str(data_dir),
        "--output-dir",
        str(calib_dir),
        "--calibration-mode",
        "auto_v2_specialist",
        "--classes",
        args.classes,
        "--steps",
        str(args.steps),
        "--full-steps",
        str(args.full_steps),
        "--lr",
        str(args.lr),
        "--class-weight",
        str(args.class_weight),
        "--huber-delta",
        str(args.huber_delta),
        "--val-ratio",
        str(args.val_ratio),
        "--split-by",
        args.split_by,
        "--gate-metric",
        args.gate_metric,
        "--gate-mode",
        args.gate_mode,
        "--batch-size",
        str(args.batch_size),
        "--seed",
        str(args.seed),
        "--reg-head-depth",
        str(args.reg_head_depth),
    ]
    if args.reg_output_mode:
        cmd.extend(["--reg-output-mode", args.reg_output_mode])
    if args.reg_range_penalty > 0:
        cmd.extend(["--reg-range-penalty", str(args.reg_range_penalty)])
    if args.reg_window_stats:
        cmd.extend(
            [
                "--reg-window-stats",
                "--reg-window-stats-mode",
                args.reg_window_stats_mode,
                "--reg-window-stats-dim",
                str(args.reg_window_stats_dim),
            ]
        )
    if args.reg_response_branch:
        cmd.extend(["--reg-response-branch", args.reg_response_branch])
    if args.reg_dct_k is not None:
        cmd.extend(["--reg-dct-k", str(args.reg_dct_k)])
    if args.reg_dct_gamma_init is not None:
        cmd.extend(["--reg-dct-gamma-init", str(args.reg_dct_gamma_init)])
    if args.reg_dct_dropout is not None:
        cmd.extend(["--reg-dct-dropout", str(args.reg_dct_dropout)])
    if args.reg_msconv_channels is not None:
        cmd.extend(["--reg-msconv-channels", str(args.reg_msconv_channels)])
    if args.reg_msconv_kernels:
        cmd.extend(["--reg-msconv-kernels", args.reg_msconv_kernels])
    if args.reg_msconv_gamma_init is not None:
        cmd.extend(["--reg-msconv-gamma-init", str(args.reg_msconv_gamma_init)])
    if args.reg_msconv_dropout is not None:
        cmd.extend(["--reg-msconv-dropout", str(args.reg_msconv_dropout)])
    if args.reg_tcn_adapter:
        cmd.append("--reg-tcn-adapter")
    if args.reg_tcn_adapter_kernel is not None:
        cmd.extend(["--reg-tcn-adapter-kernel", str(args.reg_tcn_adapter_kernel)])
    if args.reg_tcn_adapter_gamma_init is not None:
        cmd.extend(["--reg-tcn-adapter-gamma-init", str(args.reg_tcn_adapter_gamma_init)])
    if args.reg_tcn_adapter_dropout is not None:
        cmd.extend(["--reg-tcn-adapter-dropout", str(args.reg_tcn_adapter_dropout)])
    if args.reg_use_shared_trunk:
        cmd.append("--reg-use-shared-trunk")
    if args.reg_shared_trunk_dim is not None:
        cmd.extend(["--reg-shared-trunk-dim", str(args.reg_shared_trunk_dim)])
    if args.reg_gas_emb_dim is not None:
        cmd.extend(["--reg-gas-emb-dim", str(args.reg_gas_emb_dim)])
    if args.reg_residual_head_depth is not None:
        cmd.extend(["--reg-residual-head-depth", str(args.reg_residual_head_depth)])
    if args.use_reg_ratio_branch:
        cmd.append("--use-reg-ratio-branch")
    if args.reg_ratio_gamma_init is not None:
        cmd.extend(["--reg-ratio-gamma-init", str(args.reg_ratio_gamma_init)])
    if args.reg_ratio_dropout is not None:
        cmd.extend(["--reg-ratio-dropout", str(args.reg_ratio_dropout)])
    if args.cpu:
        cmd.append("--cpu")
    if args.refit_affine_full_calib:
        cmd.append("--refit-affine-full-calib")
    if args.refit_full_calib:
        cmd.append("--refit-full-calib")
    if args.refit_steps is not None:
        cmd.extend(["--refit-steps", str(args.refit_steps)])
    if args.cpu:
        cmd.append("--cpu")
    return cmd


def _build_package_command(
    args: argparse.Namespace,
    client_id: str,
    calib_dir: Path,
    package_dir: Path,
) -> List[str]:
    cmd = [
        sys.executable,
        "-m",
        "gaps_deploy.build_package",
        "--output-dir",
        str(package_dir),
        "--classifier-ckpt",
        args.classifier_ckpt,
        "--regression-ckpt",
        args.regression_ckpt,
        "--model-config",
        args.model_config,
        "--calibration-dir",
        str(calib_dir),
        "--model-version",
        args.model_version or f"per_client_{client_id}",
        "--specialist-dir",
        str(calib_dir / "specialists"),
        "--full-model-ckpt",
        str(calib_dir / "full_model.pth"),
        "--reg-head-depth",
        str(args.reg_head_depth),
    ]
    if args.reg_output_mode:
        cmd.extend(["--reg-output-mode", args.reg_output_mode])
    if args.reg_window_stats:
        cmd.extend(
            [
                "--reg-window-stats",
                "--reg-window-stats-mode",
                args.reg_window_stats_mode,
                "--reg-window-stats-dim",
                str(args.reg_window_stats_dim),
            ]
        )
    if args.reg_response_branch:
        cmd.extend(["--reg-response-branch", args.reg_response_branch])
    if args.reg_dct_k is not None:
        cmd.extend(["--reg-dct-k", str(args.reg_dct_k)])
    if args.reg_dct_gamma_init is not None:
        cmd.extend(["--reg-dct-gamma-init", str(args.reg_dct_gamma_init)])
    if args.reg_dct_dropout is not None:
        cmd.extend(["--reg-dct-dropout", str(args.reg_dct_dropout)])
    if args.reg_msconv_channels is not None:
        cmd.extend(["--reg-msconv-channels", str(args.reg_msconv_channels)])
    if args.reg_msconv_kernels:
        cmd.extend(["--reg-msconv-kernels", args.reg_msconv_kernels])
    if args.reg_msconv_gamma_init is not None:
        cmd.extend(["--reg-msconv-gamma-init", str(args.reg_msconv_gamma_init)])
    if args.reg_msconv_dropout is not None:
        cmd.extend(["--reg-msconv-dropout", str(args.reg_msconv_dropout)])
    if args.reg_tcn_adapter:
        cmd.append("--reg-tcn-adapter")
    if args.reg_tcn_adapter_kernel is not None:
        cmd.extend(["--reg-tcn-adapter-kernel", str(args.reg_tcn_adapter_kernel)])
    if args.reg_tcn_adapter_gamma_init is not None:
        cmd.extend(["--reg-tcn-adapter-gamma-init", str(args.reg_tcn_adapter_gamma_init)])
    if args.reg_tcn_adapter_dropout is not None:
        cmd.extend(["--reg-tcn-adapter-dropout", str(args.reg_tcn_adapter_dropout)])
    if args.reg_use_shared_trunk:
        cmd.append("--reg-use-shared-trunk")
    if args.reg_shared_trunk_dim is not None:
        cmd.extend(["--reg-shared-trunk-dim", str(args.reg_shared_trunk_dim)])
    if args.reg_gas_emb_dim is not None:
        cmd.extend(["--reg-gas-emb-dim", str(args.reg_gas_emb_dim)])
    if args.reg_residual_head_depth is not None:
        cmd.extend(["--reg-residual-head-depth", str(args.reg_residual_head_depth)])
    if args.use_reg_ratio_branch:
        cmd.append("--use-reg-ratio-branch")
    if args.reg_ratio_gamma_init is not None:
        cmd.extend(["--reg-ratio-gamma-init", str(args.reg_ratio_gamma_init)])
    if args.reg_ratio_dropout is not None:
        cmd.extend(["--reg-ratio-dropout", str(args.reg_ratio_dropout)])
    cmd.extend(["--qc-policy", args.qc_policy])
    return cmd


def _validate_command(
    args: argparse.Namespace,
    client_ids: Sequence[str],
    package_dirs: Dict[str, Path],
    output_root: Path,
) -> List[str]:
    cmd = [
        sys.executable,
        "-m",
        "gaps_deploy.validate_deployment_packages",
        "--clients",
        *client_ids,
        "--require-distinct-packages",
        "--expected-reg-head-depth",
        str(args.reg_head_depth),
        "--output-json",
        str(output_root / "package_preflight.json"),
    ]
    if args.reg_output_mode:
        cmd.extend(["--expected-reg-output-mode", args.reg_output_mode])
    if args.reg_window_stats:
        cmd.append("--expected-reg-window-stats")
    if args.reg_response_branch:
        cmd.extend(["--expected-reg-response-branch", args.reg_response_branch])
    if args.reg_tcn_adapter:
        cmd.append("--expected-reg-tcn-adapter")
    if args.reg_use_shared_trunk:
        cmd.append("--expected-reg-use-shared-trunk")
    if args.use_reg_ratio_branch:
        cmd.append("--expected-use-reg-ratio-branch")
    cmd.append("--client-packages")
    for client_id in client_ids:
        cmd.append(f"{client_id}={package_dirs[client_id]}")
    return cmd


def build_per_client_packages(args: argparse.Namespace) -> Dict[str, Any]:
    client_ids = _client_ids(args.clients)
    output_root = Path(args.output_root)
    calibration_root = output_root / "calibration"
    package_root = output_root / "packages"
    calibration_root.mkdir(parents=True, exist_ok=True)
    package_root.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, Any] = {
        "classifier_ckpt": args.classifier_ckpt,
        "regression_ckpt": args.regression_ckpt,
        "data_root": args.data_root,
        "clients": client_ids,
        "output_root": str(output_root),
        "reg_head_depth": args.reg_head_depth,
        "reg_output_mode": args.reg_output_mode or "checkpoint/default",
        "reg_window_stats": bool(args.reg_window_stats),
        "reg_window_stats_mode": args.reg_window_stats_mode,
        "reg_window_stats_dim": args.reg_window_stats_dim,
        "reg_response_branch": args.reg_response_branch or "checkpoint/default",
        "reg_dct_k": args.reg_dct_k,
        "reg_dct_gamma_init": args.reg_dct_gamma_init,
        "reg_dct_dropout": args.reg_dct_dropout,
        "reg_msconv_channels": args.reg_msconv_channels,
        "reg_msconv_kernels": args.reg_msconv_kernels,
        "reg_msconv_gamma_init": args.reg_msconv_gamma_init,
        "reg_msconv_dropout": args.reg_msconv_dropout,
        "reg_tcn_adapter": bool(args.reg_tcn_adapter),
        "reg_tcn_adapter_kernel": args.reg_tcn_adapter_kernel,
        "reg_tcn_adapter_gamma_init": args.reg_tcn_adapter_gamma_init,
        "reg_tcn_adapter_dropout": args.reg_tcn_adapter_dropout,
        "reg_use_shared_trunk": bool(args.reg_use_shared_trunk),
        "reg_shared_trunk_dim": args.reg_shared_trunk_dim,
        "reg_gas_emb_dim": args.reg_gas_emb_dim,
        "reg_residual_head_depth": args.reg_residual_head_depth,
        "use_reg_ratio_branch": bool(args.use_reg_ratio_branch),
        "reg_ratio_gamma_init": args.reg_ratio_gamma_init,
        "reg_ratio_dropout": args.reg_ratio_dropout,
        "calibration_dirs": {},
        "package_dirs": {},
        "commands": [],
    }

    package_dirs: Dict[str, Path] = {}
    for client_id in client_ids:
        calib_dir = calibration_root / client_id
        package_dir = package_root / client_id
        package_dirs[client_id] = package_dir
        manifest["calibration_dirs"][client_id] = str(calib_dir)
        manifest["package_dirs"][client_id] = str(package_dir)

        if not args.skip_calibration:
            cmd = _calibration_command(args, client_id, calib_dir)
            manifest["commands"].append(cmd)
            _run_or_print(cmd, args.dry_run)

        if not args.skip_package:
            cmd = _build_package_command(args, client_id, calib_dir, package_dir)
            manifest["commands"].append(cmd)
            _run_or_print(cmd, args.dry_run)

    if args.validate:
        cmd = _validate_command(args, client_ids, package_dirs, output_root)
        manifest["commands"].append(cmd)
        _run_or_print(cmd, args.dry_run)

    manifest_path = output_root / "build_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved manifest: {manifest_path}")
    print("Client package arguments:")
    print(
        " ".join(
            f"{client_id}={package_dirs[client_id]}"
            for client_id in client_ids
        )
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build target-specific calibration artifacts and deployment packages."
    )
    parser.add_argument("--classifier-ckpt", required=True)
    parser.add_argument("--regression-ckpt", required=True)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--clients", nargs="+", default=["3", "4", "5"])
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--model-version", default="")
    parser.add_argument("--qc-policy", required=True)
    parser.add_argument("--classes", default="1,2")
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--full-steps", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--class-weight", type=float, default=2.0)
    parser.add_argument("--huber-delta", type=float, default=0.2)
    parser.add_argument("--val-ratio", type=float, default=0.3)
    parser.add_argument("--split-by", default="class_concentration", choices=["class", "class_concentration"])
    parser.add_argument("--gate-metric", default="NRMSE_range", choices=["R2", "MAE", "RMSE", "NRMSE_range", "P90AE"])
    parser.add_argument("--gate-mode", default="metric", choices=["metric", "guarded"])
    parser.add_argument("--refit-affine-full-calib", action="store_true")
    parser.add_argument("--refit-full-calib", action="store_true")
    parser.add_argument("--refit-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reg-head-depth", type=int, default=4)
    parser.add_argument("--reg-output-mode", default="", choices=["", "sigmoid", "linear"])
    parser.add_argument("--reg-range-penalty", type=float, default=0.0)
    parser.add_argument("--reg-window-stats", action="store_true")
    parser.add_argument("--reg-window-stats-mode", default="global", choices=["global", "per_channel"])
    parser.add_argument("--reg-window-stats-dim", type=int, default=8)
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
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--skip-calibration", action="store_true")
    parser.add_argument("--skip-package", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    build_per_client_packages(args)


if __name__ == "__main__":
    main()
