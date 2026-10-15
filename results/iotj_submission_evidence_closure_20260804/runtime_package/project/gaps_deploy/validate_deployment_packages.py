"""Validate deployment packages before formal QC guardrail runs.

The checker validates both the artifact layout and the exact runtime load path,
including model/checkpoint compatibility, routing schema, and QC policy inputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

from .inference import DeployPredictor


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


def _parse_client_packages(raw_items: Sequence[str]) -> Dict[str, Path]:
    packages: Dict[str, Path] = {}
    for raw in raw_items:
        item = str(raw).strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Invalid client package item {item!r}; expected C3=path")
        raw_client, raw_path = item.split("=", 1)
        packages[_client_ids([raw_client])[0]] = Path(raw_path.strip().strip("\"'"))
    return packages


def _resolve_packages(
    clients: Sequence[str],
    deploy_package: str,
    client_packages: Sequence[str],
) -> Dict[str, Path]:
    client_ids = _client_ids(clients)
    per_client = _parse_client_packages(client_packages)
    if per_client:
        missing = [client_id for client_id in client_ids if client_id not in per_client]
        if missing:
            raise ValueError(f"Missing per-client packages for: {missing}")
        return {client_id: per_client[client_id] for client_id in client_ids}
    if not deploy_package:
        raise ValueError("Provide --deploy-package or --client-packages C3=path ...")
    package = Path(deploy_package)
    return {client_id: package for client_id in client_ids}


def _load_json(path: Path, errors: List[str]) -> Any:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        errors.append(f"Cannot read JSON {path}: {exc}")
        return None


def _has_response_ref_shape(ref: Any) -> bool:
    if not isinstance(ref, dict):
        return False
    return any(key in ref for key in ("center", "scale", "z_sigs"))


def _response_ref_summary(calibration_stats: Any) -> Dict[str, Any]:
    if not isinstance(calibration_stats, dict):
        return {"count": 0, "classes": [], "missing_loocv_p90": []}
    ref_root = calibration_stats.get("response_refs", calibration_stats)
    if not isinstance(ref_root, dict):
        return {"count": 0, "classes": [], "missing_loocv_p90": []}

    classes: List[int] = []
    missing_loocv: List[int] = []
    for key, ref in ref_root.items():
        try:
            cls_id = int(key)
        except (TypeError, ValueError):
            continue
        if not _has_response_ref_shape(ref):
            continue
        classes.append(cls_id)
        if isinstance(ref, dict) and "loocv_p90" not in ref:
            missing_loocv.append(cls_id)
    return {
        "count": len(classes),
        "classes": sorted(classes),
        "missing_loocv_p90": sorted(missing_loocv),
    }


def _path_from_config(package: Path, config: Dict[str, Any], key: str) -> Path:
    raw = str(config.get(key, "") or "")
    return package / raw if raw else package / "__missing__"


def _validate_one_package(
    client_id: str,
    package: Path,
    require_response_refs: bool,
    min_response_ref_classes: int,
    expected_reg_head_depth: int,
    expected_reg_output_mode: str,
    expected_reg_window_stats: bool,
    expected_reg_response_branch: str,
    expected_reg_tcn_adapter: bool,
    expected_reg_use_shared_trunk: bool = False,
    expected_use_reg_ratio_branch: bool = False,
) -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []

    if not package.exists() or not package.is_dir():
        return {
            "client_id": client_id,
            "package": str(package),
            "status": "fail",
            "errors": [f"Package directory does not exist: {package}"],
            "warnings": [],
        }

    required = {
        "deploy_config": package / "config" / "deploy_config.json",
        "model_config": package / "models" / "model_config.json",
        "qc_policy": package / "qc" / "selected_policy.json",
        "classification_model": package / "models" / "classification_model.pth",
        "regression_model": package / "models" / "regression_model.pth",
        "routing_config": package / "calibration" / "routing_config.json",
    }
    for name, path in required.items():
        if not path.exists():
            errors.append(f"Missing {name}: {path}")

    deploy_config = _load_json(required["deploy_config"], errors) if required["deploy_config"].exists() else {}
    model_config = _load_json(required["model_config"], errors) if required["model_config"].exists() else {}
    routing_config = _load_json(required["routing_config"], errors) if required["routing_config"].exists() else {}
    calibration_stats_path = package / "calibration" / "calibration_stats.json"
    calibration_stats = (
        _load_json(calibration_stats_path, errors)
        if calibration_stats_path.exists()
        else {}
    )

    if isinstance(deploy_config, dict):
        for key in ("classifier_checkpoint", "regression_checkpoint", "routing_config_path", "qc_policy_path"):
            ref_path = _path_from_config(package, deploy_config, key)
            if not ref_path.exists():
                errors.append(f"deploy_config {key} target missing: {ref_path}")
        full_model_raw = str(deploy_config.get("full_model_checkpoint", "") or "")
        if full_model_raw and not (package / full_model_raw).exists():
            errors.append(f"deploy_config full_model_checkpoint target missing: {package / full_model_raw}")

    if isinstance(model_config, dict):
        actual_depth = model_config.get("reg_head_depth")
        if expected_reg_head_depth > 0 and int(actual_depth or -1) != expected_reg_head_depth:
            errors.append(
                f"reg_head_depth expected {expected_reg_head_depth}, got {actual_depth!r}"
            )
        actual_mode = model_config.get("reg_output_mode", "sigmoid")
        if expected_reg_output_mode and str(actual_mode) != str(expected_reg_output_mode):
            errors.append(
                f"reg_output_mode expected {expected_reg_output_mode!r}, got {actual_mode!r}"
            )
        if expected_reg_window_stats and not bool(model_config.get("reg_window_stats", False)):
            errors.append("reg_window_stats expected true, got false/missing")
        actual_branch = model_config.get("reg_response_branch", "none")
        if expected_reg_response_branch and str(actual_branch) != str(expected_reg_response_branch):
            errors.append(
                f"reg_response_branch expected {expected_reg_response_branch!r}, got {actual_branch!r}"
            )
        if expected_reg_tcn_adapter and not bool(model_config.get("reg_tcn_adapter", False)):
            errors.append("reg_tcn_adapter expected true, got false/missing")
        if expected_reg_use_shared_trunk and not bool(model_config.get("reg_use_shared_trunk", False)):
            errors.append("reg_use_shared_trunk expected true, got false/missing")
        if expected_use_reg_ratio_branch and not bool(model_config.get("use_reg_ratio_branch", False)):
            errors.append("use_reg_ratio_branch expected true, got false/missing")

    selected_modes: Dict[str, Any] = {}
    if isinstance(routing_config, dict):
        raw_modes = routing_config.get("selected_modes", {})
        if isinstance(raw_modes, dict):
            selected_modes = raw_modes
        missing_classes = sorted(set(str(i) for i in range(4)) - set(selected_modes.keys()))
        if missing_classes:
            warnings.append(f"routing_config selected_modes missing classes: {missing_classes}")

    full_model_needed = any(
        str(mode).lower() == "full" for mode in selected_modes.values()
    )
    if full_model_needed and not (package / "models" / "full_model.pth").exists():
        errors.append("routing_config uses full mode but models/full_model.pth is missing")

    specialist_needed = [
        int(cls_id)
        for cls_id, mode in selected_modes.items()
        if "specialist" in str(mode).lower()
    ]
    for cls_id in specialist_needed:
        path = package / "models" / "specialists" / f"class_{cls_id}.pth"
        if not path.exists():
            errors.append(f"routing_config uses specialist for class {cls_id}, missing {path}")

    response_refs = _response_ref_summary(calibration_stats)
    if require_response_refs and response_refs["count"] < min_response_ref_classes:
        errors.append(
            "response_refs insufficient: "
            f"found classes {response_refs['classes']}, "
            f"need at least {min_response_ref_classes}"
        )
    if response_refs["missing_loocv_p90"]:
        warnings.append(
            f"response_refs missing loocv_p90 for classes {response_refs['missing_loocv_p90']}"
        )

    try:
        DeployPredictor.from_package(str(package), device="cpu")
    except Exception as exc:
        errors.append(f"Strict deployment policy/model validation failed: {exc}")

    status = "pass" if not errors else "fail"
    return {
        "client_id": client_id,
        "package": str(package),
        "status": status,
        "errors": errors,
        "warnings": warnings,
        "response_refs": response_refs,
        "selected_modes": selected_modes,
        "specialist_classes": specialist_needed,
    }


def validate_packages(
    deploy_package: str,
    client_packages: Sequence[str],
    clients: Sequence[str],
    require_distinct_packages: bool,
    require_response_refs: bool,
    min_response_ref_classes: int,
    expected_reg_head_depth: int,
    expected_reg_output_mode: str = "",
    expected_reg_window_stats: bool = False,
    expected_reg_response_branch: str = "",
    expected_reg_tcn_adapter: bool = False,
    expected_reg_use_shared_trunk: bool = False,
    expected_use_reg_ratio_branch: bool = False,
) -> Dict[str, Any]:
    package_by_client = _resolve_packages(clients, deploy_package, client_packages)
    reports = []
    global_errors: List[str] = []

    if require_distinct_packages:
        paths = [path.resolve() for path in package_by_client.values()]
        if len(set(paths)) != len(paths):
            global_errors.append(
                "--require-distinct-packages was set, but at least two clients share a package"
            )

    for client_id, package in package_by_client.items():
        reports.append(
            _validate_one_package(
                client_id=client_id,
                package=package,
                require_response_refs=require_response_refs,
                min_response_ref_classes=min_response_ref_classes,
                expected_reg_head_depth=expected_reg_head_depth,
                expected_reg_output_mode=expected_reg_output_mode,
                expected_reg_window_stats=expected_reg_window_stats,
                expected_reg_response_branch=expected_reg_response_branch,
                expected_reg_tcn_adapter=expected_reg_tcn_adapter,
                expected_reg_use_shared_trunk=expected_reg_use_shared_trunk,
                expected_use_reg_ratio_branch=expected_use_reg_ratio_branch,
            )
        )

    status = "pass"
    if global_errors or any(report["status"] != "pass" for report in reports):
        status = "fail"

    return {
        "status": status,
        "global_errors": global_errors,
        "packages": reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate deployment packages before QC guardrail evaluation."
    )
    parser.add_argument("--deploy-package", default="")
    parser.add_argument("--client-packages", nargs="*", default=[])
    parser.add_argument("--clients", nargs="+", default=["3", "4", "5"])
    parser.add_argument("--require-distinct-packages", action="store_true")
    parser.add_argument("--allow-missing-response-refs", action="store_true")
    parser.add_argument("--min-response-ref-classes", type=int, default=4)
    parser.add_argument("--expected-reg-head-depth", type=int, default=0)
    parser.add_argument("--expected-reg-output-mode", default="", choices=["", "sigmoid", "linear"])
    parser.add_argument("--expected-reg-window-stats", action="store_true")
    parser.add_argument("--expected-reg-response-branch", default="", choices=["", "none", "dct", "msconv"])
    parser.add_argument("--expected-reg-tcn-adapter", action="store_true")
    parser.add_argument("--expected-reg-use-shared-trunk", action="store_true")
    parser.add_argument("--expected-use-reg-ratio-branch", action="store_true")
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    report = validate_packages(
        deploy_package=args.deploy_package,
        client_packages=args.client_packages,
        clients=args.clients,
        require_distinct_packages=args.require_distinct_packages,
        require_response_refs=not args.allow_missing_response_refs,
        min_response_ref_classes=args.min_response_ref_classes,
        expected_reg_head_depth=args.expected_reg_head_depth,
        expected_reg_output_mode=args.expected_reg_output_mode,
        expected_reg_window_stats=args.expected_reg_window_stats,
        expected_reg_response_branch=args.expected_reg_response_branch,
        expected_reg_tcn_adapter=args.expected_reg_tcn_adapter,
        expected_reg_use_shared_trunk=args.expected_reg_use_shared_trunk,
        expected_use_reg_ratio_branch=args.expected_use_reg_ratio_branch,
    )

    print(f"overall_status={report['status']}")
    for error in report["global_errors"]:
        print(f"GLOBAL ERROR: {error}")
    for item in report["packages"]:
        print(
            f"{item['client_id']}: {item['status']} "
            f"response_refs={item['response_refs']['classes']} package={item['package']}"
        )
        for error in item["errors"]:
            print(f"  ERROR: {error}")
        for warning in item["warnings"]:
            print(f"  WARN: {warning}")

    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Saved JSON: {output}")

    raise SystemExit(0 if report["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
