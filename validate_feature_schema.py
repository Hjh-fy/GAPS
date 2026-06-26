"""Validate dataset feature schema and deployment runtime contract."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_BUNDLE = Path("results/deployment_h8_formal_c4_rescue_candidate_20260625")
DEFAULT_DATA_ROOT = Path("dataset/client_data_c12src_c345tgt_2080_timeaware_60_170_window_fullgrid")
DEFAULT_OUT_DIR = Path("results/feature_schema_validation_h8_formal_c4_rescue_20260626")
SPLITS = ["train", "calibration", "test"]
EXPECTED_SHAPE = (100, 8)


def client_num(client: str | int) -> int:
    return int(str(client).upper().replace("CLIENT_", "").replace("C", ""))


def client_name(client: str | int) -> str:
    return f"C{client_num(client)}"


def parse_clients(text: str) -> list[str]:
    return [client_name(item.strip()) for item in text.split(",") if item.strip()]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def check_record(checks: list[dict[str, Any]], name: str, status: str, message: str = "", **extra: Any) -> None:
    row = {"name": name, "status": status, "message": message}
    row.update(extra)
    checks.append(row)


def finite_array(path: Path, keys: list[str], checks: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if not path.exists():
        check_record(checks, "norm_stats_exists", "fail", f"Missing norm stats: {path}")
        return out
    try:
        data = np.load(path)
    except Exception as exc:
        check_record(checks, "norm_stats_load", "fail", str(exc))
        return out
    for key in keys:
        if key not in data.files:
            check_record(checks, f"norm_stats_{key}", "fail", f"Missing key {key}")
            continue
        arr = np.asarray(data[key])
        finite = bool(np.isfinite(arr).all())
        broadcast_ok = True
        try:
            np.zeros((1, *EXPECTED_SHAPE), dtype=np.float32) - arr
        except Exception:
            broadcast_ok = False
        status = "pass" if finite and broadcast_ok else "fail"
        check_record(
            checks,
            f"norm_stats_{key}",
            status,
            "",
            shape=list(arr.shape),
            finite=finite,
            broadcast_to_batch=bool(broadcast_ok),
        )
        out[key] = {"shape": list(arr.shape), "finite": finite, "broadcast_to_batch": bool(broadcast_ok)}
    return out


def validate_dataset(data_root: Path, clients: list[str], checks: list[dict[str, Any]], warnings: list[str]) -> None:
    for client in clients:
        cdir = data_root / f"client_{client_num(client)}"
        if not cdir.exists():
            check_record(checks, f"{client}_dir", "fail", f"Missing client directory: {cdir}")
            continue
        for split in SPLITS:
            paths = {
                "features": cdir / f"{split}_features.npy",
                "classification": cdir / f"{split}_classification_labels.npy",
                "regression": cdir / f"{split}_regression_labels.npy",
                "phase": cdir / f"{split}_phase_labels.npy",
                "metadata": cdir / f"{split}_experiment_info.json",
            }
            missing = [name for name, path in paths.items() if name != "metadata" and not path.exists()]
            if missing:
                check_record(checks, f"{client}_{split}_files", "fail", f"Missing files: {missing}")
                continue
            features = np.load(paths["features"], mmap_mode="r")
            cls = np.load(paths["classification"], mmap_mode="r")
            reg = np.load(paths["regression"], mmap_mode="r")
            phase = np.load(paths["phase"], mmap_mode="r")
            n = int(features.shape[0])
            shape_ok = features.ndim == 3 and tuple(features.shape[1:]) == EXPECTED_SHAPE
            check_record(
                checks,
                f"{client}_{split}_feature_shape",
                "pass" if shape_ok else "fail",
                "",
                shape=list(features.shape),
                expected=[None, *EXPECTED_SHAPE],
            )
            cls_ok = int(np.asarray(cls).reshape(-1).shape[0]) == n
            phase_ok = int(np.asarray(phase).reshape(-1).shape[0]) == n
            reg_ok = reg.ndim == 2 and reg.shape[0] == n and reg.shape[1] == 4
            check_record(checks, f"{client}_{split}_class_label_length", "pass" if cls_ok else "fail", "", N=n, label_N=int(np.asarray(cls).reshape(-1).shape[0]))
            check_record(checks, f"{client}_{split}_phase_label_length", "pass" if phase_ok else "fail", "", N=n, label_N=int(np.asarray(phase).reshape(-1).shape[0]))
            check_record(checks, f"{client}_{split}_regression_label_shape", "pass" if reg_ok else "fail", "", shape=list(reg.shape))
            if paths["metadata"].exists():
                meta = load_json(paths["metadata"])
                meta_ok = isinstance(meta, list) and len(meta) == n
                check_record(checks, f"{client}_{split}_metadata_length", "pass" if meta_ok else "fail", "", N=n, metadata_N=len(meta) if isinstance(meta, list) else None)
                if isinstance(meta, list) and meta:
                    response_values = {str(item.get("response_phase", "unknown")) for item in meta if isinstance(item, dict)}
                    phase_values = {str(item.get("phase_label", "unknown")) for item in meta if isinstance(item, dict)}
                    allowed_response = {"main_response", "recovery", "pre_response", "unknown"}
                    allowed_phase = {"early", "middle", "late", "unknown"}
                    bad_response = sorted(response_values - allowed_response)
                    bad_phase = sorted(phase_values - allowed_phase)
                    if bad_response:
                        warnings.append(f"{client} {split}: unexpected response_phase values {bad_response}")
                    if bad_phase:
                        warnings.append(f"{client} {split}: unexpected phase_label values {bad_phase}")
            else:
                warnings.append(f"{client} {split}: metadata JSON missing")
                check_record(checks, f"{client}_{split}_metadata_length", "warn", "Metadata JSON missing")


def validate_runtime(bundle: Path, checks: list[dict[str, Any]], warnings: list[str]) -> None:
    runtime_path = bundle / "runtime_config.json"
    if not runtime_path.exists():
        check_record(checks, "runtime_config_exists", "fail", f"Missing runtime config: {runtime_path}")
        return
    runtime = load_json(runtime_path)
    check_record(checks, "runtime_config_exists", "pass", str(runtime_path))
    input_shape = tuple(runtime.get("input_shape", []))
    check_record(checks, "runtime_input_shape", "pass" if input_shape == EXPECTED_SHAPE else "fail", "", input_shape=list(input_shape), expected=list(EXPECTED_SHAPE))
    norm_rel = runtime.get("norm_stats", "")
    norm_path = bundle / norm_rel if norm_rel else None
    if norm_path:
        finite_array(norm_path, ["mean", "std"], checks)
    else:
        check_record(checks, "norm_stats_configured", "fail", "runtime_config.norm_stats is missing")
    normalization = runtime.get("normalization", {})
    if not isinstance(normalization, dict):
        check_record(checks, "runtime_normalization", "fail", "normalization must be a dict")
    else:
        enabled = bool(normalization.get("enabled", False))
        check_record(checks, "runtime_normalization", "pass", "", enabled=enabled, reason=normalization.get("reason", ""))
    client_packages = runtime.get("client_packages", {})
    for client, rel in client_packages.items():
        exists = (bundle / str(rel)).exists()
        check_record(checks, f"runtime_client_package_{client}", "pass" if exists else "fail", str(rel))
    rich_rel = runtime.get("rich_residual_artifact", "")
    if rich_rel:
        rich_path = bundle / str(rich_rel)
        rich_exists = rich_path.exists()
        check_record(checks, "rich_residual_artifact_exists", "pass" if rich_exists else "fail", str(rich_rel))
        if rich_exists:
            artifact = load_json(rich_path)
            route = artifact.get("route_rescue_policy", {})
            schema = str(route.get("schema", ""))
            add_gates = route.get("additional_gates", [])
            v2_ok = schema.endswith(".v2") or "v2" in schema
            check_record(checks, "route_rescue_schema_v2", "pass" if v2_ok else "warn", schema)
            if add_gates:
                margin_ok = all("max_conf_margin" in gate for gate in add_gates if isinstance(gate, dict))
                check_record(checks, "route_rescue_max_conf_margin", "pass" if margin_ok else "fail", "", additional_gate_N=len(add_gates))
            else:
                warnings.append("rich residual artifact has no additional route-rescue gates")
    else:
        warnings.append("runtime_config.rich_residual_artifact is not configured")


def status_from_checks(checks: list[dict[str, Any]]) -> str:
    return "fail" if any(row["status"] == "fail" for row in checks) else "pass"


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Feature Schema Validation",
        "",
        f"- bundle: `{summary['bundle']}`",
        f"- data_root: `{summary['data_root']}`",
        f"- clients: `{', '.join(summary['clients'])}`",
        f"- status: **{summary['status']}**",
        "",
        "## Checks",
        "",
        "| check | status | message |",
        "| --- | --- | --- |",
    ]
    for row in summary["checks"]:
        lines.append(f"| {row['name']} | {row['status']} | {str(row.get('message', '')).replace('|', '/')} |")
    if summary["warnings"]:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in summary["warnings"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--clients", default="C3,C4,C5")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    checks: list[dict[str, Any]] = []
    warnings: list[str] = []
    clients = parse_clients(args.clients)
    validate_dataset(args.data_root, clients, checks, warnings)
    validate_runtime(args.bundle, checks, warnings)
    summary = {
        "bundle": str(args.bundle),
        "data_root": str(args.data_root),
        "clients": clients,
        "status": status_from_checks(checks),
        "checks": checks,
        "warnings": warnings,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "feature_schema_validation.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_markdown(args.output_dir / "feature_schema_validation.md", summary)
    print(json.dumps({"output_dir": str(args.output_dir), "status": summary["status"]}, indent=2))
    if summary["status"] == "fail":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
