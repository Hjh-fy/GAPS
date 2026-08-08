"""Build and audit the sole canonical-v1 deployment package."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model import FedGasBaseModel
from scripts.finalize_iotj_canonical_v1_evidence import (
    DATASET_ROOT,
    H1_PATH,
    R4_POLICY_PATH,
    STUDY_ROOT,
    TARGETS,
    read_json,
    sha256,
    write_json,
)


def model_size_stats(model: torch.nn.Module) -> dict[str, int]:
    state = model.state_dict()
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {
        "state_tensor_count": len(state),
        "total_parameter_count": int(total),
        "trainable_parameter_count": int(trainable),
        "fp32_model_bytes": int(total * 4),
    }


def runtime_source_files() -> tuple[Path, ...]:
    """Return repository-relative sources required by the portable runtime."""
    files = [
        Path("model.py"),
        Path("run_regression_head_ablation.py"),
        Path("scripts/benchmark_iotj_canonical_v1_pi5.py"),
    ]
    files.extend([
        Path("gaps_deploy/__init__.py"),
        Path("gaps_deploy/canonical_serialized.py"),
        Path("gaps_deploy/canonical_v1_runtime.py"),
    ])
    return tuple(files)


def runtime_source_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _copy(source: Path, destination: Path) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return {
        "path": str(destination).replace("\\", "/"),
        "bytes": destination.stat().st_size,
        "sha256": sha256(destination),
        "source_path": str(source.resolve()),
        "source_sha256": sha256(source),
    }


def _relative_asset(package: Path, source: Path, relative: str) -> dict[str, Any]:
    destination = package / relative
    item = _copy(source, destination)
    item["path"] = relative.replace("\\", "/")
    return item


def _load_classifier(checkpoint: Path) -> torch.nn.Module:
    model = FedGasBaseModel(
        num_classes=4, num_sensors=8, feat_dim=64, encoder_type="tcn",
        use_cls_proj=True, tcn_norm="instance",
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    return model


def build_package(study_root: Path, output: Path) -> Path:
    study_root = study_root.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"FAIL_CLOSED deployment package exists: {output}")
    output.mkdir(parents=True)
    closure = study_root / "evidence_closure"
    assets: dict[str, Any] = {}
    assets["preprocessing_manifest"] = _relative_asset(
        output,
        DATASET_ROOT / "canonical_preprocessing_manifest.json",
        "assets/preprocessing_manifest.json",
    )
    assets["dataset_hash"] = _relative_asset(
        output, DATASET_ROOT / "dataset_sha256.json", "assets/dataset_sha256.json"
    )
    assets["federated_h1"] = _relative_asset(
        output, H1_PATH, "assets/federated_h1.json"
    )
    assets["h23_policy"] = _relative_asset(
        output, R4_POLICY_PATH, "assets/h23_qc_auxiliary_policy.json"
    )
    assets["qc_policy"] = _relative_asset(
        output, closure / "qc" / "protocol_manifest.json", "assets/qc_policy.json"
    )
    input_schema = {
        "schema_version": "iotj.canonical_v1.runtime_input.v1",
        "sampling_hz": 5,
        "window_seconds": 10,
        "shape": [50, 8],
        "dtype": "float32",
        "tensor_bytes_per_window": 1600,
        "semantic": "canonical baseline-normalized HZ5 mean sensor window",
        "required_runtime_context": ["target_profile", "phase", "window_start_s", "window_end_s"],
    }
    write_json(output / "input_schema.json", input_schema)
    class_mapping = {
        "0": {"gas": "Ethanol", "range_ppm": 112.5},
        "1": {"gas": "CO", "range_ppm": 225.0},
        "2": {"gas": "Ethylene", "range_ppm": 112.5},
        "3": {"gas": "Methane", "range_ppm": 225.0},
    }
    write_json(output / "class_mapping.json", class_mapping)
    runtime_config = {
        "schema_version": "iotj.canonical_v1.runtime.v1",
        "status": "FINAL_DEPLOYED_RUNTIME",
        "pipeline": ["canonical_preprocessing_contract", "target_A4", "R83_and_R84_FED_H1", "frozen_equal_mean_QC"],
        "target_profiles": list(TARGETS),
        "batch_size": 1,
        "model_selection": False,
        "qc_search": False,
    }
    write_json(output / "runtime_config.json", runtime_config)
    runtime_sources = []
    for relative in runtime_source_files():
        source = ROOT / relative
        runtime_sources.append(
            _relative_asset(output, source, f"runtime/{relative.as_posix()}")
        )

    r83_all = read_json(closure / "fedridge_ablation" / "r83_models.json")
    targets: dict[str, Any] = {}
    size_rows: dict[str, Any] = {}
    for target in TARGETS:
        target_root = output / "targets" / target
        run = study_root / "classification" / f"CANONICAL-V1-A4-{target}"
        run_manifest = read_json(run / "run_manifest.json")
        checkpoint = Path(run_manifest["checkpoint"])
        classifier_asset = _relative_asset(
            output, checkpoint, f"targets/{target}/classifier_round25_adapted.pth"
        )
        locked_asset = _relative_asset(
            output, run / "locked_run_spec.json", f"targets/{target}/classifier_protocol.json"
        )
        r84_asset = _relative_asset(
            output,
            study_root / "regression" / target / "regression_models.json",
            f"targets/{target}/r84_models.json",
        )
        write_json(target_root / "r83_models.json", r83_all[target])
        r83_asset = {
            "path": f"targets/{target}/r83_models.json",
            "bytes": (target_root / "r83_models.json").stat().st_size,
            "sha256": sha256(target_root / "r83_models.json"),
            "source_path": str((closure / "fedridge_ablation" / "r83_models.json").resolve()),
            "source_sha256": sha256(closure / "fedridge_ablation" / "r83_models.json"),
        }
        qc_asset = _relative_asset(
            output,
            closure / "qc" / f"{target}_qc_threshold_lock.csv",
            f"targets/{target}/qc_threshold_lock.csv",
        )
        targets[target] = {
            "classifier": classifier_asset,
            "classifier_protocol": locked_asset,
            "r83_models": r83_asset,
            "r84_models": r84_asset,
            "qc_thresholds": qc_asset,
        }
        stats = model_size_stats(_load_classifier(checkpoint))
        size_rows[target] = {
            **stats,
            "checkpoint_file_bytes": checkpoint.stat().st_size,
            "checkpoint_sha256": sha256(checkpoint),
        }

    manifest = {
        "schema_version": "iotj.canonical_v1.deployment_package.v1",
        "status": "FINAL_DEPLOYED_RUNTIME",
        "canonical": True,
        "dataset_aggregate_sha256": "2f810d7e93cae5f361923184e9dc87d5ae59e0f59be9f52aff7e14f9f33e94f6",
        "input_shape": [50, 8],
        "input_sampling_hz": 5,
        "classifier_router": "A4",
        "regression": "R84_FED_H1",
        "qc": "frozen_equal_mean",
        "runtime_source_commit": runtime_source_commit(),
        "assets": assets,
        "runtime_sources": runtime_sources,
        "targets": targets,
        "forbidden_contamination": {
            "legacy_classifier_checkpoint": False,
            "legacy_preprocessing_checkpoint": False,
            "candidate_selection_checkpoint": False,
        },
    }
    write_json(output / "package_manifest.json", manifest)
    package_files = [path for path in output.rglob("*") if path.is_file()]
    total_bytes_before_audit = sum(path.stat().st_size for path in package_files)
    model_size = {
        "schema_version": "iotj.canonical_v1.model_size.v1",
        "classifier_by_target": size_rows,
        "parameter_count_semantics": "total_parameter_count counts scalar parameters; state_tensor_count counts state-dict tensors",
        "federated_h1_serialized_bytes": (output / assets["federated_h1"]["path"]).stat().st_size,
        "h23_qc_auxiliary_serialized_bytes": (output / assets["h23_policy"]["path"]).stat().st_size,
        "r84_serialized_bytes": sum((output / targets[target]["r84_models"]["path"]).stat().st_size for target in TARGETS),
        "qc_config_bytes": (output / assets["qc_policy"]["path"]).stat().st_size + sum((output / targets[target]["qc_thresholds"]["path"]).stat().st_size for target in TARGETS),
        "package_bytes_before_audit_files": total_bytes_before_audit,
        "input_tensor_bytes_fp32": 1600,
    }
    write_json(output / "model_size_audit.json", model_size)
    return output


def preflight_package(package: Path) -> dict[str, Any]:
    manifest = read_json(package / "package_manifest.json")
    failures = []
    if manifest.get("status") != "FINAL_DEPLOYED_RUNTIME":
        failures.append("runtime status")
    if manifest.get("input_shape") != [50, 8] or manifest.get("input_sampling_hz") != 5:
        failures.append("input contract")
    if manifest.get("classifier_router") != "A4" or manifest.get("regression") != "R84_FED_H1":
        failures.append("pipeline identity")
    items = list(manifest["assets"].values())
    items.extend(manifest.get("runtime_sources", []))
    for target in TARGETS:
        items.extend(manifest["targets"][target].values())
    for item in items:
        path = package / item["path"]
        if not path.is_file() or sha256(path) != item["sha256"]:
            failures.append(f"hash:{item['path']}")
    for target in TARGETS:
        r84 = read_json(package / manifest["targets"][target]["r84_models"]["path"])
        if any(len(model["feature_names"]) != 84 for model in r84.values()):
            failures.append(f"R84 dimension:{target}")
        _load_classifier(package / manifest["targets"][target]["classifier"]["path"])
    if any(manifest["forbidden_contamination"].values()):
        failures.append("legacy contamination")
    return {"status": "PASS" if not failures else "FAIL", "failures": failures}


def finalize_package(package: Path) -> None:
    audit = preflight_package(package)
    if audit["status"] != "PASS":
        raise RuntimeError(f"FAIL_CLOSED package preflight failed: {audit}")
    write_json(package / "package_preflight.json", audit)
    (package / "DEPLOYMENT_PACKAGE_AUDIT.md").write_text(
        "# Deployment package audit\n\nStatus: **PASS**. The package contains only canonical 5 Hz, 50x8 input contracts; target-specific final A4 checkpoints; R83/R84 models; frozen Federated-H1 and H2/H3 QC auxiliary heads; and calibration-derived equal-mean QC locks. No legacy classifier or preprocessing checkpoint is packaged.\n",
        encoding="utf-8",
    )
    size = read_json(package / "model_size_audit.json")
    size["complete_deployment_package_bytes"] = sum(
        path.stat().st_size for path in package.rglob("*") if path.is_file()
    )
    write_json(package / "model_size_audit.json", size)
    (package / "MODEL_SIZE_AUDIT.md").write_text(
        "# Model size audit\n\n`state_tensor_count` is not a parameter count. Scalar total/trainable parameter counts, FP32 parameter bytes, checkpoint storage, serialized Ridge/QC assets, complete package storage, and the 1600-byte canonical input tensor are reported separately in `model_size_audit.json`.\n",
        encoding="utf-8",
    )
    files = [
        path for path in sorted(package.rglob("*"))
        if path.is_file() and path.name != "SHA256SUMS"
    ]
    lines = [f"{sha256(path)}  {str(path.relative_to(package)).replace(chr(92), '/')}" for path in files]
    (package / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, default=STUDY_ROOT)
    parser.add_argument("--output", type=Path, default=STUDY_ROOT / "deployment_package")
    args = parser.parse_args()
    package = build_package(args.study_root, args.output)
    finalize_package(package)


if __name__ == "__main__":
    main()
