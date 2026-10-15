"""Create a portable, hash-audited Pi package for frozen runtime benchmarks."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def descriptor(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def prepare(output: Path, remote_root: PurePosixPath) -> None:
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    project = output / "project"; project.mkdir()
    shutil.copy2(ROOT / "model.py", project / "model.py")
    shutil.copy2(ROOT / "config.py", project / "config.py")
    shutil.copy2(ROOT / "scripts/benchmark_iotj_final_runtime.py", project / "benchmark_iotj_final_runtime.py")
    shutil.copy2(ROOT / "scripts/probe_iotj_runtime_cold_start.py", project / "probe_iotj_runtime_cold_start.py")
    project_scripts = project / "scripts"; project_scripts.mkdir()
    shutil.copy2(ROOT / "scripts/iotj_b5_c5_bundle_contract.py", project_scripts / "iotj_b5_c5_bundle_contract.py")
    shutil.copy2(ROOT / "scripts/benchmark_iotj_final_runtime.py", project_scripts / "benchmark_iotj_final_runtime.py")
    shutil.copytree(ROOT / "gaps_deploy", project / "gaps_deploy", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

    data = output / "data"; data.mkdir()
    data_root = ROOT / "dataset/client_data_c1234src_c5tgt_2080_timeaware_60_170_window_fullgrid/client_5"
    for name in ("test_features.npy", "test_phase_labels.npy", "test_experiment_info.json"):
        shutil.copy2(data_root / name, data / name)

    v4 = output / "v4_bundle"
    shutil.copytree(ROOT / "results/iotj_b5_c5_deployment_p1_20260722/bundle_candidate", v4)
    v4_reference = v4 / "offline_reference_1360.csv"
    shutil.copy2(ROOT / "results/iotj_b5_c5_deployment_p1_20260722/bundle_inputs/offline_reference_1360.csv", v4_reference)
    v4_manifest = json.loads((v4 / "manifest.json").read_text(encoding="utf-8"))
    v4_manifest["parity_reference"] = {"source_path": str(remote_root / "v4_bundle/offline_reference_1360.csv"), "bytes": v4_reference.stat().st_size, "sha256": sha256_file(v4_reference)}
    write_json(v4 / "manifest.json", v4_manifest)
    v5 = output / "v5_bundle"
    shutil.copytree(ROOT / "results/iotj_b5_c5_runtime_v5_candidate_20260724/runtime_v5", v5)
    v5_lineage = v5 / "calibration_lock.json"
    shutil.copy2(ROOT / "results/iotj_b5_c5_runtime_v5_candidate_20260724/target_ridge/calibration_lock.json", v5_lineage)
    v5_manifest = json.loads((v5 / "bundle_manifest.json").read_text(encoding="utf-8"))
    v5_manifest["calibration_lineage"] = {"path": str(remote_root / "v5_bundle/calibration_lock.json"), "bytes": v5_lineage.stat().st_size, "sha256": sha256_file(v5_lineage)}
    write_json(v5 / "bundle_manifest.json", v5_manifest)
    v5_qc = output / "v5_qc_bundle"
    shutil.copytree(ROOT / "results/iotj_b5_c5_runtime_v5_qc_20260725/runtime_v5_qc_bundle", v5_qc)

    contracts = output / "contracts"; contracts.mkdir()
    v4_contract = json.loads((ROOT / "results/iotj_b5_c5_deployment_p1_20260722/c5_h8_runtime_contract_b5_v4/runtime_contract.json").read_text(encoding="utf-8"))
    v4_contract["bundle_manifest"] = descriptor(v4 / "manifest.json")
    v4_contract["bundle_manifest"]["path"] = str(remote_root / "v4_bundle/manifest.json")
    write_json(contracts / "runtime_v4.json", v4_contract)

    v5_contract = json.loads((ROOT / "results/iotj_b5_c5_runtime_v5_candidate_20260724/runtime_v5/runtime_contract_v5.json").read_text(encoding="utf-8"))
    v5_contract["bundle_manifest"] = descriptor(v5 / "bundle_manifest.json")
    v5_contract["bundle_manifest"]["path"] = str(remote_root / "v5_bundle/bundle_manifest.json")
    write_json(contracts / "runtime_v5_core.json", v5_contract)

    portable_base = v5_qc / "base_runtime_contract.json"
    write_json(portable_base, v5_contract)
    qc_manifest = json.loads((v5_qc / "manifest.json").read_text(encoding="utf-8"))
    qc_manifest["assets"]["base_runtime_contract"] = {"bundle_path": "base_runtime_contract.json", "bytes": portable_base.stat().st_size, "sha256": sha256_file(portable_base)}
    write_json(v5_qc / "manifest.json", qc_manifest)
    qc_contract = json.loads((ROOT / "results/iotj_b5_c5_runtime_v5_qc_20260725/runtime_v5_qc_hc95_contract.json").read_text(encoding="utf-8"))
    qc_contract["bundle_manifest"] = descriptor(v5_qc / "manifest.json")
    qc_contract["bundle_manifest"]["path"] = str(remote_root / "v5_qc_bundle/manifest.json")
    write_json(contracts / "runtime_v5_qc2_hc95.json", qc_contract)

    original_assets = {
        "classifier": ROOT / "results/iotj_b5_c5_runtime_v5_candidate_20260724/runtime_v5/assets/classifier.pth",
        "federated_h1": ROOT / "results/iotj_b5_c5_runtime_v5_candidate_20260724/runtime_v5/assets/federated_h1.json",
        "target_ridge": ROOT / "results/iotj_b5_c5_runtime_v5_candidate_20260724/runtime_v5/assets/target_ridge_105d.json",
        "v5_qc_policy": ROOT / "results/iotj_b5_c5_runtime_v5_qc_20260725/runtime_v5_qc_bundle/qc_policy.json",
        "v4_classifier": ROOT / "results/iotj_b5_c5_deployment_p1_20260722/bundle_candidate/assets/classifier.pth",
        "v4_r4": ROOT / "results/iotj_b5_c5_deployment_p1_20260722/bundle_candidate/assets/r4_policy.json",
        "v4_qc_policy": ROOT / "results/iotj_b5_c5_deployment_p1_20260722/bundle_candidate/assets/qc_risk_policy.json",
    }
    copied_assets = {
        "classifier": v5 / "assets/classifier.pth", "federated_h1": v5 / "assets/federated_h1.json", "target_ridge": v5 / "assets/target_ridge_105d.json",
        "v5_qc_policy": v5_qc / "qc_policy.json", "v4_classifier": v4 / "assets/classifier.pth", "v4_r4": v4 / "assets/r4_policy.json", "v4_qc_policy": v4 / "assets/qc_risk_policy.json",
    }
    records = []
    for name in original_assets:
        original_sha, copied_sha = sha256_file(original_assets[name]), sha256_file(copied_assets[name])
        if original_sha != copied_sha:
            raise RuntimeError(f"portable package asset drift: {name}")
        records.append({"name": name, "original_path": str(original_assets[name]), "package_path": str(copied_assets[name].relative_to(output)), "bytes": copied_assets[name].stat().st_size, "sha256": copied_sha, "status": "PASS"})
    write_json(output / "portable_package_manifest.json", {"schema_version": "iotj.final_benchmark_portable_package.v1", "status": "PASS", "remote_root": str(remote_root), "frozen_asset_byte_identity": records, "contract_relocation_only": True, "model_or_policy_changed": False})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--remote-root", default="/home/gaps/iotj_final_system_benchmark_20260725", type=PurePosixPath)
    args = parser.parse_args()
    prepare(args.output, args.remote_root)


if __name__ == "__main__":
    main()
