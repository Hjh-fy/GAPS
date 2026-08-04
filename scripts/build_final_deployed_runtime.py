"""Build a byte-audited portable package for the frozen final runtime."""

from __future__ import annotations

import argparse, hashlib, json, shutil, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from model import FedGasBaseModel
from gaps_flower.state_fingerprint import checkpoint_provenance, model_parameter_inventory


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build(output: Path, r4_policy: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    project = output / "project"; project.mkdir()
    shutil.copy2(ROOT / "model.py", project / "model.py")
    shutil.copy2(ROOT / "run_regression_head_ablation.py", project / "run_regression_head_ablation.py")
    shutil.copy2(ROOT / "scripts/iotj_b5_c5_bundle_contract.py", project / "iotj_b5_c5_bundle_contract.py")
    shutil.copy2(ROOT / "scripts/benchmark_final_deployed_runtime.py", project / "benchmark_final_deployed_runtime.py")
    shutil.copytree(ROOT / "gaps_deploy", project / "gaps_deploy", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    data = output / "data"; data.mkdir()
    data_root = ROOT / "dataset/client_data_c1234src_c5tgt_2080_timeaware_60_170_window_fullgrid/client_5"
    for name in ("test_features.npy", "test_phase_labels.npy", "test_experiment_info.json"):
        shutil.copy2(data_root / name, data / name)
    assets = output / "assets"; assets.mkdir()
    sources = {
        "classifier": ROOT / "results/iotj_final_classification_le1_20260804/FCL-E4-A4/remote_server/server_round_025_adapted.pth",
        "federated_h1": ROOT / "results/iotj_h1_federated_ridge_equivalence_20260724/federated_h1_manifest.json",
        "regression_models": ROOT / "results/iotj_final_end_to_end_a4_20260804/regression/regression_models.json",
        "r4_policy": r4_policy,
        "qc_threshold_lock": ROOT / "results/iotj_final_end_to_end_a4_20260804/qc/qc_threshold_lock.csv",
    }
    names = {"classifier": "classifier_a4_round25.pth", "federated_h1": "federated_h1.json", "regression_models": "regression_models.json", "r4_policy": "r4_policy.json", "qc_threshold_lock": "qc_threshold_lock.csv"}
    records = {}
    for key, source in sources.items():
        destination = assets / names[key]
        shutil.copy2(source, destination)
        if sha(source) != sha(destination):
            raise RuntimeError(f"asset copy drift: {key}")
        records[key] = {"path": f"assets/{names[key]}", "source_path": str(source.resolve()), "sha256": sha(destination), "bytes": destination.stat().st_size}
    model = FedGasBaseModel(num_classes=4, num_sensors=8, feat_dim=64, encoder_type="tcn", use_cls_proj=True, tcn_norm="instance")
    payload = torch.load(sources["classifier"], map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state"], strict=True)
    identity = checkpoint_provenance(sources["classifier"])
    identity.update(model_parameter_inventory(model))
    manifest = {
        "schema_version": "iotj.final_deployed_runtime.v1",
        "status": "FINAL_DEPLOYED_RUNTIME",
        "frozen_result_baseline_commit": "ceb6c78",
        "model_or_qc_formula_changed": False,
        "classifier_checkpoint_identity": identity,
        "deployed_pipeline": ["A4 classifier", "R84_FED_H1", "final equal-mean QC"],
        "qc_formula": "equal_mean_of_calibration_p95_normalized_components",
        "assets": records,
        "runtime_code": [
            {"path": str(path.relative_to(output)).replace("\\", "/"), "sha256": sha(path), "bytes": path.stat().st_size}
            for path in (
                project / "model.py",
                project / "run_regression_head_ablation.py",
                project / "iotj_b5_c5_bundle_contract.py",
                project / "gaps_deploy/final_a4_runtime.py",
                project / "benchmark_final_deployed_runtime.py",
            )
        ],
    }
    (output / "FINAL_DEPLOYMENT_MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--r4-policy", type=Path, required=True)
    args = parser.parse_args(); build(args.output, args.r4_policy)
