"""Build the independent fixed-DA C12->C345 deployment bundle."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


OUTPUT_FIELDS = [
    "gas_class",
    "gas_name",
    "class_prob",
    "base_r3ak16_raw_ppm",
    "routed_pred_ppm",
    "final_ppm",
    "co_corrected_ppm",
    "auto_output_ppm",
    "qc_decision",
    "risk_score",
]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def copy_file(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def load_qc_policy(path: Path) -> dict[str, Any]:
    policies = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            client = str(row["Client"])
            score = str(row["score"])
            policies.append({
                "policy_name": f"fixed_da_r25_{client}_{score}",
                "group": client,
                "scores": [score],
                "thresholds": {score: 1.0},
                "low_ratio": float(row["low_threshold"]),
                "high_ratio": float(row["high_threshold"]),
            })
    if {p["group"] for p in policies} != {"C3", "C4", "C5"}:
        raise ValueError("QC policy must contain exactly C3, C4, and C5")
    return {"policies": policies}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_runtime_source(repo_root: Path, out: Path) -> None:
    runtime = out / "runtime_src"
    for name in ("config.py", "model.py", "utils.py", "federated_dataset.py"):
        copy_file(repo_root / name, runtime / name)
    for name in (
        "__init__.py",
        "inference.py",
        "calibration.py",
        "deploy_config.py",
        "package_contract.py",
        "qc_policy.py",
        "r4a_residual.py",
        "rich_residual.py",
        "final_runtime.py",
    ):
        copy_file(repo_root / "gaps_deploy" / name, runtime / "gaps_deploy" / name)
    for name in ("__init__.py", "regression_task.py"):
        copy_file(repo_root / "gaps_flower" / name, runtime / "gaps_flower" / name)


def run_inference_text() -> str:
    return '''from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "runtime_src"))
if "--bundle" not in sys.argv:
    sys.argv[1:1] = ["--bundle", str(ROOT)]

from gaps_deploy.final_runtime import main

if __name__ == "__main__":
    main()
'''


def main() -> None:
    parser = argparse.ArgumentParser(description="Build final fixed-DA deployment bundle")
    parser.add_argument("--output-dir", default="results/deployment_fixed_da_c12_c345_final")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--classifier", default="results/timeaware_2080_c12src_c345tgt_fixed_da_r25/expB_strong_da_fixed_da_r25/server_latest_adapted.pth")
    parser.add_argument("--semantic-protos", default="results/timeaware_2080_c12src_c345tgt_fixed_da_r25/expB_strong_da_fixed_da_r25/semantic_protos_latest.json")
    parser.add_argument("--regression", default="results/timeaware_2080_c12src_c345tgt_fixed_da_r25_r3ak16_auto_v2_eval/fixed_da_r25/regression_fedavg_global.pt")
    parser.add_argument("--packages", default="results/timeaware_2080_c12src_c345tgt_fixed_da_r25_r3ak16_auto_v2_eval/fixed_da_r25/auto_v2_packages/packages")
    parser.add_argument("--qc-policies", default="results/timeaware_2080_c12src_c345tgt_fixed_da_r25_r3ak16_auto_v2_eval/fixed_da_r25/qc_selected_policies.csv")
    parser.add_argument("--co-params", default="results/timeaware_2080_flower_expB_r3ak16_auto_v2_eval/co_specific_correction/deployment_candidate/co_guarded_ridge_params.json")
    parser.add_argument("--rich-residual-artifact", default="results/deployment_candidates_20260624/c12_c345_a1_rich_residual_plus_c4_rescue.json")
    parser.add_argument("--norm-stats", default="dataset/client_data_c12src_c345tgt_2080_timeaware_60_170_window_fullgrid/norm_stats.npz")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    out = (repo_root / args.output_dir).resolve()
    if out.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists: {out}; pass --overwrite to replace it")
        if repo_root not in out.parents or out == repo_root:
            raise ValueError(f"Refusing to remove unsafe output path: {out}")
        shutil.rmtree(out)
    out.mkdir(parents=True)

    classifier = repo_root / args.classifier
    semantic = repo_root / args.semantic_protos
    regression = repo_root / args.regression
    packages = repo_root / args.packages
    qc_csv = repo_root / args.qc_policies
    co_params_src = repo_root / args.co_params
    rich_residual_src = repo_root / args.rich_residual_artifact if args.rich_residual_artifact else None
    norm_stats = repo_root / args.norm_stats

    copy_file(classifier, out / "classifier.pth")
    copy_file(semantic, out / "semantic_protos_latest.json")
    copy_file(regression, out / "regression_fedavg_global.pt")
    copy_file(norm_stats, out / "norm_stats.npz")

    qc_policy = load_qc_policy(qc_csv)
    write_json(out / "qc_policy.json", qc_policy)

    client_packages = {}
    for client_num in (3, 4, 5):
        client = f"C{client_num}"
        src = packages / client
        dst_name = f"client_{client_num}_auto_v2_package"
        dst = out / dst_name
        shutil.copytree(src, dst)
        write_json(dst / "qc" / "selected_policy.json", qc_policy)
        client_packages[client] = dst_name

    co_params = json.loads(co_params_src.read_text(encoding="utf-8"))
    co_params["source_enabled"] = bool(co_params.get("enabled", False))
    co_params["enabled"] = False
    co_params["compatibility_note"] = (
        "Parameters were fitted on the prior frozen C12->C345 mainline and are disabled. "
        "On fixed-DA replay they triggered 131 review CO rows and worsened Coverage+Review RMSE."
    )
    write_json(out / "co_guarded_ridge_params.json", co_params)
    write_json(out / "co_specific_correction_manifest.json", {
        "name": co_params.get("name"),
        "status": "disabled: incompatible with fixed-DA residual distribution",
        "enabled": False,
        "gate_fields": ["client_id", "pred_class", "routed_pred_ppm", "qc_decision"],
        "forbidden_gate_fields": ["true_ppm", "true_class", "oracle_route", "test_label"],
        "output_field": "co_corrected_ppm",
        "does_not_overwrite": "final_ppm",
        "fixed_da_replay": {
            "rows": 5400,
            "would_trigger_count_if_enabled": 131,
            "accepted_trigger_count": 0,
            "non_co_trigger_count": 0,
            "coverage_review_rmse_before": 11.82774376508843,
            "coverage_review_rmse_after": 13.00028053966749,
            "trigger_subset_rmse_before": 19.4816914216467,
            "trigger_subset_rmse_after": 36.012365167731,
        },
        "source_params": args.co_params,
    })
    if rich_residual_src:
        copy_file(rich_residual_src, out / "rich_residual_candidate.json")

    model_config = json.loads(
        (out / client_packages["C3"] / "models" / "model_config.json").read_text(encoding="utf-8")
    )
    write_json(out / "model_config.json", model_config)
    write_json(out / "label_mapping.json", {
        "0": "Ethanol",
        "1": "CO",
        "2": "Ethylene",
        "3": "Methane",
    })
    runtime_config = {
        "bundle_version": "fixed-da-c12-c345-r25-v1",
        "client_packages": client_packages,
        "norm_stats": "norm_stats.npz",
        "normalization": {
            "enabled": False,
            "reason": "The frozen time-aware features and training/runtime package use normalize=False.",
        },
        "co_correction_params": "co_guarded_ridge_params.json",
        "rich_residual_artifact": "rich_residual_candidate.json" if rich_residual_src else "",
        "output_fields": OUTPUT_FIELDS,
        "input_shape": [100, 8],
        "device_default": "cpu",
    }
    write_json(out / "runtime_config.json", runtime_config)
    (out / "run_inference.py").write_text(run_inference_text(), encoding="utf-8")
    copy_runtime_source(repo_root, out)

    files = []
    for path in sorted(p for p in out.rglob("*") if p.is_file() and p.name != "manifest.json"):
        files.append({
            "path": path.relative_to(out).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    write_json(out / "manifest.json", {
        "name": "GAPS fixed-DA C12->C345 final deployment bundle",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "classifier_round": 25,
        "source_clients": ["C1", "C2"],
        "target_clients": ["C3", "C4", "C5"],
        "input_shape": [100, 8],
        "output_fields": OUTPUT_FIELDS,
        "runtime_dependencies": ["python>=3.10", "numpy", "torch", "scikit-learn", "matplotlib"],
        "files": files,
    })
    print(json.dumps({"output_dir": str(out), "files": len(files)}, indent=2))


if __name__ == "__main__":
    main()
