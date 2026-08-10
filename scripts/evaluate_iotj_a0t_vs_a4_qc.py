"""Apply the frozen canonical-v1 QC policy to A0T and GAPS/A4 outputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gaps_deploy.c5_h8_runtime import FixedH8Policy, SerializedRidge
from run_regression_head_ablation import CLASS_RANGES
from scripts.evaluate_iotj_feature_metadata_ablation import profile_feature_dict
from scripts.finalize_iotj_canonical_v1_evidence import (
    _classification_uncertainty,
    annotate_qc,
    summarize_qc_workpoints,
)
from scripts.run_iotj_canonical_v1_r84 import enriched_oracle_rows


DEFAULT_ROOT = ROOT / "results" / "iotj_canonical_v1_final" / "a0t_vs_a4_regression"
FROZEN_ROOT = ROOT / "results" / "iotj_canonical_v1_final_20260808"
H23_PATH = FROZEN_ROOT / "deployment_package" / "assets" / "h23_qc_auxiliary_policy.json"
R83_PATH = FROZEN_ROOT / "evidence_closure" / "fedridge_ablation" / "r83_models.json"
H23_EXPECTED_SHA256 = "18b6c14373018474807eec2bd19a0b508b75adfbf994b0821a786a11def9c263"
R83_EXPECTED_SHA256 = "b470ce910a6a9ab5c8e3853cd09d43db7e7388df3db10a6d9c9cb07aba57e9f1"
THRESHOLD_EXPECTED_SHA256 = {
    "C3": "a2674e7ae439a6815fc8f2572ea1980348a2f458ad0bf05e283adca50e14b14a",
    "C4": "612c3e4ba1deb4f6ec7da9384dd3a7d32228f19d6e9f972b8d9a9f0ac7137555",
    "C5": "45c7ccfd52043d9322b1982bf862b28eef5281b7f57cfe2d6f73c7de83647713",
}
METHODS = ("A0T", "A4")
TARGETS = ("C3", "C4", "C5")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"FAIL_CLOSED empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fields} for row in rows)


def frozen_qc_asset_paths(target: str) -> dict[str, Path]:
    if target not in TARGETS:
        raise ValueError(target)
    return {
        "h23": H23_PATH,
        "r83": R83_PATH,
        "thresholds": FROZEN_ROOT / "evidence_closure" / "qc" / f"{target}_qc_threshold_lock.csv",
    }


def load_frozen_thresholds(target: str) -> list[dict[str, str]]:
    path = frozen_qc_asset_paths(target)["thresholds"]
    observed = sha256(path)
    if observed != THRESHOLD_EXPECTED_SHA256[target]:
        raise RuntimeError(f"FAIL_CLOSED {target} QC threshold hash differs: {observed}")
    rows = read_csv(path)
    if any(row["target_test_used_for_selection"].lower() != "false" for row in rows):
        raise RuntimeError("FAIL_CLOSED frozen QC lock used target test for selection")
    return rows


def _load_assets() -> tuple[FixedH8Policy, dict[str, dict[int, SerializedRidge]]]:
    if sha256(H23_PATH) != H23_EXPECTED_SHA256 or sha256(R83_PATH) != R83_EXPECTED_SHA256:
        raise RuntimeError("FAIL_CLOSED frozen QC auxiliary asset hash differs")
    h23_payload = json.loads(H23_PATH.read_text(encoding="utf-8"))
    h23 = FixedH8Policy.from_json(h23_payload["source_aug_target_ridge_policy"])
    r83_payload = json.loads(R83_PATH.read_text(encoding="utf-8"))
    r83 = {
        target: {int(key): SerializedRidge.from_json(value) for key, value in payload.items()}
        for target, payload in r83_payload.items()
    }
    return h23, r83


def _enrich_records(
    method: str,
    target: str,
    root: Path,
    h23: FixedH8Policy,
    r83: Mapping[int, SerializedRidge],
) -> list[dict[str, Any]]:
    experiment_id = f"CAN-V1-REG-{method}-{target}-S42"
    raw_rows = read_csv(root / "endpoints" / experiment_id / "test_s_all.csv")
    features_by_id = {
        int(row["sample_index"]): row for row in enriched_oracle_rows(target, "test")
    }
    output: list[dict[str, Any]] = []
    for raw in raw_rows:
        sample_index = int(raw["sample_index"])
        feature_row = features_by_id.get(sample_index)
        if feature_row is None:
            raise RuntimeError(f"FAIL_CLOSED {experiment_id} missing feature row {sample_index}")
        route = int(raw["pred_class"])
        full = feature_row["feature_dict"]
        sensor = profile_feature_dict(full, "M83_SENSOR")
        pred83 = float(r83[route].predict(sensor))
        h1_route = float(raw["H1_federated_source_ridge_ppm"])
        h2_route = float(h23.source_mlp[route].predict(full))
        shared = dict(full)
        shared["route_class"] = route
        h3_route = float(h23.shared_mlp.predict(shared))
        class_range = float(CLASS_RANGES[route])
        item: dict[str, Any] = {**raw}
        item.update(
            {
                "method": method,
                "target": target,
                "pred_83d_ppm": pred83,
                "classification_uncertainty_risk": _classification_uncertainty(raw),
                "regression_disagreement_risk": abs(float(raw["pred_84d_h1_ppm"]) - pred83) / class_range,
                "source_prior_disagreement_risk": (
                    max(h1_route, h2_route, h3_route) - min(h1_route, h2_route, h3_route)
                ) / class_range,
                "true_ppm": float(raw["true_ppm"]),
                "pred_84d_h1_ppm": float(raw["pred_84d_h1_ppm"]),
                "true_class": int(raw["true_class"]),
            }
        )
        output.append(item)
    if len(output) != len(features_by_id):
        raise RuntimeError(f"FAIL_CLOSED {experiment_id} test row count differs")
    return annotate_qc(output, load_frozen_thresholds(target))


def qc_summary_rows(
    method: str, target: str, records: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    return [
        {"method": method, "target": target, **row}
        for row in summarize_qc_workpoints(f"{method}_{target}", records)
    ]


def evaluate_qc(root: Path = DEFAULT_ROOT) -> dict[str, Any]:
    root = root.resolve()
    if not (root / "protocol_manifest.json").is_file():
        raise RuntimeError("FAIL_CLOSED regression comparison is incomplete")
    h23, r83 = _load_assets()
    all_summaries: list[dict[str, Any]] = []
    asset_manifest: dict[str, Any] = {
        "status": "FROZEN_QC_COMPLETE",
        "policy": "exact canonical-v1 A4 equal-mean QC locks reused for both methods",
        "threshold_refit": False,
        "qc_threshold_changed": False,
        "h23_sha256": sha256(H23_PATH),
        "r83_sha256": sha256(R83_PATH),
        "threshold_sha256": {},
    }
    for method in METHODS:
        for target in TARGETS:
            records = _enrich_records(method, target, root, h23, r83[target])
            endpoint = root / "endpoints" / f"CAN-V1-REG-{method}-{target}-S42"
            write_csv(endpoint / "test_qc_frozen.csv", records)
            all_summaries.extend(qc_summary_rows(method, target, records))
            asset_manifest["threshold_sha256"][target] = sha256(
                frozen_qc_asset_paths(target)["thresholds"]
            )
    write_csv(root / "qc_comparison_raw.csv", all_summaries)
    (root / "qc_protocol_manifest.json").write_text(
        json.dumps(asset_manifest, indent=2) + "\n", encoding="utf-8"
    )
    return asset_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    result = evaluate_qc(parser.parse_args().root)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
