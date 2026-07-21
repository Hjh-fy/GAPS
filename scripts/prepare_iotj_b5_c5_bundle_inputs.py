"""Prepare explicit, fail-closed inputs for the B5/C5 deployment bundle.

This script only binds already-fitted B5/C5 artifacts.  It neither trains a
classifier nor refits any regression/QC component.  The generated parity CSV
is deliberately kept outside the runtime bundle by the packager.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


GAS_NAMES = ("Ethanol", "CO", "Ethylene", "Methane")
CONCENTRATION_RANGES = {
    0: {"min_ppm": 12.5, "max_ppm": 125.0},
    1: {"min_ppm": 25.0, "max_ppm": 250.0},
    2: {"min_ppm": 12.5, "max_ppm": 125.0},
    3: {"min_ppm": 25.0, "max_ppm": 250.0},
}
PARITY_FIELDS = ("sample_index", "pred_class", "selected_profile", "qc_decision", "final_ppm")
EXPECTED_PARITY_ROWS = 1360
SELECTED_PROFILE = "b5_c5_r4_h23_hc90"
R4_FINAL_FIELD = "target_ridge_plus_source_preds_ppm"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def _classifier_sha(policy: Mapping[str, Any], label: str) -> str:
    value = policy.get("classifier_sha256")
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{label} has no valid classifier_sha256")
    return value


def _require_empty_or_new(output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite prepared bundle inputs: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def _validated_hc90_rows(source: Path) -> list[dict[str, str]]:
    """Read a valid R4-bound HC90 stream before any candidate output is created."""
    with Path(source).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"sample_index", "pred_class", "qc_decision", "qc_workpoint", R4_FINAL_FIELD}
        missing = sorted(required - set(reader.fieldnames or ()))
        if missing:
            raise ValueError(f"HC90 source missing fields: {missing}")
        rows = list(reader)
    if len(rows) != EXPECTED_PARITY_ROWS:
        raise ValueError(f"expected {EXPECTED_PARITY_ROWS} HC90 rows; got {len(rows)}")
    indexes = [int(str(row["sample_index"])) for row in rows]
    if len(set(indexes)) != EXPECTED_PARITY_ROWS:
        raise ValueError("HC90 source has duplicate sample_index")
    if any(str(row["qc_workpoint"]) != "HC90" for row in rows):
        raise ValueError("parity source must be the HC90 workpoint")
    for row in rows:
        try:
            r4_final = float(str(row[R4_FINAL_FIELD]))
        except ValueError as error:
            raise ValueError("parity source contains non-numeric final ppm") from error
        if not math.isfinite(r4_final):
            raise ValueError("parity source contains non-finite final ppm")
    return rows


def write_parity_reference(source: Path, output: Path) -> None:
    """Reduce the validated frozen HC90 stream to the five runtime parity fields."""
    rows = _validated_hc90_rows(source)
    with Path(output).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PARITY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "sample_index": int(str(row["sample_index"])),
                    "pred_class": int(str(row["pred_class"])),
                    "selected_profile": SELECTED_PROFILE,
                    "qc_decision": str(row["qc_decision"]),
                    "final_ppm": str(row[R4_FINAL_FIELD]),
                }
            )


def prepare_bundle_inputs(
    *,
    classifier: Path,
    r4_policy: Path,
    h23_reference: Path,
    qc_dir: Path,
    hc90_reference: Path,
    output_dir: Path,
    frozen_commit: str,
    source_archive_sha256: str,
) -> dict[str, str]:
    """Emit immutable metadata and an explicit input map for one B5 candidate."""
    classifier = Path(classifier).resolve()
    r4_policy = Path(r4_policy).resolve()
    h23_reference = Path(h23_reference).resolve()
    qc_dir = Path(qc_dir).resolve()
    hc90_reference = Path(hc90_reference).resolve()
    output_dir = Path(output_dir).resolve()
    for label, path in {
        "classifier": classifier,
        "r4_policy": r4_policy,
        "h23_reference": h23_reference,
        "hc90_reference": hc90_reference,
    }.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing {label}: {path}")
    if len(frozen_commit) != 40 or len(source_archive_sha256) != 64:
        raise ValueError("frozen_commit and source_archive_sha256 must be full SHA-1/SHA-256 values")
    classifier_sha = _sha256(classifier)
    r4_payload = _read_json(r4_policy)
    h23_payload = _read_json(h23_reference)
    if _classifier_sha(r4_payload, "r4_policy") != classifier_sha:
        raise ValueError("r4_policy classifier hash does not bind the supplied classifier")
    if _classifier_sha(h23_payload, "h23_reference") != classifier_sha:
        raise ValueError("h23_reference classifier hash does not bind the supplied classifier")
    class_ids = r4_payload.get("source_aug_target_ridge_policy", {}).get("switch_rule", {}).get("class_ids")
    if sorted(class_ids or ()) != [0, 1, 2, 3]:
        raise ValueError("R4 policy must explicitly route all four gas classes")
    qc_manifest = _read_json(qc_dir / "manifest.json")
    if qc_manifest.get("pred_key") != R4_FINAL_FIELD:
        raise ValueError("QC manifest does not bind HC90 decisions to the B5 R4 prediction")
    if qc_manifest.get("secondary_workpoint") != "HC90":
        raise ValueError("QC manifest does not declare HC90 as the secondary workpoint")
    _validated_hc90_rows(hc90_reference)

    _require_empty_or_new(output_dir)
    feature_schema = output_dir / "feature_schema.json"
    class_map = output_dir / "class_map.json"
    normalization = output_dir / "normalization.json"
    parity_reference = output_dir / "offline_reference_1360.csv"
    _write_json(
        feature_schema,
        {
            "schema_version": "iotj.b5_c5_feature_schema.v1",
            "classifier": {
                "architecture": "FedGasBaseModel",
                "encoder": "tcn",
                "tcn_norm": "instance",
                "num_classes": 4,
                "input_shape": [100, 8],
                "feature_dimension": 64,
                "parameter_count": 22765,
                "checkpoint_sha256": classifier_sha,
                "strict_load_verified": True,
            },
            "window": {"layout": "time_sensor", "shape": [100, 8], "dtype": "float32"},
            "r4_feature_names": r4_payload["source_aug_target_ridge_policy"]["feature_names"],
            "provenance": {
                "frozen_commit": frozen_commit,
                "source_archive_sha256": source_archive_sha256,
                "source_paths": ["config.py", "model.py", "utils.py", "gaps_flower/task.py"],
            },
        },
    )
    _write_json(
        class_map,
        {
            "schema_version": "iotj.b5_c5_class_map.v1",
            "classes": [
                {"class_id": class_id, "gas": name, **CONCENTRATION_RANGES[class_id]}
                for class_id, name in enumerate(GAS_NAMES)
            ],
        },
    )
    _write_json(
        normalization,
        {
            "schema_version": "iotj.b5_c5_normalization.v1",
            "input_normalization": {
                "enabled": False,
                "reason": "frozen Flower loader uses normalize=False",
                "dtype": "float32",
            },
            "architecture_note": "TCN instance normalization is part of the classifier checkpoint, not external input normalization.",
        },
    )
    write_parity_reference(hc90_reference, parity_reference)
    qc_paths = {
        "qc_risk_policy": qc_dir / "risk_policy.json",
        "qc_component_calibrator": qc_dir / "component_calibrator.json",
        "qc_feature_reference": qc_dir / "feature_reference.json",
        "qc_risk_selection": qc_dir / "risk_selection.json",
    }
    for key, path in qc_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing {key}: {path}")
    input_map = {
        "classifier": str(classifier),
        "r4_policy": str(r4_policy),
        "h23_reference": str(h23_reference),
        **{key: str(path) for key, path in qc_paths.items()},
        "feature_schema": str(feature_schema),
        "class_map": str(class_map),
        "normalization": str(normalization),
        "offline_reference_1360": str(parity_reference),
    }
    _write_json(output_dir / "input_map.json", input_map)
    return input_map


def verify_classifier_structure(classifier: Path) -> None:
    """Fail if the selected checkpoint cannot strictly load the frozen B5 architecture."""
    import torch

    from config import FLConfig
    from utils import create_model_by_config

    payload = torch.load(Path(classifier), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("model_state"), dict):
        raise ValueError("classifier checkpoint has no model_state")
    model = create_model_by_config(FLConfig(), with_reg_head=False)
    model.load_state_dict(payload["model_state"], strict=True)
    if sum(parameter.numel() for parameter in model.parameters()) != 22765:
        raise ValueError("unexpected classifier parameter count")


def verify_frozen_model_code(frozen_commit: str) -> None:
    command = ["git", "diff", "--quiet", frozen_commit, "--", "config.py", "model.py", "utils.py"]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise ValueError("current model/config code differs from the declared frozen commit")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classifier", required=True, type=Path)
    parser.add_argument("--r4-policy", required=True, type=Path)
    parser.add_argument("--h23-reference", required=True, type=Path)
    parser.add_argument("--qc-dir", required=True, type=Path)
    parser.add_argument("--hc90-reference", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--frozen-commit", required=True)
    parser.add_argument("--source-archive-sha256", required=True)
    args = parser.parse_args()
    verify_frozen_model_code(args.frozen_commit)
    verify_classifier_structure(args.classifier)
    inputs = prepare_bundle_inputs(
        classifier=args.classifier,
        r4_policy=args.r4_policy,
        h23_reference=args.h23_reference,
        qc_dir=args.qc_dir,
        hc90_reference=args.hc90_reference,
        output_dir=args.output_dir,
        frozen_commit=args.frozen_commit,
        source_archive_sha256=args.source_archive_sha256,
    )
    print(json.dumps({"status": "ready", "asset_keys": sorted(inputs)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
