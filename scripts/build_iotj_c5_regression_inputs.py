"""Build aligned C5 regression inputs from a frozen C12-to-C5 classifier."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = Path("dataset/client_data_c1234src_c5tgt_2080_timeaware_60_170_window_fullgrid")
DEFAULT_REGRESSION_CHECKPOINT = Path(
    "results/R3aK16_flower_reg_depth4_dct_src12/regression_fedavg_global.pt"
)
EXPECTED_COUNTS = {"calibration": 320, "test": 1360}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _as_int(value: Any) -> int:
    return int(float(value))


def _as_float(value: Any) -> float:
    return float(value)


def convert_pipeline_record(row: dict[str, Any], split: str) -> dict[str, Any]:
    """Convert current pipeline records to the target-head feature contract."""
    client = str(row.get("client") or f"C{_as_int(row['client_id'])}")
    if client != "C5":
        raise ValueError(f"only C5 is permitted, got {client}")
    sample_index = _as_int(row.get("row_id", row.get("sample_index")))
    true_class = _as_int(row.get("true_class", row.get("true_cls")))
    pred_class = _as_int(row.get("pred_cls", row.get("pred_class")))
    final_ppm = _as_float(row.get("final_calibrated_ppm", row.get("pred_cal_ppm")))
    converted = dict(row)
    converted.update(
        {
            "client": "C5",
            "client_id": "C5",
            "split": split,
            "sample_index": sample_index,
            "true_class": true_class,
            "pred_class": pred_class,
            "route_class": _as_int(row.get("route_cls", pred_class)),
            "route_correct": int(_as_int(row.get("route_cls", pred_class)) == true_class),
            "class_correct": int(pred_class == true_class),
            "true_ppm": _as_float(row["true_ppm"]),
            "raw_ppm": _as_float(row.get("base_raw_ppm", row.get("pred_raw_ppm"))),
            "final_ppm": final_ppm,
            "auto_v2_ppm": final_ppm,
            "confidence": _as_float(row.get("class_confidence", 0.0)),
            "top1_confidence": _as_float(row.get("class_confidence", 0.0)),
            "confidence_margin": _as_float(row.get("class_margin", 0.0)),
            "risk_score": _as_float(row.get("composite_response_risk", 0.0)),
            "risk_classifier_uncertainty": _as_float(row.get("classifier_entropy_risk", 0.0)),
            "risk_margin_risk": max(0.0, 1.0 - _as_float(row.get("class_margin", 0.0))),
            "risk_route_response_risk": _as_float(row.get("route_response_risk", 0.0)),
        }
    )
    return converted


def validate_rows(rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    keys: set[tuple[str, int]] = set()
    counts: dict[str, int] = {"calibration": 0, "test": 0}
    for row in rows:
        if row.get("client") != "C5":
            raise ValueError("non-C5 row in primary regression input")
        split = str(row.get("split"))
        if split not in counts:
            raise ValueError(f"unexpected split: {split}")
        key = (split, _as_int(row.get("sample_index")))
        if key in keys:
            raise ValueError(f"duplicate C5 row key: {key}")
        keys.add(key)
        counts[split] += 1
    if counts != EXPECTED_COUNTS:
        raise ValueError(f"C5 row counts must equal {EXPECTED_COUNTS}; got {counts}")
    return counts


def _run(command: Sequence[str]) -> None:
    subprocess.run(list(command), cwd=REPO_ROOT, check=True)


def build_inputs(args: argparse.Namespace) -> dict[str, Any]:
    classifier = Path(args.classifier_checkpoint).resolve()
    regression = Path(args.regression_checkpoint).resolve()
    data_root = Path(args.data_root).resolve()
    output_dir = Path(args.output_dir)
    if not classifier.is_file() or not regression.is_file():
        raise FileNotFoundError(f"missing checkpoint: classifier={classifier}, regression={regression}")
    output_dir.mkdir(parents=True, exist_ok=True)

    combined: list[dict[str, Any]] = []
    for split in ("calibration", "test"):
        split_dir = output_dir / "r3ak16_pipeline" / split
        _run(
            [
                sys.executable,
                "-m",
                "gaps_flower.evaluate_regression_pipeline",
                "--classifier-ckpt", str(classifier),
                "--regression-ckpt", str(regression),
                "--data-root", str(data_root),
                "--client-ids", "5",
                "--split", split,
                "--route-source", "predicted",
                "--device", args.device,
                "--batch-size", str(args.batch_size),
                "--output-dir", str(split_dir),
            ]
        )
        records = _read_csv(split_dir / f"{split}_records.csv")
        converted = [convert_pipeline_record(row, split) for row in records]
        if len(converted) != EXPECTED_COUNTS[split]:
            raise ValueError(
                f"{split} expected {EXPECTED_COUNTS[split]} C5 rows; got {len(converted)}"
            )
        combined.extend(converted)

    counts = validate_rows(combined)
    predictions_path = output_dir / "c5_target_layer_predictions.csv"
    _write_csv(predictions_path, combined)

    backbone_dir = output_dir / "backbone_features"
    _run(
        [
            sys.executable,
            "export_backbone_features.py",
            "--checkpoint", str(classifier),
            "--data-root", str(data_root),
            "--clients", "5",
            "--splits", "calibration,test",
            "--pred-prefix", "iotj_c12_c5",
            "--output-dir", str(backbone_dir),
            "--device", args.device,
            "--batch-size", str(args.batch_size),
        ]
    )
    manifest = {
        "schema_version": 1,
        "protocol": {"source_clients": [1, 2], "target_clients": [5]},
        "classifier_checkpoint": str(classifier),
        "classifier_sha256": _sha256(classifier),
        "regression_reference": "R3aK16 source C1/C2",
        "regression_checkpoint": str(regression),
        "regression_sha256": _sha256(regression),
        "data_root": str(data_root),
        "counts": counts,
        "outputs": {
            "target_predictions": str(predictions_path),
            "backbone_calibration": str(backbone_dir / "backbone_features_calibration.csv"),
            "backbone_test": str(backbone_dir / "backbone_features_test.csv"),
        },
        "training_performed": False,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classifier-checkpoint", required=True)
    parser.add_argument("--regression-checkpoint", default=str(DEFAULT_REGRESSION_CHECKPOINT))
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args(argv)
    manifest = build_inputs(args)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
