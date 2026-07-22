"""Execute and record formal 1,360-row B5/C5 fixed-H8 runtime parity."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gaps_deploy.c5_h8_runtime import C5H8Runtime
from scripts.validate_iotj_b5_c5_runtime_parity import validate_c5_h8_parity


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_row_map(path: Path, contract_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != "iotj.c5_h8_row_map.v1" or payload.get("status") != "ready" or payload.get("row_count") != 1360:
        raise ValueError("runtime row map is not ready for 1360 rows")
    if payload.get("contract_sha256") != _sha256(contract_path):
        raise ValueError("runtime row map contract hash differs")
    rows = payload.get("rows")
    if not isinstance(rows, list) or len(rows) != 1360:
        raise ValueError("runtime row map rows differ")
    runtime_keys = {int(row["runtime_index"]) for row in rows}
    reference_keys = {int(row["reference_index"]) for row in rows}
    if runtime_keys != set(range(1360)) or reference_keys != set(range(1360)):
        raise ValueError("runtime row map is not a 0..1359 bijection")
    return rows


def _write_runtime_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ("sample_index", "pred_class", "h8_ppm", "deployment_risk_full", "qc_decision", "auto_output_ppm", "qc_workpoint", "selected_profile", "filename", "repeat_id")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def run_c5_h8_parity(*, contract_path: Path, row_map_path: Path, output_dir: Path, workpoint: str = "HC95") -> dict[str, Any]:
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite parity output: {output_dir}")
    contract_path, row_map_path = Path(contract_path), Path(row_map_path)
    runtime = C5H8Runtime.from_runtime_contract(contract_path)
    reference_path = runtime.contract_reference(workpoint)
    windows, metadata, phases = runtime.load_contract_inputs()
    mapping = _load_row_map(row_map_path, contract_path)
    raw_rows = runtime.predict_batch(windows, metadata, phases, workpoint=workpoint)
    remapped: dict[int, dict[str, Any]] = {}
    for item in mapping:
        runtime_index, reference_index = int(item["runtime_index"]), int(item["reference_index"])
        row = dict(raw_rows[runtime_index])
        if (row["filename"], str(row["repeat_id"])) != (str(item["filename"]), str(item["repeat_id"])):
            raise ValueError(f"runtime row-map key differs at runtime index {runtime_index}")
        row["sample_index"] = reference_index
        remapped[reference_index] = row
    rows = [remapped[index] for index in range(1360)]
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime_path = output_dir / "runtime_rows.csv"
    _write_runtime_rows(runtime_path, rows)
    report = validate_c5_h8_parity(reference_path, runtime_path, workpoint)
    if report["status"] != "equivalent":
        raise RuntimeError("C5 H8 runtime parity failed; no success report was written")
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("C5 H8 parity provenance is blocked: code commit unavailable") from error
    classifier_hash = runtime.bundle.manifest["assets"]["classifier"]["sha256"] if runtime.bundle else None
    report.update({
        "bundle_manifest_sha256": _sha256(runtime.bundle.root / "manifest.json"),
        "classifier_sha256": classifier_hash,
        "runtime_contract_sha256": _sha256(contract_path),
        "row_map_sha256": _sha256(row_map_path),
        "reference_sha256": _sha256(reference_path),
        "runtime_rows_sha256": _sha256(runtime_path),
        "code_commit": commit,
        "evidence_boundary": "deployment runtime parity only; no training, refit, or metric promotion",
    })
    if not all(report.get(key) for key in ("bundle_manifest_sha256", "classifier_sha256", "runtime_contract_sha256", "row_map_sha256", "reference_sha256", "runtime_rows_sha256", "code_commit")):
        raise RuntimeError("C5 H8 parity provenance is incomplete")
    (output_dir / "parity_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--row-map", type=Path, required=True)
    parser.add_argument("--workpoint", choices=("HC95", "HC90"), default="HC95")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = run_c5_h8_parity(contract_path=args.contract, row_map_path=args.row_map, output_dir=args.output_dir, workpoint=args.workpoint)
    print(json.dumps({"status": report["status"], "workpoint": report["workpoint"], "runtime_rows": report["runtime_rows"], "max_abs_h8_ppm_delta": report["max_abs_h8_ppm_delta"], "max_abs_risk_delta": report["max_abs_risk_delta"]}))


if __name__ == "__main__":
    main()
