"""Freeze an evaluation-only recovery runtime for the completed fold-1 run."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import tarfile
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "results"
    / "lab_three_gas_nominal_three_node_r25le3_20260729_postflight"
    / "recovery_source_v1"
)
TRAINING_SOURCE_SHA = (
    "4354e9f3a4a6a20eeefd2f54180b3962cb4e9e7a8ce5cae80365dd8f79846e60"
)
ROOT_FILES = (
    "client.py",
    "config.py",
    "federated_dataset.py",
    "model.py",
    "utils.py",
)
SCRIPT_FILES = (
    "scripts/__init__.py",
    "scripts/lab_three_gas_3class/evaluate_exposure_checkpoint.py",
    "scripts/lab_three_gas_3class/evaluate_source_target_run.py",
    "scripts/lab_three_gas_3class/train_centralized_baseline.py",
    "scripts/lab_three_gas_3class/validate_three_node_run.py",
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def paths() -> list[Path]:
    selected = [PROJECT_ROOT / name for name in ROOT_FILES]
    selected.extend(sorted((PROJECT_ROOT / "gaps_flower").glob("*.py")))
    selected.extend(PROJECT_ROOT / name for name in SCRIPT_FILES)
    missing = [str(path) for path in selected if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    return sorted(
        set(selected),
        key=lambda path: path.relative_to(PROJECT_ROOT).as_posix(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite: {output}")
    output.mkdir(parents=True)

    archive = output / "postflight_recovery_source.tar"
    members = []
    with tarfile.open(archive, "x", format=tarfile.PAX_FORMAT) as bundle:
        for path in paths():
            relative = path.relative_to(PROJECT_ROOT).as_posix()
            data = path.read_bytes()
            info = tarfile.TarInfo(relative)
            info.size = len(data)
            info.mode = 0o644
            info.mtime = 0
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            bundle.addfile(info, io.BytesIO(data))
            members.append(
                {
                    "relative_path": relative,
                    "byte_size": len(data),
                    "sha256": sha256_bytes(data),
                }
            )
    manifest = {
        "schema_version": "gaps.lab_three_gas.postflight_recovery_source.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "evaluation_and_audit_only_no_training",
        "parent_training_source_sha256": TRAINING_SOURCE_SHA,
        "source_archive_sha256": sha256_file(archive),
        "members": members,
        "recovery_reason": (
            "The frozen training runtime omitted "
            "train_centralized_baseline.py, so the planned evaluator could "
            "not import metric helpers after all 25 training rounds completed."
        ),
    }
    manifest_path = output / "source_manifest.json"
    with manifest_path.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
