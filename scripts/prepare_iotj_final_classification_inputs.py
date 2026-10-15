"""Import read-only formal inputs with ordered content verification."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from gaps_flower.state_fingerprint import checkpoint_provenance


def import_checkpoint(source: str | Path, destination: str | Path) -> dict[str, Any]:
    source_path = Path(source).resolve()
    destination_path = Path(destination).resolve()
    if not source_path.is_file():
        raise RuntimeError(f"FAIL_CLOSED source checkpoint missing: {source_path}")
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    source_info = checkpoint_provenance(source_path)
    if destination_path.exists():
        copy_info = checkpoint_provenance(destination_path)
    else:
        shutil.copy2(source_path, destination_path)
        copy_info = checkpoint_provenance(destination_path)
    equality_verified = (
        source_info["ordered_state_content_fingerprint"]
        == copy_info["ordered_state_content_fingerprint"]
    )
    if not equality_verified:
        raise RuntimeError("FAIL_CLOSED imported checkpoint ordered content mismatch")
    if source_info["formal_round"] != 25 or copy_info["formal_round"] != 25:
        raise RuntimeError("FAIL_CLOSED formal source checkpoint must be round 25")
    return {
        "schema_version": "iotj.final_classification.input_import.v1",
        "formal_round": 25,
        "source": source_info,
        "copy": copy_info,
        "equality_verified": True,
        "equality_basis": "ordered_state_content_fingerprint",
        "whole_file_sha256_role": "provenance_only",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    manifest = import_checkpoint(args.source, args.destination)
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
