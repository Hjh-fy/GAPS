"""Create a non-overwriting, hash-bound C5/H8 runtime-contract layer."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


EXPECTED_ROWS = 1360
EXPECTED_WINDOW_SHAPE = (100, 8)
SCHEMA_VERSION = "iotj.c5_h8_runtime_contract.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bound_file(path: Path, label: str) -> dict[str, Any]:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    return {"path": path.as_posix(), "bytes": path.stat().st_size, "sha256": _sha256(path)}


def _verify_features(path: Path) -> None:
    array = np.load(path, mmap_mode="r")
    if tuple(array.shape) != (EXPECTED_ROWS, *EXPECTED_WINDOW_SHAPE):
        raise ValueError(f"runtime features must have shape {(EXPECTED_ROWS, *EXPECTED_WINDOW_SHAPE)}; got {tuple(array.shape)}")
    if array.dtype != np.float32:
        raise ValueError(f"runtime features must be float32; got {array.dtype}")
    if not np.isfinite(array).all():
        raise ValueError("runtime features contain non-finite values")


def _verify_metadata(path: Path) -> None:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, list) or len(value) != EXPECTED_ROWS:
        raise ValueError(f"runtime metadata must contain exactly {EXPECTED_ROWS} rows")


def prepare_runtime_contract(
    *,
    bundle_dir: Path,
    classifier_model: Mapping[str, Any],
    input_features: Path,
    input_metadata: Path,
    hc95_reference: Path,
    hc90_reference: Path,
    output_dir: Path,
) -> Path:
    """Bind immutable runtime-only metadata without changing frozen assets."""
    bundle_dir = Path(bundle_dir).resolve()
    manifest = bundle_dir / "manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"missing bundle manifest: {manifest}")
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    if manifest_payload.get("schema_version") != "iotj.b5_c5_deployment_bundle.v1" or manifest_payload.get("status") != "ready":
        raise ValueError("bundle manifest is not a ready B5/C5 deployment bundle")
    _verify_features(Path(input_features))
    _verify_metadata(Path(input_metadata))
    if not isinstance(classifier_model, Mapping) or not classifier_model:
        raise ValueError("classifier model contract must be a non-empty object")
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite runtime contract: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "ready",
        "bundle_manifest": _bound_file(manifest, "bundle manifest"),
        "classifier_model": dict(classifier_model),
        "inputs": {
            "features": _bound_file(Path(input_features), "runtime features"),
            "metadata": _bound_file(Path(input_metadata), "runtime metadata"),
            "row_count": EXPECTED_ROWS,
            "window_shape": list(EXPECTED_WINDOW_SHAPE),
            "dtype": "float32",
        },
        "references": {
            "HC95": _bound_file(Path(hc95_reference), "HC95 reference"),
            "HC90": _bound_file(Path(hc90_reference), "HC90 reference"),
        },
    }
    (output_dir / "runtime_contract.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--classifier-model", type=Path, required=True)
    parser.add_argument("--input-features", type=Path, required=True)
    parser.add_argument("--input-metadata", type=Path, required=True)
    parser.add_argument("--hc95-reference", type=Path, required=True)
    parser.add_argument("--hc90-reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    model = json.loads(args.classifier_model.read_text(encoding="utf-8"))
    output = prepare_runtime_contract(
        bundle_dir=args.bundle_dir,
        classifier_model=model,
        input_features=args.input_features,
        input_metadata=args.input_metadata,
        hc95_reference=args.hc95_reference,
        hc90_reference=args.hc90_reference,
        output_dir=args.output_dir,
    )
    print(json.dumps({"status": "ready", "output": output.as_posix()}))


if __name__ == "__main__":
    main()
