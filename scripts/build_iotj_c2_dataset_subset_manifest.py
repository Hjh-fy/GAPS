"""Derive an immutable client-only dataset manifest for remote C2 transfer."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence


_HASH_RE = re.compile(r"^[0-9a-f]{64}$")


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def build_client_subset_manifest(
    full_manifest: Mapping[str, Any], *, client_id: int
) -> dict[str, Any]:
    """Select exactly one client's paths from a validated full dataset manifest."""
    parent_hash = full_manifest.get("dataset_manifest_sha256")
    if not isinstance(parent_hash, str) or not _HASH_RE.fullmatch(parent_hash):
        raise ValueError("full dataset manifest SHA-256 is invalid")
    if not isinstance(client_id, int) or isinstance(client_id, bool) or client_id <= 0:
        raise ValueError("client_id must be a positive integer")
    files = full_manifest.get("files")
    if not isinstance(files, list):
        raise ValueError("full dataset manifest files must be a list")
    prefix = f"client_{client_id}/"
    selected: list[dict[str, Any]] = []
    for row in files:
        if not isinstance(row, Mapping):
            raise ValueError("full dataset manifest file row is invalid")
        relative_path = row.get("relative_path")
        digest = row.get("sha256")
        byte_size = row.get("byte_size")
        if not isinstance(relative_path, str) or not isinstance(digest, str):
            raise ValueError("full dataset manifest file identity is invalid")
        if not _HASH_RE.fullmatch(digest) or not isinstance(byte_size, int) or byte_size < 0:
            raise ValueError("full dataset manifest file digest or size is invalid")
        if relative_path.startswith(prefix):
            selected.append(
                {
                    "relative_path": relative_path,
                    "sha256": digest,
                    "byte_size": byte_size,
                }
            )
    if not selected:
        raise ValueError(f"full dataset manifest has no files for client_{client_id}")
    selected.sort(key=lambda row: str(row["relative_path"]))
    payload: dict[str, Any] = {
        "schema_version": 1,
        "client_id": client_id,
        "parent_dataset_manifest_sha256": parent_hash,
        "files": selected,
    }
    payload["dataset_subset_manifest_sha256"] = _canonical_sha256(payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--client-id", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        full = json.loads(args.dataset_manifest.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("dataset manifest is not valid JSON") from exc
    if not isinstance(full, Mapping):
        raise ValueError("dataset manifest must be an object")
    payload = build_client_subset_manifest(full, client_id=args.client_id)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite subset manifest: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "status": "created",
                "client_id": args.client_id,
                "file_count": len(payload["files"]),
                "dataset_subset_manifest_sha256": payload[
                    "dataset_subset_manifest_sha256"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
