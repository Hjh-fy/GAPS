"""Verify every file recorded by a canonical-v1 dataset hash index."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def verify(root: Path) -> dict[str, object]:
    root = root.resolve()
    index = json.loads((root / "dataset_sha256.json").read_text(encoding="utf-8"))
    bad: list[str] = []
    for relative, expected in index["files"].items():
        path = root / relative
        observed = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        if observed != expected:
            bad.append(relative)
    return {
        "status": "PASS" if not bad else "FAIL",
        "aggregate_sha256": index["aggregate_sha256"],
        "checked_files": len(index["files"]),
        "bad_files": bad,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    result = verify(parser.parse_args().root)
    print(json.dumps(result, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
