#!/usr/bin/env python3
"""Verify that Runtime-v5 UI packages fail closed on contract tampering."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import tempfile
from typing import Callable

from edge_ai_runtime import EdgeAIPackage, EdgeAIRuntime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-dir", required=True)
    return parser.parse_args()


def expect_failure(label: str, action: Callable[[], object], contains: str) -> None:
    try:
        action()
    except Exception as exc:
        if contains.lower() not in str(exc).lower():
            raise AssertionError(
                f"{label}: wrong failure message: {exc}"
            ) from exc
        return
    raise AssertionError(f"{label}: package unexpectedly loaded")


def main() -> int:
    source = Path(parse_args().package_dir).expanduser().resolve()
    with tempfile.TemporaryDirectory(prefix="gaps_runtime_v5_fail_closed_") as temp:
        root = Path(temp)

        bad_asset = root / "bad_asset"
        shutil.copytree(source, bad_asset)
        with (
            bad_asset / "runtime_v5_core/assets/classifier.pth"
        ).open("ab") as handle:
            handle.write(b"\x00")
        expect_failure(
            "model asset tamper",
            lambda: EdgeAIRuntime(bad_asset),
            "portable asset bytes/sha256 differ",
        )

        bad_mode = root / "bad_mode"
        shutil.copytree(source, bad_mode)
        manifest_path = bad_mode / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["input"]["feature_mode"] = "raw_adc"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        expect_failure(
            "input contract tamper",
            lambda: EdgeAIPackage(bad_mode),
            "baseline_duration_s",
        )

        bad_code = root / "bad_code"
        shutil.copytree(source, bad_code)
        with (bad_code / "runtime_code/model.py").open("ab") as handle:
            handle.write(b"\n# tamper\n")
        expect_failure(
            "runtime code tamper",
            lambda: EdgeAIPackage(bad_code),
            "code file identity differs",
        )

    print("Runtime-v5 fail-closed self-test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
