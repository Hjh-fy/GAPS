"""Validate Stage-A Flower command manifests before launching long runs."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path
from typing import Any


REQUIRED_STRONG_DA = {
    "domain_adapt_steps": 100,
    "domain_adapt_warmup": 0,
    "da_use_adversarial": True,
    "da_lambda_coral": 0.5,
    "da_lambda_adv": 0.5,
    "da_server_opt_lr": 0.0005,
    "use_adapted_as_global": True,
}

FLAG_NAMES = {
    "domain_adapt_steps": "--domain-adapt-steps",
    "domain_adapt_warmup": "--domain-adapt-warmup",
    "da_use_adversarial": "--da-use-adversarial",
    "da_lambda_coral": "--da-lambda-coral",
    "da_lambda_adv": "--da-lambda-adv",
    "da_server_opt_lr": "--da-server-opt-lr",
    "use_adapted_as_global": "--use-adapted-as-global",
}


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def values_equal(actual: Any, expected: Any) -> bool:
    if isinstance(expected, bool):
        return parse_bool(actual) is expected
    if isinstance(expected, int):
        try:
            return int(actual) == expected
        except (TypeError, ValueError):
            return False
    if isinstance(expected, float):
        try:
            return abs(float(actual) - expected) <= 1e-12
        except (TypeError, ValueError):
            return False
    return actual == expected


def command_flags(command: str) -> dict[str, str]:
    parts = shlex.split(command)
    out: dict[str, str] = {}
    index = 0
    while index < len(parts):
        item = parts[index]
        if item.startswith("--"):
            if "=" in item:
                flag, value = item.split("=", 1)
                out[flag] = value
            elif index + 1 < len(parts) and not parts[index + 1].startswith("--"):
                out[item] = parts[index + 1]
                index += 1
            else:
                out[item] = "true"
        index += 1
    return out


def validate_manifest(path: Path) -> list[str]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    expected_config = manifest.get("expected_da_config", {})
    flags = command_flags(str(manifest.get("server_command", "")))
    errors = []

    if flags.get("--da-preset") != "fixed_da_strong":
        errors.append("server_command must include --da-preset fixed_da_strong")

    for key, expected in REQUIRED_STRONG_DA.items():
        actual = expected_config.get(key)
        if not values_equal(actual, expected):
            errors.append(
                f"expected_da_config.{key}={actual!r}, expected {expected!r}"
            )
        flag = FLAG_NAMES[key]
        if flag not in flags:
            errors.append(f"server_command missing {flag}")
            continue
        if not values_equal(flags[flag], expected):
            errors.append(
                f"server_command {flag}={flags[flag]!r}, expected {expected!r}"
            )
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Stage-A matrix command manifest")
    parser.add_argument("manifest", help="Path to command_manifest.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.manifest)
    errors = validate_manifest(path)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
    print(f"OK: {path}")


if __name__ == "__main__":
    main()
