"""Evaluate one frozen A4 checkpoint on stable, early, and full target scopes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

if __package__:
    from .evaluate_source_target_run import evaluate
else:
    from evaluate_source_target_run import evaluate  # type: ignore[no-redef]


SCOPE_SPLITS = {
    "stable360": "test",
    "early60": "early",
    "full420": "full",
}


def evaluate_scopes(
    *,
    checkpoint: Path,
    data_root: Path,
    target_client: int,
    output_dir: Path,
    device: str,
    evaluator: Callable[..., dict[str, Any]] = evaluate,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite evaluation: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    scopes = {}
    for scope_name, split in SCOPE_SPLITS.items():
        scopes[scope_name] = evaluator(
            checkpoint,
            data_root,
            [target_client],
            split,
            output_dir / f"target_{scope_name}.json",
            device,
        )
    summary = {
        "checkpoint": str(checkpoint),
        "data_root": str(data_root),
        "target_client": target_client,
        "scopes": scopes,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--target-client", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    summary = evaluate_scopes(
        checkpoint=args.checkpoint,
        data_root=args.data_root,
        target_client=args.target_client,
        output_dir=args.output_dir,
        device=args.device,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
