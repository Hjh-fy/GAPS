"""Evaluate one locked Flower checkpoint without target-test model selection.

The default policy evaluates the final configured round. Source calibration is
still scored for monitoring, but it does not choose the checkpoint. The legacy
source-calibration policy remains available only for explicit reproduction.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVALUATOR = PROJECT_ROOT / "scripts" / "lab_three_gas_3class" / "evaluate_exposure_checkpoint.py"


def parse_client_ids(text: str) -> list[int]:
    ids = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not ids:
        raise argparse.ArgumentTypeError("client list must not be empty")
    return ids


def evaluate(
    checkpoint: Path,
    data_root: Path,
    client_ids: list[int],
    split: str,
    output: Path,
    device: str,
) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(EVALUATOR),
        "--checkpoint",
        str(checkpoint),
        "--data-root",
        str(data_root),
        "--client-ids",
        ",".join(str(cid) for cid in client_ids),
        "--split",
        split,
        "--device",
        device,
        "--output",
        str(output),
    ]
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)
    return json.loads(output.read_text(encoding="utf-8"))


def select_source_row(
    rows: list[dict[str, Any]],
    policy: str,
) -> tuple[dict[str, Any], str]:
    if not rows:
        raise ValueError("Cannot select from an empty round list")
    if policy == "last_round":
        return (
            rows[-1],
            "fixed final configured round; source calibration is monitoring only",
        )
    if policy == "source_calibration":
        selected = max(
            rows,
            key=lambda row: (
                row["source_validation_exposure_macro_f1"],
                row["source_validation_window_macro_f1"],
                -row["round"],
            ),
        )
        return (
            selected,
            "max source calibration exposure Macro-F1; tie-break by source "
            "window Macro-F1, then earliest round",
        )
    raise ValueError(f"Unknown selection policy: {policy}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--source-clients", type=parse_client_ids, required=True)
    parser.add_argument("--target-client", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=25)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--selection-policy",
        choices=("last_round", "source_calibration"),
        default="last_round",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Explicit new evaluation destination. Defaults to "
            "<run-dir>/formal_evaluation."
        ),
    )
    args = parser.parse_args()

    output_dir = args.output_dir or args.run_dir / "formal_evaluation"
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite an existing evaluation: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    source_rows: list[dict[str, Any]] = []
    for round_idx in range(1, args.rounds + 1):
        checkpoint = args.run_dir / f"server_round_{round_idx:03d}.pth"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        payload = evaluate(
            checkpoint,
            args.data_root,
            args.source_clients,
            "calibration",
            output_dir / f"source_calibration_round_{round_idx:03d}.json",
            args.device,
        )
        source_rows.append(
            {
                "round": round_idx,
                "checkpoint": str(checkpoint),
                "source_validation_exposure_macro_f1": float(
                    payload["global"]["exposure"]["macro_f1"]
                ),
                "source_validation_window_macro_f1": float(
                    payload["global"]["window"]["macro_f1"]
                ),
                "payload": payload,
            }
        )

    selected, selection_rule = select_source_row(
        source_rows,
        args.selection_policy,
    )
    selected_round = int(selected["round"])
    checkpoints = {
        "unadapted": args.run_dir / f"server_round_{selected_round:03d}.pth",
        "adapted": args.run_dir
        / f"server_round_{selected_round:03d}_adapted.pth",
    }

    final: dict[str, Any] = {}
    for kind, checkpoint in checkpoints.items():
        if not checkpoint.is_file():
            if kind == "adapted":
                continue
            raise FileNotFoundError(checkpoint)
        final[kind] = {
            "checkpoint": str(checkpoint),
            "target_calibration": evaluate(
                checkpoint,
                args.data_root,
                [args.target_client],
                "calibration",
                output_dir / f"target_calibration_{kind}.json",
                args.device,
            ),
            # This is the only target-test access in the selection pipeline.
            "target_test": evaluate(
                checkpoint,
                args.data_root,
                [args.target_client],
                "test",
                output_dir / f"target_test_{kind}.json",
                args.device,
            ),
        }

    summary = {
        "selection_policy": args.selection_policy,
        "selection_rule": selection_rule,
        "source_clients": args.source_clients,
        "target_client": args.target_client,
        "selected_round": selected_round,
        "selected_source_validation_exposure_macro_f1": selected[
            "source_validation_exposure_macro_f1"
        ],
        "rounds": [
            {key: value for key, value in row.items() if key != "payload"}
            for row in source_rows
        ],
        "final": final,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
