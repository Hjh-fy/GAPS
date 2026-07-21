"""Run the formal C5 regression and high-coverage QC suite for one classifier."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Sequence


def build_suite_commands(
    *,
    classifier_checkpoint: Path,
    regression_checkpoint: Path,
    data_root: Path,
    output_root: Path,
    device: str,
    seed: int,
    n_random: int,
) -> list[list[str]]:
    inputs = output_root / "inputs"
    h23 = output_root / "h23_plus"
    h8 = output_root / "h8_no_rescue"
    qc = output_root / "high_coverage_qc"
    ladder = output_root / "r0_r7"
    target_predictions = inputs / "c5_target_layer_predictions.csv"
    backbone_calibration = inputs / "backbone_features" / "backbone_features_calibration.csv"
    backbone_test = inputs / "backbone_features" / "backbone_features_test.csv"
    return [
        [
            sys.executable,
            "scripts/build_iotj_c5_regression_inputs.py",
            "--classifier-checkpoint",
            str(classifier_checkpoint),
            "--regression-checkpoint",
            str(regression_checkpoint),
            "--data-root",
            str(data_root),
            "--output-dir",
            str(inputs),
            "--device",
            device,
        ],
        [
            sys.executable,
            "scripts/run_iotj_c5_h23_plus.py",
            "--data-root",
            str(data_root),
            "--target-predictions",
            str(target_predictions),
            "--backbone-calibration",
            str(backbone_calibration),
            "--backbone-test",
            str(backbone_test),
            "--seed",
            str(seed),
            "--runtime-reference-output",
            str(h23 / "h23_reference.json"),
            "--classifier-checkpoint",
            str(classifier_checkpoint),
            "--output-dir",
            str(h23),
        ],
        [
            sys.executable,
            "run_source_augmented_target_ridge_eval.py",
            "--data-root",
            str(data_root),
            "--target-predictions",
            str(target_predictions),
            "--source-clients",
            "1,2",
            "--target-clients",
            "5",
            "--disable-c4-rescue",
            "--seed",
            str(seed),
            "--runtime-policy-output",
            str(h8 / "r4_policy.json"),
            "--classifier-checkpoint",
            str(classifier_checkpoint),
            "--output-dir",
            str(h8),
        ],
        [
            sys.executable,
            "scripts/evaluate_iotj_high_coverage_qc.py",
            "--target-inputs",
            str(target_predictions),
            "--h23-validation",
            str(h23 / "c5_h23_plus_validation_predictions.csv"),
            "--h23-test",
            str(h23 / "c5_h23_plus_test_predictions.csv"),
            "--h8-validation",
            str(h8 / "target_validation_plus_source_preds.csv"),
            "--h8-test",
            str(h8 / "target_predictions_plus_source_preds.csv"),
            "--h8-test-oracle",
            str(h8 / "target_predictions_plus_source_preds_oracle_route.csv"),
            "--backbone-calibration",
            str(backbone_calibration),
            "--backbone-test",
            str(backbone_test),
            "--pred-key",
            "target_ridge_plus_source_preds_ppm",
            "--n-random",
            str(n_random),
            "--seed",
            str(seed),
            "--output-dir",
            str(qc),
        ],
        [
            sys.executable,
            "scripts/assemble_iotj_c5_regression_ladder.py",
            "--validation-scored",
            str(qc / "calibration_validation_scored.csv"),
            "--test-scored",
            str(qc / "test_scored.csv"),
            "--risk-selection",
            str(qc / "risk_selection.json"),
            "--output-dir",
            str(ladder),
        ],
    ]


def expected_outputs(output_root: Path) -> tuple[Path, ...]:
    return (
        output_root / "inputs" / "manifest.json",
        output_root / "h23_plus" / "manifest.json",
        output_root / "h8_no_rescue" / "manifest.json",
        output_root / "high_coverage_qc" / "manifest.json",
        output_root / "high_coverage_qc" / "operational_summary.json",
        output_root / "r0_r7" / "r0_r7_summary.csv",
        output_root / "r0_r7" / "manifest.json",
    )


def run_commands(commands: Sequence[Sequence[str]], repo_root: Path) -> None:
    for command in commands:
        subprocess.run(list(command), cwd=repo_root, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classifier-checkpoint", type=Path, required=True)
    parser.add_argument("--classifier-id", required=True)
    parser.add_argument(
        "--regression-checkpoint",
        type=Path,
        default=Path("results/R3aK16_flower_reg_depth4_dct_src12/regression_fedavg_global.pt"),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("dataset/client_data_c1234src_c5tgt_2080_timeaware_60_170_window_fullgrid"),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-random", type=int, default=1000)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    commands = build_suite_commands(
        classifier_checkpoint=args.classifier_checkpoint,
        regression_checkpoint=args.regression_checkpoint,
        data_root=args.data_root,
        output_root=args.output_root,
        device=args.device,
        seed=args.seed,
        n_random=args.n_random,
    )
    if args.dry_run:
        print(json.dumps(commands, indent=2, ensure_ascii=False))
        return 0
    args.output_root.mkdir(parents=True, exist_ok=True)
    run_commands(commands, repo_root)
    missing = [str(path) for path in expected_outputs(args.output_root) if not path.is_file()]
    if missing:
        raise FileNotFoundError("regression suite outputs are incomplete:\n" + "\n".join(missing))
    manifest = {
        "schema_version": 1,
        "classifier_id": args.classifier_id,
        "classifier_checkpoint": str(args.classifier_checkpoint),
        "regression_checkpoint": str(args.regression_checkpoint),
        "data_root": str(args.data_root),
        "seed": args.seed,
        "device": args.device,
        "n_random": args.n_random,
        "commands": commands,
        "outputs": [str(path) for path in expected_outputs(args.output_root)],
        "training_location_required": "Alibaba Cloud ECS",
    }
    (args.output_root / "suite_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
