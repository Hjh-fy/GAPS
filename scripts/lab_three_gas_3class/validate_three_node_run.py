"""Fail-closed postflight audit for one laboratory three-gas Flower run."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np


DIRECTION_ROLES = {
    "P2_to_P3": ([2], 3),
    "P2_to_P1": ([2], 1),
    "P1_to_P3": ([1], 3),
    "P12_to_P3": ([1, 2], 3),
}


def resolve_run_roles(direction: str) -> tuple[list[int], int]:
    try:
        sources, target = DIRECTION_ROLES[direction]
    except KeyError as exc:
        raise ValueError(f"Unsupported direction: {direction}") from exc
    return list(sources), target


def parse_client_ids(text: str) -> list[int]:
    clients = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not clients:
        raise argparse.ArgumentTypeError("client list must not be empty")
    return clients


def load_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def require_finite(value: Any, path: str = "payload") -> None:
    if isinstance(value, float):
        require(math.isfinite(value), f"Non-finite value at {path}: {value}")
    elif isinstance(value, dict):
        for key, item in value.items():
            require_finite(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            require_finite(item, f"{path}[{index}]")


def checkpoint_rounds(run_dir: Path, adapted: bool) -> list[int]:
    suffix = "_adapted.pth" if adapted else ".pth"
    pattern = re.compile(
        rf"^server_round_(\d{{3}}){re.escape(suffix)}$"
    )
    rounds = []
    for path in run_dir.glob("server_round_*.pth"):
        match = pattern.match(path.name)
        if match:
            rounds.append(int(match.group(1)))
    return sorted(rounds)


def validate_evaluation(
    evaluation: dict[str, Any],
    expected_rounds: int,
    source_clients: list[int],
    expected_target_scope: dict[str, dict[str, int]],
    expected_input_dim: int,
    selection_policy: str,
    target_client: int,
) -> dict[str, Any]:
    require(
        evaluation["source_clients"] == source_clients,
        "Evaluation source client identity mismatch",
    )
    require(
        evaluation["target_client"] == target_client,
        "Evaluation target client identity mismatch",
    )
    rows = evaluation["rounds"]
    require(len(rows) == expected_rounds, "Incomplete selection-round coverage")
    require(
        [row["round"] for row in rows] == list(range(1, expected_rounds + 1)),
        "Selection rounds are not exactly 1..N",
    )
    if selection_policy == "last_round":
        expected_selected = expected_rounds
    else:
        expected_selected = max(
            rows,
            key=lambda row: (
                row["source_validation_exposure_macro_f1"],
                row["source_validation_window_macro_f1"],
                -row["round"],
            ),
        )["round"]
    require(
        evaluation["selected_round"] == expected_selected,
        f"Selected round does not follow policy={selection_policy}",
    )
    require(
        evaluation.get("selection_policy", "source_calibration")
        == selection_policy,
        "Evaluation selection policy mismatch",
    )
    require(
        set(evaluation["final"]) == {"unadapted", "adapted"},
        "Both unadapted and adapted selected checkpoints are required",
    )
    metrics: dict[str, Any] = {}
    for variant, payload in evaluation["final"].items():
        for split in ("target_calibration", "target_test"):
            result = payload[split]
            dataset_split = (
                "calibration" if split == "target_calibration" else "test"
            )
            require(
                result["split"] == dataset_split,
                f"{variant} {split} split identity mismatch",
            )
            config = result["model_config"]
            require(
                config
                == {
                    "num_classes": 3,
                    "input_dim": expected_input_dim,
                    "num_clients": 3,
                    "num_phases": 1,
                    "seq_len": 100,
                },
                f"{variant} {split} model contract mismatch",
            )
            require(
                result["global"]["window"]["n_samples"]
                == expected_target_scope[dataset_split]["n_windows"],
                (
                    f"{variant} {split} expected "
                    f"{expected_target_scope[dataset_split]['n_windows']} windows"
                ),
            )
            require(
                result["global"]["exposure"]["n_exposures"]
                == expected_target_scope[dataset_split]["n_exposures"],
                (
                    f"{variant} {split} expected "
                    f"{expected_target_scope[dataset_split]['n_exposures']} "
                    "independent exposures"
                ),
            )
        metrics[variant] = {
            "target_test_window_accuracy": payload["target_test"]["global"][
                "window"
            ]["accuracy"],
            "target_test_window_macro_f1": payload["target_test"]["global"][
                "window"
            ]["macro_f1"],
            "target_test_exposure_accuracy": payload["target_test"]["global"][
                "exposure"
            ]["accuracy"],
            "target_test_exposure_macro_f1": payload["target_test"]["global"][
                "exposure"
            ]["macro_f1"],
        }
    return {
        "selected_round": expected_selected,
        "metrics": metrics,
    }


def expected_target_scope(target_data_dir: Path) -> dict[str, dict[str, int]]:
    scope: dict[str, dict[str, int]] = {}
    for split in ("calibration", "test"):
        labels_path = target_data_dir / f"{split}_classification_labels.npy"
        manifest_path = target_data_dir / f"{split}_window_manifest.csv"
        require(labels_path.is_file(), f"Missing target labels: {labels_path}")
        require(
            manifest_path.is_file(),
            f"Missing target window manifest: {manifest_path}",
        )
        labels = np.load(labels_path, mmap_mode="r")
        with manifest_path.open(
            "r", encoding="utf-8-sig", newline=""
        ) as handle:
            rows = list(csv.DictReader(handle))
        require(
            len(rows) == len(labels),
            f"{split} target manifest/label length mismatch",
        )
        exposure_ids = {
            str(row.get("exposure_id", "")).strip() for row in rows
        }
        require(
            "" not in exposure_ids,
            f"{split} target manifest has missing exposure_id",
        )
        scope[split] = {
            "n_windows": int(len(labels)),
            "n_exposures": int(len(exposure_ids)),
        }
    return scope


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--direction",
        choices=tuple(DIRECTION_ROLES),
        required=True,
    )
    parser.add_argument("--source-clients", type=parse_client_ids)
    parser.add_argument("--target-client", type=int)
    parser.add_argument("--rounds", type=int, default=25)
    parser.add_argument("--local-epochs", type=int, default=3)
    parser.add_argument("--da-steps", type=int, default=100)
    parser.add_argument("--input-dim", type=int, default=6)
    parser.add_argument(
        "--profile",
        choices=("strong_cls", "proto_replay"),
        default="strong_cls",
    )
    parser.add_argument(
        "--da-mode",
        choices=("legacy_strong", "corrected_b2"),
        default="legacy_strong",
    )
    parser.add_argument("--target-ce-weight", type=float, default=0.0)
    parser.add_argument(
        "--selection-policy",
        choices=("last_round", "source_calibration"),
        default="last_round",
    )
    parser.add_argument("--target-data-dir", type=Path, required=True)
    parser.add_argument("--evaluation-dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    source_clients, target_client = resolve_run_roles(args.direction)
    if args.source_clients is not None:
        require(
            args.source_clients == source_clients,
            "Explicit source clients do not match direction",
        )
    if args.target_client is not None:
        require(
            args.target_client == target_client,
            "Explicit target client does not match direction",
        )
    expected_round_list = list(range(1, args.rounds + 1))
    require(
        checkpoint_rounds(run_dir, adapted=False) == expected_round_list,
        "Base checkpoint set is incomplete or contains unexpected rounds",
    )
    require(
        checkpoint_rounds(run_dir, adapted=True) == expected_round_list,
        "Adapted checkpoint set is incomplete or contains unexpected rounds",
    )

    config = load_json(run_dir / "run_config.json")["args"]
    expected_config = {
        "rounds": args.rounds,
        "min_clients": len(source_clients),
        "num_classes": 3,
        "input_dim": args.input_dim,
        "num_clients": 3,
        "num_phases": 1,
        "domain_adapt_steps": args.da_steps,
        "strict_calibration_split": True,
        "use_domain_adapt": True,
        "use_adapted_as_global": True,
        "profile": args.profile,
        "da_lambda_target_ce": args.target_ce_weight,
    }
    for key, expected in expected_config.items():
        require(config.get(key) == expected, f"run_config {key} mismatch")
    if args.da_mode == "corrected_b2":
        corrected_expected = {
            "da_preset": "none",
            "da_use_coral": False,
            "da_use_mmd": True,
            "da_use_adversarial": False,
            "use_proto_mmd": False,
            "da_mmd_objective": "mmd2",
            "da_stage_alignment": "cross_domain_same_class_phase",
            "da_adv_feature_objective": "wasserstein_min",
            "da_lambda_coral": 0.0,
            "da_lambda_adv": 0.0,
            "da_lambda_proto_mmd": 0.0,
            "da_lambda_stage_mmd": 0.0,
        }
        for key, expected in corrected_expected.items():
            require(
                config.get(key) == expected,
                f"corrected_b2 run_config {key} mismatch",
            )
    else:
        legacy_expected = {
            "da_preset": "fixed_da_strong",
            "da_use_coral": True,
            "da_use_mmd": True,
            "da_use_adversarial": True,
            "use_proto_mmd": True,
            "da_mmd_objective": "legacy_quartic",
            "da_stage_alignment": "legacy_intra_domain",
            "da_adv_feature_objective": "legacy_grl_plus",
        }
        for key, expected in legacy_expected.items():
            require(
                config.get(key) == expected,
                f"legacy_strong run_config {key} mismatch",
            )
    require(
        config["server_calib_data"].endswith(f"/client_{target_client}"),
        f"Server calibration must be target client_{target_client}",
    )
    for client_id in source_clients:
        require(
            f"/client_{client_id}" in config["server_val_data"],
            f"Source validation missing client_{client_id}",
        )

    history = load_json(run_dir / "history.json")
    rounds = history["rounds"]
    require(len(rounds) == args.rounds, "history does not contain all rounds")
    for index, row in enumerate(rounds, start=1):
        require(row["round"] == index, f"History round mismatch at {index}")
        require(
            row["fit_clients"] == len(source_clients),
            f"Round {index} fit client count mismatch",
        )
        require(row["fit_failures"] == 0, f"Round {index} fit failure")
        require(
            row["evaluate_clients"] == len(source_clients),
            f"Round {index} evaluate client count mismatch",
        )
        require(row["evaluate_failures"] == 0, f"Round {index} eval failure")
        require(
            row["fit_metrics"]["local_epochs"] == float(args.local_epochs),
            f"Round {index} local epoch mismatch",
        )
        da = row["domain_adapt_summary"]
        require(da["num_steps"] == args.da_steps, f"Round {index} DA steps")
        require(
            da["checkpoint_changed_tensors"] > 0,
            f"Round {index} adapted checkpoint did not change",
        )
        require_finite(row, f"history.rounds[{index - 1}]")

    evaluation_dir = (
        args.evaluation_dir.resolve()
        if args.evaluation_dir
        else run_dir / "formal_evaluation"
    )
    require(
        len(list(evaluation_dir.glob("source_calibration_round_*.json")))
        == args.rounds,
        "Source-calibration evaluation files are incomplete",
    )
    require(
        not list(evaluation_dir.glob("target_test_round_*.json")),
        "Per-round target-test files violate the locked test boundary",
    )
    evaluation = load_json(evaluation_dir / "summary.json")
    target_scope = expected_target_scope(args.target_data_dir.resolve())
    evaluation_audit = validate_evaluation(
        evaluation,
        args.rounds,
        source_clients,
        target_scope,
        args.input_dim,
        args.selection_policy,
        target_client,
    )
    require_finite(evaluation, "formal_evaluation")

    audit = {
        "schema_version": "gaps.lab_three_gas.attempt_audit.v1",
        "status": "valid",
        "direction": args.direction,
        "source_clients": source_clients,
        "target_client": target_client,
        "target_scope": target_scope,
        "rounds": args.rounds,
        "local_epochs": args.local_epochs,
        "da_steps_per_round": args.da_steps,
        "model_profile": args.profile,
        "domain_adaptation_mode": args.da_mode,
        "target_ce_weight": args.target_ce_weight,
        "selection_policy": args.selection_policy,
        "selection_boundary": (
            "final configured round fixed before target test; source calibration "
            "is monitoring only"
            if args.selection_policy == "last_round"
            else (
                "all rounds scored on source calibration only; target test opened "
                "after the selected round was locked"
            )
        ),
        "evidence_boundary": "preliminary_nominal_boundary_screening",
        **evaluation_audit,
    }
    output = args.output or run_dir / "attempt_audit.json"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite audit: {output}")
    output.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
