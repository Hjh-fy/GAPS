"""Generate frozen B2/B5 manifests for the cross-direction IoT-J study."""
from __future__ import annotations

import argparse
import hashlib
import json
import shlex
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

if __package__:
    from scripts.generate_iotj_classification_ablation_commands import (
        BATCH_SIZE,
        CLIENT_LR,
        DA_LR,
        DA_STEPS,
        LOCAL_EPOCHS,
        ROUNDS,
        V3_SPECS,
        _bool,
        _client_command,
        _git_revision,
    )
else:
    from generate_iotj_classification_ablation_commands import (
        BATCH_SIZE,
        CLIENT_LR,
        DA_LR,
        DA_STEPS,
        LOCAL_EPOCHS,
        ROUNDS,
        V3_SPECS,
        _bool,
        _client_command,
        _git_revision,
    )


APPROVED_DIRECTION_IDS = (
    "F1_C1_TO_C5",
    "R1_C5_TO_C1",
    "R2_C45_TO_C1",
)
APPROVED_GROUPS = ("B2", "B5")
APPROVED_SEEDS = (42, 43, 44, 45, 46)
PI_PROJECT = "/home/gaps/GAPS/flower_runtime"
PI_PYTHON = "/home/gaps/GAPS/gaps_rpi_env/bin/python"
ECS_PYTHON = "/root/gaps_env/bin/python"


@dataclass(frozen=True)
class DirectionSpec:
    direction_id: str
    data_root: str
    source_clients: tuple[int, ...]
    target_client: int
    executors: dict[int, str]
    expected_source_train: dict[int, int]
    expected_target_counts: dict[str, int]
    split_seed: int


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_text_lf(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(content)


def _int_mapping(value: Mapping[str, Any], *, field: str) -> dict[int, Any]:
    try:
        return {int(key): item for key, item in value.items()}
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} keys must be integer client IDs") from exc


def load_direction_specs(path: Path) -> tuple[DirectionSpec, ...]:
    payload = load_json(path)
    if payload.get("schema_version") != 1:
        raise ValueError("cross-direction config requires schema_version=1")
    if tuple(payload.get("groups", ())) != APPROVED_GROUPS:
        raise ValueError("cross-direction groups must be exactly B2 and B5")
    split_seed = int(payload.get("split_seed", -1))
    if split_seed != 42:
        raise ValueError("cross-direction split seed must remain 42")

    specs: list[DirectionSpec] = []
    for raw in payload.get("directions", ()):
        sources = tuple(int(item) for item in raw["source_clients"])
        executors = {
            client_id: str(executor)
            for client_id, executor in _int_mapping(
                raw["executors"], field="executors"
            ).items()
        }
        expected_source = {
            client_id: int(count)
            for client_id, count in _int_mapping(
                raw["expected_source_train"], field="expected_source_train"
            ).items()
        }
        spec = DirectionSpec(
            direction_id=str(raw["direction_id"]),
            data_root=str(raw["data_root"]),
            source_clients=sources,
            target_client=int(raw["target_client"]),
            executors=executors,
            expected_source_train=expected_source,
            expected_target_counts={
                str(key): int(value)
                for key, value in raw["expected_target_counts"].items()
            },
            split_seed=split_seed,
        )
        _validate_direction_spec(spec)
        specs.append(spec)
    if tuple(spec.direction_id for spec in specs) != APPROVED_DIRECTION_IDS:
        raise ValueError(
            f"directions must remain in approved order {APPROVED_DIRECTION_IDS}"
        )
    return tuple(specs)


def _validate_direction_spec(spec: DirectionSpec) -> None:
    if not spec.source_clients:
        raise ValueError(f"{spec.direction_id} must have a source client")
    if len(set(spec.source_clients)) != len(spec.source_clients):
        raise ValueError(f"{spec.direction_id} has duplicate source clients")
    if spec.target_client in spec.source_clients:
        raise ValueError(f"{spec.direction_id} target overlaps its sources")
    if spec.target_client in {3, 4}:
        raise ValueError("C3/C4 cannot be target clients")
    if set(spec.executors) != set(spec.source_clients):
        raise ValueError(f"{spec.direction_id} executor mapping does not match sources")
    if set(spec.expected_source_train) != set(spec.source_clients):
        raise ValueError(f"{spec.direction_id} source counts do not match sources")
    if set(spec.executors.values()) - {"pi", "pc"}:
        raise ValueError(f"{spec.direction_id} has an unsupported executor")
    if list(spec.executors.values()).count("pi") > 1:
        raise ValueError(f"{spec.direction_id} assigns multiple clients to Pi")
    if list(spec.executors.values()).count("pc") > 1:
        raise ValueError(f"{spec.direction_id} assigns multiple clients to PC")
    if set(spec.expected_target_counts) != {"calibration", "test"}:
        raise ValueError(f"{spec.direction_id} target counts are incomplete")


def _assert_balanced_labels(path: Path, expected_n: int) -> None:
    labels = np.load(path, mmap_mode="r")
    if labels.shape != (expected_n,):
        raise ValueError(f"unexpected label shape {labels.shape}: {path}")
    values, counts = np.unique(labels, return_counts=True)
    if values.tolist() != [0, 1, 2, 3] or len(set(counts.tolist())) != 1:
        raise ValueError(f"classification labels are not four-class balanced: {path}")


def _validate_split_arrays(client_dir: Path, split: str, expected_n: int) -> None:
    required = (
        f"{split}_features.npy",
        f"{split}_classification_labels.npy",
        f"{split}_phase_labels.npy",
        f"{split}_regression_labels.npy",
    )
    for name in required:
        path = client_dir / name
        if not path.is_file():
            raise FileNotFoundError(path)
        array = np.load(path, mmap_mode="r")
        if int(array.shape[0]) != expected_n:
            raise ValueError(
                f"expected {expected_n} rows, found {array.shape[0]}: {path}"
            )
    _assert_balanced_labels(
        client_dir / f"{split}_classification_labels.npy", expected_n
    )


def validate_direction_data(spec: DirectionSpec, repo_root: Path) -> Path:
    data_root = repo_root / "dataset" / spec.data_root
    for name in ("split_info.json", "norm_stats.npz"):
        if not (data_root / name).is_file():
            raise FileNotFoundError(data_root / name)
    for client_id, expected_n in spec.expected_source_train.items():
        _validate_split_arrays(data_root / f"client_{client_id}", "train", expected_n)
    for split, expected_n in spec.expected_target_counts.items():
        _validate_split_arrays(
            data_root / f"client_{spec.target_client}", split, expected_n
        )
    return data_root


def _client_paths(data_root: str, clients: Iterable[int]) -> str:
    return ",".join(f"dataset/{data_root}/client_{client}" for client in clients)


def _run_name(direction: DirectionSpec, group_id: str, seed: int) -> str:
    da_label = "corrected_full_da" if group_id == "B5" else "corrected_server_da"
    return (
        f"{group_id}_proto_replay_{da_label}_"
        f"{direction.direction_id.lower()}_s{seed}_r25"
    )


def _server_command(
    direction: DirectionSpec,
    group_id: str,
    run_name: str,
    seed: int,
    results_root: str,
) -> list[str]:
    spec = V3_SPECS[group_id]
    return [
        ECS_PYTHON,
        "-m",
        "gaps_flower.server_app",
        "--server-address", "0.0.0.0:8080",
        "--rounds", str(ROUNDS),
        "--min-clients", str(len(direction.source_clients)),
        "--strategy", spec.strategy,
        "--profile", spec.profile,
        "--seed", str(seed),
        "--run-name", run_name,
        "--output-dir", f"{results_root}/{run_name}",
        "--save-history", "true",
        "--use-selective-agg", _bool(spec.use_selective_agg),
        "--use-proto-mmd", _bool(spec.use_proto_mmd_diagnostics),
        "--da-preset", spec.da_preset,
        "--use-domain-adapt", _bool(spec.use_domain_adapt),
        "--server-val-data", _client_paths(direction.data_root, direction.source_clients),
        "--server-calib-data", _client_paths(direction.data_root, (direction.target_client,)),
        "--domain-adapt-steps", str(DA_STEPS),
        "--domain-adapt-warmup", "0",
        "--da-use-coral", _bool(spec.da_use_coral),
        "--da-use-mmd", _bool(spec.da_use_mmd),
        "--da-use-adversarial", _bool(spec.da_use_adversarial),
        "--da-mmd-objective", spec.da_mmd_objective,
        "--da-stage-alignment", spec.da_stage_alignment,
        "--da-adv-feature-objective", spec.da_adv_feature_objective,
        "--da-coral-class-conditional", "true",
        "--strict-calibration-split", "true",
        "--da-device", "cpu",
        "--use-adapted-as-global", _bool(spec.use_domain_adapt),
        "--da-lambda-coral", str(spec.da_lambda_coral),
        "--da-lambda-global-mmd", str(spec.da_lambda_global_mmd),
        "--da-lambda-class-mmd", str(spec.da_lambda_class_mmd),
        "--da-lambda-proto-anchor", str(spec.da_lambda_proto_anchor),
        "--da-lambda-adv", str(spec.da_lambda_adv),
        "--da-lambda-target-ce", str(spec.da_lambda_target_ce),
        "--da-lambda-proto", str(spec.da_lambda_proto),
        "--da-lambda-consistency", str(spec.da_lambda_consistency),
        "--da-lambda-residual", str(spec.da_lambda_residual),
        "--da-lambda-proto-mmd", str(spec.da_lambda_proto_mmd),
        "--da-lambda-stage-mmd", str(spec.da_lambda_stage_mmd),
        "--da-target-ce-label-smoothing", "0.0",
        "--da-target-ce-class-balanced", "false",
        "--da-server-opt-lr", str(DA_LR),
    ]


def _active_file_hashes(direction: DirectionSpec, data_root: Path) -> dict[str, str]:
    paths = [data_root / "split_info.json", data_root / "norm_stats.npz"]
    for client_id in direction.source_clients:
        paths.extend(
            data_root / f"client_{client_id}" / f"train_{suffix}.npy"
            for suffix in (
                "features",
                "classification_labels",
                "phase_labels",
                "regression_labels",
            )
        )
    for split in ("calibration", "test"):
        paths.extend(
            data_root / f"client_{direction.target_client}" / f"{split}_{suffix}.npy"
            for suffix in (
                "features",
                "classification_labels",
                "phase_labels",
                "regression_labels",
            )
        )
    return {
        str(path.relative_to(data_root)).replace("\\", "/"): _sha256(path)
        for path in paths
    }


def build_run_manifest(
    direction: DirectionSpec,
    group_id: str,
    seed: int,
    *,
    repo_root: Path,
    results_root: str,
) -> dict[str, Any]:
    if group_id not in APPROVED_GROUPS:
        raise ValueError("cross-direction study permits only B2 and B5")
    if seed not in APPROVED_SEEDS:
        raise ValueError(f"unsupported training seed: {seed}")
    _validate_direction_spec(direction)
    data_root = validate_direction_data(direction, repo_root)
    spec = V3_SPECS[group_id]
    run_name = _run_name(direction, group_id, seed)
    clients: list[dict[str, Any]] = []
    for client_id in direction.source_clients:
        executor = direction.executors[client_id]
        if executor == "pi":
            runtime_root = f"{PI_PROJECT}/dataset/{direction.data_root}"
            python_bin = PI_PYTHON
            extension = "sh"
        else:
            runtime_root = str(data_root.resolve())
            python_bin = "python"
            extension = "ps1"
        clients.append(
            {
                "client_id": client_id,
                "executor": executor,
                "script_name": f"client_c{client_id}_{executor}_command.{extension}",
                "command": _client_command(
                    client_id,
                    spec.profile,
                    seed,
                    runtime_root,
                    python_bin,
                    "cpu",
                ),
            }
        )
    manifest = {
        "schema_version": 1,
        "study_id": "IOTJ-B2-B5-CROSS-DIRECTION-20260713",
        "direction_id": direction.direction_id,
        "group_id": group_id,
        "method_version": "v3_corrected",
        "run_name": run_name,
        "scheduled_for_training": True,
        "execution_stage": (
            "cross_direction_screening"
            if seed == 42
            else "cross_direction_confirmation"
        ),
        "protocol": {
            "source_clients": list(direction.source_clients),
            "target_clients": [direction.target_client],
            "data_root": direction.data_root,
            "split_seed": direction.split_seed,
            "training_seed": seed,
            "expected_source_train": {
                str(key): value for key, value in direction.expected_source_train.items()
            },
            "expected_target_counts": direction.expected_target_counts,
        },
        "training": {
            "rounds": ROUNDS,
            "local_epochs": LOCAL_EPOCHS,
            "batch_size": BATCH_SIZE,
            "client_lr": CLIENT_LR,
            "profile": spec.profile,
            "strategy": spec.strategy,
            "use_selective_agg": spec.use_selective_agg,
            "use_proto_mmd_diagnostics": spec.use_proto_mmd_diagnostics,
        },
        "server_adaptation": {
            "enabled": spec.use_domain_adapt,
            "preset": spec.da_preset,
            "steps": DA_STEPS,
            "lr": DA_LR,
            "use_coral": spec.da_use_coral,
            "use_mmd": spec.da_use_mmd,
            "use_adversarial": spec.da_use_adversarial,
            "mmd_objective": spec.da_mmd_objective,
            "stage_alignment": spec.da_stage_alignment,
            "adv_feature_objective": spec.da_adv_feature_objective,
            "lambda_coral": spec.da_lambda_coral,
            "lambda_global_mmd": spec.da_lambda_global_mmd,
            "lambda_class_mmd": spec.da_lambda_class_mmd,
            "lambda_proto_anchor": spec.da_lambda_proto_anchor,
            "lambda_adv": spec.da_lambda_adv,
            "lambda_target_ce": spec.da_lambda_target_ce,
            "lambda_proto": spec.da_lambda_proto,
            "lambda_consistency": spec.da_lambda_consistency,
            "lambda_residual": spec.da_lambda_residual,
            "lambda_proto_mmd": spec.da_lambda_proto_mmd,
            "lambda_stage_mmd": spec.da_lambda_stage_mmd,
        },
        "topology": {
            "server": "Alibaba Cloud ECS",
            "source_executors": {
                str(key): value for key, value in direction.executors.items()
            },
            "target_role": "server-side calibration only; target test excluded from training",
        },
        "commands": {
            "server_ecs": _server_command(
                direction, group_id, run_name, seed, results_root
            ),
            "clients": clients,
        },
        "provenance": {
            "code_revision": _git_revision(repo_root),
            "active_file_sha256": _active_file_hashes(direction, data_root),
        },
    }
    validate_manifest(manifest)
    return manifest


def _command_value(command: Sequence[str], option: str) -> str:
    try:
        return command[command.index(option) + 1]
    except (ValueError, IndexError) as exc:
        raise ValueError(f"command is missing {option}") from exc


def validate_manifest(manifest: dict[str, Any]) -> None:
    if manifest["group_id"] not in APPROVED_GROUPS:
        raise ValueError("cross-direction study permits only B2 and B5")
    protocol = manifest["protocol"]
    training = manifest["training"]
    adaptation = manifest["server_adaptation"]
    if training["rounds"] != 25 or training["local_epochs"] != 5:
        raise ValueError("classification schedule must remain 25 rounds and 5 epochs")
    if training["batch_size"] != 32 or training["client_lr"] != 5e-4:
        raise ValueError("classification optimizer contract changed")
    if training["profile"] != "proto_replay" or not training["use_selective_agg"]:
        raise ValueError("B2/B5 semantic base changed")
    expected_common = {
        "lambda_proto_anchor": 0.3,
        "lambda_proto": 0.05,
        "lambda_consistency": 2.0,
        "lambda_residual": 0.1,
        "lambda_proto_mmd": 0.0,
        "lambda_target_ce": 0.0,
        "lambda_global_mmd": 0.5,
        "lambda_class_mmd": 0.5,
    }
    for key, expected in expected_common.items():
        if adaptation[key] != expected:
            raise ValueError(f"frozen adaptation field changed: {key}")
    expected_extra = (
        {"lambda_coral": 0.0, "lambda_stage_mmd": 0.0, "lambda_adv": 0.0}
        if manifest["group_id"] == "B2"
        else {"lambda_coral": 0.5, "lambda_stage_mmd": 0.2, "lambda_adv": 0.5}
    )
    for key, expected in expected_extra.items():
        if adaptation[key] != expected:
            raise ValueError(f"frozen {manifest['group_id']} field changed: {key}")
    if adaptation["mmd_objective"] != "mmd2":
        raise ValueError("B2/B5 require conventional MMD-squared")
    if set(protocol["source_clients"]) & set(protocol["target_clients"]):
        raise ValueError("source and target clients overlap")
    server = manifest["commands"]["server_ecs"]
    if int(_command_value(server, "--min-clients")) != len(protocol["source_clients"]):
        raise ValueError("server min-clients does not match source clients")
    expected_calib = f"dataset/{protocol['data_root']}/client_{protocol['target_clients'][0]}"
    if _command_value(server, "--server-calib-data") != expected_calib:
        raise ValueError("server calibration path does not match target client")
    client_rows = manifest["commands"]["clients"]
    if [row["client_id"] for row in client_rows] != protocol["source_clients"]:
        raise ValueError("client commands do not match source clients")
    for row in client_rows:
        command = row["command"]
        if int(_command_value(command, "--client-id")) != row["client_id"]:
            raise ValueError("client command ID mismatch")
        if protocol["data_root"] not in _command_value(command, "--data-root"):
            raise ValueError("client command data root mismatch")


def _write_command_files(run_dir: Path, manifest: dict[str, Any]) -> None:
    _write_text_lf(
        run_dir / "command_manifest.json",
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
    )
    _write_text_lf(
        run_dir / "server_command.sh",
        "#!/usr/bin/env bash\nset -euo pipefail\ncd /root/GAPS\n"
        + shlex.join(manifest["commands"]["server_ecs"])
        + "\n",
    )
    for row in manifest["commands"]["clients"]:
        path = run_dir / row["script_name"]
        if row["executor"] == "pi":
            content = (
                "#!/usr/bin/env bash\nset -euo pipefail\n"
                f"cd {shlex.quote(PI_PROJECT)}\n"
                + shlex.join(row["command"])
                + "\n"
            )
        else:
            args = ",\n    ".join(json.dumps(arg) for arg in row["command"])
            content = (
                '$ErrorActionPreference = "Stop"\n$argsList = @(\n    '
                + args
                + "\n)\n& $argsList[0] $argsList[1..($argsList.Count - 1)]\n"
            )
        _write_text_lf(path, content)


def generate_manifests(
    config_path: Path,
    output_root: Path,
    *,
    repo_root: Path,
    results_root: str,
    seeds: Sequence[int],
) -> list[dict[str, Any]]:
    directions = load_direction_specs(config_path)
    manifests: list[dict[str, Any]] = []
    for seed in seeds:
        if seed not in APPROVED_SEEDS:
            raise ValueError(f"unsupported training seed: {seed}")
        for direction in directions:
            for group_id in APPROVED_GROUPS:
                manifest = build_run_manifest(
                    direction,
                    group_id,
                    seed,
                    repo_root=repo_root,
                    results_root=results_root,
                )
                _write_command_files(output_root / manifest["run_name"], manifest)
                manifests.append(manifest)
    index = {
        "schema_version": 1,
        "study_id": "IOTJ-B2-B5-CROSS-DIRECTION-20260713",
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "seeds": list(seeds),
        "training_runs": [row["run_name"] for row in manifests],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_text_lf(
        output_root / "command_index.json",
        json.dumps(index, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
    )
    return manifests


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/iotj_b2_b5_cross_direction_20260713.json"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/iotj_b2_b5_cross_direction_20260713_commands"),
    )
    parser.add_argument(
        "--results-root", default="results/iotj_b2_b5_cross_direction_20260713"
    )
    parser.add_argument(
        "--seeds",
        default="42",
        help="Comma-separated approved training seeds",
    )
    parser.add_argument("--seed", type=int, help="Generate one approved seed")
    args = parser.parse_args(argv)
    seeds = (
        (args.seed,)
        if args.seed is not None
        else tuple(int(item.strip()) for item in args.seeds.split(",") if item.strip())
    )
    repo_root = Path(__file__).resolve().parents[1]
    manifests = generate_manifests(
        args.config,
        args.output_root,
        repo_root=repo_root,
        results_root=args.results_root,
        seeds=seeds,
    )
    print(
        json.dumps(
            {
                "runs": len(manifests),
                "seeds": list(seeds),
                "output_root": str(args.output_root),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
