"""Generate frozen C1/C2-to-C5 cloud-edge classification commands."""
from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


DATA_ROOT_NAME = "client_data_c1234src_c5tgt_2080_timeaware_60_170_window_fullgrid"
SOURCE_CLIENTS = (1, 2)
TARGET_CLIENTS = (5,)
ROUNDS = 25
LOCAL_EPOCHS = 5
BATCH_SIZE = 32
CLIENT_LR = 5e-4
DA_STEPS = 100
DA_LR = 5e-4
SCREENING_SEED = 42
CONFIRMATION_SEEDS = (42, 43, 44, 45, 46)
CONFIRMATION_GROUPS = frozenset({"A0", "A0T", "A4", "A4S", "A5", "A7"})
CORE_SCREENING_GROUPS = ("A0", "A0T", "A2", "A3", "A4", "A4S", "A5", "A6", "A7")
V3_SCREENING_GROUPS = ("B1", "B2", "B3", "B4", "B5")
V3_CONFIRMATION_GROUPS = ("A0", "A0T", "A6", "B5")
APPENDIX_GROUPS = (
    "A7-noCORAL",
    "A7-noMMD",
    "A7-noADV",
    "A7-noSemantic",
    "A7-noStage",
)


@dataclass(frozen=True)
class AblationSpec:
    group_id: str
    profile: str
    strategy: str
    use_selective_agg: bool
    use_proto_mmd_diagnostics: bool
    use_domain_adapt: bool
    da_preset: str
    da_use_coral: bool = False
    da_use_mmd: bool = False
    da_use_adversarial: bool = False
    da_lambda_coral: float = 0.0
    da_lambda_global_mmd: float = 0.0
    da_lambda_class_mmd: float = 0.0
    da_lambda_proto_anchor: float = 0.0
    da_lambda_adv: float = 0.0
    da_lambda_proto: float = 0.0
    da_lambda_consistency: float = 0.0
    da_lambda_residual: float = 0.0
    da_lambda_proto_mmd: float = 0.0
    da_lambda_stage_mmd: float = 0.0
    da_lambda_target_ce: float = 0.0
    da_mmd_objective: str = "legacy_quartic"
    da_stage_alignment: str = "legacy_intra_domain"
    da_adv_feature_objective: str = "legacy_grl_plus"


SPECS = {
    "A0": AblationSpec("A0", "ce_only", "fedavg", False, False, False, "none"),
    "A0T": AblationSpec(
        "A0T", "ce_only", "gaps", False, False, True, "none",
        da_lambda_target_ce=1.0,
    ),
    "A1": AblationSpec("A1", "ce_only", "gaps", False, False, False, "none"),
    # These groups use GAPS only to exchange prototypes. With selective
    # aggregation disabled, parameter aggregation is FedAvg-equivalent.
    "A2": AblationSpec("A2", "align_only", "gaps", False, False, False, "none"),
    "A3": AblationSpec("A3", "replay_only", "gaps", False, False, False, "none"),
    "A4": AblationSpec("A4", "align_replay", "gaps", False, False, False, "none"),
    "A4S": AblationSpec("A4S", "align_replay", "gaps", True, False, False, "none"),
    "A5": AblationSpec(
        "A5", "align_replay", "gaps", True, False, True, "none",
        da_use_coral=True,
        da_use_mmd=True,
        da_use_adversarial=True,
        da_lambda_coral=0.5,
        da_lambda_global_mmd=0.5,
        da_lambda_class_mmd=0.5,
        da_lambda_adv=0.5,
    ),
    "A6": AblationSpec(
        "A6", "proto_replay", "gaps", True, False, True, "none",
        da_lambda_proto_anchor=0.3,
        da_lambda_proto=0.05,
        da_lambda_consistency=2.0,
        da_lambda_residual=0.1,
        da_lambda_proto_mmd=0.2,
    ),
    "A7": AblationSpec(
        "A7", "proto_replay", "gaps", True, False, True, "fixed_da_strong",
        da_use_coral=True,
        da_use_mmd=True,
        da_use_adversarial=True,
        da_lambda_coral=0.5,
        da_lambda_global_mmd=0.5,
        da_lambda_class_mmd=0.5,
        da_lambda_proto_anchor=0.3,
        da_lambda_adv=0.5,
        da_lambda_proto=0.05,
        da_lambda_consistency=2.0,
        da_lambda_residual=0.1,
        da_lambda_proto_mmd=0.2,
        da_lambda_stage_mmd=0.2,
    ),
    "A7-noCORAL": AblationSpec(
        "A7-noCORAL", "proto_replay", "gaps", True, False, True, "none",
        da_use_mmd=True, da_use_adversarial=True,
        da_lambda_global_mmd=0.5, da_lambda_class_mmd=0.5,
        da_lambda_proto_anchor=0.3, da_lambda_adv=0.5,
        da_lambda_proto=0.05, da_lambda_consistency=2.0,
        da_lambda_residual=0.1, da_lambda_proto_mmd=0.2,
        da_lambda_stage_mmd=0.2,
    ),
    "A7-noMMD": AblationSpec(
        "A7-noMMD", "proto_replay", "gaps", True, False, True, "none",
        da_use_coral=True, da_use_adversarial=True,
        da_lambda_coral=0.5, da_lambda_proto_anchor=0.3,
        da_lambda_adv=0.5, da_lambda_proto=0.05,
        da_lambda_consistency=2.0, da_lambda_residual=0.1,
    ),
    "A7-noADV": AblationSpec(
        "A7-noADV", "proto_replay", "gaps", True, False, True, "none",
        da_use_coral=True, da_use_mmd=True,
        da_lambda_coral=0.5, da_lambda_global_mmd=0.5,
        da_lambda_class_mmd=0.5, da_lambda_proto_anchor=0.3,
        da_lambda_proto=0.05, da_lambda_consistency=2.0,
        da_lambda_residual=0.1, da_lambda_proto_mmd=0.2,
        da_lambda_stage_mmd=0.2,
    ),
    "A7-noSemantic": AblationSpec(
        "A7-noSemantic", "align_replay", "gaps", True, False, True, "none",
        da_use_coral=True, da_use_mmd=True, da_use_adversarial=True,
        da_lambda_coral=0.5, da_lambda_global_mmd=0.5,
        da_lambda_class_mmd=0.5, da_lambda_adv=0.5,
        da_lambda_stage_mmd=0.2,
    ),
    "A7-noStage": AblationSpec(
        "A7-noStage", "proto_replay", "gaps", True, False, True, "none",
        da_use_coral=True, da_use_mmd=True, da_use_adversarial=True,
        da_lambda_coral=0.5, da_lambda_global_mmd=0.5,
        da_lambda_class_mmd=0.5, da_lambda_proto_anchor=0.3,
        da_lambda_adv=0.5, da_lambda_proto=0.05,
        da_lambda_consistency=2.0, da_lambda_residual=0.1,
        da_lambda_proto_mmd=0.2,
    ),
}


_V3_COMMON = {
    "da_lambda_proto_anchor": 0.3,
    "da_lambda_proto": 0.05,
    "da_lambda_consistency": 2.0,
    "da_lambda_residual": 0.1,
    "da_lambda_proto_mmd": 0.0,
    "da_mmd_objective": "mmd2",
    "da_stage_alignment": "cross_domain_same_class_phase",
    "da_adv_feature_objective": "wasserstein_min",
}

V3_SPECS = {
    "B1": AblationSpec(
        "B1", "proto_replay", "gaps", True, False, True, "none",
        da_use_coral=True,
        da_lambda_coral=0.5,
        **_V3_COMMON,
    ),
    "B2": AblationSpec(
        "B2", "proto_replay", "gaps", True, False, True, "none",
        da_use_mmd=True,
        da_lambda_global_mmd=0.5,
        da_lambda_class_mmd=0.5,
        **_V3_COMMON,
    ),
    "B3": AblationSpec(
        "B3", "proto_replay", "gaps", True, False, True, "none",
        da_use_mmd=True,
        da_lambda_stage_mmd=0.2,
        **_V3_COMMON,
    ),
    "B4": AblationSpec(
        "B4", "proto_replay", "gaps", True, False, True, "none",
        da_use_adversarial=True,
        da_lambda_adv=0.5,
        **_V3_COMMON,
    ),
    "B5": AblationSpec(
        "B5", "proto_replay", "gaps", True, False, True, "none",
        da_use_coral=True,
        da_use_mmd=True,
        da_use_adversarial=True,
        da_lambda_coral=0.5,
        da_lambda_global_mmd=0.5,
        da_lambda_class_mmd=0.5,
        da_lambda_adv=0.5,
        da_lambda_stage_mmd=0.2,
        **_V3_COMMON,
    ),
}

ALL_SPECS = {**SPECS, **V3_SPECS}


def _bool(value: bool) -> str:
    return "true" if value else "false"


def _write_text_lf(path: Path, content: str) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(content)


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _client_paths(data_root: str, clients: Iterable[int]) -> str:
    return ",".join(f"dataset/{data_root}/client_{client}" for client in clients)


def _run_name(group_id: str, seed: int) -> str:
    spec = ALL_SPECS[group_id]
    if group_id == "B5":
        da_label = "corrected_full_da"
    elif group_id.startswith("B"):
        da_label = "corrected_server_da"
    else:
        da_label = "full_da" if group_id == "A7" else ("server_da" if spec.use_domain_adapt else "no_da")
    return f"{group_id}_{spec.profile}_{da_label}_c12_to_c5_s{seed}_r25"


def _server_command(spec: AblationSpec, run_name: str, seed: int, results_root: str) -> list[str]:
    command = [
        "/root/gaps_env/bin/python",
        "-m",
        "gaps_flower.server_app",
        "--server-address", "0.0.0.0:8080",
        "--rounds", str(ROUNDS),
        "--min-clients", str(len(SOURCE_CLIENTS)),
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
        "--server-val-data", _client_paths(DATA_ROOT_NAME, SOURCE_CLIENTS),
        "--server-calib-data", _client_paths(DATA_ROOT_NAME, TARGET_CLIENTS),
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
    return command


def _client_command(
    client_id: int,
    profile: str,
    seed: int,
    data_root: str,
    python_bin: str,
    device: str,
) -> list[str]:
    return [
        python_bin,
        "-m",
        "gaps_flower.client_app",
        "--server-address", "127.0.0.1:18080",
        "--client-id", str(client_id),
        "--data-root", data_root,
        "--device", device,
        "--local-epochs", str(LOCAL_EPOCHS),
        "--batch-size", str(BATCH_SIZE),
        "--profile", profile,
        "--seed", str(seed),
    ]


def build_run_manifest(
    group_id: str,
    seed: int,
    *,
    repo_root: Path,
    results_root: str,
) -> dict[str, Any]:
    if group_id not in ALL_SPECS:
        raise ValueError(f"unknown group: {group_id}")
    if seed not in CONFIRMATION_SEEDS:
        raise ValueError(f"unsupported seed: {seed}")
    spec = ALL_SPECS[group_id]
    run_name = _run_name(group_id, seed)
    data_root = repo_root / "dataset" / DATA_ROOT_NAME
    scheduled = group_id != "A1"
    if group_id == "A1":
        execution_stage = "contract_only"
    elif group_id in APPENDIX_GROUPS:
        execution_stage = "appendix_conditional"
    elif group_id in V3_SCREENING_GROUPS and seed == SCREENING_SEED:
        execution_stage = "v3_correction_screening"
    elif group_id in V3_SCREENING_GROUPS:
        execution_stage = "v3_confirmation"
    elif seed == SCREENING_SEED:
        execution_stage = "core_screening"
    else:
        execution_stage = "confirmation"
    manifest = {
        "schema_version": 1,
        "group_id": group_id,
        "method_version": "v3_corrected" if group_id.startswith("B") else "v2_legacy",
        "run_name": run_name,
        "scheduled_for_training": scheduled,
        "contract_only": group_id == "A1",
        "execution_stage": execution_stage,
        "protocol": {
            "source_clients": list(SOURCE_CLIENTS),
            "target_clients": list(TARGET_CLIENTS),
            "data_root": DATA_ROOT_NAME,
            "split_seed": 42,
            "training_seed": seed,
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
        "causal_factors": {
            "prototype_alignment": spec.profile in {
                "align_only", "align_replay", "proto_only", "proto_replay"
            },
            "replay_distillation": spec.profile in {
                "replay_only", "align_replay", "proto_replay"
            },
            "device_residual_statistics": spec.profile in {"proto_only", "proto_replay"},
            "selective_aggregation": spec.use_selective_agg,
            "server_distribution_adaptation": bool(
                spec.da_use_coral or spec.da_use_mmd or spec.da_use_adversarial
            ),
            "server_semantic_adaptation": bool(
                spec.da_lambda_proto_anchor
                or spec.da_lambda_proto
                or spec.da_lambda_consistency
                or spec.da_lambda_residual
                or spec.da_lambda_proto_mmd
            ),
            "server_stage_mmd": bool(spec.da_lambda_stage_mmd),
            "target_supervised_ce": bool(spec.da_lambda_target_ce),
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
            "C1": "physical Raspberry Pi CPU",
            "C2": "physical Windows PC CPU",
            "C5": "server-side calibration only; no target test labels in training",
        },
        "commands": {
            "server_ecs": _server_command(spec, run_name, seed, results_root),
            "client_c1_pi": _client_command(
                1,
                spec.profile,
                seed,
                f"/home/gaps/GAPS/flower_runtime/dataset/{DATA_ROOT_NAME}",
                "/home/gaps/GAPS/gaps_rpi_env/bin/python",
                "cpu",
            ),
            "client_c2_pc": _client_command(
                2,
                spec.profile,
                seed,
                str(data_root.resolve()),
                "python",
                "cpu",
            ),
        },
        "provenance": {
            "code_revision": _git_revision(repo_root),
            "split_info_sha256": _sha256(data_root / "split_info.json"),
            "norm_stats_sha256": _sha256(data_root / "norm_stats.npz"),
        },
    }
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest: dict[str, Any]) -> None:
    protocol = manifest["protocol"]
    training = manifest["training"]
    adaptation = manifest["server_adaptation"]
    if protocol["source_clients"] != [1, 2] or protocol["target_clients"] != [5]:
        raise ValueError("primary protocol must be C1/C2 source and C5-only target")
    if any(client in protocol["target_clients"] for client in (3, 4)):
        raise ValueError("C3/C4 cannot be primary targets")
    if training["rounds"] != 25 or training["local_epochs"] != 5:
        raise ValueError("classification schedule must be 25 rounds and 5 local epochs")
    if training["batch_size"] != 32 or training["client_lr"] != 5e-4:
        raise ValueError("classification optimizer contract changed")
    if adaptation["steps"] != 100 or adaptation["lr"] != 5e-4:
        raise ValueError("server adaptation optimizer contract changed")
    group_id = manifest["group_id"]
    if group_id == "A0T":
        if adaptation["lambda_target_ce"] != 1.0:
            raise ValueError("A0T must use the frozen target CE weight")
    elif adaptation["lambda_target_ce"] != 0.0:
        raise ValueError("target CE must remain disabled outside A0T")
    if group_id in {"A2", "A3", "A4"} and training["use_selective_agg"]:
        raise ValueError(f"{group_id} must isolate client losses from selective aggregation")
    if group_id in {"A4S", "A5", "A6", "A7", *V3_SCREENING_GROUPS} and not training["use_selective_agg"]:
        raise ValueError(f"{group_id} requires the selective-aggregation base")
    if training["use_proto_mmd_diagnostics"]:
        raise ValueError("timing-neutral primary runs keep prototype MMD diagnostics disabled")
    if group_id in V3_SCREENING_GROUPS:
        if adaptation["lambda_proto_mmd"] != 0.0:
            raise ValueError("v3 correction groups must disable detached prototype pair-L2")
        expected_modes = {
            "mmd_objective": "mmd2",
            "stage_alignment": "cross_domain_same_class_phase",
            "adv_feature_objective": "wasserstein_min",
        }
        for key, expected in expected_modes.items():
            if adaptation[key] != expected:
                raise ValueError(f"{group_id} requires {key}={expected}")


def _write_command_files(run_dir: Path, manifest: dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_text_lf(
        run_dir / "command_manifest.json",
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
    )
    commands = manifest["commands"]
    _write_text_lf(
        run_dir / "server_command.sh",
        "#!/usr/bin/env bash\nset -euo pipefail\ncd /root/GAPS\n" + shlex.join(commands["server_ecs"]) + "\n",
    )
    _write_text_lf(
        run_dir / "client_c1_pi_command.sh",
        "#!/usr/bin/env bash\nset -euo pipefail\ncd /home/gaps/GAPS/flower_runtime\n"
        + shlex.join(commands["client_c1_pi"])
        + "\n",
    )
    pc_args = ",\n    ".join(json.dumps(arg) for arg in commands["client_c2_pc"])
    _write_text_lf(
        run_dir / "client_c2_pc_command.ps1",
        "$ErrorActionPreference = \"Stop\"\n$argsList = @(\n    "
        + pc_args
        + "\n)\n& $argsList[0] $argsList[1..($argsList.Count - 1)]\n",
    )


def generate_manifests(
    output_root: Path,
    *,
    repo_root: Path,
    results_root: str,
    include_confirmation_seeds: bool,
    suite: str = "v2",
) -> list[dict[str, Any]]:
    if suite not in {"v2", "v3"}:
        raise ValueError(f"unsupported suite: {suite}")
    rows: list[dict[str, Any]] = []
    screening_groups = tuple(SPECS) if suite == "v2" else V3_SCREENING_GROUPS
    confirmation_groups = (
        tuple(sorted(CONFIRMATION_GROUPS))
        if suite == "v2"
        else V3_CONFIRMATION_GROUPS
    )
    schedule: list[tuple[str, int]] = [
        (group, SCREENING_SEED) for group in screening_groups
    ]
    if include_confirmation_seeds:
        schedule.extend(
            (group, seed)
            for group in confirmation_groups
            for seed in CONFIRMATION_SEEDS
            if seed != SCREENING_SEED
        )
    for group_id, seed in schedule:
        manifest = build_run_manifest(
            group_id, seed, repo_root=repo_root, results_root=results_root
        )
        _write_command_files(output_root / manifest["run_name"], manifest)
        rows.append(manifest)
    index = {
        "schema_version": 1,
        "suite": suite,
        "protocol": "C1/C2 source -> C5 target only",
        "training_runs": [row["run_name"] for row in rows if row["scheduled_for_training"]],
        "contract_only_runs": [row["run_name"] for row in rows if row["contract_only"]],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (output_root / "command_index.json").write_text(
        json.dumps(index, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/iotj_classification_ablation_20260711_v2_commands"),
    )
    parser.add_argument(
        "--results-root",
        default="results/iotj_classification_ablation_20260711_v2",
    )
    parser.add_argument("--suite", choices=("v2", "v3"), default="v2")
    parser.add_argument("--include-confirmation-seeds", action="store_true")
    args = parser.parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    rows = generate_manifests(
        args.output_root,
        repo_root=repo_root,
        results_root=args.results_root,
        include_confirmation_seeds=args.include_confirmation_seeds,
        suite=args.suite,
    )
    training_count = sum(row["scheduled_for_training"] for row in rows)
    print(
        f"Wrote {len(rows)} manifests to {args.output_root}; "
        f"{training_count} scheduled training runs"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
