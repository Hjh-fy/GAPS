"""Run formal classifier-aligned C5 regression suites on Alibaba Cloud ECS."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import sys
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_iotj_c5_regression_suite import expected_outputs


@dataclass(frozen=True)
class ClassifierSpec:
    classifier_id: str
    local_checkpoint: Path


def parse_classifier_spec(value: str) -> ClassifierSpec:
    if "=" not in value:
        raise ValueError("classifier specification must use ID=PATH")
    classifier_id, path = value.split("=", 1)
    classifier_id = classifier_id.strip()
    path = path.strip()
    if not classifier_id:
        raise ValueError("classifier ID must not be empty")
    if not path:
        raise ValueError("classifier checkpoint path must not be empty")
    return ClassifierSpec(classifier_id=classifier_id, local_checkpoint=Path(path))


def merge_cloud_manifest(existing: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    compatibility_fields = (
        "schema_version",
        "training_location",
        "seed",
        "n_random",
        "device",
        "remote_output_base",
    )
    for field in compatibility_fields:
        if existing.get(field) != update.get(field):
            raise ValueError(
                f"cannot append cloud manifest with incompatible {field}: "
                f"{existing.get(field)!r} != {update.get(field)!r}"
            )

    recovered = dict(existing.get("recovered", {}))
    for classifier_id, path in update.get("recovered", {}).items():
        previous = recovered.get(classifier_id)
        if previous is not None and previous != path:
            raise ValueError(
                f"classifier {classifier_id} has conflicting recovered paths: "
                f"{previous!r} != {path!r}"
            )
        recovered[classifier_id] = path

    classifiers = list(existing.get("classifiers", []))
    for classifier_id in update.get("classifiers", []):
        if classifier_id not in classifiers:
            classifiers.append(classifier_id)

    merged = dict(existing)
    merged["classifiers"] = classifiers
    merged["recovered"] = recovered
    return merged


def build_remote_suite_command(
    *,
    classifier_id: str,
    remote_checkpoint: Path | PurePosixPath,
    remote_regression_checkpoint: Path | PurePosixPath,
    remote_data_root: Path | PurePosixPath,
    remote_output_root: Path | PurePosixPath,
    python_bin: str,
    device: str,
    seed: int,
    n_random: int,
) -> list[str]:
    return [
        python_bin,
        "scripts/run_iotj_c5_regression_suite.py",
        "--classifier-checkpoint",
        str(remote_checkpoint).replace("\\", "/"),
        "--classifier-id",
        classifier_id,
        "--regression-checkpoint",
        str(remote_regression_checkpoint).replace("\\", "/"),
        "--data-root",
        str(remote_data_root).replace("\\", "/"),
        "--output-root",
        str(remote_output_root).replace("\\", "/"),
        "--device",
        device,
        "--seed",
        str(seed),
        "--n-random",
        str(n_random),
    ]


def _run(command: Sequence[str], *, timeout: int = 300) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        list(command),
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(f"command failed ({result.returncode}): {' '.join(command)}\n{detail}")
    return result


def _ssh(host: str, command: str, *, timeout: int = 300) -> subprocess.CompletedProcess[str]:
    return _run(
        [
            "ssh",
            "-n",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=10",
            "-o",
            "ServerAliveInterval=30",
            "-o",
            "ServerAliveCountMax=4",
            host,
            command,
        ],
        timeout=timeout,
    )


def _scp_to_remote(paths: Sequence[Path], destination: str, *, recursive: bool = False) -> None:
    flag = "-pr" if recursive else "-p"
    _run(["scp", flag, *[str(path) for path in paths], destination], timeout=900)


def _sync_code(ecs_host: str, ecs_project: PurePosixPath) -> None:
    _ssh(
        ecs_host,
        f"mkdir -p {shlex.quote(str(ecs_project))}",
    )
    root_python = sorted(REPO_ROOT.glob("*.py"))
    if not root_python:
        raise FileNotFoundError("no root Python files found for ECS synchronization")
    _scp_to_remote(root_python, f"{ecs_host}:{ecs_project}/")
    for directory in ("scripts", "gaps_flower", "gaps_deploy"):
        _scp_to_remote(
            [REPO_ROOT / directory],
            f"{ecs_host}:{ecs_project}/",
            recursive=True,
        )


def _sync_checkpoints(
    ecs_host: str,
    checkpoint_dir: PurePosixPath,
    classifiers: Sequence[ClassifierSpec],
    regression_checkpoint: Path,
) -> tuple[dict[str, PurePosixPath], PurePosixPath]:
    _ssh(ecs_host, f"mkdir -p {shlex.quote(str(checkpoint_dir))}")
    remote_classifiers: dict[str, PurePosixPath] = {}
    for spec in classifiers:
        if not spec.local_checkpoint.is_file():
            raise FileNotFoundError(spec.local_checkpoint)
        remote_path = checkpoint_dir / f"{spec.classifier_id}.pth"
        _scp_to_remote([spec.local_checkpoint], f"{ecs_host}:{remote_path}")
        remote_classifiers[spec.classifier_id] = remote_path
    if not regression_checkpoint.is_file():
        raise FileNotFoundError(regression_checkpoint)
    remote_regression = checkpoint_dir / "R3aK16_source_regression.pt"
    _scp_to_remote([regression_checkpoint], f"{ecs_host}:{remote_regression}")
    return remote_classifiers, remote_regression


def _remote_state(ecs_host: str, output_root: PurePosixPath) -> str:
    suite_manifest = output_root / "suite_manifest.json"
    command = (
        f"if test -f {shlex.quote(str(suite_manifest))}; then echo COMPLETE; "
        f"elif test -d {shlex.quote(str(output_root))} && "
        f"test -n \"$(find {shlex.quote(str(output_root))} -mindepth 1 -print -quit 2>/dev/null)\"; "
        "then echo PARTIAL; else echo EMPTY; fi"
    )
    return _ssh(ecs_host, command).stdout.strip()


def _run_remote_suite(
    ecs_host: str,
    ecs_project: PurePosixPath,
    output_root: PurePosixPath,
    command: Sequence[str],
) -> None:
    state = _remote_state(ecs_host, output_root)
    if state == "COMPLETE":
        return
    if state != "EMPTY":
        raise RuntimeError(f"refusing partial remote regression output: {output_root}")
    log_path = output_root / "cloud_suite.log"
    quoted = " ".join(shlex.quote(part) for part in command)
    shell = (
        f"mkdir -p {shlex.quote(str(output_root))} && "
        f"cd {shlex.quote(str(ecs_project))} && "
        f"{quoted} > {shlex.quote(str(log_path))} 2>&1"
    )
    try:
        _ssh(ecs_host, shell, timeout=21600)
    except Exception as error:
        tail = _ssh(
            ecs_host,
            f"tail -n 80 {shlex.quote(str(log_path))} 2>/dev/null || true",
        ).stdout
        raise RuntimeError(f"remote regression suite failed for {output_root}\n{tail}") from error
    if _remote_state(ecs_host, output_root) != "COMPLETE":
        raise RuntimeError(f"remote suite exited without completion manifest: {output_root}")


def _recover_suite(
    ecs_host: str,
    remote_output: PurePosixPath,
    local_base: Path,
) -> Path:
    local_base.mkdir(parents=True, exist_ok=True)
    local_output = local_base / remote_output.name
    if local_output.exists():
        if (local_output / "suite_manifest.json").is_file():
            return local_output
        raise RuntimeError(f"refusing to overwrite partial local regression output: {local_output}")
    _run(["scp", "-pr", f"{ecs_host}:{remote_output}", str(local_base)], timeout=1800)
    required = (*expected_outputs(local_output), local_output / "suite_manifest.json")
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("recovered regression suite is incomplete:\n" + "\n".join(missing))
    return local_output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classifier", action="append", required=True, help="ID=local checkpoint path")
    parser.add_argument(
        "--regression-checkpoint",
        type=Path,
        default=Path("results/R3aK16_flower_reg_depth4_dct_src12/regression_fedavg_global.pt"),
    )
    parser.add_argument("--ecs-host", default="root@121.40.139.213")
    parser.add_argument("--ecs-project", default="/root/GAPS")
    parser.add_argument("--python-bin", default="/root/gaps_env/bin/python")
    parser.add_argument(
        "--remote-data-root",
        default="/root/GAPS/dataset/client_data_c1234src_c5tgt_2080_timeaware_60_170_window_fullgrid",
    )
    parser.add_argument("--remote-output-base", default="/root/GAPS/results/iotj_c5_formal_regression_20260713")
    parser.add_argument(
        "--local-output-base",
        type=Path,
        default=Path("results/iotj_c5_formal_regression_20260713"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-random", type=int, default=1000)
    parser.add_argument("--skip-code-sync", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    classifiers = [parse_classifier_spec(value) for value in args.classifier]
    ids = [spec.classifier_id for spec in classifiers]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate classifier IDs: {ids}")
    ecs_project = PurePosixPath(args.ecs_project)
    checkpoint_dir = ecs_project / "results" / "iotj_reg_checkpoints_20260713"
    output_base = PurePosixPath(args.remote_output_base)
    preview = []
    for spec in classifiers:
        preview.append(
            build_remote_suite_command(
                classifier_id=spec.classifier_id,
                remote_checkpoint=checkpoint_dir / f"{spec.classifier_id}.pth",
                remote_regression_checkpoint=checkpoint_dir / "R3aK16_source_regression.pt",
                remote_data_root=PurePosixPath(args.remote_data_root),
                remote_output_root=output_base / spec.classifier_id,
                python_bin=args.python_bin,
                device=args.device,
                seed=args.seed,
                n_random=args.n_random,
            )
        )
    if args.dry_run:
        print(json.dumps(preview, indent=2, ensure_ascii=False))
        return 0

    if not args.skip_code_sync:
        _sync_code(args.ecs_host, ecs_project)
    remote_classifiers, remote_regression = _sync_checkpoints(
        args.ecs_host,
        checkpoint_dir,
        classifiers,
        args.regression_checkpoint,
    )
    recovered: dict[str, str] = {}
    for spec in classifiers:
        remote_output = output_base / spec.classifier_id
        command = build_remote_suite_command(
            classifier_id=spec.classifier_id,
            remote_checkpoint=remote_classifiers[spec.classifier_id],
            remote_regression_checkpoint=remote_regression,
            remote_data_root=PurePosixPath(args.remote_data_root),
            remote_output_root=remote_output,
            python_bin=args.python_bin,
            device=args.device,
            seed=args.seed,
            n_random=args.n_random,
        )
        _run_remote_suite(args.ecs_host, ecs_project, remote_output, command)
        recovered[spec.classifier_id] = str(
            _recover_suite(args.ecs_host, remote_output, args.local_output_base)
        )
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "training_location": "Alibaba Cloud ECS",
        "classifiers": ids,
        "seed": args.seed,
        "n_random": args.n_random,
        "device": args.device,
        "remote_output_base": str(output_base),
        "recovered": recovered,
    }
    args.local_output_base.mkdir(parents=True, exist_ok=True)
    manifest_path = args.local_output_base / "cloud_run_manifest.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest = merge_cloud_manifest(existing, manifest)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
