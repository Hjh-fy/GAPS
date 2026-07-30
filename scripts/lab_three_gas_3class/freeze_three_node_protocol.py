"""Freeze the nominal three-gas, three-host screening protocol.

This creates an immutable source archive plus dataset, topology, and protocol
manifests.  It deliberately does not modify or reuse the frozen public-data
IoT-J evidence identities.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = (
    PROJECT_ROOT / "dataset" / "client_data_lab_3gas_5fold_nominal_v1"
)
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "lab_3gas_three_node_20260729"
    / "protocol_v2"
)

ROOT_SOURCE_FILES = (
    "client.py",
    "config.py",
    "federated_dataset.py",
    "model.py",
    "utils.py",
)
SCRIPT_SOURCE_FILES = (
    "scripts/__init__.py",
    "scripts/remote_launch_flower_client_clean.py",
    "scripts/remote_launch_flower_server_clean.py",
    "scripts/lab_three_gas_3class/evaluate_exposure_checkpoint.py",
    "scripts/lab_three_gas_3class/evaluate_source_target_run.py",
    "scripts/lab_three_gas_3class/remote_runtime_preflight.py",
    "scripts/lab_three_gas_3class/train_centralized_baseline.py",
    "scripts/lab_three_gas_3class/validate_three_node_run.py",
)
CONTROLLER_PATH = (
    PROJECT_ROOT
    / "scripts"
    / "lab_three_gas_3class"
    / "run_lab_three_node_fold.ps1"
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(payload: Any) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def write_json_exclusive(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(
            (
                json.dumps(
                    payload,
                    indent=2,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            ).encode("utf-8")
        )


def copy_exclusive(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as handle:
        handle.write(source.read_bytes())


def source_paths() -> list[Path]:
    paths = [PROJECT_ROOT / relative for relative in ROOT_SOURCE_FILES]
    paths.extend(
        sorted(
            path
            for path in (PROJECT_ROOT / "gaps_flower").glob("*.py")
            if path.is_file()
        )
    )
    paths.extend(PROJECT_ROOT / relative for relative in SCRIPT_SOURCE_FILES)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing source files: {missing}")
    return sorted(set(paths), key=lambda path: path.relative_to(PROJECT_ROOT).as_posix())


def build_source_archive(output_root: Path) -> tuple[Path, dict[str, Any]]:
    archive = output_root / "source" / "lab_three_gas_source.tar"
    archive.parent.mkdir(parents=True, exist_ok=False)
    members: list[dict[str, Any]] = []
    with tarfile.open(archive, "x", format=tarfile.PAX_FORMAT) as bundle:
        for path in source_paths():
            relative = path.relative_to(PROJECT_ROOT).as_posix()
            data = path.read_bytes()
            info = tarfile.TarInfo(relative)
            info.size = len(data)
            info.mode = 0o644
            info.mtime = 0
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            bundle.addfile(info, io.BytesIO(data))
            members.append(
                {
                    "relative_path": relative,
                    "byte_size": len(data),
                    "sha256": sha256_bytes(data),
                }
            )
    manifest = {
        "schema_version": "gaps.lab_three_gas.source.v2",
        "source_archive": str(archive.relative_to(PROJECT_ROOT)),
        "source_archive_sha256": sha256_file(archive),
        "members": members,
    }
    write_json_exclusive(output_root / "source" / "source_manifest.json", manifest)
    return archive, manifest


def build_dataset_manifest(
    output_root: Path, data_root: Path
) -> dict[str, Any]:
    if not data_root.is_dir():
        raise FileNotFoundError(data_root)
    files = []
    for path in sorted(
        (path for path in data_root.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(data_root).as_posix(),
    ):
        files.append(
            {
                "relative_path": path.relative_to(data_root).as_posix(),
                "byte_size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    dataset_summary = json.loads(
        (data_root / "dataset_summary.json").read_text(encoding="utf-8")
    )
    build_config = json.loads(
        (data_root / "build_config.json").read_text(encoding="utf-8")
    )
    core = {
        "schema_version": "gaps.lab_three_gas.dataset.v2",
        "dataset_identity": data_root.name,
        "boundary_mode": dataset_summary["boundary_mode"],
        "split_protocol": dataset_summary.get(
            "split_protocol", "concentration_group_aware_5fold"
        ),
        "folds": sorted(
            int(path.name.removeprefix("fold_"))
            for path in data_root.glob("fold_*")
            if path.is_dir() and path.name.removeprefix("fold_").isdigit()
        ),
        "normalization_fit_clients": build_config["normalization_clients"],
        "final_evidence_eligible": False,
        "files": files,
    }
    manifest = {
        **core,
        "dataset_manifest_sha256": sha256_bytes(canonical_bytes(core)),
    }
    write_json_exclusive(output_root / "dataset_manifest.json", manifest)
    return manifest


def build_topology_manifest(
    output_root: Path, source_sha: str, dataset_name: str
) -> dict[str, Any]:
    core = {
        "schema_version": "gaps.lab_three_gas.topology.v2",
        "controller": {
            "role": "orchestration_only",
            "training": False,
        },
        "hosts": {
            "server": {
                "ssh": "root@121.40.139.213",
                "project_root": "/root/GAPS",
                "python": "/root/gaps_env/bin/python",
                "roles": ["flower_server", "server_domain_adaptation", "P3_target"],
                "runtime_src": f"/root/GAPS/lab_3gas_confirmation_runtime/{source_sha}/src",
                "data_root": f"/root/GAPS/dataset/{dataset_name}",
            },
            "C2": {
                "ssh": "root@114.55.171.63",
                "project_root": "/root/GAPS",
                "python": "/root/gaps_c2_cpu_env/bin/python",
                "roles": ["logical_C2", "P2_source"],
                "runtime_src": f"/root/GAPS/confirmation_runtime_c2/{source_sha}/src",
                "data_root": f"/root/GAPS/lab_3gas_data/{dataset_name}",
            },
            "C1": {
                "ssh": "gaps@192.168.137.172",
                "project_root": "/home/gaps/GAPS",
                "python": "/home/gaps/GAPS/gaps_rpi_env/bin/python",
                "roles": ["logical_C1", "P1_source"],
                "runtime_src": f"/home/gaps/GAPS/confirmation_runtime/{source_sha}/src",
                "data_root": f"/home/gaps/GAPS/lab_3gas_data/{dataset_name}",
            },
        },
        "flower_transport": {
            "server_bind": "127.0.0.1:8080",
            "controller_forward": "127.0.0.1:18080",
            "client_reverse_endpoint": "127.0.0.1:18080",
            "public_flower_port": False,
        },
    }
    manifest = {
        **core,
        "execution_topology_manifest_sha256": sha256_bytes(canonical_bytes(core)),
    }
    write_json_exclusive(output_root / "execution_topology_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--direction",
        choices=("P12_to_P3", "P2_to_P3", "both"),
        default="both",
    )
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
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite protocol: {output_root}")
    output_root.mkdir(parents=True)

    archive, source = build_source_archive(output_root)
    dataset = build_dataset_manifest(output_root, args.data_root.resolve())
    topology = build_topology_manifest(
        output_root,
        source["source_archive_sha256"],
        args.data_root.resolve().name,
    )
    controller_snapshot = (
        output_root / "controller" / CONTROLLER_PATH.name
    )
    copy_exclusive(CONTROLLER_PATH, controller_snapshot)
    class_schema = json.loads(
        (args.data_root.resolve() / "class_schema.json").read_text(
            encoding="utf-8"
        )
    )
    protocol_core = {
        "schema_version": "gaps.lab_three_gas.protocol.v2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": "three_class_gas_classification_only",
        "directions": (
            ["P1+P2->P3", "P2->P3"]
            if args.direction == "both"
            else [
                "P1+P2->P3"
                if args.direction == "P12_to_P3"
                else "P2->P3"
            ]
        ),
        "folds": dataset["folds"],
        "split_protocol": dataset["split_protocol"],
        "rounds": 25,
        "local_epochs": 3,
        "batch_size": 32,
        "seed": 42,
        "num_classes": 3,
        "input_shape": class_schema["input_shape"],
        "selected_channels": class_schema["selected_channels"],
        "num_phases": 1,
        "model_profile": args.profile,
        "domain_adaptation_mode": args.da_mode,
        "target_ce_weight": args.target_ce_weight,
        "server_da_steps_per_round": 100,
        "round_selection": "fixed final configured round",
        "source_archive_sha256": source["source_archive_sha256"],
        "dataset_manifest_sha256": dataset["dataset_manifest_sha256"],
        "execution_topology_manifest_sha256": topology[
            "execution_topology_manifest_sha256"
        ],
        "controller_path": str(CONTROLLER_PATH.relative_to(PROJECT_ROOT)),
        "controller_snapshot": str(
            controller_snapshot.relative_to(PROJECT_ROOT)
        ),
        "controller_sha256": sha256_file(CONTROLLER_PATH),
        "evidence_boundary": {
            "status": "preliminary_nominal_boundary_screening",
            "may_modify_frozen_iotj_evidence": False,
            "may_claim_final_lab_performance": False,
            "reason": (
                "Gas onset/end times still use the nominal schedule; rebuild "
                "from reviewed exact boundaries before final reporting."
            ),
        },
    }
    protocol = {
        **protocol_core,
        "protocol_manifest_sha256": sha256_bytes(canonical_bytes(protocol_core)),
    }
    write_json_exclusive(output_root / "protocol_manifest.json", protocol)
    print(
        json.dumps(
            {
                "status": "frozen",
                "output_root": str(output_root),
                "archive": str(archive),
                "source_archive_sha256": source["source_archive_sha256"],
                "dataset_manifest_sha256": dataset["dataset_manifest_sha256"],
                "protocol_manifest_sha256": protocol[
                    "protocol_manifest_sha256"
                ],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
