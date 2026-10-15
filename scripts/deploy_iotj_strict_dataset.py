"""Deploy the frozen strict dataset to the authorized three-machine topology."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "dataset/iotj_canonical_v1_strict_nonoverlap"
OUTPUT = ROOT / "results/iotj_canonical_v1_scientific_validation_20260809/strict_nonoverlap/deployment"
EXPECTED_AGGREGATE = "881de29938460ad1a7564aca1f01a2b3f41cdc4820284397a05a0b3b218816c4"


def deployment_targets() -> list[tuple[str, str]]:
    return [
        ("root@121.40.139.213", "/root/GAPS/dataset"),
        ("gaps@192.168.137.172", "/home/gaps/GAPS/flower_runtime/dataset"),
        ("root@114.55.171.63", "/root/GAPS/confirmation_c2_data"),
    ]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command: list[str], timeout: float) -> str:
    return subprocess.run(command, cwd=ROOT, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout).stdout


def ssh(host: str, command: str, timeout: float = 120.0) -> str:
    return run(["ssh", "-n", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", host, command], timeout)


def deploy() -> dict[str, Any]:
    index = json.loads((DATASET / "dataset_sha256.json").read_text(encoding="utf-8"))
    if index["aggregate_sha256"] != EXPECTED_AGGREGATE:
        raise RuntimeError("FAIL_CLOSED strict dataset aggregate differs")
    index_sha = sha256(DATASET / "dataset_sha256.json")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    archive = OUTPUT / f"{DATASET.name}_{index_sha[:12]}.tar"
    if not archive.exists():
        subprocess.run(["tar", "-cf", str(archive), "-C", str(DATASET.parent), DATASET.name], cwd=ROOT, check=True, timeout=300)
    archive_sha = sha256(archive)
    deployments = []
    for host, parent in deployment_targets():
        target = f"{parent}/{DATASET.name}"
        marker = f"{parent}/.{DATASET.name}.dataset_index_sha256"
        observed = ssh(host, f"if test -f {shlex.quote(marker)}; then cat {shlex.quote(marker)}; fi").strip()
        if observed:
            if observed != index_sha:
                raise RuntimeError(f"FAIL_CLOSED remote strict marker differs: {host}")
            deployments.append({"host": host, "target": target, "status": "REUSED_HASH_MATCH", "dataset_index_sha256": index_sha})
            continue
        if ssh(host, f"if test -e {shlex.quote(target)}; then echo EXISTS; fi").strip():
            raise RuntimeError(f"FAIL_CLOSED unhashed strict dataset target exists: {host}:{target}")
        remote_archive = f"/tmp/{archive.name}"
        run(["scp", "-p", str(archive), f"{host}:{remote_archive}"], 1200)
        remote = " && ".join([
            f"mkdir -p {shlex.quote(parent)}",
            f"tar -xf {shlex.quote(remote_archive)} -C {shlex.quote(parent)}",
            f"test \"$(sha256sum {shlex.quote(target + '/dataset_sha256.json')} | cut -d ' ' -f 1)\" = {shlex.quote(index_sha)}",
            f"printf '%s\\n' {shlex.quote(index_sha)} > {shlex.quote(marker)}",
        ])
        ssh(host, remote, timeout=600)
        deployments.append({"host": host, "target": target, "status": "DEPLOYED_HASH_VERIFIED", "dataset_index_sha256": index_sha})
    payload = {"schema_version": "iotj.strict_dataset.deployment.v1", "status": "PASS", "dataset_aggregate_sha256": EXPECTED_AGGREGATE, "dataset_index_sha256": index_sha, "archive": str(archive), "archive_sha256": archive_sha, "deployments": deployments}
    (OUTPUT / "deployment_manifest.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(json.dumps(deploy(), indent=2))


if __name__ == "__main__":
    main()
