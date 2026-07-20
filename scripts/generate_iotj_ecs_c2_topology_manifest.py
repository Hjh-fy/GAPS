"""Generate the immutable execution-placement manifest for ECS-hosted C2.

This manifest is deliberately separate from the frozen algorithm protocol.  It
binds the remote-C2 placement to every frozen run configuration, without
rewriting the historical command manifests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence


_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_C2_HOST = "root@114.55.171.63"


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    data = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _required_hash(payload: Mapping[str, Any], key: str, label: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not _HASH_RE.fullmatch(value):
        raise ValueError(f"{label} {key} must be a lowercase SHA-256")
    return value


def build_execution_topology_manifest(protocol: Mapping[str, Any]) -> dict[str, Any]:
    """Build a deterministic placement manifest from a frozen protocol object."""
    source_archive_sha256 = _required_hash(
        protocol, "source_archive_sha256", "protocol"
    )
    schedule = protocol.get("schedule")
    if not isinstance(schedule, list) or not schedule:
        raise ValueError("protocol schedule must be a non-empty list")
    config_by_run: dict[str, str] = {}
    for row in schedule:
        if not isinstance(row, Mapping):
            raise ValueError("protocol schedule row must be an object")
        run_id = row.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("protocol schedule run_id is invalid")
        if run_id in config_by_run:
            raise ValueError(f"protocol schedule has duplicate run_id: {run_id}")
        config_by_run[run_id] = _required_hash(
            row, "algorithm_config_sha256", f"protocol schedule {run_id}"
        )
    return {
        "schema_version": 1,
        "topology_id": "ecs_c2_pi_c1",
        "source_archive_sha256": source_archive_sha256,
        "algorithm_config_sha256_by_run": config_by_run,
        "hosts": {
            "C1": {"host_id": "pi-c1"},
            "C2": {"host_id": "ecs-c2", "ssh_host": _C2_HOST},
            "server": {"host_id": "ecs-server"},
        },
        "transport": {
            "flower_endpoint": "loopback_reverse_tunnel",
            "public_flower_port_exposure": False,
        },
    }


def write_execution_topology_manifest(protocol_path: Path, output: Path) -> dict[str, Any]:
    """Read an immutable protocol and write a new topology manifest once."""
    try:
        protocol = json.loads(Path(protocol_path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("protocol manifest is not valid JSON") from exc
    if not isinstance(protocol, Mapping):
        raise ValueError("protocol manifest must be an object")
    payload = build_execution_topology_manifest(protocol)
    payload["execution_topology_manifest_sha256"] = _canonical_sha256(payload)
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite topology manifest: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = write_execution_topology_manifest(args.protocol_manifest, args.output)
    print(
        json.dumps(
            {
                "status": "created",
                "output": str(args.output),
                "execution_topology_manifest_sha256": payload[
                    "execution_topology_manifest_sha256"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
