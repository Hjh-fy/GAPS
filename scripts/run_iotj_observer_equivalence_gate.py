"""Fail-closed OFF-A/ON/OFF-B numerical equivalence Gate.

This command owns only a deterministic local, synthetic two-client/two-round
Flower fixture.  It never reads a project dataset or a formal checkpoint.  The
real server/client CLIs are launched as subprocesses; observability is the sole
switch between the three attempts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import socket
import subprocess
import sys
import time
from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path
from typing import AbstractSet, Any

import numpy as np
import torch


VOLATILE_JSON_PATHS = {
    ("run_config", "args", "observer_context"),
    ("run_config", "args", "observer_events"),
    ("metrics", "fit_seconds"),
    ("metrics", "evaluate_seconds"),
    ("provenance", "wall_time_utc"),
    ("provenance", "pid"),
    ("provenance", "path"),
}

_GROUPS = ("B2", "B5")
_MODES = ("off_a", "on", "off_b")
_TIMING_PATHS = {
    ("metrics", "fit_seconds"),
    ("metrics", "evaluate_seconds"),
}
_OBSERVER_PATHS = {
    ("run_config", "args", "observer_context"),
    ("run_config", "args", "observer_events"),
}


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _is_reparse_or_link(path: Path) -> bool:
    if path.is_symlink():
        return True
    try:
        attributes = path.lstat().st_file_attributes  # type: ignore[attr-defined]
    except (AttributeError, FileNotFoundError, OSError):
        return False
    return bool(attributes & getattr(__import__("stat"), "FILE_ATTRIBUTE_REPARSE_POINT", 0))


def _require_no_link_ancestors(path: Path) -> None:
    candidate = path.absolute()
    for ancestor in (candidate, *candidate.parents):
        if os.path.lexists(ancestor) and _is_reparse_or_link(ancestor):
            raise ValueError(f"symlink/reparse path component is forbidden: {ancestor}")


def _require_regular_file(path: Path) -> Path:
    _require_no_link_ancestors(path.parent)
    if _is_reparse_or_link(path):
        raise ValueError(f"symlink/reparse input is forbidden: {path}")
    if not path.is_file():
        raise ValueError(f"required regular file is missing: {path}")
    return path


def _finite_json(value: Any, path: tuple[Any, ...] = ()) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite JSON value at {path!r}")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _finite_json(item, path + (key,))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _finite_json(item, path + (index,))


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    cpu = tensor.detach().cpu().contiguous()
    return cpu.view(torch.uint8).numpy().tobytes(order="C")


def _walk_tensors(
    value: Any,
    *,
    prefix: tuple[str, ...] = (),
    found: list[tuple[str, torch.Tensor]] | None = None,
) -> list[tuple[str, torch.Tensor]]:
    if found is None:
        found = []
    if isinstance(value, torch.Tensor):
        found.append((".".join(prefix), value))
    elif isinstance(value, Mapping):
        for key, item in value.items():
            _walk_tensors(item, prefix=prefix + (str(key),), found=found)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _walk_tensors(item, prefix=prefix + (str(index),), found=found)
    return found


def tensor_fingerprint(checkpoint: Path) -> dict[str, Any]:
    """Fingerprint every checkpoint tensor in insertion order and raw bytes."""

    path = _require_regular_file(Path(checkpoint))
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(f"cannot load checkpoint {path}: {exc}") from exc
    walked = _walk_tensors(payload)
    if not walked:
        raise ValueError(f"checkpoint contains no tensors: {path}")

    records: OrderedDict[str, dict[str, Any]] = OrderedDict()
    comparison_records: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for key, tensor in walked:
        raw = _tensor_bytes(tensor)
        record = {
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
            "numel": int(tensor.numel()),
            "raw_bytes": len(raw),
            "raw_sha256": _sha256_bytes(raw),
            "_raw": raw,
        }
        records[key] = record
        comparison_records[key] = {
            name: item for name, item in record.items() if name != "_raw"
        }
    comparison = {
        "kind": "tensor_checkpoint",
        "key_order": list(records),
        "tensors": comparison_records,
    }
    return {
        "kind": "tensor_checkpoint",
        "artifact_sha256": _sha256_file(path),
        "content_sha256": _sha256_bytes(_canonical_bytes(comparison)),
        "key_order": list(records),
        "tensors": records,
        "comparison": comparison,
    }


def _normalize_json(
    value: Any,
    volatile_paths: AbstractSet[tuple],
    path: tuple[Any, ...] = (),
) -> Any:
    if path in volatile_paths:
        if path in _TIMING_PATHS:
            if type(value) is float:
                return 0.0
            if type(value) is int:
                return 0
            raise ValueError(f"timing scalar has invalid type at {path!r}")
        return {"__ignored_exact_volatile_leaf__": ".".join(map(str, path))}
    if isinstance(value, Mapping):
        return {
            key: _normalize_json(item, volatile_paths, path + (key,))
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _normalize_json(item, volatile_paths, path + (index,))
            for index, item in enumerate(value)
        ]
    return value


def json_fingerprint(
    path: Path, volatile_paths: AbstractSet[tuple]
) -> dict[str, Any]:
    """Fingerprint JSON after normalizing only exact allowlisted leaf paths."""

    input_path = _require_regular_file(Path(path))
    raw = input_path.read_bytes()
    try:
        value = json.loads(raw.decode("utf-8"), parse_constant=lambda item: (_ for _ in ()).throw(ValueError(item)))
    except Exception as exc:
        raise ValueError(f"invalid JSON input {input_path}: {exc}") from exc
    _finite_json(value)
    normalized = _normalize_json(value, volatile_paths)
    content = _canonical_bytes(normalized)
    return {
        "kind": "json",
        "artifact_sha256": _sha256_bytes(raw),
        "content_sha256": _sha256_bytes(content),
        "comparison": normalized,
    }


def _comparison_view(value: Any) -> Any:
    if isinstance(value, Mapping):
        if "comparison" in value and (
            "content_sha256" in value or "artifact_sha256" in value
        ):
            return value["comparison"]
        return {key: _comparison_view(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_comparison_view(item) for item in value]
    return value


def _first_mismatches(
    left: Any, right: Any, path: tuple[Any, ...] = (), limit: int = 50
) -> list[str]:
    if type(left) is not type(right):
        return [f"{'.'.join(map(str, path)) or '<root>'}: type differs"]
    if isinstance(left, Mapping):
        if set(left) != set(right):
            return [f"{'.'.join(map(str, path)) or '<root>'}: keys differ"]
        output: list[str] = []
        for key in sorted(left, key=str):
            output.extend(_first_mismatches(left[key], right[key], path + (key,), limit))
            if len(output) >= limit:
                break
        return output[:limit]
    if isinstance(left, list):
        if len(left) != len(right):
            return [f"{'.'.join(map(str, path)) or '<root>'}: list length differs"]
        output = []
        for index, (l_item, r_item) in enumerate(zip(left, right)):
            output.extend(_first_mismatches(l_item, r_item, path + (index,), limit))
            if len(output) >= limit:
                break
        return output[:limit]
    if left != right:
        return [f"{'.'.join(map(str, path)) or '<root>'}: value differs"]
    return []


def _artifact_hashes(value: Mapping[str, Any]) -> dict[str, str]:
    output: dict[str, str] = {}
    for key, item in value.items():
        if isinstance(item, Mapping) and isinstance(item.get("artifact_sha256"), str):
            output[str(key)] = str(item["artifact_sha256"])
    return output


def _dtype_to_numpy(dtype: str) -> np.dtype[Any] | None:
    mapping = {
        "torch.float16": np.dtype("float16"),
        "torch.float32": np.dtype("float32"),
        "torch.float64": np.dtype("float64"),
        "torch.int8": np.dtype("int8"),
        "torch.uint8": np.dtype("uint8"),
        "torch.int16": np.dtype("int16"),
        "torch.int32": np.dtype("int32"),
        "torch.int64": np.dtype("int64"),
        "torch.bool": np.dtype("bool"),
    }
    return mapping.get(dtype)


def _max_tensor_delta(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    maximum = 0.0
    for name in sorted(set(left) & set(right)):
        l_item, r_item = left[name], right[name]
        if not isinstance(l_item, Mapping) or not isinstance(r_item, Mapping):
            continue
        l_tensors, r_tensors = l_item.get("tensors"), r_item.get("tensors")
        if not isinstance(l_tensors, Mapping) or not isinstance(r_tensors, Mapping):
            continue
        for key in set(l_tensors) & set(r_tensors):
            l_record, r_record = l_tensors[key], r_tensors[key]
            if not isinstance(l_record, Mapping) or not isinstance(r_record, Mapping):
                continue
            if l_record.get("raw_sha256") == r_record.get("raw_sha256"):
                continue
            dtype = _dtype_to_numpy(str(l_record.get("dtype")))
            if dtype is None or dtype != _dtype_to_numpy(str(r_record.get("dtype"))):
                maximum = max(maximum, 1.0)
                continue
            l_raw, r_raw = l_record.get("_raw"), r_record.get("_raw")
            if not isinstance(l_raw, bytes) or not isinstance(r_raw, bytes):
                maximum = max(maximum, 1.0)
                continue
            l_array = np.frombuffer(l_raw, dtype=dtype)
            r_array = np.frombuffer(r_raw, dtype=dtype)
            if l_array.shape != r_array.shape:
                maximum = max(maximum, 1.0)
                continue
            if np.issubdtype(dtype, np.number):
                delta = np.max(np.abs(l_array.astype(np.float64) - r_array.astype(np.float64)), initial=0.0)
                maximum = max(maximum, float(delta))
            else:
                maximum = max(maximum, 1.0)
    return maximum


def compare_fingerprints(
    off_a: Mapping[str, Any],
    on: Mapping[str, Any],
    off_b: Mapping[str, Any],
) -> dict[str, Any]:
    """Classify OFF drift before checking whether ON mutated the numerical path."""

    off_a_view = _comparison_view(off_a)
    on_view = _comparison_view(on)
    off_b_view = _comparison_view(off_b)
    off_mismatches = _first_mismatches(off_a_view, off_b_view)
    on_mismatches = _first_mismatches(off_a_view, on_view)
    if off_mismatches:
        status = "environment_nondeterminism"
        selected = off_mismatches
        max_abs_delta = _max_tensor_delta(off_a, off_b)
    elif on_mismatches:
        status = "observer_path_mutation"
        selected = on_mismatches
        max_abs_delta = _max_tensor_delta(off_a, on)
    else:
        status = "equivalent"
        selected = []
        max_abs_delta = 0.0
    result = {
        "status": status,
        "equivalent": status == "equivalent",
        "off_pair_equal": not off_mismatches,
        "on_equal_to_off": not on_mismatches,
        "max_abs_delta": float(max_abs_delta),
        "mismatches": selected,
        "artifact_hashes": {
            "off_a": _artifact_hashes(off_a),
            "on": _artifact_hashes(on),
            "off_b": _artifact_hashes(off_b),
        },
        "content_set_sha256": {
            "off_a": _sha256_bytes(_canonical_bytes(off_a_view)),
            "on": _sha256_bytes(_canonical_bytes(on_view)),
            "off_b": _sha256_bytes(_canonical_bytes(off_b_view)),
        },
    }
    return result


def _write_fixture(root: Path) -> dict[str, str]:
    data_root = root / "fixture_data"
    data_root.mkdir()
    rng = np.random.RandomState(7001)
    labels = np.asarray([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int64)
    phases = np.asarray([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64)
    regression = np.zeros((8, 4), dtype=np.float32)
    for client_id in (1, 2, 5):
        client_dir = data_root / f"client_{client_id}"
        client_dir.mkdir()
        for split in ("train", "test", "calibration"):
            features = rng.standard_normal((8, 100, 8)).astype(np.float32)
            np.save(client_dir / f"{split}_features.npy", features, allow_pickle=False)
            np.save(client_dir / f"{split}_classification_labels.npy", labels, allow_pickle=False)
            np.save(client_dir / f"{split}_phase_labels.npy", phases, allow_pickle=False)
            np.save(client_dir / f"{split}_regression_labels.npy", regression, allow_pickle=False)
    hashes = {
        str(path.relative_to(root)).replace("\\", "/"): _sha256_file(path)
        for path in sorted(data_root.rglob("*.npy"))
    }
    manifest = root / "fixture_manifest.json"
    manifest.write_bytes(_canonical_bytes({"seed": 7001, "files": hashes}) + b"\n")
    hashes[str(manifest.relative_to(root)).replace("\\", "/")] = _sha256_file(manifest)
    return hashes


def _reserve_port(used: set[int]) -> int:
    for _ in range(100):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            port = int(sock.getsockname()[1])
        if port not in used:
            used.add(port)
            return port
    raise RuntimeError("unable to reserve a unique loopback port")


def _observer_context(
    *, group_id: str, producer: str, client_id: str | None, fixture_sha: str
) -> dict[str, Any]:
    group_lower = group_id.lower()
    return {
        "run_id": f"c12_to_c5__{group_lower}__s42",
        "attempt_id": f"c12_to_c5__{group_lower}__s42__a001",
        "group_id": group_id,
        "training_seed": 42,
        "client_id": client_id,
        "host_id": "local-loopback",
        "producer": producer,
        "confirmation_commit": "0" * 40,
        "source_archive_sha256": fixture_sha,
        "dataset_manifest_sha256": fixture_sha,
        "algorithm_config_sha256": _sha256_bytes(
            _canonical_bytes({"group": group_id, "rounds": 2, "seed": 42})
        ),
    }


def _write_context(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_bytes(_canonical_bytes(payload) + b"\n")


def _bool(value: bool) -> str:
    return "true" if value else "false"


def _server_command(
    *, group_id: str, port: int, data_root: Path, output_dir: Path
) -> list[str]:
    b5 = group_id == "B5"
    source = f"{data_root / 'client_1'},{data_root / 'client_2'}"
    target = str(data_root / "client_5")
    return [
        sys.executable,
        "-m",
        "gaps_flower.server_app",
        "--server-address",
        f"127.0.0.1:{port}",
        "--rounds",
        "2",
        "--min-clients",
        "2",
        "--output-dir",
        str(output_dir),
        "--run-name",
        f"observer_equivalence_{group_id.lower()}",
        "--seed",
        "42",
        "--strategy",
        "gaps",
        "--profile",
        "proto_replay",
        "--save-history",
        "true",
        "--use-selective-agg",
        "true",
        "--use-proto-mmd",
        "false",
        "--da-preset",
        "none",
        "--use-domain-adapt",
        "true",
        "--server-val-data",
        source,
        "--server-calib-data",
        target,
        "--domain-adapt-steps",
        "1",
        "--domain-adapt-warmup",
        "0",
        "--da-use-coral",
        _bool(b5),
        "--da-use-mmd",
        "true",
        "--da-use-adversarial",
        _bool(b5),
        "--da-mmd-objective",
        "mmd2",
        "--da-stage-alignment",
        "cross_domain_same_class_phase",
        "--da-adv-feature-objective",
        "wasserstein_min",
        "--da-coral-class-conditional",
        "true",
        "--strict-calibration-split",
        "true",
        "--da-device",
        "cpu",
        "--use-adapted-as-global",
        "true",
        "--da-lambda-coral",
        "0.5" if b5 else "0.0",
        "--da-lambda-global-mmd",
        "0.5",
        "--da-lambda-class-mmd",
        "0.5",
        "--da-lambda-proto-anchor",
        "0.3",
        "--da-lambda-adv",
        "0.5" if b5 else "0.0",
        "--da-lambda-target-ce",
        "0.0",
        "--da-lambda-proto",
        "0.05",
        "--da-lambda-consistency",
        "2.0",
        "--da-lambda-residual",
        "0.1",
        "--da-lambda-proto-mmd",
        "0.0",
        "--da-lambda-stage-mmd",
        "0.2" if b5 else "0.0",
        "--da-target-ce-label-smoothing",
        "0.0",
        "--da-target-ce-class-balanced",
        "false",
        "--da-server-opt-lr",
        "0.0005",
    ]


def _client_command(*, client_id: int, port: int, data_root: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "gaps_flower.client_app",
        "--server-address",
        f"127.0.0.1:{port}",
        "--client-id",
        str(client_id),
        "--data-root",
        str(data_root),
        "--device",
        "cpu",
        "--local-epochs",
        "1",
        "--batch-size",
        "4",
        "--profile",
        "proto_replay",
        "--seed",
        "42",
    ]


def _popen_flags() -> dict[str, Any]:
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def _stop_process(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name == "nt":
            process.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=5)
    except Exception:
        try:
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
            else:
                os.killpg(process.pid, signal.SIGKILL)
        except Exception:
            process.kill()


def _tail(path: Path, limit: int = 8000) -> str:
    if not path.is_file():
        return "<missing>"
    return path.read_text(encoding="utf-8", errors="replace")[-limit:]


def _wait_server(port: int, process: subprocess.Popen[Any], timeout: float = 45.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited before readiness with {process.returncode}")
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.settimeout(0.2)
            try:
                probe.connect(("127.0.0.1", port))
                return
            except OSError:
                time.sleep(0.1)
    raise TimeoutError("server readiness timeout")


def _launch_attempt(
    *,
    root: Path,
    fixture_root: Path,
    group_id: str,
    mode: str,
    port: int,
    fixture_sha: str,
) -> dict[str, Any]:
    attempt = root / mode
    attempt.mkdir()
    # Keep the runtime output path identical across OFF-A/ON/OFF-B.  The
    # adapted checkpoint contains a diagnostic path, so mode-specific runtime
    # paths would make the raw checkpoint bytes differ despite identical
    # tensors.  Each active directory is still freshly and exclusively made,
    # then moved into its immutable attempt evidence directory after exit.
    active_server_output = root / "_active_server_output"
    if os.path.lexists(active_server_output):
        raise FileExistsError(f"stale active server output: {active_server_output}")
    active_server_output.mkdir()
    server_output = attempt / "server_output"
    commands = attempt / "commands.json"
    server_command = _server_command(
        group_id=group_id,
        port=port,
        data_root=fixture_root,
        output_dir=active_server_output,
    )
    client_commands = [
        _client_command(client_id=client_id, port=port, data_root=fixture_root)
        for client_id in (1, 2)
    ]
    contexts: list[tuple[Path, Path]] = []
    if mode == "on":
        server_context = attempt / "server_context.json"
        server_events = attempt / "server_events.jsonl"
        _write_context(
            server_context,
            _observer_context(
                group_id=group_id,
                producer="server",
                client_id=None,
                fixture_sha=fixture_sha,
            ),
        )
        server_command.extend(
            ["--observer-context", str(server_context), "--observer-events", str(server_events)]
        )
        contexts.append((server_context, server_events))
        for index, client_id in enumerate((1, 2)):
            context = attempt / f"client_c{client_id}_context.json"
            events = attempt / f"client_c{client_id}_events.jsonl"
            _write_context(
                context,
                _observer_context(
                    group_id=group_id,
                    producer="client",
                    client_id=f"C{client_id}",
                    fixture_sha=fixture_sha,
                ),
            )
            client_commands[index].extend(
                ["--observer-context", str(context), "--observer-events", str(events)]
            )
            contexts.append((context, events))
    commands.write_bytes(
        _canonical_bytes(
            {
                "server": server_command,
                "clients": client_commands,
                "environment": {
                    "PYTHONHASHSEED": "0",
                    "OMP_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                },
            }
        )
        + b"\n"
    )

    env = os.environ.copy()
    env.update(
        {"PYTHONHASHSEED": "0", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    )
    processes: list[subprocess.Popen[Any]] = []
    handles: list[Any] = []
    repo_root = Path(__file__).resolve().parents[1]
    try:
        server_stdout = (attempt / "server.stdout.log").open("xb")
        server_stderr = (attempt / "server.stderr.log").open("xb")
        handles.extend((server_stdout, server_stderr))
        server = subprocess.Popen(
            server_command,
            cwd=repo_root,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=server_stdout,
            stderr=server_stderr,
            **_popen_flags(),
        )
        processes.append(server)
        _wait_server(port, server)
        clients: list[subprocess.Popen[Any]] = []
        for client_id, command in zip((1, 2), client_commands):
            stdout = (attempt / f"client_c{client_id}.stdout.log").open("xb")
            stderr = (attempt / f"client_c{client_id}.stderr.log").open("xb")
            handles.extend((stdout, stderr))
            client = subprocess.Popen(
                command,
                cwd=repo_root,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                **_popen_flags(),
            )
            clients.append(client)
            processes.append(client)
        for client_id, client in zip((1, 2), clients):
            try:
                return_code = client.wait(timeout=240)
            except subprocess.TimeoutExpired as exc:
                raise TimeoutError(f"client C{client_id} timeout") from exc
            if return_code != 0:
                raise RuntimeError(f"client C{client_id} exit {return_code}")
        try:
            server_code = server.wait(timeout=120)
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError("server timeout") from exc
        if server_code != 0:
            raise RuntimeError(f"server exit {server_code}")
        active_server_output.rename(server_output)
    except Exception as exc:
        for handle in handles:
            handle.flush()
        detail = {
            "server_stdout": _tail(attempt / "server.stdout.log"),
            "server_stderr": _tail(attempt / "server.stderr.log"),
            "client_c1_stderr": _tail(attempt / "client_c1.stderr.log"),
            "client_c2_stderr": _tail(attempt / "client_c2.stderr.log"),
        }
        raise RuntimeError(f"{mode} loopback failed: {exc}; logs={detail}") from exc
    finally:
        for process in reversed(processes):
            _stop_process(process)
        for handle in handles:
            handle.close()

    expected_sidecars = [events for _context, events in contexts]
    if mode == "on":
        missing = [str(path) for path in expected_sidecars if not path.is_file()]
        if missing:
            raise RuntimeError(f"ON did not create expected sidecars: {missing}")
    else:
        unexpected = sorted(attempt.glob("*_events.jsonl"))
        if unexpected:
            raise RuntimeError(f"OFF created observer sidecars: {unexpected}")
    return {"attempt": attempt, "server_output": server_output}


def _json_projection(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    projection_path = path.with_suffix(path.suffix + ".gate.json")
    projection_path.write_bytes(_canonical_bytes(payload) + b"\n")
    return json_fingerprint(projection_path, VOLATILE_JSON_PATHS)


def _capture_artifacts(attempt: Mapping[str, Any]) -> dict[str, Any]:
    attempt_root = Path(attempt["attempt"])
    output = Path(attempt["server_output"])
    artifacts: OrderedDict[str, Any] = OrderedDict()
    artifacts["final_aggregated_checkpoint"] = tensor_fingerprint(output / "server_latest.pth")
    artifacts["final_adapted_checkpoint"] = tensor_fingerprint(output / "server_latest_adapted.pth")
    for label in ("final_aggregated_checkpoint", "final_adapted_checkpoint"):
        raw_sha = artifacts[label]["artifact_sha256"]
        artifacts[f"{label}_raw"] = {
            "kind": "raw_checkpoint",
            "artifact_sha256": raw_sha,
            "content_sha256": raw_sha,
            "comparison": {"raw_file_sha256": raw_sha},
        }

    raw_config = json.loads((output / "run_config.json").read_text(encoding="utf-8"))
    excluded = {"server_address", "output_dir", "run_name"}
    projected_args = {
        key: value for key, value in raw_config["args"].items() if key not in excluded
    }
    artifacts["run_config"] = _json_projection(
        attempt_root / "run_config_projection.json",
        {
            "run_config": {"args": projected_args},
            "provenance": {
                "wall_time_utc": "not-collected",
                "pid": 0,
                "path": str(output),
            },
        },
    )
    for round_idx in (1, 2):
        for stem in ("prototype_stats", "semantic_protos"):
            source = output / f"{stem}_round_{round_idx:03d}.json"
            artifacts[f"{stem}_round_{round_idx}"] = json_fingerprint(
                source, VOLATILE_JSON_PATHS
            )
        client_stats_path = output / f"client_stats_round_{round_idx:03d}.json"
        client_stats = json.loads(client_stats_path.read_text(encoding="utf-8"))
        for client in client_stats["clients"]:
            client_id = int(client["client_id"])
            artifacts[f"client_stats_round_{round_idx}_c{client_id}"] = _json_projection(
                attempt_root / f"client_stats_round_{round_idx}_c{client_id}.json",
                {"metrics": client},
            )
        diagnostics_path = output / f"domain_adapt_round_{round_idx:03d}.json"
        diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
        semantic_path = diagnostics.pop("semantic_protos_after_da", None)
        artifacts[f"domain_adapt_round_{round_idx}"] = _json_projection(
            attempt_root / f"domain_adapt_round_{round_idx}.json",
            {
                "metrics": diagnostics,
                "provenance": {
                    "wall_time_utc": "not-collected",
                    "pid": 0,
                    "path": semantic_path,
                },
            },
        )
    return artifacts


def _read_events(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _message_fingerprints(on_attempt: Mapping[str, Any]) -> list[dict[str, Any]]:
    events = _read_events(Path(on_attempt["attempt"]) / "server_events.jsonl")
    messages = []
    for event in events:
        event_type = event.get("event_type")
        if event_type == "flower_fitins_prepared":
            audit = event["payload"]["downlink_audit"]
            direction = "downlink"
        elif event_type == "flower_fitres_available":
            audit = event["payload"]["uplink_audit"]
            direction = "uplink"
        else:
            continue
        messages.append(
            {
                "round": int(event["round"]),
                "direction": direction,
                "client_id": event.get("client_id") if direction == "uplink" else None,
                "application_message_bytes": int(audit["application_message_bytes"]),
                "application_message_sha256": str(audit["application_message_sha256"]),
                "logical": audit["logical"],
            }
        )
    messages.sort(
        key=lambda item: (
            item["round"], item["direction"], item["client_id"] or "", item["application_message_sha256"]
        )
    )
    if len(messages) != 8:
        raise RuntimeError(f"expected 8 audited FitIns/FitRes messages, got {len(messages)}")
    if sum(item["direction"] == "downlink" for item in messages) != 4:
        raise RuntimeError("expected exactly four FitIns message fingerprints")
    if sum(item["direction"] == "uplink" for item in messages) != 4:
        raise RuntimeError("expected exactly four FitRes message fingerprints")
    return messages


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    if os.path.lexists(temporary):
        raise FileExistsError(f"temporary report path already exists: {temporary}")
    with temporary.open("xb") as handle:
        handle.write(_canonical_bytes(payload) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _prepare_output_root(output_root: Path) -> Path:
    root = Path(output_root)
    if os.path.lexists(root):
        raise FileExistsError(f"output_root must not already exist: {root}")
    _require_no_link_ancestors(root.parent)
    parent = root.parent.resolve(strict=True)
    resolved = parent / root.name
    resolved.mkdir()
    return resolved


def run_local_gate(output_root: Path, group_id: str) -> dict[str, Any]:
    """Run real B2/B5 Flower CLIs OFF-A, ON, OFF-B on synthetic data."""

    group = str(group_id).upper()
    if group not in _GROUPS:
        raise ValueError(f"group_id must be one of {_GROUPS}, got {group_id!r}")
    root = _prepare_output_root(Path(output_root))
    fixture_hashes = _write_fixture(root)
    fixture_root = root / "fixture_data"
    fixture_sha = _sha256_bytes(_canonical_bytes(fixture_hashes))
    ports: set[int] = set()
    attempts: dict[str, dict[str, Any]] = {}
    artifacts: dict[str, dict[str, Any]] = {}
    try:
        for mode in _MODES:
            attempts[mode] = _launch_attempt(
                root=root,
                fixture_root=fixture_root,
                group_id=group,
                mode=mode,
                port=_reserve_port(ports),
                fixture_sha=fixture_sha,
            )
            artifacts[mode] = _capture_artifacts(attempts[mode])
        comparison = compare_fingerprints(
            artifacts["off_a"], artifacts["on"], artifacts["off_b"]
        )
        messages = _message_fingerprints(attempts["on"])
        content_hashes = comparison["content_set_sha256"]
        report = {
            "schema_version": "iotj.observer_equivalence.v1",
            "group_id": group,
            "fixture": {
                "numpy_seed": 7001,
                "training_seed": 42,
                "rounds": 2,
                "clients": ["C1", "C2"],
                "local_epochs": 1,
                "batch_size": 4,
                "window_shape": [100, 8],
                "rows_per_source": 8,
                "input_hashes": fixture_hashes,
                "input_set_sha256": fixture_sha,
            },
            "execution_order": ["OFF-A", "ON", "OFF-B"],
            "status": comparison["status"],
            "equivalent": comparison["equivalent"],
            "max_abs_delta": comparison["max_abs_delta"],
            "off_pair_equal": comparison["off_pair_equal"],
            "on_equal_to_off": comparison["on_equal_to_off"],
            "mismatches": comparison["mismatches"],
            "artifact_hashes": comparison["artifact_hashes"],
            "artifact_content_set_sha256": content_hashes,
            "final_checkpoint_sha256": {
                mode: artifacts[mode]["final_adapted_checkpoint"]["artifact_sha256"]
                for mode in _MODES
            },
            "final_checkpoint_raw_sha256": {
                mode: {
                    "aggregated": artifacts[mode]["final_aggregated_checkpoint"]["artifact_sha256"],
                    "adapted": artifacts[mode]["final_adapted_checkpoint"]["artifact_sha256"],
                }
                for mode in _MODES
            },
            "final_checkpoint_tensor_content_sha256": {
                mode: {
                    "aggregated": artifacts[mode]["final_aggregated_checkpoint"]["content_sha256"],
                    "adapted": artifacts[mode]["final_adapted_checkpoint"]["content_sha256"],
                }
                for mode in _MODES
            },
            "message_fingerprints": messages,
            "message_fingerprint_sha256": _sha256_bytes(_canonical_bytes(messages)),
            "observer_sidecars": {
                "off_a": 0,
                "on": 3,
                "off_b": 0,
            },
            "boundaries": {
                "dataset": "synthetic-only; no C5 test or project dataset opened",
                "topology": "local loopback; formal ECS/Pi/PC smoke is a later gate",
                "message_capture": "application fingerprints are read from ON sidecars; OFF equivalence is established by exact downstream numerical artifacts",
            },
        }
    except Exception as exc:
        report = {
            "schema_version": "iotj.observer_equivalence.v1",
            "group_id": group,
            "status": "gate_execution_error",
            "equivalent": False,
            "max_abs_delta": None,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "fixture_input_hashes": fixture_hashes,
        }
    _atomic_write_json(root / "observer_equivalence_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the local OFF-A/ON/OFF-B observer equivalence Gate"
    )
    parser.add_argument("--group", required=True, choices=_GROUPS)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = run_local_gate(args.output_root, args.group)
    except Exception as exc:
        print(f"observer equivalence Gate refused to run: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    raise SystemExit(0 if report.get("status") == "equivalent" else 2)


if __name__ == "__main__":
    main()
