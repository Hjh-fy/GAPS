"""Finalize the frozen A4 classifier and its downstream C5 replay evidence.

This module never trains the classifier.  Existing classification assets are
read-only and every downstream artifact is written to a new output root.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gaps_flower.state_fingerprint import checkpoint_provenance


SCHEMA_VERSION = "iotj.final_a4_end_to_end.v1"


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"FAIL_CLOSED JSON object required: {path}")
    return payload


def checkpoint_identity(path: str | Path) -> dict[str, Any]:
    """Return serialization-independent state identity plus file provenance."""
    return checkpoint_provenance(Path(path))


def prepare_output_root(path: str | Path) -> Path:
    """Create a new output root and refuse to overwrite any existing content."""
    output = Path(path)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    return output


def _value_after(argv: Sequence[str], flag: str) -> str:
    try:
        index = list(argv).index(flag)
    except ValueError as exc:
        raise RuntimeError(f"FAIL_CLOSED required protocol flag missing: {flag}") from exc
    if index + 1 >= len(argv):
        raise RuntimeError(f"FAIL_CLOSED protocol flag has no value: {flag}")
    return str(argv[index + 1])


def _bool_after(argv: Sequence[str], flag: str) -> bool:
    value = _value_after(argv, flag).strip().lower()
    if value not in {"true", "false"}:
        raise RuntimeError(f"FAIL_CLOSED invalid Boolean for {flag}: {value}")
    return value == "true"


def build_classifier_manifest(classification_root: str | Path) -> dict[str, Any]:
    """Build the fail-closed final classifier manifest from frozen A4 assets."""
    root = Path(classification_root).resolve()
    run_root = root / "FCL-E4-A4"
    locked = read_json(run_root / "locked_run_spec.json")
    completed = read_json(run_root / "fixed_endpoint_complete.json")
    evaluation = read_json(run_root / "final_evaluation_C5.json")

    protocol = locked.get("protocol")
    server = locked.get("server")
    client_c1 = locked.get("client_c1")
    client_c2 = locked.get("client_c2")
    if not all(isinstance(value, list) for value in (server, client_c1, client_c2)):
        raise RuntimeError("FAIL_CLOSED locked A4 command arrays are unavailable")
    if not isinstance(protocol, dict):
        raise RuntimeError("FAIL_CLOSED locked A4 protocol object is unavailable")

    checkpoint = run_root / "remote_server" / "server_round_025_adapted.pth"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"FAIL_CLOSED A4 round-25 checkpoint missing: {checkpoint}")
    target_metrics = evaluation.get("target_metrics")
    if not isinstance(target_metrics, dict):
        raise RuntimeError("FAIL_CLOSED C5 final target metrics missing")

    observed = {
        "rounds": int(protocol.get("rounds", -1)),
        "local_epochs": int(_value_after(client_c1, "--local-epochs")),
        "batch_size": int(protocol.get("batch_size", -1)),
        "seed": int(protocol.get("seed", -1)),
        "target_ce_weight": float(_value_after(server, "--da-lambda-target-ce")),
        "selective_aggregation": _bool_after(server, "--use-selective-agg"),
        "ablation_variant": _value_after(server, "--ablation-variant"),
        "target_information_method": _value_after(server, "--target-information-method"),
        "endpoint_round": int(completed.get("fixed_endpoint", {}).get("round", -1)),
    }
    expected = {
        "rounds": 25,
        "local_epochs": 1,
        "batch_size": 32,
        "seed": 42,
        "target_ce_weight": 0.0,
        "selective_aggregation": False,
        "ablation_variant": "A4",
        "target_information_method": "a4",
        "endpoint_round": 25,
    }
    if observed != expected:
        raise RuntimeError(
            f"FAIL_CLOSED frozen A4 protocol differs: observed={observed}"
        )

    missing_reason = (
        "No immutable same-protocol server-centric A4 round-25 endpoint was "
        "found in the local result root or audited server result root; a "
        "full-GAPS endpoint is not an A4 substitute."
    )
    identity = checkpoint_identity(checkpoint)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "partial_complete_c5_only",
        "protocol": {
            "method": "server-centric A4",
            "rounds": 25,
            "local_epochs": 1,
            "batch_size": 32,
            "seed": 42,
            "source_clients": ["C1", "C2"],
            "target_ce_weight": 0.0,
            "selective_aggregation": False,
            "fixed_endpoint_only": True,
        },
        "targets": {
            "C3": {"status": "blocked", "checkpoint": None, "reason": missing_reason},
            "C4": {"status": "blocked", "checkpoint": None, "reason": missing_reason},
            "C5": {
                "status": "complete",
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_identity": identity,
                "accuracy": float(target_metrics["accuracy"]),
                "macro_f1": float(target_metrics["macro_f1"]),
                "nll": float(target_metrics["nll"]),
                "ece": float(target_metrics["ece"]),
                "num_examples": int(target_metrics["num_examples"]),
                "selection_role": "none_fixed_endpoint_only",
            },
        },
        "classification_retrained": False,
        "full_gaps_endpoint_substituted_for_a4": False,
        "source_root": str(root),
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--classification-root",
        default="results/iotj_final_classification_le1_20260804",
    )
    parser.add_argument(
        "--output-root",
        default="results/iotj_final_end_to_end_a4_20260804",
    )
    parser.add_argument("--freeze-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = prepare_output_root(args.output_root)
    manifest = build_classifier_manifest(args.classification_root)
    write_json(output / "final_classifier_manifest.json", manifest)
    if not args.freeze_only:
        raise RuntimeError(
            "Regression replay is not implemented in this task; use --freeze-only"
        )


if __name__ == "__main__":
    main()
