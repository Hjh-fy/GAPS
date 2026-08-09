"""Post-run numerical and control-variate audit for canonical SCAFFOLD."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping

import torch


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results/iotj_canonical_v1_scientific_validation_20260809/comparators/source_fl/CAN-V1-CMP-SCAFFOLD"
GATE = ROOT / "results/iotj_canonical_v1_scientific_validation_20260809/comparators/preflight/scaffold_source_numerical_gate.json"
DEFAULT_OUTPUT = ROOT / "results/iotj_canonical_v1_scientific_validation_20260809/comparators/scaffold_audit"


def audit_history(history: Mapping[str, Any], gate: Mapping[str, Any]) -> dict[str, Any]:
    rounds = list(history.get("rounds", []))
    fingerprints = [str(row.get("scaffold", {}).get("server_control_fingerprint", "")) for row in rounds]
    finite = all(
        math.isfinite(float(value))
        for row in rounds
        for value in (row["fit_metrics"]["train_ce_mean"], row["fit_metrics"]["train_accuracy"], row["evaluate_loss"], row["evaluate_metrics"]["accuracy"])
    ) if rounds else False
    checks = {
        "source_only_numerical_gate": gate.get("passed") is True,
        "rounds_1_to_25": [row.get("round") for row in rounds] == list(range(1, 26)),
        "two_clients_every_round": all(row.get("fit_clients") == 2 and row.get("evaluate_clients") == 2 for row in rounds),
        "zero_fit_evaluate_failures": all(row.get("fit_failures") == 0 and row.get("evaluate_failures") == 0 for row in rounds),
        "canonical_sgd": all(row.get("scaffold", {}).get("optimizer") == "SGD" for row in rounds),
        "fixed_lr_5e_4": all(float(row.get("scaffold", {}).get("optimizer_lr", -1)) == 5e-4 and float(row["fit_metrics"].get("scaffold_optimizer_lr", -1)) == 5e-4 for row in rounds),
        "local_epochs_1": all(float(row["fit_metrics"].get("local_epochs", -1)) == 1.0 for row in rounds),
        "positive_local_steps": all(float(row["fit_metrics"].get("scaffold_local_steps", 0)) > 0 for row in rounds),
        "no_adam_state": all(float(row["fit_metrics"].get("scaffold_adam_state_present", 1)) == 0.0 for row in rounds),
        "server_control_updates": len(rounds) == 25 and len(set(fingerprints)) > 1 and all(row.get("scaffold", {}).get("server_control_rounds_completed") == row.get("round") for row in rounds),
        "finite_round_metrics": finite,
    }
    return {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "round_count": len(rounds)}


def _tensor_state(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "model_state" in payload:
        payload = payload["model_state"]
    return {str(key): value.detach().float() for key, value in payload.items() if torch.is_tensor(value)}


def _norm(state: Mapping[str, torch.Tensor]) -> float:
    return math.sqrt(sum(float(torch.sum(value * value)) for value in state.values()))


def _delta_norm(left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]) -> float:
    return math.sqrt(sum(float(torch.sum((right[key] - value) ** 2)) for key, value in left.items()))


def build(output: Path) -> dict[str, Any]:
    remote = RUN / "remote_server"
    history = json.loads((remote / "history.json").read_text(encoding="utf-8"))
    gate = json.loads(GATE.read_text(encoding="utf-8"))
    verdict = audit_history(history, gate)
    if verdict["status"] != "PASS":
        raise RuntimeError(f"FAIL_CLOSED SCAFFOLD history audit failed: {verdict['checks']}")
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    previous_model = None
    parameter_alignment = True
    for item in history["rounds"]:
        index = int(item["round"])
        model = _tensor_state(remote / f"server_round_{index:03d}.pth")
        control = _tensor_state(remote / f"scaffold_server_control_round_{index:03d}.pth")
        client_stats = json.loads((remote / f"client_stats_round_{index:03d}.json").read_text(encoding="utf-8"))
        server_fingerprints = {row.get("server_parameters_fingerprint") for row in client_stats["clients"]}
        parameter_alignment = parameter_alignment and len(server_fingerprints) == 1
        rows.append({
            "round": index, "train_ce_mean": item["fit_metrics"]["train_ce_mean"],
            "train_accuracy": item["fit_metrics"]["train_accuracy"],
            "evaluate_loss": item["evaluate_loss"], "evaluate_accuracy": item["evaluate_metrics"]["accuracy"],
            "server_control_norm": _norm(control),
            "server_model_delta_norm": "" if previous_model is None else _delta_norm(previous_model, model),
            "server_control_fingerprint": item["scaffold"]["server_control_fingerprint"],
            "optimizer": item["scaffold"]["optimizer"], "optimizer_lr": item["scaffold"]["optimizer_lr"],
            "local_epochs": item["fit_metrics"]["local_epochs"], "local_steps": item["fit_metrics"]["scaffold_local_steps"],
            "adam_state_present": item["fit_metrics"]["scaffold_adam_state_present"],
            "participation_count": item["fit_clients"], "parameter_alignment": len(server_fingerprints) == 1,
        })
        previous_model = model
    with (output / "scaffold_roundwise_diagnostics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    status = "PASS_WITH_LIMITATION" if parameter_alignment else "FAIL"
    lines = [
        "# SCAFFOLD sanity audit", "", f"Status: `{status}`.", "",
        "The source-only numerical gate and every one of 25 runtime rounds pass finite-value, participation, failure-count, canonical SGD, fixed-LR, LE1, positive-step, no-Adam-state, and server-control-update checks. Server control norms, model-delta norms, loss, and accuracy are preserved in `scaffold_roundwise_diagnostics.csv`.", "",
        f"Both clients received the same server-parameter fingerprint in every round: **{parameter_alignment}**. Server control fingerprints and saved tensors change across rounds, confirming server-c updates.", "",
        "Per-client control-variate persistence and the gradient correction `grad L + c - c_i` are enforced by the canonical implementation tests (`test_scaffold_client_control_variate_persists`, `test_scaffold_gradient_contains_control_variate_correction`, and related tests). Flower's aggregated history does not retain the clients' string-valued before/after control fingerprints, so per-client control norms cannot be reconstructed post hoc from the server bundle; this is the stated audit limitation.", "",
        "No learning-rate search or target information was used.",
    ]
    (output / "SCAFFOLD_SANITY_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    payload = {**verdict, "status": status, "parameter_alignment": parameter_alignment, "client_control_runtime_evidence": "implementation_tests_plus_continuous_client_process; string fingerprints unavailable in aggregate history"}
    (output / "scaffold_sanity_audit.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(build(args.output.resolve()), indent=2))


if __name__ == "__main__":
    main()
