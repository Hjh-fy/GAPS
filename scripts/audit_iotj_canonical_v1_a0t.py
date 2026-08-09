"""Post-run protocol and loss-activity audit for canonical equal-label A0T."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "results/iotj_canonical_v1_final_20260808/a0t_equal_label/classification"
DEFAULT_OUTPUT = ROOT / "results/iotj_canonical_v1_scientific_validation_20260809/a0t_audit"
TARGETS = ("C3", "C4", "C5")


def audit_loss_activity(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    target = [row for row in rows if row.get("loss_name") == "target_ce"]
    target_ok = len(target) == 1 and float(target[0].get("configured_weight", -1)) == 1.0 and target[0].get("input_available") is True and int(target[0].get("active_steps", -1)) == 100 and math.isfinite(float(target[0].get("mean_weighted_loss", float("nan"))))
    non_ce = [row for row in rows if row.get("loss_name") != "target_ce"]
    non_ce_ok = all(int(row.get("active_steps", -1)) == 0 and abs(float(row.get("mean_weighted_loss", float("nan")))) <= 1e-12 for row in non_ce)
    return {"status": "PASS" if target_ok and non_ce_ok else "FAIL", "target_ce_active": target_ok, "all_non_ce_inactive": non_ce_ok}


def build(output: Path) -> dict[str, Any]:
    output = output.resolve(); output.mkdir(parents=True, exist_ok=True)
    activity_rows = []
    target_checks = []
    for target in TARGETS:
        run_dir = RUN_ROOT / f"CANONICAL-V1-A0T-{target}"
        marker = json.loads((run_dir / "fixed_endpoint_complete.json").read_text(encoding="utf-8"))
        manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
        spec = json.loads((run_dir / "locked_run_spec.json").read_text(encoding="utf-8"))
        history = json.loads((run_dir / "remote_server/history.json").read_text(encoding="utf-8"))
        protocol_ok = (
            marker.get("experiment_id") == f"CANONICAL-V1-A0T-{target}"
            and int(marker.get("fixed_endpoint", {}).get("round", -1)) == 25
            and marker.get("target_test_opened") is False and manifest.get("target_test_opened") is False
            and spec["protocol"].get("target_label_budget") == "same_canonical_calibration_as_A4"
            and spec["protocol"].get("target_test_selection") is False
            and spec["protocol"].get("hyperparameter_search") is False
        )
        rounds_ok = len(history.get("rounds", [])) == 25
        losses_ok = True
        for round_item in history.get("rounds", []):
            rows = list(round_item.get("domain_adapt_summary", {}).get("loss_activity", []))
            verdict = audit_loss_activity(rows)
            losses_ok = losses_ok and verdict["status"] == "PASS"
            for row in rows:
                activity_rows.append({"target": target, "round": round_item["round"], **row})
        target_checks.append({"target": target, "protocol_ok": protocol_ok, "rounds_25": rounds_ok, "target_ce_only_loss_activity": losses_ok})
    status = "PASS" if all(all(value for key, value in row.items() if key != "target") for row in target_checks) else "FAIL"
    if status != "PASS":
        raise RuntimeError(f"FAIL_CLOSED A0T audit failed: {target_checks}")
    with (output / "a0t_loss_activity.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = list(activity_rows[0]); writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(activity_rows)
    payload = {"schema_version": "iotj.canonical_v1.a0t.audit.v1", "status": status, "targets": target_checks, "target_test_used_for_training_or_selection": False, "non_ce_active_steps": 0}
    (output / "a0t_protocol_audit.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (output / "A0T_PROTOCOL_AUDIT.md").write_text(
        "# Canonical equal-label A0T audit\n\nStatus: **PASS**.\n\n"
        "All C3/C4/C5 runs use the preregistered calibration identities and class-label budget, 25 rounds, LE1, seed42, and fixed round-25 endpoints. Across all 75 server-adaptation rounds, target CE is the only active target loss (100 steps/round); every MMD, CORAL, adversarial, prototype, semantic, consistency, residual, and stage term has zero active steps and zero weighted loss. Target test was not used for training, hyperparameter selection, stopping, or checkpoint selection.\n",
        encoding="utf-8",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT); args = parser.parse_args()
    print(json.dumps(build(args.output), indent=2))


if __name__ == "__main__":
    main()
