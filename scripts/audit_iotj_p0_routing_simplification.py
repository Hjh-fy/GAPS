"""Fail-closed audit for the IoT-J P0 routing simplification study."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

EXPECTED_METHODS = {"source_only", "simple_target_ce", "full_target_adapter"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def require(condition: bool, message: str) -> None:
    if not condition: raise RuntimeError(f"FAIL_CLOSED: {message}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(); root = args.result_root.resolve()
    source = root / "P0A_PURE_FEDAVG_LE1_S42"; commission = root / "P0B_ROUNDWISE_COMMISSIONING_S42"
    source_manifest = json.loads((source / "protocol_manifest.json").read_text(encoding="utf-8"))
    commission_manifest = json.loads((commission / "protocol_manifest.json").read_text(encoding="utf-8"))
    training = source_manifest["training"]
    require(source_manifest["dataset"].endswith("client_data_c1234src_c5tgt_2080_timeaware_60_170_window_fullgrid"), "dataset path")
    require(source_manifest["source_clients"] == ["C1", "C2"] and source_manifest["target_client"] == "C5", "client roles")
    require(source_manifest["seed"] == 42 and training["rounds"] == 25 and training["local_epochs"] == 1, "seed/round/local epoch")
    require(training["profile"] == "ce_only" and training["aggregation"] == "sample_weighted_FedAvg" and training["fedprox_mu"] == 0.0, "pure FedAvg contract")
    require(source_manifest["target_access"] == "none" and not source_manifest["target_test_used_for_selection"], "P0-A target isolation")
    index = json.loads((source / "checkpoint_index.json").read_text(encoding="utf-8"))
    require([item["round"] for item in index] == list(range(1, 26)), "exactly 25 ordered checkpoints")
    for item in index:
        path = Path(item["absolute_local_path"]); require(path.is_file() and sha256(path) == item["sha256"], f"checkpoint hash round {item['round']}")
    curve = json.loads((source / "client_training_curve.json").read_text(encoding="utf-8"))
    require(len(curve) == 50 and {(row["round"], row["client_id"]) for row in curve} == {(r, c) for r in range(1, 26) for c in ("C1", "C2")}, "client curve completeness")
    require(all(row["local_epochs"] == 1 and row["train_ce_averaging"] == "sample_weighted_over_local_minibatches" for row in curve), "training instrumentation convention")
    curve_csv = rows(source / "client_training_curve.csv")
    require(len(curve_csv) == 50, "client training CSV completeness")
    require(commission_manifest["source_checkpoint_count"] == 25 and set(commission_manifest["methods"]) == EXPECTED_METHODS, "commissioning methods")
    require(commission_manifest["commissioning_steps"] == 100 and commission_manifest["commissioning_lr"] == 5e-4, "commissioning budget")
    require(commission_manifest["formal_comparison_round"] == 25 and not commission_manifest["adapted_checkpoint_inheritance"], "round25/no inheritance")
    require(not commission_manifest["target_test_used_for_selection"] and commission_manifest["target_test_role"] == "post_hoc_evaluation_only", "test role")
    metrics = rows(commission / "roundwise_routing_metrics.csv")
    require(len(metrics) == 75, "75 round-method metric rows")
    require({(int(row["source_round"]), row["method"]) for row in metrics} == {(r, m) for r in range(1, 26) for m in EXPECTED_METHODS}, "complete round-method grid")
    require(all(int(row["num_examples"]) == 1360 and row["selection_role"] == "post_hoc_diagnostic_only" for row in metrics), "sealed C5 evaluation scope")
    require(len(rows(commission / "simple_ce_commissioning_diagnostics.csv")) == 2500, "simple CE step count")
    full = rows(commission / "full_da_commissioning_diagnostics.csv"); require(len(full) == 2500, "full DA step count")
    activity = {row["loss_name"]: row for row in rows(commission / "server_loss_activity_summary.csv")}
    require(activity["target_ce_loss"]["activity_status"] == "ZERO_BY_CONFIGURATION", "full DA target CE disabled")
    require(activity["mmd_proto_loss"]["activity_status"] == "ZERO_BY_CONFIGURATION", "proto MMD disabled")
    for name in ("proto_loss", "consist_loss", "residual_loss"):
        require(activity[name]["activity_status"] == "ZERO_NO_INPUT_STATISTICS", f"{name} has no fabricated statistics")
    formal = rows(commission / "round25_routing_comparison.csv"); require(len(formal) == 3 and {row["method"] for row in formal} == EXPECTED_METHODS, "fixed round25 comparison")

    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=False)
    main_files = [source / "protocol_manifest.json", source / "checkpoint_index.json", source / "client_training_curve.json", source / "client_training_curve.csv", commission / "protocol_manifest.json", commission / "roundwise_routing_metrics.csv", commission / "round25_routing_comparison.csv", commission / "simple_ce_commissioning_diagnostics.csv", commission / "full_da_commissioning_diagnostics.csv", commission / "server_loss_activity_summary.csv"]
    hashes = [{"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)} for path in main_files]
    (output / "sha256_index.json").write_text(json.dumps(hashes, indent=2) + "\n", encoding="utf-8")
    report = """# P0 Routing Simplification Experiment Audit\n\n## Verdict: approved for seed42 descriptive evidence\n\nAll 20 frozen contract checks passed: dataset/client roles, seed, 25xLE1 CE-only FedAvg, target isolation, 25 checkpoint hashes, independent commissioning, locked 100-step/5e-4 settings, inactive-statistics disclosure, sealed test role, and fixed round-25 comparison.\n\n## Limitations\n\nRound-wise C5 curves are retrospective diagnostics only. Seed42 cannot support uncertainty, significance, or stability claims. This new LE1 protocol is not a single-factor ablation against historical B5.\n"""
    (output / "EXPERIMENT_AUDIT.md").write_text(report, encoding="utf-8")
    print(json.dumps({"status": "PASS", "output_dir": str(output)}))


if __name__ == "__main__": main()
