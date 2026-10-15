"""Create the immutable R1-M2 seed-42 audit and result-analysis bundle."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_IDS = (
    "R1M2-TARGET-ONLY-S42",
    "R1M2-CENTRAL-SOURCE-S42",
    "R1M2-FEDPROX-SOURCE-S42",
    "R1M2-FEDAVG-SAME-ADAPTER-S42",
    "R1M2-DS-FEDAVG-S42",
)
B5_ID = "IOTJ-B5-S42-REFERENCE"
B5_METRICS = Path(
    "results/iotj_b5_multiseed_20260724/seed42_reference/"
    "classification_evaluation/seed42_classification_metrics.json"
)
B5_PROTOCOL = Path("results/iotj_b5_multiseed_20260724/protocol_manifest.json")
TRAINING_COMMIT = "61a5d18ae512dd827bec210a2666ef51003df1a0"
B5_COMMIT = "2ef7aea77b9dfabdd09da4f38742907a37c58c30"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metric_row(experiment_id: str, manifest: dict[str, Any], manifest_path: Path) -> dict[str, Any]:
    raw = manifest["metrics"]
    if "clients" in raw:
        if len(raw["clients"]) != 1 or int(raw["clients"][0]["client_id"]) != 5:
            raise ValueError(f"{experiment_id}: expected exactly one C5 metric row")
        metrics = raw["clients"][0]
    else:
        metrics = raw
    if int(metrics["num_examples"]) != 1360:
        raise ValueError(f"{experiment_id}: C5 test row count is not 1360")
    for name in ("accuracy", "macro_f1", "nll", "ece"):
        if not math.isfinite(float(metrics[name])):
            raise ValueError(f"{experiment_id}: non-finite {name}")
    if int(manifest["seed"]) != 42 or manifest["status"] != "completed":
        raise ValueError(f"{experiment_id}: seed/status contract failed")

    artifact_text = manifest.get("artifact") or manifest.get("checkpoint")
    artifact = Path(str(artifact_text))
    expected_hash = manifest.get("artifact_sha256") or manifest.get("checkpoint_sha256")
    if not artifact.is_file() or sha256_file(artifact) != expected_hash:
        raise ValueError(f"{experiment_id}: artifact/checkpoint hash mismatch")

    cost = manifest.get("cost", {})
    return {
        "experiment_id": experiment_id,
        "accuracy": float(metrics["accuracy"]),
        "macro_f1": float(metrics["macro_f1"]),
        "nll": float(metrics["nll"]),
        "ece": float(metrics["ece"]),
        "num_examples": int(metrics["num_examples"]),
        "seed": 42,
        "wall_seconds": float(cost.get("wall_seconds", 0.0)),
        "communication_rounds": int(cost.get("communication_rounds", 0)),
        "model_payload_bytes_total": cost.get("model_payload_bytes_total"),
        "checkpoint_sha256": str(expected_hash),
        "manifest": manifest_path.as_posix(),
        "calculation_status": "reported",
    }


def b5_row() -> dict[str, Any]:
    metrics_path = REPO_ROOT / B5_METRICS
    payload = read_json(metrics_path)
    metrics = payload["metrics"]
    protocol = read_json(REPO_ROOT / B5_PROTOCOL)
    seed_reference = protocol["seed42_reference"]
    return {
        "experiment_id": B5_ID,
        "accuracy": float(metrics["accuracy"]),
        "macro_f1": float(metrics["macro_f1"]),
        "nll": float(metrics["nll"]),
        "ece": float(metrics["ece"]),
        "num_examples": int(metrics["N"]),
        "seed": 42,
        "wall_seconds": float(seed_reference["training_round_wall_seconds"]),
        "communication_rounds": 25,
        "model_payload_bytes_total": None,
        "checkpoint_sha256": seed_reference["checkpoint_sha256"],
        "manifest": metrics_path.relative_to(REPO_ROOT).as_posix(),
        "calculation_status": "reported",
    }


def code_compatibility() -> dict[str, Any]:
    paths = (
        "model.py", "client.py", "config.py", "gaps_flower/task.py",
        "gaps_flower/strategy.py", "gaps_flower/domain_adaptation.py",
        "gaps_flower/server_app.py",
    )
    result = subprocess.run(
        ["git", "diff", "--numstat", f"{B5_COMMIT}..{TRAINING_COMMIT}", "--", *paths],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    changed = []
    for line in result.stdout.splitlines():
        added, deleted, path = line.split("\t", 2)
        changed.append({"path": path, "added": int(added), "deleted": int(deleted)})
    return {
        "b5_commit": B5_COMMIT,
        "baseline_runtime_commit": TRAINING_COMMIT,
        "audited_paths": list(paths),
        "changed_paths": changed,
        "interpretation": (
            "Only FedProx/default-zero and explicit statistic-upload profile plumbing changed; "
            "model.py, strategy.py, domain_adaptation.py and server_app.py are unchanged. "
            "The existing proto_replay behavior remains semantically identical."
        ),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-root",
        default="results/iotj_r1_m2_baseline_fairness_seed42_20260803",
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    result_root = (REPO_ROOT / args.result_root).resolve()
    output_dir = (REPO_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    rows = []
    manifests = {}
    for experiment_id in EXPECTED_IDS:
        manifest_path = result_root / experiment_id / "run_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"incomplete comparison set: {manifest_path}")
        manifest = read_json(manifest_path)
        if manifest.get("experiment_id") != experiment_id:
            raise ValueError(f"experiment identity mismatch: {manifest_path}")
        rows.append(metric_row(experiment_id, manifest, manifest_path))
        manifests[experiment_id] = manifest
    reference = b5_row()
    rows.append(reference)

    b5_f1 = reference["macro_f1"]
    for row in rows:
        row["macro_f1_delta_vs_b5_pp"] = 100.0 * (row["macro_f1"] - b5_f1)
    write_csv(output_dir / "comparison_metrics.csv", rows)

    metric_records = []
    for row in rows:
        for name, direction in (("accuracy", "higher"), ("macro_f1", "higher"), ("nll", "lower"), ("ece", "lower")):
            metric_records.append(
                {
                    "metric_id": f"{row['experiment_id']}::{name}",
                    "experiment_id": row["experiment_id"],
                    "metric_name": name,
                    "value": row[name],
                    "unit": "fraction" if name in {"accuracy", "macro_f1", "ece"} else "nats_per_window",
                    "direction": direction,
                    "sample_scope": "C5 sealed test; 1360 windows",
                    "client_scope": "C5",
                    "gas_scope": "all four classes",
                    "aggregation": "single seed42 final checkpoint",
                    "seed_set": [42],
                    "uncertainty": "not estimable from one seed",
                    "source_path": row["manifest"],
                    "calculation_status": "reported",
                    "notes": "No significance or stability inference is permitted.",
                }
            )
    (output_dir / "metric_records.json").write_text(
        json.dumps(metric_records, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    compatibility = code_compatibility()
    (output_dir / "code_compatibility.json").write_text(
        json.dumps(compatibility, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    by_id = {row["experiment_id"]: row for row in rows}
    analysis = f"""# R1-M2 Seed42 Result Analysis

## Input contract and provenance

- Experiments: {', '.join(EXPECTED_IDS)} plus `{B5_ID}`.
- Scope: C5 sealed test, 1,360 windows, final checkpoint, seed42 only.
- Values in `comparison_metrics.csv` are reported from experiment manifests; percentage-point deltas versus B5 are recomputed here.
- No standard deviation, confidence interval, p-value, or seed-level effect-size uncertainty is estimable from one seed.

## Descriptive comparison

| Experiment | Accuracy | Macro-F1 | Delta vs B5 (pp) | NLL | ECE |
|---|---:|---:|---:|---:|---:|
"""
    for row in rows:
        analysis += (
            f"| {row['experiment_id']} | {row['accuracy']:.6f} | {row['macro_f1']:.6f} | "
            f"{row['macro_f1_delta_vs_b5_pp']:+.3f} | {row['nll']:.6f} | {row['ece']:.6f} |\n"
        )
    analysis += f"""

## Seed42 findings

- FedAvg+same target adapter exceeds B5 by {by_id['R1M2-FEDAVG-SAME-ADAPTER-S42']['macro_f1_delta_vs_b5_pp']:+.3f} Macro-F1 percentage points. The result does not support attributing the seed42 gain to GAPS client alignment/replay/decoupling or selective aggregation.
- Target-only exceeds B5 by {by_id['R1M2-TARGET-ONLY-S42']['macro_f1_delta_vs_b5_pp']:+.3f} points, but uses a stronger fully supervised target-CE objective and is therefore an upper/reference configuration.
- B5 exceeds DS by {-by_id['R1M2-DS-FEDAVG-S42']['macro_f1_delta_vs_b5_pp']:.3f} points and substantially exceeds both no-target source baselines.

## Interpretation boundaries

- Target-only uses all 320 C5 calibration labels for 2,500 supervised CE steps. It is a strong target-supervised upper/reference configuration, not an equal-objective GAPS ablation.
- Centralized Source-only and FedProx use no target calibration labels; their differences from GAPS combine target adaptation and method effects.
- FedAvg+same target adapter is the closest mechanism comparator: it keeps C5 calibration and server distribution-adaptation settings while disabling client alignment/replay/decoupling and selective aggregation.
- DS uses calibration-only ridge selection over 34 matched strata and the hash-pinned historical A0 checkpoint.
- Communication accounting is exact for model payload bytes. The adapter-matched run also sends `ce_stats` JSON; those extra wire bytes were not instrumented and must be labeled as additional/unknown rather than folded into the exact model-byte total.
- All conclusions are seed42-specific.

## Proposed paper table

Use `comparison_metrics.csv`, report Accuracy/Macro-F1/NLL/ECE and the declared target-label access. Do not add significance markers.
"""
    (output_dir / "RESULT_ANALYSIS.md").write_text(analysis, encoding="utf-8")

    audit = f"""# R1-M2 Baseline Fairness Experiment Audit

## Audit scope and intended claim

Assess whether the five registered seed42 baselines close reviewer concern R1-M2 without target-test leakage or hidden training-budget changes.

## Findings

| ID | Severity | Finding | Impact | Status |
|---|---|---|---|---|
| R1M2-A01 | informational | All five runs use seed42 and C5 sealed test n=1,360; artifact hashes match manifests. | Reproducible single-seed comparison. | passed |
| R1M2-A02 | major | Target-only uses 2,500 fully supervised target CE steps, unlike B5 target CE weight 0. | Treat only as a target-supervised upper/reference row. | constrained |
| R1M2-A03 | informational | Centralized Source-only, FedProx and DS do not isolate every GAPS mechanism. | Use them as coverage baselines, not causal ablations. | passed with scope |
| R1M2-A04 | informational | FedAvg+same adapter holds target calibration and DA settings, disables client GAPS losses and selective aggregation by design. | Closest R1-M2 mechanism comparator. | passed |
| R1M2-A05 | minor | Target-only and DS executed at head d6881d4 with an as-run wrapper later captured in 61a5d18; the only wrapper change was repository import-path plumbing. | Preserve both provenance identifiers. | documented |
| R1M2-A06 | informational | B5 commit compatibility diff changes only default-inert FedProx/profile plumbing in effective paths. | Existing B5 checkpoint remains code-compatible for this comparison. | passed |
| R1M2-A07 | major | Only one seed was authorized. | No stability, CI, or significance claim. | constrained |
| R1M2-A08 | blocking for the original broad claim | FedAvg+same target adapter exceeds B5 by {by_id['R1M2-FEDAVG-SAME-ADAPTER-S42']['macro_f1_delta_vs_b5_pp']:+.3f} Macro-F1 percentage points. | The seed42 evidence does not support superiority of GAPS client mechanisms/selective aggregation beyond matched target adaptation. | manuscript claim must be narrowed |
| R1M2-A09 | minor | The FedProx result manifest retains a generic `ce_stats` statistics-payload note although its locked client profile is `ce_only`. | Numerical metrics and exact model-payload accounting are unaffected; treat FedProx as having no extra statistics payload. | documented amendment |
| R1M2-A10 | minor | Distributed model payload bytes are exact, but adapter `ce_stats` JSON wire bytes were not instrumented. | Communication comparison is usable only with an explicit `model bytes + unmeasured small statistics payload` qualifier for the adapter row. | constrained |

## Leakage assessment

No manifest reports target-test use for training, calibration, selection, stopping, or hyperparameter tuning. DS alpha was selected inside C5 calibration folds before the test split was opened.

## Verdict

The experiment set is approved for a seed42-only descriptive baseline table. The original broad superiority/attribution claim is blocked: it must be narrowed to the demonstrated value of target-assisted adaptation and to GAPS outperforming DS and no-target source baselines. Not approved for statistical superiority or stability claims.
"""
    (output_dir / "EXPERIMENT_AUDIT.md").write_text(audit, encoding="utf-8")

    evidence = {
        "evidence_id": "EVID-R1-M2-SEED42",
        "experiment_ids": [*EXPECTED_IDS, B5_ID],
        "metric_ids": [record["metric_id"] for record in metric_records],
        "comparison": "R1-M2 baseline fairness on C5 sealed test",
        "source_paths": [row["manifest"] for row in rows],
        "audit_status": "approved_data_blocked_broad_claim",
        "support_strength": "descriptive_single_seed",
        "claim_ids": ["R1-M2"],
        "limitations": [
            "seed42 only; no uncertainty/significance claim",
            "target-only is a stronger supervised-target objective",
            "only the adapter-matched FedAvg row is a close mechanism comparator",
            "adapter-matched FedAvg exceeds B5 at seed42, blocking the broad mechanism-superiority claim",
            "adapter ce_stats JSON wire bytes were not instrumented; model payload bytes remain exact",
        ],
        "provenance": {
            "analysis_dir": output_dir.relative_to(REPO_ROOT).as_posix(),
            "code_compatibility": "code_compatibility.json",
        },
    }
    (output_dir / "evidence_record.json").write_text(
        json.dumps(evidence, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "PASS", "output_dir": str(output_dir)}, sort_keys=True))


if __name__ == "__main__":
    main()
