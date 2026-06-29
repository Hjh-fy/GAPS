"""Summarize communication-compression Flower classification experiments."""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import ConcatDataset, DataLoader

from gaps_flower.evaluate_checkpoint import (
    evaluate_classification,
    load_checkpoint_model,
    make_loader,
    resolve_device,
)


CHECKPOINT_RE = re.compile(r"^server_round_(?P<round>\d{3})(?P<adapted>_adapted)?\.pth$")


@dataclass(frozen=True)
class CheckpointSpec:
    path: Path
    round_id: int
    variant: str


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def make_target_loader(data_root: str | Path, client_ids: list[int], split: str, batch_size: int) -> DataLoader:
    datasets = [make_loader(data_root, client_id, split, batch_size).dataset for client_id in client_ids]
    dataset = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)


def macro_f1_from_confusion(confusion: list[list[int]]) -> float:
    matrix = torch.tensor(confusion, dtype=torch.float64)
    scores = []
    for cls_id in range(matrix.size(0)):
        tp = matrix[cls_id, cls_id]
        fp = matrix[:, cls_id].sum() - tp
        fn = matrix[cls_id, :].sum() - tp
        denom = (2.0 * tp) + fp + fn
        if denom > 0:
            scores.append(float((2.0 * tp / denom).item()))
    return float(sum(scores) / len(scores)) if scores else 0.0


def discover_round_checkpoints(run_dir: Path) -> list[CheckpointSpec]:
    specs = []
    for path in sorted(run_dir.glob("server_round_*.pth")):
        match = CHECKPOINT_RE.match(path.name)
        if not match:
            continue
        specs.append(
            CheckpointSpec(
                path=path,
                round_id=int(match.group("round")),
                variant="adapted" if match.group("adapted") else "base",
            )
        )
    return sorted(specs, key=lambda item: (item.round_id, item.variant))


def evaluate_spec(
    spec: CheckpointSpec,
    *,
    data_root: str | Path,
    target_clients: list[int],
    split: str,
    device: torch.device,
    batch_size: int,
    num_classes: int,
    ece_bins: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model, config, checkpoint = load_checkpoint_model(str(spec.path), device, batch_size)
    loader = make_target_loader(data_root, target_clients, split, config.BATCH_SIZE)
    metrics = evaluate_classification(
        model=model,
        loader=loader,
        device=device,
        num_classes=num_classes,
        ece_bins=ece_bins,
        inference_mode="logits",
    )
    row = {
        "checkpoint": str(spec.path),
        "round": int(spec.round_id),
        "variant": spec.variant,
        "num_examples": int(metrics["num_examples"]),
        "accuracy": float(metrics["accuracy"]),
        "macro_f1": macro_f1_from_confusion(metrics["confusion_matrix"]),
        "ece": float(metrics["ece"]),
        "nll": float(metrics["nll"]),
        "macro_accuracy": float(metrics["macro_accuracy"]),
        "mean_confidence": float(metrics["mean_confidence"]),
        "mean_margin": float(metrics["mean_margin"]),
        "adaptive": bool(checkpoint.get("adaptive", False)),
        "post_da": bool(checkpoint.get("post_da", False)),
        "post_da_steps": int(checkpoint.get("post_da_steps", 0) or 0),
    }
    client_rows = []
    for client_id in target_clients:
        client_loader = make_loader(data_root, client_id, split, config.BATCH_SIZE)
        client_metrics = evaluate_classification(
            model=model,
            loader=client_loader,
            device=device,
            num_classes=num_classes,
            ece_bins=ece_bins,
            inference_mode="logits",
        )
        client_rows.append(
            {
                "client_id": int(client_id),
                "num_examples": int(client_metrics["num_examples"]),
                "accuracy": float(client_metrics["accuracy"]),
                "macro_f1": macro_f1_from_confusion(client_metrics["confusion_matrix"]),
                "ece": float(client_metrics["ece"]),
                "nll": float(client_metrics["nll"]),
                "macro_accuracy": float(client_metrics["macro_accuracy"]),
            }
        )
    return row, client_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def add_baseline_gaps(rows: list[dict[str, Any]], baseline_profile: str) -> None:
    baseline = next((row for row in rows if row["profile_id"] == baseline_profile), None)
    if not baseline:
        return
    base_acc = float(baseline["accuracy"])
    base_f1 = float(baseline["macro_f1"])
    base_nll = float(baseline["nll"])
    base_ece = float(baseline["ece"])
    base_rounds = max(1, int(baseline["communication_rounds"]))
    for row in rows:
        row["accuracy_gap_to_baseline"] = float(row["accuracy"]) - base_acc
        row["macro_f1_gap_to_baseline"] = float(row["macro_f1"]) - base_f1
        row["nll_gap_to_baseline"] = float(row["nll"]) - base_nll
        row["ece_gap_to_baseline"] = float(row["ece"]) - base_ece
        row["communication_reduction_vs_baseline"] = 1.0 - (
            float(row["communication_rounds"]) / float(base_rounds)
        )


def write_report(
    path: Path,
    *,
    profiles_config: str,
    matrix_config: str,
    results_root: Path,
    data_root: str,
    split: str,
    final_rows: list[dict[str, Any]],
    missing: list[str],
) -> None:
    lines = [
        "# Communication Compression Classification Summary",
        "",
        f"- profiles_config: `{profiles_config}`",
        f"- matrix_config: `{matrix_config}`",
        f"- results_root: `{results_root}`",
        f"- data_root: `{data_root}`",
        f"- split: `{split}`",
        "",
        "## Final Profile Metrics",
        "",
        "| profile | rounds | post-DA steps | accuracy | macro-F1 | ECE | NLL | acc gap vs 25R | communication reduction |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in final_rows:
        lines.append(
            "| {profile_id} | {communication_rounds} | {post_da_steps} | {accuracy:.4f} | "
            "{macro_f1:.4f} | {ece:.4f} | {nll:.4f} | {accuracy_gap_to_baseline:.4f} | "
            "{communication_reduction_vs_baseline:.2%} |".format(**row)
        )
    if not final_rows:
        lines.append("| _none_ | | | | | | | | |")

    lines.extend(
        [
            "",
            "## Interpretation Rule",
            "",
            "The compression claim is supported when `10R_strong_DA` or a post-DA profile keeps target accuracy/macro-F1 close to `25R_strong_DA` while using 60% fewer communication rounds.",
            "",
            "Post-DA profiles are server-only adaptation after the 10R communication run. They do not add client-server communication rounds and must use calibration splits only.",
        ]
    )
    if missing:
        lines.extend(["", "## Missing Checkpoints", ""])
        lines.extend(f"- {item}" for item in missing)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize communication-compression classification runs")
    parser.add_argument("--matrix-config", default="configs/communication_compression_matrix_20260630.json")
    parser.add_argument("--profiles-config", default="configs/communication_compression_profiles_20260630.json")
    parser.add_argument("--results-root", default="results/communication_compression_20260630")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", default="results/communication_compression_20260630_summary")
    parser.add_argument("--split", choices=("test", "calibration", "full"), default="test")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument("--ece-bins", type=int, default=15)
    parser.add_argument("--skip-rounds", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    matrix = load_json(args.matrix_config)
    profiles = load_json(args.profiles_config)
    runs = {str(run["run_id"]): run for run in matrix.get("runs", [])}
    results_root = Path(args.results_root)
    output_dir = Path(args.output_dir)
    device = resolve_device(args.device)

    per_round_rows: list[dict[str, Any]] = []
    final_rows: list[dict[str, Any]] = []
    client_rows: list[dict[str, Any]] = []
    missing: list[str] = []

    if not args.skip_rounds:
        for run_id, run in runs.items():
            run_dir = results_root / run_id
            if not run_dir.exists():
                missing.append(f"{run_id}: missing run directory")
                continue
            target_clients = [int(cid) for cid in run.get("target_clients", [])]
            for spec in discover_round_checkpoints(run_dir):
                row, _client_rows = evaluate_spec(
                    spec,
                    data_root=args.data_root,
                    target_clients=target_clients,
                    split=args.split,
                    device=device,
                    batch_size=args.batch_size,
                    num_classes=args.num_classes,
                    ece_bins=args.ece_bins,
                )
                row.update(
                    {
                        "run_id": run_id,
                        "communication_rounds": int(run.get("rounds", spec.round_id)),
                        "source_clients": ",".join(str(cid) for cid in run.get("source_clients", [])),
                        "target_clients": ",".join(str(cid) for cid in target_clients),
                    }
                )
                per_round_rows.append(row)

    for profile in profiles.get("profiles", []):
        run_id = str(profile["run_id"])
        run = runs.get(run_id)
        if run is None:
            missing.append(f"{profile['profile_id']}: unknown run_id {run_id}")
            continue
        checkpoint = results_root / run_id / str(profile["checkpoint"])
        if not checkpoint.exists():
            missing.append(f"{profile['profile_id']}: missing checkpoint {checkpoint}")
            continue
        target_clients = [int(cid) for cid in run.get("target_clients", [])]
        spec = CheckpointSpec(
            path=checkpoint,
            round_id=int(profile.get("communication_rounds", run.get("rounds", -1))),
            variant=str(profile["profile_id"]),
        )
        row, profile_client_rows = evaluate_spec(
            spec,
            data_root=args.data_root,
            target_clients=target_clients,
            split=args.split,
            device=device,
            batch_size=args.batch_size,
            num_classes=args.num_classes,
            ece_bins=args.ece_bins,
        )
        row.update(
            {
                "profile_id": str(profile["profile_id"]),
                "run_id": run_id,
                "communication_rounds": int(profile.get("communication_rounds", run.get("rounds", -1))),
                "post_da_steps": int(profile.get("post_da_steps", 0)),
                "role": str(profile.get("role", "")),
                "checkpoint_name": str(profile["checkpoint"]),
            }
        )
        final_rows.append(row)
        for client_row in profile_client_rows:
            client_row.update(
                {
                    "profile_id": str(profile["profile_id"]),
                    "run_id": run_id,
                    "communication_rounds": int(row["communication_rounds"]),
                    "post_da_steps": int(row["post_da_steps"]),
                }
            )
            client_rows.append(client_row)

    add_baseline_gaps(final_rows, str(profiles.get("baseline_profile", "")))
    write_csv(output_dir / "per_round_target_metrics.csv", per_round_rows)
    write_csv(output_dir / "final_profile_metrics.csv", final_rows)
    write_csv(output_dir / "profile_client_metrics.csv", client_rows)
    write_report(
        output_dir / "communication_compression_report.md",
        profiles_config=args.profiles_config,
        matrix_config=args.matrix_config,
        results_root=results_root,
        data_root=args.data_root,
        split=args.split,
        final_rows=final_rows,
        missing=missing,
    )
    print(f"per_round: {output_dir / 'per_round_target_metrics.csv'} ({len(per_round_rows)} rows)")
    print(f"final: {output_dir / 'final_profile_metrics.csv'} ({len(final_rows)} rows)")
    print(f"clients: {output_dir / 'profile_client_metrics.csv'} ({len(client_rows)} rows)")
    print(f"report: {output_dir / 'communication_compression_report.md'}")
    if missing:
        print("missing:")
        for item in missing:
            print(f"  - {item}")


if __name__ == "__main__":
    main()
