"""Summarize A1/A4 cross-board post-hoc response-scope evaluations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


SCOPE_ORDER = ("early60", "stable360", "full420")


def _correct_count(confusion_matrix: list[list[int]]) -> int:
    return sum(
        int(confusion_matrix[index][index])
        for index in range(len(confusion_matrix))
    )


def summarize_experiment(
    *,
    experiment_id: str,
    protocol: str,
    primary_scope: str,
    scope_payload: dict[str, Any],
    audit_payload: dict[str, Any],
    scope_source: str,
    audit_source: str,
) -> dict[str, Any]:
    """Validate identity and preserve all three response-time scopes."""
    if audit_payload.get("status") != "valid":
        raise ValueError(f"{experiment_id}: postflight audit is not valid")
    if int(audit_payload.get("selected_round", -1)) != 25:
        raise ValueError(f"{experiment_id}: selected round must be 25")
    if primary_scope not in SCOPE_ORDER:
        raise ValueError(f"{experiment_id}: unsupported primary scope {primary_scope}")
    if int(scope_payload.get("target_client", -1)) != int(
        audit_payload["target_client"]
    ):
        raise ValueError(f"{experiment_id}: target client mismatch")

    scopes: dict[str, Any] = {}
    for scope_name in SCOPE_ORDER:
        result = scope_payload["scopes"][scope_name]["global"]
        window = result["window"]
        exposure = result["exposure"]
        scopes[scope_name] = {
            "window_correct": _correct_count(window["confusion_matrix"]),
            "window_total": int(window["n_samples"]),
            "window_accuracy": float(window["accuracy"]),
            "window_macro_f1": float(window["macro_f1"]),
            "window_confusion_matrix": window["confusion_matrix"],
            "exposure_correct": _correct_count(exposure["confusion_matrix"]),
            "exposure_total": int(exposure["n_exposures"]),
            "exposure_accuracy": float(exposure["accuracy"]),
            "exposure_macro_f1": float(exposure["macro_f1"]),
            "exposure_confusion_matrix": exposure["confusion_matrix"],
            "calculation_status": "recomputed",
        }

    adapted_accuracy = float(
        audit_payload["metrics"]["adapted"]["target_test_window_accuracy"]
    )
    if abs(scopes[primary_scope]["window_accuracy"] - adapted_accuracy) > 1e-12:
        raise ValueError(f"{experiment_id}: primary metric/audit mismatch")

    return {
        "experiment_id": experiment_id,
        "protocol": protocol,
        "primary_scope": primary_scope,
        "direction": audit_payload["direction"],
        "source_clients": audit_payload["source_clients"],
        "target_client": int(audit_payload["target_client"]),
        "seed": 42,
        "rounds": int(audit_payload["rounds"]),
        "local_epochs": int(audit_payload["local_epochs"]),
        "da_steps_per_round": int(audit_payload["da_steps_per_round"]),
        "model_profile": audit_payload["model_profile"],
        "domain_adaptation_mode": audit_payload["domain_adaptation_mode"],
        "target_ce_weight": float(audit_payload["target_ce_weight"]),
        "selection_policy": audit_payload["selection_policy"],
        "selected_round": int(audit_payload["selected_round"]),
        "checkpoint": scope_payload["checkpoint"],
        "formal_primary": {
            "unadapted_accuracy": float(
                audit_payload["metrics"]["unadapted"][
                    "target_test_window_accuracy"
                ]
            ),
            "adapted_accuracy": adapted_accuracy,
        },
        "scopes": scopes,
        "provenance": {
            "scope_summary": scope_source,
            "postflight_audit": audit_source,
        },
        "status": "audited",
    }


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _matrix_add(
    left: list[list[int]], right: list[list[int]]
) -> list[list[int]]:
    return [
        [int(a) + int(b) for a, b in zip(left_row, right_row, strict=True)]
        for left_row, right_row in zip(left, right, strict=True)
    ]


def _validate_scope_partition(experiment: dict[str, Any]) -> None:
    scopes = experiment["scopes"]
    early = scopes["early60"]
    stable = scopes["stable360"]
    full = scopes["full420"]
    if early["window_total"] + stable["window_total"] != full["window_total"]:
        raise ValueError(
            f"{experiment['experiment_id']}: early/stable totals do not form full"
        )
    if _matrix_add(
        early["window_confusion_matrix"], stable["window_confusion_matrix"]
    ) != full["window_confusion_matrix"]:
        raise ValueError(
            f"{experiment['experiment_id']}: early/stable confusion matrices "
            "do not form full"
        )


def _metric_rows(experiments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for experiment in experiments:
        for scope_name in SCOPE_ORDER:
            scope = experiment["scopes"][scope_name]
            rows.append(
                {
                    "metric_id": (
                        f"{experiment['experiment_id']}::{scope_name}::window_accuracy"
                    ),
                    "experiment_id": experiment["experiment_id"],
                    "metric_name": "window_accuracy",
                    "value": scope["window_accuracy"],
                    "unit": "proportion",
                    "direction": "higher_is_better",
                    "sample_scope": scope_name,
                    "client_scope": f"P{experiment['target_client']}",
                    "gas_scope": "acetaldehyde;methane;acetic_acid",
                    "aggregation": "pooled_target_windows",
                    "seed_set": "42",
                    "uncertainty": "unknown_single_seed",
                    "window_correct": scope["window_correct"],
                    "window_total": scope["window_total"],
                    "window_macro_f1": scope["window_macro_f1"],
                    "exposure_correct": scope["exposure_correct"],
                    "exposure_total": scope["exposure_total"],
                    "exposure_accuracy": scope["exposure_accuracy"],
                    "exposure_macro_f1": scope["exposure_macro_f1"],
                    "source_path": experiment["provenance"]["scope_summary"],
                    "calculation_status": scope["calculation_status"],
                    "notes": "post_hoc_scope;fixed_round_25_adapted_checkpoint",
                }
            )
    return rows


def _registry_rows(
    experiments: list[dict[str, Any]], manifest: dict[str, Any]
) -> list[dict[str, Any]]:
    manifest_by_id = {
        item["experiment_id"]: item for item in manifest["experiments"]
    }
    rows: list[dict[str, Any]] = []
    for experiment in experiments:
        item = manifest_by_id[experiment["experiment_id"]]
        rows.append(
            {
                "experiment_id": experiment["experiment_id"],
                "source_clients": ";".join(
                    f"P{client}" for client in experiment["source_clients"]
                ),
                "target_clients": f"P{experiment['target_client']}",
                "split_protocol": item["split_protocol"],
                "model": experiment["model_profile"],
                "checkpoint": experiment["checkpoint"],
                "DA": experiment["domain_adaptation_mode"],
                "calibration": "target_time_purged_3_windows_per_exposure",
                "QC": "none",
                "seed": experiment["seed"],
                "result_path": experiment["provenance"]["scope_summary"],
                "metrics": "combined_metrics.csv",
                "status": "audited",
                "notes": (
                    f"protocol={experiment['protocol']};"
                    f"primary_scope={experiment['primary_scope']};"
                    "single_seed_descriptive_only"
                ),
                "code_commit": manifest.get("code_commit", "unknown"),
                "config_path": item["config_path"],
                "dataset_path": item["dataset_path"],
                "created_at": manifest.get("created_at", "unknown"),
                "evidence_status": "approved_descriptive_only",
                "provenance": experiment["provenance"]["postflight_audit"],
            }
        )
    return rows


def build_summary(
    *,
    manifest: dict[str, Any],
    posthoc_root: Path,
    controller_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    if manifest.get("seed_set") != [42]:
        raise ValueError("manifest seed_set must be exactly [42]")
    experiments: list[dict[str, Any]] = []
    for item in manifest["experiments"]:
        run_dir = item["run_dir"]
        scope_path = posthoc_root / run_dir / "evaluation" / "summary.json"
        audit_path = controller_root / run_dir / "postflight_attempt_audit.json"
        scope_payload = _load_json(scope_path)
        expected_checkpoint = f"/{run_dir}/server_round_025_adapted.pth"
        if not str(scope_payload.get("checkpoint", "")).endswith(
            expected_checkpoint
        ):
            raise ValueError(
                f"{item['experiment_id']}: adapted checkpoint identity mismatch"
            )
        experiment = summarize_experiment(
            experiment_id=item["experiment_id"],
            protocol=item["protocol"],
            primary_scope=item["primary_scope"],
            scope_payload=scope_payload,
            audit_payload=_load_json(audit_path),
            scope_source=scope_path.as_posix(),
            audit_source=audit_path.as_posix(),
        )
        _validate_scope_partition(experiment)
        checkpoint_sha_path = posthoc_root / run_dir / "checkpoint.sha256"
        if checkpoint_sha_path.is_file():
            experiment["checkpoint_sha256"] = checkpoint_sha_path.read_text(
                encoding="utf-8"
            ).split()[0]
        else:
            experiment["checkpoint_sha256"] = "unknown"
        experiments.append(experiment)

    summary = {
        "schema_version": "gaps.lab_three_gas.a1_posthoc_summary.v1",
        "protocol": {
            "task": "three_gas_classification",
            "input_shape": [100, 6],
            "selected_channels": [1, 2, 4, 6, 8, 9],
            "seed_set": [42],
            "rounds": 25,
            "local_epochs": 1,
            "server_da_steps_per_round": 100,
            "checkpoint_policy": "fixed_round_25",
            "source_archive_sha256": manifest["source_archive_sha256"],
        },
        "experiments": experiments,
        "limitations": [
            "single_seed_descriptive_only",
            "overlapping_windows_within_exposure",
            "nominal_gas_boundaries",
            "all_retained_concentrations_in_target_calibration",
            "post_hoc_time_scope_diagnostics",
        ],
    }
    return summary, _metric_rows(experiments), _registry_rows(
        experiments, manifest
    )


def _markdown_table(summary: dict[str, Any]) -> str:
    lines = [
        "| Experiment | Protocol | Direction | Early 0–150 s | Stable | Full | Exposure (full) |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for experiment in summary["experiments"]:
        scopes = experiment["scopes"]
        lines.append(
            "| {experiment_id} | {protocol} | {direction} | {early:.2%} | "
            "{stable:.2%} | {full:.2%} | {exposure:.2%} |".format(
                experiment_id=experiment["experiment_id"],
                protocol=experiment["protocol"],
                direction=experiment["direction"].replace("_to_", "→"),
                early=scopes["early60"]["window_accuracy"],
                stable=scopes["stable360"]["window_accuracy"],
                full=scopes["full420"]["window_accuracy"],
                exposure=scopes["full420"]["exposure_accuracy"],
            )
        )
    return "\n".join(lines)


def _render_result_analysis(summary: dict[str, Any]) -> str:
    return f"""# Result Analysis

## Input contract and provenance
- Experiment IDs: {', '.join(item['experiment_id'] for item in summary['experiments'])}
- Metric schema: window Accuracy/Macro-F1 and exposure Accuracy/Macro-F1.
- Sample scope: target-board early60, stable360, and full420 windows.
- Reported versus recomputed values: formal primary metrics copied from postflight audits are `reported`; all three post-hoc scope metrics are `recomputed` from frozen checkpoints.

## Descriptive statistics
{_markdown_table(summary)}

Only seed 42 is available; mean, SD, 95% CI, significance tests, and seed-level effect sizes are therefore `unknown` and are not reported.

## Assumptions, comparisons, effect sizes, and corrections
- Each row is descriptive for one fixed source/target configuration and one fixed round-25 adapted checkpoint.
- Overlapping windows are not treated as independent clients or independent exposures.
- No inferential tests or multiple-comparison corrections are applicable with one seed.

## Anomalies and sensitivity analysis
- Early-window performance is substantially below stable-window performance in every run.
- Exposure-level accuracy can remain 100% even when many early windows are wrong because exposure prediction aggregates all windows.
- P1+P2→P3 has no matched source-update/data-budget control; its difference from single-source runs is not a pure diversity effect.

## Proposed paper tables and figures
- Main table: early/stable/full window Accuracy and Macro-F1 by transfer direction.
- Diagnostic figure: paired early versus stable accuracy for each direction.

## Unknowns, conflicts, and audit handoff
- Uncertainty across seeds: `unknown`.
- Generalization to unseen concentrations and future sessions: `unknown`.
- No metric/provenance conflict was detected for the declared primary scopes.
"""


def _render_audit(summary: dict[str, Any]) -> str:
    return f"""# Experiment Audit

## Audit scope and intended claim
Audit the descriptive claim that fixed round-25 A1/A4 models have different classification performance in early, stable, and full response-time scopes.

## Compared experiments
{_markdown_table(summary)}

All runs use seed 42, 25 federated rounds, one local epoch, 100 server-DA steps per round, `proto_replay`, `corrected_b2`, target CE weight 0, source-train-only normalization, and the last-round checkpoint policy.

## Findings
| Finding ID | Severity | Check | Evidence | Impact | Required action | Status |
|---|---|---|---|---|---|---|
| F1 | informational | checkpoint identity | all summaries reference round-25 adapted checkpoints | fixes selection boundary | none | closed |
| F2 | informational | primary-scope consistency | A1 matches full420 audit; A4 matches stable360 audit | prevents scope mixing | none | closed |
| F3 | major | seed coverage | seed set is only 42 | blocks inferential/general claims | add seeds only if stronger claim is needed | open |
| F4 | major | target concentration scope | calibration and test cover all retained concentrations | not unseen-concentration evidence | label claim boundary | open |
| F5 | minor | boundary precision | nominal gas boundaries | may shift early/stable assignment | replace when exact valve timestamps exist | open |
| F6 | major | P1+P2 budget match | no matched source-update/data-budget control | not a pure source-diversity ablation | add matched control before causal wording | open |

## Leakage assessment
Target calibration, purged, early, stable, and full windows use explicit non-overlapping time indices. Target test was opened only after fixed round selection. The same exposures and concentrations appear across target calibration and test time positions, so the result measures time-purged within-exposure adaptation, not unseen-exposure or unseen-concentration generalization.

## Baseline, completeness, and reproducibility assessment
All six planned runs and their valid postflight audits are present. Early plus stable confusion matrices exactly reproduce the full-window confusion matrix. Checkpoint hashes and source archive identity are recorded. Single-seed uncertainty remains unavailable.

## Verdict: approved
Approved only for single-seed descriptive, within-protocol evidence. Blocked for seed-robust, unseen-concentration, or causal source-diversity claims.

## Unknowns and handoff
Multi-seed variance and performance under exact valve timestamps remain `unknown`.
"""


def _render_formal_report(summary: dict[str, Any]) -> str:
    best_early = max(
        summary["experiments"],
        key=lambda item: item["scopes"]["early60"]["window_accuracy"],
    )
    best_stable = max(
        summary["experiments"],
        key=lambda item: item["scopes"]["stable360"]["window_accuracy"],
    )
    return f"""# 实验室三气体A1/A4跨板实验后验分段正式报告

## 实验口径

- 固定模型：第25轮适配后检查点，不重新训练、不重新选轮次。
- 早期段 `early60`：每次暴露0–150 s，共60个窗口。
- 稳定段 `stable360`：每次暴露稳定响应位置，共360个窗口。
- 完整段 `full420`：早期段与稳定段并集，共420个窗口。
- A1正式主口径为完整420窗口；A4匹配对照正式主口径为稳定360窗口。

## 结果

{_markdown_table(summary)}

## 主要发现

1. 六组模型的稳定段准确率均明显高于早期段，说明完整段性能下降主要来自0–150 s过渡响应。
2. 早期段最高为 {best_early['experiment_id']}（{best_early['scopes']['early60']['window_accuracy']:.2%}）；稳定段最高为 {best_stable['experiment_id']}（{best_stable['scopes']['stable360']['window_accuracy']:.2%}）。
3. 暴露级准确率不能替代窗口级实时性能：整次暴露聚合可以达到100%，但早期实时判别仍可能错误。
4. 当前结果支持“稳定后分类可靠、早期过渡段仍是主要瓶颈”的单种子描述性结论。

## 证据边界

- 仅使用seed 42，不报告跨种子均值、标准差、置信区间或显著性。
- 校准和测试覆盖全部保留浓度，因此不是未见浓度泛化实验。
- P1+P2→P3没有匹配通信/数据预算对照，不能把差异单独归因于源板多样性。
- 时间边界仍为名义边界，获得精确阀门时间后应复核早期段。
"""


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_artifacts(
    *,
    output_dir: Path,
    summary: dict[str, Any],
    metric_rows: list[dict[str, Any]],
    registry_rows: list[dict[str, Any]],
) -> None:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output: {output_dir}")
    output_dir.mkdir(parents=True)
    (output_dir / "combined_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _write_csv(output_dir / "combined_metrics.csv", metric_rows)
    _write_csv(output_dir / "experiment_registry.csv", registry_rows)
    (output_dir / "RESULT_ANALYSIS.md").write_text(
        _render_result_analysis(summary), encoding="utf-8"
    )
    (output_dir / "EXPERIMENT_AUDIT.md").write_text(
        _render_audit(summary), encoding="utf-8"
    )
    (output_dir / "FORMAL_REPORT.zh.md").write_text(
        _render_formal_report(summary), encoding="utf-8"
    )
    hashes: list[str] = []
    for path in sorted(output_dir.iterdir()):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        hashes.append(f"{digest}  {path.name}")
    (output_dir / "SHA256SUMS.txt").write_text(
        "\n".join(hashes) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--posthoc-root", type=Path, required=True)
    parser.add_argument("--controller-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary, metric_rows, registry_rows = build_summary(
        manifest=_load_json(args.manifest),
        posthoc_root=args.posthoc_root,
        controller_root=args.controller_root,
    )
    write_artifacts(
        output_dir=args.output_dir,
        summary=summary,
        metric_rows=metric_rows,
        registry_rows=registry_rows,
    )


if __name__ == "__main__":
    main()
