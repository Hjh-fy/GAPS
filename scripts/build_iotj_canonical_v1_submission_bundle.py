"""Build the Git-trackable canonical-v1 IoT-J submission evidence closure."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DATASET_HASH = "2f810d7e93cae5f361923184e9dc87d5ae59e0f59be9f52aff7e14f9f33e94f6"
REQUIRED_EVIDENCE_FILES = (
    "01_dataset_manifest.json",
    "02_final_experiment_state.json",
    "03_classification_final.csv",
    "04_regression_final.csv",
    "05_qc_final.csv",
    "06_quality_robustness.csv",
    "07_pi5_benchmark.csv",
    "08_model_size.json",
    "09_a0t_equal_label.csv",
    "10_fedridge_83d_84d.csv",
    "11_window_overlap.csv",
    "12_reproducibility_manifest.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    fields: list[str] = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fields} for row in rows)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def record(path: Path) -> dict[str, Any]:
    return {
        "source_artifact_path": str(path.resolve()),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
        "dataset_hash": DATASET_HASH,
        "canonical": True,
    }


def select(rows: list[dict[str, str]], **values: object) -> dict[str, str]:
    matches = [row for row in rows if all(str(row.get(key)) == str(value) for key, value in values.items())]
    if len(matches) != 1:
        raise RuntimeError(f"expected one row for {values}, found {len(matches)}")
    return matches[0]


def build(study: Path, docs: Path) -> None:
    study, docs = study.resolve(), docs.resolve()
    docs.mkdir(parents=True, exist_ok=True)
    closure = study / "evidence_closure"
    sources = {
        "dataset": ROOT / "dataset/iotj_canonical_v1/canonical_preprocessing_manifest.json",
        "state": study / "FINAL_EXPERIMENT_STATE.json",
        "classification": study / "classification_evaluation/classification_metrics.csv",
        "regression": study / "regression/cross_target_r84_summary.csv",
        "qc": closure / "qc/qc_equal_mean_summary.csv",
        "random_qc": closure / "qc/qc_random_reference.csv",
        "quality": closure / "quality/quality_stratified_summary.csv",
        "pi5": study / "pi5_benchmark/pi5_benchmark_summary.csv",
        "pi5_environment": study / "pi5_benchmark/benchmark_environment.json",
        "model_size": study / "deployment_package/model_size_audit.json",
        "fedridge": closure / "fedridge_ablation/canonical_83d_vs_84d.csv",
        "overlap": closure / "overlap/window_overlap_summary.csv",
        "package": study / "deployment_package/package_manifest.json",
        "package_archive": study / "deployment_package_f3d1577.tar.gz",
    }
    missing = [str(path) for path in sources.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"submission evidence source missing: {missing}")

    shutil.copy2(sources["dataset"], docs / REQUIRED_EVIDENCE_FILES[0])
    shutil.copy2(sources["state"], docs / REQUIRED_EVIDENCE_FILES[1])
    shutil.copy2(sources["classification"], docs / REQUIRED_EVIDENCE_FILES[2])
    shutil.copy2(sources["regression"], docs / REQUIRED_EVIDENCE_FILES[3])
    shutil.copy2(sources["qc"], docs / REQUIRED_EVIDENCE_FILES[4])
    shutil.copy2(sources["quality"], docs / REQUIRED_EVIDENCE_FILES[5])
    shutil.copy2(sources["pi5"], docs / REQUIRED_EVIDENCE_FILES[6])
    shutil.copy2(sources["model_size"], docs / REQUIRED_EVIDENCE_FILES[7])
    a0t_rows = [{
        "target": target,
        "status": "BLOCKED_NOT_RUN",
        "reason": "preregistered canonical equal-label run exists, but execution authorization was not granted",
        "accuracy": "",
        "macro_f1": "",
        "nll": "",
        "ece": "",
        "canonical": False,
        "test_opened": False,
    } for target in ("C3", "C4", "C5")]
    write_csv(docs / REQUIRED_EVIDENCE_FILES[8], a0t_rows)
    shutil.copy2(sources["fedridge"], docs / REQUIRED_EVIDENCE_FILES[9])
    shutil.copy2(sources["overlap"], docs / REQUIRED_EVIDENCE_FILES[10])

    state = json.loads(sources["state"].read_text(encoding="utf-8"))
    code_commit = git_head()
    provenance = {name: record(path) for name, path in sources.items()}
    provenance["checkpoints"] = state["classification_checkpoint_hashes"]
    reproducibility = {
        "schema_version": "iotj.canonical_v1.submission_evidence.v1",
        "status": "EVIDENCE_CLOSURE_WITH_DISCLOSED_BLOCKERS",
        "dataset_hash": DATASET_HASH,
        "branch": "codex/iotj-final-classification-le1",
        "bundle_generation_commit": code_commit,
        "formal_experiment_commit": state["code_commit"],
        "deployment_runtime_commit": "f3d15775f3c3372dc1589989746d864fa0c332d3",
        "deployment_archive_sha256": sha256(sources["package_archive"]),
        "checkpoint_hashes": state["classification_checkpoint_hashes"],
        "source_artifacts": provenance,
        "canonical_protocol": {
            "source_clients": ["C1", "C2"], "targets": ["C3", "C4", "C5"],
            "preprocessing": "HZ5_MEAN_W10S", "input_shape": [50, 8],
            "router": "A4", "rounds": 25, "local_epochs": 1, "seed": 42,
            "regression": "R84_FED_H1", "qc": "frozen_equal_mean",
        },
        "blocked_items": ["canonical_equal_label_A0T", "strict_grouped_non_overlap_robustness", "canonical_figure_regeneration", "manuscript_v7_consistency_scan"],
    }
    write_json(docs / REQUIRED_EVIDENCE_FILES[11], reproducibility)

    classification = read_csv(sources["classification"])
    regression = read_csv(sources["regression"])
    qc = read_csv(sources["qc"])
    random_qc = read_csv(sources["random_qc"])
    fedridge = read_csv(sources["fedridge"])
    overlap = read_csv(sources["overlap"])
    pi5 = read_csv(sources["pi5"])
    quality = read_csv(sources["quality"])
    model_size = json.loads(sources["model_size"].read_text(encoding="utf-8"))

    master: list[dict[str, Any]] = []
    for row in classification:
        master.append({"evidence": "classification", "scope": row["scope"], "metric_1": "accuracy", "value_1": row["accuracy"], "metric_2": "macro_f1", "value_2": row["macro_f1"], "status": "COMPLETE", "canonical": True})
    for row in regression:
        if row["evaluation_scope"] == "S_ALL":
            master.append({"evidence": "regression", "scope": row["target"], "metric_1": "RMSE_ppm", "value_1": row["RMSE"], "metric_2": "NRMSE", "value_2": row["NRMSE"], "status": "COMPLETE", "canonical": True})
    for scope in ("ALL", "C3", "C4", "C5"):
        for workpoint, target_coverage in (("HC90", "0.9"), ("HC95", "0.95")):
            row = select(qc, scope=scope, workpoint=workpoint, population="accepted")
            random = select(random_qc, scope=scope, target_coverage=target_coverage)
            master.append({"evidence": "qc", "scope": f"{scope}_{workpoint}", "metric_1": "coverage", "value_1": row["coverage"], "metric_2": "NRMSE", "value_2": row["NRMSE_range"], "metric_3": "RMSE_gain_vs_random", "value_3": random["QC_RMSE_improvement_over_random"], "status": "COMPLETE", "canonical": True})
    master.extend([
        {"evidence": "pi5", "scope": "C5_total_pipeline", "metric_1": "P50_ms", "value_1": select(pi5, component="total_pipeline_ms")["P50_ms"], "metric_2": "P95_ms", "value_2": select(pi5, component="total_pipeline_ms")["P95_ms"], "metric_3": "P99_ms", "value_3": select(pi5, component="total_pipeline_ms")["P99_ms"], "status": "COMPLETE", "canonical": True},
        {"evidence": "model_size", "scope": "classifier", "metric_1": "total_parameter_count", "value_1": model_size["classifier_by_target"]["C5"]["total_parameter_count"], "metric_2": "fp32_model_bytes", "value_2": model_size["classifier_by_target"]["C5"]["fp32_model_bytes"], "status": "COMPLETE", "canonical": True},
        {"evidence": "equal_label_fairness", "scope": "C3_C4_C5", "status": "BLOCKED_NOT_RUN", "canonical": False},
        {"evidence": "window_overlap", "scope": "C3_C4_C5", "metric_1": "raw_time_overlap", "value_1": "present", "status": "SUBMISSION_BLOCKER", "canonical": True},
    ])
    write_csv(study / "FINAL_RESULT_MASTER_TABLE.csv", master)

    panels = [
        {"figure": "Fig.1", "panel": "all", "content": "cloud-edge-sensor architecture", "status": "LEGACY_ONLY", "source_csv": "N/A", "source_json": "deployment_package/package_manifest.json", "checkpoint_hash": "N/A", "script": "MISSING", "caption": "Must be checked against canonical 5 Hz, 50x8, LE1 protocol."},
        {"figure": "Fig.2", "panel": "all", "content": "device-domain shift", "status": "NEEDS_REGEN", "source_csv": "canonical sensor-shift evidence not bundled", "source_json": "01_dataset_manifest.json", "checkpoint_hash": "N/A", "script": "MISSING", "caption": "Regenerate from canonical-v1 only."},
        {"figure": "Fig.3", "panel": "all", "content": "cross-target classification", "status": "NEEDS_REGEN", "source_csv": "03_classification_final.csv", "source_json": "02_final_experiment_state.json", "checkpoint_hash": "C3/C4/C5 in state manifest", "script": "MISSING", "caption": "A4, LE1, round25, seed42."},
        {"figure": "Fig.4", "panel": "all", "content": "C5 classification and A0-A6", "status": "LEGACY_ONLY", "source_csv": "canonical A4 exists; canonical ablation table absent", "source_json": "02_final_experiment_state.json", "checkpoint_hash": "C5:3965ec86...", "script": "MISSING", "caption": "Do not relabel historical ablation as canonical."},
        {"figure": "Fig.5", "panel": "all", "content": "concentration/per-gas", "status": "NEEDS_REGEN", "source_csv": "04_regression_final.csv + regression/cross_target_r84_per_gas.csv", "source_json": "regression/protocol_manifest.json", "checkpoint_hash": "C3/C4/C5 in state manifest", "script": "MISSING", "caption": "Canonical A4+R84_FED_H1."},
        {"figure": "Fig.6", "panel": "all", "content": "83D/84D plus calibration budget", "status": "NEEDS_REGEN", "source_csv": "10_fedridge_83d_84d.csv", "source_json": "CALIBRATION_BUDGET_GAP.md", "checkpoint_hash": "C3/C4/C5 in state manifest", "script": "MISSING", "caption": "Budget panel is not yet supported by canonical evidence."},
        {"figure": "Fig.7", "panel": "all", "content": "QC curve/random/HC90/HC95", "status": "NEEDS_REGEN", "source_csv": "05_qc_final.csv + qc_random_reference.csv", "source_json": "qc/protocol_manifest.json", "checkpoint_hash": "C3/C4/C5 in state manifest", "script": "MISSING", "caption": "Frozen equal-mean QC; same-budget random seed 20260804."},
        {"figure": "Fig.8", "panel": "all", "content": "communication/Pi5/physical validation", "status": "NEEDS_REGEN", "source_csv": "07_pi5_benchmark.csv + 06_quality_robustness.csv", "source_json": "08_model_size.json", "checkpoint_hash": "C5:3965ec86...", "script": "MISSING", "caption": "Pi5 formal package f3d1577; communication must not infer 50% reduction."},
        {"figure": "Main table", "panel": "classification/regression/QC", "content": "formal numerical results", "status": "READY", "source_csv": "FINAL_RESULT_MASTER_TABLE.csv", "source_json": "12_reproducibility_manifest.json", "checkpoint_hash": "manifested", "script": "N/A", "caption": "Canonical values only."},
    ]
    write_csv(study / "FINAL_FIGURE_MANIFEST.csv", panels)
    tracker = "# Figure/table panel tracker\n\n" + "\n".join(
        f"- **{row['figure']} {row['panel']} - {row['status']}**: {row['content']}; source `{row['source_csv']}`; script `{row['script']}`. {row['caption']}"
        for row in panels
    ) + "\n"
    (docs / "FIGURE_TABLE_PANEL_TRACKER.md").write_text(tracker, encoding="utf-8")

    calibration_gap = """# Calibration-budget evidence gap

Status: **MISSING_CANONICAL_EVIDENCE**. Historical preprocessing/budget studies are not admissible as canonical-v1 quantitative evidence. No new budget search was started. A future preregistered backend-only sensitivity study should freeze A4 and vary only the R84 calibration subset at 100%/50%/25%; an end-to-end commissioning budget study is a separate supplementary question.
"""
    (docs / "CALIBRATION_BUDGET_GAP.md").write_text(calibration_gap, encoding="utf-8")

    qc_all90 = select(qc, scope="ALL", workpoint="HC90", population="accepted")
    qc_all95 = select(qc, scope="ALL", workpoint="HC95", population="accepted")
    rnd90 = select(random_qc, scope="ALL", target_coverage="0.9")
    rnd95 = select(random_qc, scope="ALL", target_coverage="0.95")
    total_pi = select(pi5, component="total_pipeline_ms")
    all83 = select(fedridge, scope="ALL", gas="ALL", variant="R83_TARGET_ONLY")
    all84 = select(fedridge, scope="ALL", gas="ALL", variant="R84_FED_H1")
    repeat1 = select(quality, scope="C5", slice="Methane_225ppm_repeat1")
    repeat2 = select(quality, scope="C5", slice="Methane_225ppm_repeat2")

    claims = f"""# Final claim-evidence matrix

| Claim | Evidence | Allowed wording | Status |
|---|---|---|---|
| Cross-target classification | `03_classification_final.csv` | A4 reaches 99.851%, 99.779%, 99.412% accuracy on C3/C4/C5 | SUPPORTED |
| Concentration regression | `04_regression_final.csv` | R84 S_ALL RMSE is 9.333/13.808/18.477 ppm | SUPPORTED |
| QC improves retained error over random retention | QC/random CSVs | At achieved HC90/HC95 accepted coverage, aggregate RMSE gain is {float(rnd90['QC_RMSE_improvement_over_random']):.3f}/{float(rnd95['QC_RMSE_improvement_over_random']):.3f} ppm | SUPPORTED_DESCRIPTIVELY |
| Edge runtime | `07_pi5_benchmark.csv` | P50/P95/P99 {float(total_pi['P50_ms']):.3f}/{float(total_pi['P95_ms']):.3f}/{float(total_pi['P99_ms']):.3f} ms; {float(total_pi['throughput_windows_per_second']):.2f} windows/s | SUPPORTED |
| 50% temporal input reduction | dataset manifest/model-size audit | 100x8 to 50x8 points, 3200 to 1600 raw FP32 bytes | SUPPORTED; do not infer 50% latency or FL communication |
| Equal-label superiority | `09_a0t_equal_label.csv` | No comparative claim permitted | BLOCKED |
| Independent calibration/test raw observations | `11_window_overlap.csv` | Exact window identities are disjoint, but raw-time overlap is present | CONTRADICTED/BLOCKED |
| Calibration-budget robustness | `CALIBRATION_BUDGET_GAP.md` | No canonical quantitative claim permitted | MISSING |
"""
    (study / "FINAL_CLAIM_EVIDENCE_MATRIX.md").write_text(claims, encoding="utf-8")

    status_md = """# Final experiment status

- Canonical A4 classification: COMPLETE (C3/C4/C5, 25 rounds, local_epochs=1, seed42).
- Canonical R84_FED_H1 regression: COMPLETE.
- Frozen equal-mean QC/random reference/quality audit: COMPLETE.
- Canonical 83D vs 84D ablation: COMPLETE.
- FINAL_DEPLOYED_RUNTIME package and Pi 5 benchmark: COMPLETE.
- Canonical equal-label A0T: BLOCKED_NOT_RUN; preregistered only.
- Strict grouped non-overlap robustness: NOT_RUN; overlap audit proposal only.
- Further algorithm search: STOPPED by protocol.
"""
    (study / "FINAL_EXPERIMENT_STATUS.md").write_text(status_md, encoding="utf-8")

    audit = f"""# Final submission audit

Status: **EVIDENCE CLOSURE COMPLETE WITH BLOCKERS**.

Passed: frozen dataset hash; target-specific A4 checkpoint hashes; no target-test selection; R84 and QC provenance; 1,000-repeat same-budget random QC; no quality-based deletion; portable package preflight; exact Pi 5 package SHA; parameter-count semantics corrected; canonical 83D/84D comparison.

Blockers: (1) canonical equal-label A0T has no executed result; (2) calibration/test raw-time overlap is {', '.join(row['target'] for row in overlap)} and a strict grouped non-overlap robustness run is absent; (3) canonical figures need regeneration; (4) the requested manuscript v7 source was not available in the repository or discovered manuscript directories, so a six-way manuscript consistency PASS cannot be issued; (5) canonical calibration-budget evidence is absent.

The C5 methane 225 ppm repeat1 anomaly is retained: S_ALL RMSE {float(repeat1['S_ALL_RMSE']):.3f} ppm versus {float(repeat2['S_ALL_RMSE']):.3f} ppm for repeat2. No sample was deleted.
"""
    (study / "FINAL_SUBMISSION_AUDIT.md").write_text(audit, encoding="utf-8")

    consistency = """# Final consistency audit

Code -> dataset manifest -> experiment state -> result CSV -> deployment package are consistent with HZ5_MEAN_W10S, 50x8 input, A4, local_epochs=1, 25 rounds, seed42, R84_FED_H1, and frozen equal-mean QC. The formal deployment archive contains runtime commit f3d1577 and passed its package preflight and Pi 5 hash check.

Manuscript status: **BLOCKED_NOT_AUDITED_AS_V7**. No v7 `main.tex` was found. The available v5/v6 manuscripts are historical and were not treated as the canonical final manuscript. Therefore legacy-token removal from the final Methods/Results (`10 Hz`, `100x8`, `SEQ_LEN=100`, `local_epochs=5`, old C5/QC/deployment values) remains a writing-stage action.
"""
    (docs / "FINAL_CONSISTENCY_AUDIT.md").write_text(consistency, encoding="utf-8")

    index = "# Canonical-v1 final evidence index\n\n" + "\n".join(
        f"- `{name}` - SHA256 `{sha256(docs / name)}`"
        for name in REQUIRED_EVIDENCE_FILES
    ) + "\n\nAll quantitative files are canonical unless their own status explicitly says `BLOCKED_NOT_RUN`. Source paths and hashes are in `12_reproducibility_manifest.json`.\n"
    (docs / "FINAL_EVIDENCE_INDEX.md").write_text(index, encoding="utf-8")

    readiness = f"""# Final submission readiness

| Evidence item | Status | Main result | Artifact | Canonical? |
|---|---|---|---|---|
| Dataset | PASS | hash `{DATASET_HASH[:12]}...`; 5 Hz, 50x8 | `01_dataset_manifest.json` | Yes |
| Classification | PASS | C3/C4/C5 accuracy 99.851/99.779/99.412% | `03_classification_final.csv` | Yes |
| Regression | PASS | RMSE 9.333/13.808/18.477 ppm | `04_regression_final.csv` | Yes |
| QC HC90 | PASS | coverage {float(qc_all90['coverage'])*100:.2f}%, NRMSE {float(qc_all90['NRMSE_range']):.5f} | `05_qc_final.csv` | Yes |
| QC HC95 | PASS | coverage {float(qc_all95['coverage'])*100:.2f}%, NRMSE {float(qc_all95['NRMSE_range']):.5f} | `05_qc_final.csv` | Yes |
| Random reference | PASS | aggregate RMSE gain {float(rnd90['QC_RMSE_improvement_over_random']):.3f}/{float(rnd95['QC_RMSE_improvement_over_random']):.3f} ppm | source index | Yes |
| Quality robustness | PASS/WARNING | methane225 repeat1 RMSE {float(repeat1['S_ALL_RMSE']):.3f} ppm, retained | `06_quality_robustness.csv` | Yes |
| Equal-label fairness | BLOCKED | A0T not run | `09_a0t_equal_label.csv` | No result |
| FedRidge ablation | PASS/MIXED | ALL RMSE {float(all83['RMSE']):.3f}->{float(all84['RMSE']):.3f} ppm ({float(all84['relative_RMSE_reduction_vs_83D'])*100:.2f}%); C4 RMSE worsens | `10_fedridge_83d_84d.csv` | Yes |
| Window overlap | BLOCKER | exact overlap 0; raw-time overlap present for C3/C4/C5 | `11_window_overlap.csv` | Yes |
| Deployment package | PASS | SHA256 `{sha256(sources['package_archive'])[:12]}...` | package manifest | Yes |
| Pi 5 | PASS | P50/P95/P99 {float(total_pi['P50_ms']):.3f}/{float(total_pi['P95_ms']):.3f}/{float(total_pi['P99_ms']):.3f} ms; {float(total_pi['throughput_windows_per_second']):.2f} windows/s; {int(total_pi['peak_rss_bytes'])/2**20:.2f} MiB | `07_pi5_benchmark.csv` | Yes |
| Model size | PASS | 22,765 params; 91,060 FP32 bytes | `08_model_size.json` | Yes |
| Figure/table tracker | NEEDS_REGEN | canonical data mapped, final plots absent | tracker | Mixed |
| Manuscript consistency | BLOCKED | v7 source unavailable | consistency audit | N/A |

Decision: stop algorithm/preprocessing/R84/QC exploration. The project may enter final writing and canonical figure regeneration, but is **not yet submission-ready** until the A0T fairness gap, raw-time-overlap robustness, final figures, and v7 manuscript consistency are resolved. Calibration-budget evidence is an additional claim-dependent gap.
"""
    (study / "FINAL_SUBMISSION_READINESS.md").write_text(readiness, encoding="utf-8")

    # Copy concise closure documents into the Git evidence bundle.
    for name in (
        "FINAL_CLAIM_EVIDENCE_MATRIX.md", "FINAL_SUBMISSION_AUDIT.md",
        "FINAL_EXPERIMENT_STATUS.md", "FINAL_SUBMISSION_READINESS.md",
    ):
        shutil.copy2(study / name, docs / name)
    shutil.copy2(study / "FINAL_RESULT_MASTER_TABLE.csv", docs / "FINAL_RESULT_MASTER_TABLE.csv")
    shutil.copy2(study / "FINAL_FIGURE_MANIFEST.csv", docs / "FINAL_FIGURE_MANIFEST.csv")

    required = [docs / name for name in REQUIRED_EVIDENCE_FILES]
    if any(not path.is_file() for path in required):
        raise RuntimeError("required evidence bundle is incomplete")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=ROOT / "results/iotj_canonical_v1_final_20260808")
    parser.add_argument("--docs", type=Path, default=ROOT / "docs/experiments/iotj_canonical_v1_final")
    args = parser.parse_args()
    build(args.study, args.docs)


if __name__ == "__main__":
    main()
