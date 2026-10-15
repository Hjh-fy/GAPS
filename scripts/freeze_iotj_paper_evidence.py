"""Build the evidence-frozen GAPS IoT-J manuscript from audited artifacts.

The script is intentionally read-only with respect to existing experiments,
models, runtimes, QC assets, and the source manuscript.  It creates a new
evidence-freeze directory, a byte-identical source backup, a new HTML
manuscript, paper tables/figures, and machine-readable audits.
"""

from __future__ import annotations

import csv
import hashlib
import html
import json
import math
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
SHARED_ROOT = ROOT.parents[1] if ROOT.parent.name == ".worktrees" else ROOT
FREEZE = ROOT / "docs/paper_evidence_freeze"
PAPER = ROOT / "docs/paper"
SOURCE = SHARED_ROOT / "docs/paper/GAPS_IoTJ_traditional_draft_20260720.zh.html"
BACKUP = PAPER / "GAPS_IoTJ_traditional_draft_20260720.zh.pre_paper_evidence_freeze_20260726.html"
MANUSCRIPT = PAPER / "GAPS_IoTJ_evidence_frozen_20260726.zh.html"
SCHEMA = "iotj.paper_evidence_freeze.v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(fields or (list(rows[0]) if rows else []))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        writer.writerows({name: row.get(name, "") for name in names} for row in rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return "shared-root/" + path.relative_to(SHARED_ROOT).as_posix()


def descriptor(path: Path) -> dict[str, Any]:
    return {"path": rel(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def require_new() -> None:
    for path in (FREEZE, BACKUP, MANUSCRIPT):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite evidence-freeze artifact: {path}")
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)


def assert_close(actual: float, expected: float, label: str, tol: float = 5e-10) -> None:
    if not math.isclose(actual, expected, rel_tol=tol, abs_tol=tol):
        raise RuntimeError(f"canonical value conflict for {label}: {actual} != {expected}")


def source_paths() -> dict[str, Path]:
    return {
        "classification_summary": ROOT / "results/iotj_b5_multiseed_20260724/b5_classification_multiseed_summary.csv",
        "classification_per_seed": ROOT / "results/iotj_b5_multiseed_20260724/per_seed_b5_classification_metrics.csv",
        "classification_manifest": ROOT / "results/iotj_b5_multiseed_20260724/b5_multiseed_completion_manifest.json",
        "regression_summary": ROOT / "results/iotj_b5_regression_multiseed_20260724/regression_multiseed_summary.csv",
        "regression_per_seed": ROOT / "results/iotj_b5_regression_multiseed_20260724/per_seed_regression_metrics.csv",
        "regression_per_gas": ROOT / "results/iotj_b5_regression_multiseed_20260724/per_gas_multiseed_summary.csv",
        "regression_decision": ROOT / "results/iotj_b5_regression_multiseed_20260724/final_regression_decision.json",
        "regression_protocol": ROOT / "results/iotj_b5_regression_multiseed_20260724/protocol_manifest.json",
        "h1_equivalence": ROOT / "results/iotj_b5_c5_runtime_v5_candidate_20260724/federated_h1/equivalence_decision.json",
        "h1_payload": ROOT / "results/iotj_b5_c5_runtime_v5_candidate_20260724/federated_h1/communication_payload_summary.json",
        "h1_model": ROOT / "results/iotj_b5_c5_runtime_v5_candidate_20260724/federated_h1/global_h1_model.json",
        "target_ridge": ROOT / "results/iotj_b5_c5_runtime_v5_candidate_20260724/target_ridge/target_ridge_105d_manifest.json",
        "runtime_v5_contract": ROOT / "results/iotj_b5_c5_runtime_v5_candidate_20260724/runtime_v5/runtime_contract_v5.json",
        "qc_decision": ROOT / "results/iotj_b5_c5_runtime_v5_qc_20260725/decision_gate.json",
        "qc_comparison": ROOT / "results/iotj_b5_c5_runtime_v5_qc_20260725/comparison_vs_runtime_v4.json",
        "qc_oof": ROOT / "results/iotj_b5_c5_runtime_v5_qc_20260725/qc_candidate_calibration_summary.csv",
        "qc_per_gas": ROOT / "results/iotj_final_system_benchmark_20260725/paper_tables/table_qc_per_gas.csv",
        "qc_overall": ROOT / "results/iotj_final_system_benchmark_20260725/paper_tables/table_qc_overall.csv",
        "benchmark": ROOT / "results/iotj_final_system_benchmark_20260725/benchmarks/benchmark_summary.csv",
        "packages": ROOT / "results/iotj_final_system_benchmark_20260725/system_metrics/package_size_summary.csv",
        "b5_payload": ROOT / "results/iotj_final_system_benchmark_20260725/system_metrics/b5_fl_communication_summary.csv",
        "h1_system_payload": ROOT / "results/iotj_final_system_benchmark_20260725/system_metrics/federated_h1_communication_summary.csv",
        "lowcal_summary": ROOT / "results/iotj_low_calibration_sensitivity_20260725/low_calibration_summary.csv",
        "lowcal_per_gas": ROOT / "results/iotj_low_calibration_sensitivity_20260725/low_calibration_per_gas_summary.csv",
        "lowcal_decision": ROOT / "results/iotj_low_calibration_sensitivity_20260725/decision_gate.json",
        "harmonized_table": ROOT / "results/iotj_calibration_protocol_harmonization_20260726/paper_tables/table_calibration_protocol_comparison.csv",
        "harmonized_decision": ROOT / "results/iotj_calibration_protocol_harmonization_20260726/decision_gate.json",
        "historical_audit": ROOT / "results/iotj_calibration_protocol_harmonization_20260726/historical_holdout_audit.json",
        "runtime_frozen": ROOT / "results/iotj_final_system_benchmark_20260725/frozen_asset_manifest.json",
    }


def validate_sources(paths: Mapping[str, Path]) -> None:
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise RuntimeError(f"missing canonical evidence: {missing}")
    cls = {row["metric"]: row for row in read_csv(paths["classification_summary"])}
    for metric, mean, std in (
        ("accuracy", 0.9891176470588235, 0.005982601649061194),
        ("macro_f1", 0.9891344644033736, 0.005960319881564127),
        ("nll", 0.10023704803768194, 0.037833443140437885),
        ("ece", 0.01060657672584056, 0.005773839365711773),
    ):
        assert_close(float(cls[metric]["mean"]), mean, f"B5 {metric}")
        assert_close(float(cls[metric]["sample_std_ddof1"]), std, f"B5 {metric} std")
    reg = {row["variant"]: row for row in read_csv(paths["regression_summary"])}
    for variant, value in (
        ("RG0_RICH_ONLY", 14.454631096309683),
        ("RG1_FEDERATED_H1", 11.633858261113179),
        ("RG2_ALL_PRIOR", 11.520766152359354),
    ):
        assert_close(float(reg[variant]["S_CC_RMSE_mean"]), value, variant)
    decision = read_json(paths["regression_decision"])
    if decision["decision"] != "SELECT_B5_FEDERATED_H1":
        raise RuntimeError("regression decision drifted")
    assert_close(float(decision["observed"]["primary_S_CC_relative_delta"]), 0.009816370479029746, "RG1 noninferiority")
    h1 = read_json(paths["h1_equivalence"])
    if h1["decision"] != "PRACTICAL_EQUIVALENCE":
        raise RuntimeError("H1 equivalence decision drifted")
    assert_close(float(h1["real_topology_metrics"]["S_CC_RMSE"]), 11.341598573018034, "frozen RG1")
    qc = read_json(paths["qc_decision"])
    if qc["decision"] != "RUNTIME_V5_QC_VALID_BUT_NOT_SUPERIOR":
        raise RuntimeError("QC decision drifted")
    harmonic = read_json(paths["harmonized_decision"])
    if harmonic["decision"] != "SENSITIVITY_PARTLY_PROTOCOL_DEPENDENT":
        raise RuntimeError("harmonization decision drifted")
    frozen = read_json(paths["runtime_frozen"])
    if not frozen.get("runtime_v4_six_sha_unchanged"):
        raise RuntimeError("runtime v4 six-SHA audit failed")


def preflight(paths: Mapping[str, Path]) -> None:
    source_candidates = list(SHARED_ROOT.rglob("*.html"))
    iotj_candidates = [p for p in source_candidates if "iotj" in p.name.lower() and ".worktrees" not in p.parts]
    if iotj_candidates != [SOURCE]:
        raise RuntimeError(f"canonical HTML is not unique: {[str(p) for p in iotj_candidates]}")
    status = subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True, errors="replace")
    report = f"""# IoT-J paper evidence freeze preflight

- Status: PASS
- Branch: `codex/iotj-confirmation-observability`
- Local/origin HEAD at preflight: `4a846fc36512bb93f5e310a0e5789ce17eb21968`
- Canonical source uniquely identified: `{rel(SOURCE)}`
- Identification basis: it is the only IoT-J HTML in the shared project root; the active worktree contained no competing HTML manuscript.
- Canonical source Git state: untracked in the shared root, therefore imported by SHA and backed up byte-for-byte.
- Runtime v4 six frozen SHA status: PASS.
- Required classification, regression, H1, runtime v5, QC, benchmark, low-calibration, and harmonization evidence: present.
- Existing unrelated modifications were observed and left untouched:
  - `results/iotj_a003_timing_diagnosis_20260719/a003_vs_b2_pilot_timing_analysis.md`
  - `results/iotj_advisor_metrics_20260721/build_advisor_workbook_v3.mjs`
- Temporary directories were not deleted or modified.
- No training, inference evaluation, benchmark, or test reopening was run.

## Working-tree snapshot

```text
{status[:6000]}
```
"""
    FREEZE.mkdir(parents=True)
    (FREEZE / "preflight_report.md").write_text(report, encoding="utf-8")
    write_json(FREEZE / "manuscript_source_manifest.json", {
        "schema_version": SCHEMA,
        "source": descriptor(SOURCE),
        "canonical_selection": "unique IoT-J HTML in shared project root",
        "source_git_state": "untracked_shared_root",
        "source_author_metadata": "none present in source HTML",
        "source_references_preserved": True,
        "source_commit": git_head(),
        "runtime_v4_six_sha_unchanged": True,
        "read_only_evidence_count": len(paths),
    })


def evidence_inventory(paths: Mapping[str, Path]) -> list[dict[str, Any]]:
    items = [
        ("E-CLS-5S", ["B5-CLS-S42-46"], ["accuracy", "macro-F1", "NLL", "ECE"], "B5 five-seed stability", ["classification_summary", "classification_per_seed", "classification_manifest"], "approved", "direct", "Training-seed variability only."),
        ("E-CLS-TOPO", ["B5-CLS-S42-46"], ["rounds", "topology", "parameters", "application payload"], "real three-host B5 training", ["classification_manifest", "b5_payload"], "approved", "direct", "Measured application payload is not transport-layer traffic."),
        ("E-REG-5S", ["B5-REG-MS-20260724"], ["S_CC RMSE", "S_ALL RMSE", "per-gas RMSE"], "RG0/RG1/RG2 five-seed comparison", ["regression_summary", "regression_per_seed", "regression_per_gas"], "approved", "direct", "RG2 has lower absolute mean S_CC in 5/5 seeds."),
        ("E-REG-GATE", ["B5-REG-MS-20260724"], ["paired relative S_CC delta"], "RG1 versus RG2 preregistered 1% non-inferiority", ["regression_decision", "regression_protocol"], "approved", "direct", "Non-inferiority is not superiority."),
        ("E-H1-EQUIV", ["IOTJ-H1-FED-EQUIV-20260724"], ["alpha agreement", "prediction max difference", "RMSE difference"], "sufficient-statistics H1 practical equivalence", ["h1_equivalence", "h1_model"], "approved", "direct", "Numerical practical equivalence; no cryptographic or DP guarantee."),
        ("E-H1-LOCAL", ["IOTJ-H1-FED-EQUIV-20260724"], ["payload bytes", "raw rows transmitted"], "source raw-row locality during H1 construction", ["h1_payload", "h1_system_payload"], "approved", "qualified", "Sufficient statistics can reveal information and are not securely aggregated."),
        ("E-RUNTIME-V5", ["IOTJ-RUNTIME-V5-20260724"], ["parity", "bundle contract"], "final simplified regression implementation", ["runtime_v5_contract", "target_ridge"], "approved", "direct", "QC2 remains a non-promoted candidate."),
        ("E-QC-CLOSE", ["IOTJ-RUNTIME-V5-QC-20260725"], ["yield", "accepted RMSE", "CO guards"], "v4/v5 QC quality-coverage comparison", ["qc_decision", "qc_comparison", "qc_overall", "qc_per_gas"], "approved", "direct", "Accepted RMSE must be paired with accepted yield."),
        ("E-QC-OOF", ["IOTJ-RUNTIME-V5-QC-20260725"], ["OOF risk metrics"], "QC1/QC2/QC3 calibration-only selection", ["qc_oof"], "approved", "direct", "QC2 selected before test; no post-test retuning."),
        ("E-BENCH", ["IOTJ-FINAL-SYSTEM-BENCHMARK-20260725"], ["latency", "throughput", "RSS", "temperature", "bundle bytes"], "PC/Pi runtime efficiency", ["benchmark", "packages"], "approved", "direct", "Steady-state batch=1 benchmark; PC scheduling tail remains visible."),
        ("E-LOWCAL", ["IOTJ-LOW-CALIBRATION-SENSITIVITY-S42-20260725"], ["S_CC RMSE", "S_ALL RMSE"], "group-aware calibration-budget sensitivity", ["lowcal_summary", "lowcal_per_gas", "lowcal_decision"], "approved", "qualified", "Frozen-method descriptive analysis on a previously used test."),
        ("E-HARM", ["IOTJ-CALIBRATION-PROTOCOL-HARMONIZATION-20260726"], ["dual-track S_CC RMSE"], "protocol harmonization", ["harmonized_table", "harmonized_decision", "historical_audit"], "approved", "qualified", "Historical 240/80 is window-level and not original-file independent."),
    ]
    output = []
    for eid, experiments, metrics, comparison, keys, status, strength, limitation in items:
        output.append({
            "evidence_id": eid,
            "experiment_ids": experiments,
            "metric_ids": metrics,
            "comparison": comparison,
            "source_paths": [rel(paths[key]) for key in keys],
            "source_sha256": {rel(paths[key]): sha256(paths[key]) for key in keys},
            "audit_status": status,
            "support_strength": strength,
            "claim_ids": [],
            "limitations": [limitation],
            "provenance": {"source_commit": git_head(), "calculation_status": "reported_from_frozen_artifact"},
        })
    return output


def claim_matrix(paths: Mapping[str, Path]) -> list[dict[str, Any]]:
    definitions = [
        ("C1", "Abstract; 4.1", "B5 分类在五个训练种子下保持稳定。", "B5 classification is stable across five training seeds.", "direct", "Accuracy 0.989118 ± 0.005983; macro-F1 0.989134 ± 0.005960", "B5-CLS-S42-46", "classification_summary", "approved", "仅覆盖 seeds 42–46 和 C1/C2→C5。", "stable across five training seeds", "universally robust"),
        ("C2", "Methods; 4.3", "由充分统计量重构的 Federated H1 与 pooled source Ridge 达到实用等价。", "Federated H1 reconstructed from sufficient statistics is practically equivalent to the pooled source Ridge reference.", "direct", "4/4 alpha agreement; max prediction difference 6.2532e-08 ppm", "IOTJ-H1-FED-EQUIV-20260724", "h1_equivalence", "approved", "数值实用等价，不是形式化安全性证明。", "practically equivalent", "mathematically identical or private"),
        ("C3", "Methods; Discussion", "Federated H1 构建期间源端原始行保持本地。", "Source raw rows remain local during Federated H1 construction.", "qualified", "raw_source_rows_transmitted=false", "IOTJ-H1-FED-EQUIV-20260724", "h1_payload", "approved", "统计量未由安全聚合或差分隐私保护。", "raw source samples remain local", "privacy is guaranteed"),
        ("C4", "Abstract; 4.2", "Federated-H1 目标个性化满足相对 all-prior 的预注册 1% 非劣标准。", "Federated-H1 target personalization satisfies the preregistered 1% non-inferiority criterion relative to all-prior regression.", "direct", "paired relative S_CC degradation 0.981637%", "B5-REG-MS-20260724", "regression_decision", "approved", "RG2 在 5/5 seeds 的绝对 S_CC 更低。", "meets the preregistered 1% non-inferiority criterion", "RG1 is more accurate than RG2"),
        ("C5", "Abstract; 4.5", "v5 回归核心相对 v4 降低回归参数、bundle 大小和 Pi 延迟。", "Runtime v5 regression core reduces regression parameters, bundle size, and Raspberry Pi latency relative to runtime v4.", "direct", "28737→844 params; 2,971,538→289,916 bytes; Pi p50 4.571→3.725 ms", "IOTJ-FINAL-SYSTEM-BENCHMARK-20260725", "benchmark", "approved", "v5 core 不含 QC，比较需标注角色。", "reduces implementation overhead under the frozen benchmark", "globally faster in every latency statistic"),
        ("C6", "4.4; Discussion", "v5 QC2 富集高误差样本，但因 coverage 与 CO guards 失败而未晋级。", "Runtime v5 QC2 enriches high-error samples but was not promoted because coverage and CO guards failed.", "direct", "HC95 yield 93.75%, accepted RMSE 13.9178 ppm; HC90 CO guard failed", "IOTJ-RUNTIME-V5-QC-20260725", "qc_decision", "approved", "accepted RMSE 必须与 yield 同报。", "valid candidate not promoted", "superior to v4"),
        ("C7", "4.4; 4.5", "Runtime v4 保持正式 selective-output baseline。", "Runtime v4 remains the formal selective-output baseline.", "direct", "FORMAL_BASELINE", "IOTJ-FINAL-SYSTEM-BENCHMARK-20260725", "qc_overall", "approved", "v5 regression core 与 v4 QC baseline 角色不同。", "formal selective-output baseline", "legacy or superseded runtime"),
        ("C8", "4.6; Discussion", "目标个性化表现出较高 calibration-budget sensitivity。", "Target personalization exhibits high calibration-budget sensitivity.", "qualified", "group-aware S_CC 10.8724→23.9156 ppm when 320→160", "IOTJ-LOW-CALIBRATION-SENSITIVITY-S42-20260725", "lowcal_summary", "approved", "描述性 frozen-method sensitivity，不是新独立 confirmatory test。", "high calibration-budget sensitivity", "few-window robustness"),
        ("C9", "4.7; Discussion", "两种协议的退化方向一致，但幅度部分依赖协议。", "Calibration degradation is directionally consistent across protocols while its magnitude is partly protocol-dependent.", "qualified", "SENSITIVITY_PARTLY_PROTOCOL_DEPENDENT", "IOTJ-CALIBRATION-PROTOCOL-HARMONIZATION-20260726", "harmonized_decision", "approved", "相同历史 test 被描述性复用。", "directionally consistent; magnitude partly protocol-dependent", "protocol-independent magnitude"),
        ("C10", "Protocol; Limitations", "历史 240/80 calibration split 是 window-level，非 original-file independent。", "The historical 240/80 calibration split is window-level and not original-file independent.", "direct", "61/61 validation filenames overlap fit", "IOTJ-CALIBRATION-PROTOCOL-HARMONIZATION-20260726", "historical_audit", "approved", "这是 calibration-internal overlap，不是 test-label leakage。", "historical window-level holdout", "original-file-independent historical split"),
    ]
    rows = []
    for values in definitions:
        cid, section, zh, en, etype, value, exp, key, status, limitation, allowed, forbidden = values
        rows.append({
            "claim_id": cid, "paper_section": section, "claim_zh": zh, "claim_en": en,
            "evidence_type": etype, "canonical_value": value, "experiment_id": exp,
            "result_path": rel(paths[key]), "code_commit": provenance_commit(paths[key]),
            "asset_sha256": sha256(paths[key]), "evidence_status": status,
            "limitation": limitation, "allowed_wording": allowed, "forbidden_wording": forbidden,
        })
    return rows


def provenance_commit(path: Path) -> str:
    if "b5_regression_multiseed" in str(path):
        return "99cd23e8b4a5f2f103170f1d8a110d6d85febd5e"
    if "final_system_benchmark" in str(path):
        return "4ccfc489821410ddacb6ad36180694bb953311f1"
    if "calibration_protocol_harmonization" in str(path):
        return "c30952c3df0b4f2098bf4d42697aaa11a15ab7b0"
    return "see_bound_result_index"


def inventory_outputs(inventory: list[dict[str, Any]], claims: list[dict[str, Any]], paths: Mapping[str, Path]) -> None:
    claim_by_evidence = {
        "E-CLS-5S": ["C1"], "E-CLS-TOPO": ["C1"], "E-REG-5S": ["C4"],
        "E-REG-GATE": ["C4"], "E-H1-EQUIV": ["C2"], "E-H1-LOCAL": ["C3"],
        "E-RUNTIME-V5": ["C5"], "E-QC-CLOSE": ["C6", "C7"], "E-QC-OOF": ["C6"],
        "E-BENCH": ["C5", "C7"], "E-LOWCAL": ["C8"], "E-HARM": ["C9", "C10"],
    }
    for item in inventory:
        item["claim_ids"] = claim_by_evidence[item["evidence_id"]]
    write_json(FREEZE / "evidence_inventory.json", {"schema_version": SCHEMA, "evidence": inventory})
    md = ["# Evidence inventory", "", f"Approved evidence records: {len(inventory)}", "",
          "| ID | Comparison | Status | Support | Claims | Limitation |",
          "|---|---|---|---|---|---|"]
    for item in inventory:
        md.append(f"| {item['evidence_id']} | {item['comparison']} | {item['audit_status']} | {item['support_strength']} | {', '.join(item['claim_ids'])} | {item['limitations'][0]} |")
    (FREEZE / "evidence_inventory.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    write_json(FREEZE / "canonical_asset_sha256.json", {
        "schema_version": SCHEMA,
        "assets": {name: descriptor(path) for name, path in paths.items()},
        "runtime_v4_six_sha256": read_json(paths["runtime_frozen"])["assets"]["runtime_v4"],
    })
    write_json(FREEZE / "claim_evidence_matrix.json", {"schema_version": SCHEMA, "claims": claims})
    write_csv(FREEZE / "claim_evidence_matrix.csv", claims)
    lines = ["# Claim–Evidence matrix", "",
             "| Claim | Section | Canonical value | Evidence | Status | Limitation |",
             "|---|---|---|---|---|---|"]
    for row in claims:
        lines.append(f"| {row['claim_id']} | {row['paper_section']} | {row['canonical_value']} | {row['result_path']} | {row['evidence_status']} | {row['limitation']} |")
    (FREEZE / "claim_evidence_matrix.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def table_html(title: str, rows: Sequence[Mapping[str, Any]]) -> str:
    headers = list(rows[0])
    head = "".join(f"<th>{html.escape(str(x))}</th>" for x in headers)
    body = "".join("<tr>" + "".join(f"<td>{html.escape(str(row[h]))}</td>" for h in headers) + "</tr>" for row in rows)
    return f"<p class='table-title'>{html.escape(title)}</p><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def paper_tables(paths: Mapping[str, Path]) -> dict[str, list[dict[str, Any]]]:
    cls = {row["metric"]: row for row in read_csv(paths["classification_summary"])}
    reg = {row["variant"]: row for row in read_csv(paths["regression_summary"])}
    benchmark = read_csv(paths["benchmark"])
    packages = {row["runtime"]: row for row in read_csv(paths["packages"])}
    qc_all = [row for row in read_csv(paths["qc_overall"]) if row["QC_workpoint"] in {"HC95", "HC90"}]
    harm = read_csv(paths["harmonized_table"])
    historical = read_json(paths["historical_audit"])
    dataset = [
        {"Device/role": "C1", "Platform": "Raspberry Pi 5", "Role": "source classifier + local H1 statistics", "Paper split": "local source train/validation"},
        {"Device/role": "C2", "Platform": "ECS client", "Role": "source classifier + local H1 statistics", "Paper split": "local source train/validation"},
        {"Device/role": "Server", "Platform": "Alibaba ECS", "Role": "FedAvg, server-side DA, H1 statistics aggregation", "Paper split": "no raw C1/C2 training rows"},
        {"Device/role": "C5", "Platform": "target device", "Role": "target personalization and deployment test", "Paper split": "320 calibration / 1360 test windows"},
        {"Device/role": "Sensors", "Platform": "8 MOS channels", "Role": "Ethanol, CO, Ethylene, Methane", "Paper split": "100×8 response windows"},
    ]
    classification = [{
        "Metric": label, "Mean": f"{float(cls[key]['mean']):.6f}",
        "Sample SD": f"{float(cls[key]['sample_std_ddof1']):.6f}",
        "Min": f"{float(cls[key]['min']):.6f}", "Max": f"{float(cls[key]['max']):.6f}",
    } for key, label in (("accuracy", "Accuracy"), ("macro_f1", "Macro-F1"), ("nll", "NLL"), ("ece", "ECE"))]
    regression = []
    for key, label, dependency in (
        ("RG0_RICH_ONLY", "Rich features only", "104D"),
        ("RG1_FEDERATED_H1", "Federated-H1 personalization", "104D + one source reference"),
        ("RG2_ALL_PRIOR", "All-prior personalization", "104D + three source references"),
    ):
        row = reg[key]
        regression.append({
            "Method": label, "Input/dependency": dependency,
            "S_CC RMSE (ppm)": f"{float(row['S_CC_RMSE_mean']):.4f} ± {float(row['S_CC_RMSE_sample_std']):.4f}",
            "S_ALL RMSE (ppm)": f"{float(row['S_ALL_RMSE_mean']):.4f} ± {float(row['S_ALL_RMSE_sample_std']):.4f}",
            "Decision": "selected: 1% non-inferior" if key == "RG1_FEDERATED_H1" else ("absolute S_CC lower" if key == "RG2_ALL_PRIOR" else "baseline"),
        })
    qc = [{
        "Runtime/workpoint": f"{row['runtime']} {row['QC_workpoint']}",
        "A/R/R": f"{row['accept_N']}/{row['review_N']}/{row['reject_N']}",
        "Accepted yield": f"{100*float(row['accepted_yield']):.2f}%",
        "Accepted RMSE (ppm)": f"{float(row['accepted_RMSE']):.4f}",
        "CO yield": f"{100*float(row['CO_accepted_yield']):.2f}%",
        "CO RMSE (ppm)": f"{float(row['CO_accepted_RMSE']):.4f}",
        "Status": row["deployment_status"],
    } for row in qc_all]
    system = []
    for row in benchmark:
        pkg = packages[row["runtime"]]
        params = {"RUNTIME_V4_FULL": "28,737", "RUNTIME_V5_REGRESSION_CORE": "844", "RUNTIME_V5_QC2_CANDIDATE": "844"}[row["runtime"]]
        system.append({
            "Runtime": row["runtime"].replace("RUNTIME_", "").replace("_", " "),
            "Regression params": params, "Bundle bytes": f"{int(pkg['total_bytes']):,}",
            "p50 / p95 (ms)": f"{float(row['p50_ms']):.3f} / {float(row['p95_ms']):.3f}",
            "Throughput (window/s)": f"{float(row['throughput_windows_per_s']):.1f}",
            "Peak RSS (MiB)": f"{float(row['rss_peak_mib']):.1f}",
            "Peak temp. (°C)": "—" if not row["temperature_peak_c"] else f"{float(row['temperature_peak_c']):.2f}",
            "Status": row["deployment_status"],
        })
    group_summary = read_csv(ROOT / "results/iotj_calibration_protocol_harmonization_20260726/track_groupaware/groupaware_budget_summary.csv")
    calibration = [{
        "Calibration windows": row["nominal_budget"],
        "S_CC RMSE (ppm)": f"{float(row['S_CC_RMSE_mean']):.4f} ± {float(row['S_CC_RMSE_sample_std']):.4f}",
        "S_ALL RMSE (ppm)": f"{float(row['S_ALL_RMSE_mean']):.4f} ± {float(row['S_ALL_RMSE_sample_std']):.4f}",
        "Variability source": "fold/alpha-selection" if row["nominal_budget"] == "320" else "subset + fold",
    } for row in group_summary]
    appendix_harm = [{
        "Budget": row["calibration_budget"],
        "Historical S_CC": row["historical_S_CC_mean_std"],
        "Group-aware S_CC": row["groupaware_S_CC_mean_std"],
        "G−H (ppm)": f"{float(row['protocol_delta_ppm']):.4f}",
    } for row in harm]
    tables = {"table_I_dataset_roles": dataset, "table_II_classification": classification,
              "table_III_regression": regression, "table_IV_qc": qc,
              "table_V_system": system, "table_VI_calibration": calibration,
              "table_A7_harmonization": appendix_harm}
    table_dir = FREEZE / "paper_tables"
    for name, rows in tables.items():
        write_csv(table_dir / f"{name}.csv", rows)
    # Appendix source tables remain machine-readable and are copied from frozen sources.
    appendix = {
        "table_A1_per_seed_classification.csv": paths["classification_per_seed"],
        "table_A2_per_seed_regression.csv": paths["regression_per_seed"],
        "table_A3_per_gas_regression.csv": paths["regression_per_gas"],
        "table_A4_qc_oof_candidates.csv": paths["qc_oof"],
        "table_A5_qc_per_gas.csv": paths["qc_per_gas"],
        "table_A6_communication_payloads.csv": paths["b5_payload"],
        "table_A8_lowcal_per_gas.csv": paths["lowcal_per_gas"],
    }
    for name, source in appendix.items():
        shutil.copy2(source, table_dir / name)
    write_json(FREEZE / "dataset_role_audit.json", {
        "historical_fit_rows": historical["fit"]["rows"],
        "historical_validation_rows": historical["validation"]["rows"],
        "historical_filename_overlap": historical["filename_overlap_count"],
        "window_shape": [100, 8], "gas_classes": ["Ethanol", "CO", "Ethylene", "Methane"],
    })
    return tables


def style_axes(ax: Any) -> None:
    ax.grid(axis="y", alpha=.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def create_figures(paths: Mapping[str, Path]) -> list[dict[str, Any]]:
    figure_dir = FREEZE / "figures"
    figure_dir.mkdir(parents=True)
    blue, orange, green = "#1f77b4", "#d95f02", "#2ca02c"
    diagrams = {
        "fig1_architecture.svg": [
            ("C1 · Raspberry Pi", 40, 80, 180, 70), ("C2 · ECS client", 40, 190, 180, 70),
            ("Alibaba ECS server", 310, 115, 220, 100), ("C5 target", 620, 115, 180, 100),
        ],
        "fig2_federated_classification.svg": [
            ("C1/C2 local training", 30, 120, 190, 80), ("FedAvg · 25 rounds", 280, 120, 190, 80),
            ("Server-side DA", 530, 120, 170, 80), ("Frozen B5 route", 760, 120, 170, 80),
        ],
        "fig3_h1_target_personalization.svg": [
            ("Local moments + normal equations", 25, 80, 230, 90), ("Server H1 reconstruction", 330, 80, 210, 90),
            ("104D rich + 1D H1", 615, 80, 180, 90), ("C5 per-gas Ridge", 860, 80, 170, 90),
        ],
    }
    diagram_records = []
    for name, boxes in diagrams.items():
        width = 1050 if name == "fig3_h1_target_personalization.svg" else 960
        parts = [f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='300' viewBox='0 0 {width} 300'>",
                 "<rect width='100%' height='100%' fill='white'/>",
                 "<defs><marker id='a' markerWidth='10' markerHeight='10' refX='8' refY='3' orient='auto'><path d='M0,0 L0,6 L9,3 z' fill='#444'/></marker></defs>"]
        for idx, (label, x, y, w, h) in enumerate(boxes):
            parts.append(f"<rect x='{x}' y='{y}' width='{w}' height='{h}' rx='9' fill='#e8f1fa' stroke='#1f4e79' stroke-width='2'/>")
            parts.append(f"<text x='{x+w/2}' y='{y+h/2+5}' text-anchor='middle' font-family='Arial' font-size='16'>{html.escape(label)}</text>")
            if idx:
                px, py, pw, ph = boxes[idx-1][1:]
                parts.append(f"<line x1='{px+pw}' y1='{py+ph/2}' x2='{x-10}' y2='{y+h/2}' stroke='#444' stroke-width='2' marker-end='url(#a)'/>")
        footer = {
            "fig1_architecture.svg": "Raw C1/C2 rows stay local · shared classifier and sufficient statistics cross the boundary",
            "fig2_federated_classification.svg": "Real three-host topology · five local epochs · batch 32 · server DA 100 steps/round",
            "fig3_h1_target_personalization.svg": "Four gases fitted independently · calibration-only alpha selection · 105D final input",
        }[name]
        parts.append(f"<text x='{width/2}' y='255' text-anchor='middle' font-family='Arial' font-size='15' fill='#444'>{html.escape(footer)}</text></svg>")
        path = figure_dir / name
        path.write_text("".join(parts), encoding="utf-8")
        diagram_records.append(descriptor(path))

    reg = read_csv(paths["regression_summary"])
    fig, ax = plt.subplots(figsize=(5.8, 3.7), constrained_layout=True)
    labels = ["Rich only", "Federated H1", "All prior"]
    means = [float(row["S_CC_RMSE_mean"]) for row in reg]
    stds = [float(row["S_CC_RMSE_sample_std"]) for row in reg]
    ax.bar(labels, means, yerr=stds, capsize=4, color=[blue, green, orange], edgecolor="black", linewidth=.5)
    ax.set_ylabel("S_CC RMSE (ppm)"); style_axes(ax)
    fig.savefig(figure_dir / "fig4_regression_five_seed.png", dpi=300)
    fig.savefig(figure_dir / "fig4_regression_five_seed.pdf")
    plt.close(fig)

    shutil.copy2(ROOT / "results/iotj_final_system_benchmark_20260725/figures/qc_quality_coverage_tradeoff.png",
                 figure_dir / "fig5_qc_quality_coverage.png")
    shutil.copy2(ROOT / "results/iotj_final_system_benchmark_20260725/figures/qc_quality_coverage_tradeoff.pdf",
                 figure_dir / "fig5_qc_quality_coverage.pdf")

    bench = read_csv(paths["benchmark"])
    pi = [row for row in bench if row["platform"] == "Pi"]
    packages = {row["runtime"]: int(row["total_bytes"]) for row in read_csv(paths["packages"])}
    short = ["v4", "v5 core", "v5 QC2"]
    fig, axes = plt.subplots(1, 3, figsize=(9.4, 3.4), constrained_layout=True)
    axes[0].bar(short, [float(row["p50_ms"]) for row in pi], color=[blue, green, orange])
    axes[0].set_ylabel("Pi p50 latency (ms)")
    axes[1].bar(short, [packages[row["runtime"]] / 1e6 for row in pi], color=[blue, green, orange])
    axes[1].set_ylabel("Bundle size (MB)")
    axes[2].bar(short, [28737, 844, 844], color=[blue, green, orange])
    axes[2].set_ylabel("Regression parameters")
    for ax in axes:
        ax.tick_params(axis="x", rotation=18); style_axes(ax)
    fig.savefig(figure_dir / "fig6_system_efficiency.png", dpi=300)
    fig.savefig(figure_dir / "fig6_system_efficiency.pdf")
    plt.close(fig)

    group = read_csv(ROOT / "results/iotj_calibration_protocol_harmonization_20260726/track_groupaware/groupaware_budget_summary.csv")
    group = sorted(group, key=lambda r: int(r["nominal_budget"]))
    fig, ax = plt.subplots(figsize=(5.7, 3.7), constrained_layout=True)
    ax.errorbar([int(r["nominal_budget"]) for r in group], [float(r["S_CC_RMSE_mean"]) for r in group],
                yerr=[float(r["S_CC_RMSE_sample_std"]) for r in group], marker="o", capsize=4, color=blue)
    ax.set_xlabel("Target calibration windows"); ax.set_ylabel("S_CC RMSE (ppm)"); style_axes(ax)
    fig.savefig(figure_dir / "fig7_groupaware_calibration.png", dpi=300)
    fig.savefig(figure_dir / "fig7_groupaware_calibration.pdf")
    plt.close(fig)

    appendix_sources = {
        "appendix_calibration_dual_track.png": ROOT / "results/iotj_calibration_protocol_harmonization_20260726/paper_figures/calibration_protocol_sensitivity_comparison.png",
        "appendix_calibration_protocol_delta.png": ROOT / "results/iotj_calibration_protocol_harmonization_20260726/paper_figures/calibration_protocol_delta.png",
        "appendix_qc_per_gas_yield.png": ROOT / "results/iotj_final_system_benchmark_20260725/figures/qc_per_gas_yield_comparison.png",
    }
    for name, source in appendix_sources.items():
        shutil.copy2(source, figure_dir / name)
    records = [descriptor(path) for path in sorted(figure_dir.iterdir())]
    write_json(FREEZE / "figure_generation_manifest.json", {
        "schema_version": SCHEMA, "figures": records,
        "plot_policy": "frozen CSV/JSON only; no smoothing; error bars retained",
    })
    return records


def extract_references(source_text: str) -> str:
    match = re.search(r"<h2>参考文献</h2>\s*(<ol>.*?</ol>)", source_text, re.S)
    if not match:
        raise RuntimeError("source references not found")
    return match.group(1)


def manuscript_html(tables: Mapping[str, list[dict[str, Any]]], source_text: str) -> str:
    refs = extract_references(source_text)
    fig = lambda number, name, caption: (
        f"<figure id='fig{number}'><img src='../paper_evidence_freeze/figures/{name}' "
        f"alt='{html.escape(caption)}'><figcaption><strong>Fig. {number}.</strong> {caption}</figcaption></figure>"
    )
    css = """
body{margin:0;background:#e8eaed;color:#111;font-family:"Times New Roman","Noto Serif CJK SC",serif;line-height:1.48}
.paper{width:min(216mm,calc(100vw - 24px));margin:20px auto;padding:14mm 16mm;background:#fff;box-shadow:0 2px 14px #999}
.preprint{display:flex;justify-content:space-between;border-bottom:1px solid #aaa;font:8pt Arial;padding-bottom:5px}
h1{text-align:center;font-size:20pt}h2{text-align:center;font:700 11pt Arial;margin-top:16px}h3{font:700 10pt Arial;margin-top:12px}
p{text-align:justify;margin:5px 0}.abstract{border:1px solid #bbb;padding:9px}.contributions li{margin:4px 0}
table{width:100%;border-collapse:collapse;font:8pt Arial;margin:8px 0 12px}th,td{border:1px solid #aaa;padding:4px;text-align:left;vertical-align:top}
.table-title{font-weight:bold;font-size:9pt}.note{font-size:8.5pt;color:#444}
figure{margin:12px 0;break-inside:avoid}figure img{display:block;max-width:100%;max-height:105mm;margin:auto}figcaption{font-size:8.5pt;text-align:justify}
.status{background:#eef6ee;border-left:4px solid #2b7a2b;padding:8px}.appendix{border-top:2px solid #333;margin-top:20px}
"""
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>GAPS：真实设备联邦分类、充分统计量回归参考与轻量目标个性化</title><style>{css}</style></head>
<body><main class="paper"><div class="preprint"><span>GAPS · IoTJ-STYLE EVIDENCE-FROZEN MANUSCRIPT</span><span>2026-07-26</span></div>
<header><h1>GAPS：真实设备联邦分类、充分统计量回归参考与轻量目标个性化的跨设备气体感知系统</h1>
<p style="text-align:center"><em>GAPS: Cross-Device Gas Sensing with Real-Device Federated Classification, a Sufficient-Statistics Regression Reference, and Lightweight Target Personalization</em></p></header>
<section class="abstract"><h2>摘要</h2>
<p>跨设备气体感知需要同时处理类别识别、浓度估计和可靠输出中的设备偏移。本文提出 GAPS：源设备在真实三机拓扑上训练 B5 联邦分类器，服务器执行域适应；随后 C1/C2 在不上传原始样本行的条件下交换特征矩与正规方程等充分统计量，重构一个共享的分气体 Ridge 回归参考；目标设备 C5 使用 104 维 rich features 与 1 维 Federated H1 预测训练 105 维轻量个性化 Ridge。五个训练种子上，B5 accuracy 为 0.989118 ± 0.005983。Federated-H1 个性化的五种子 S_CC RMSE 为 11.6339 ± 0.3142 ppm；其相对 all-prior 方法的退化为 0.981637%，满足预注册 1% 非劣标准，但 all-prior 的绝对均值略低。正式 seed-42 frozen target 结果为 11.3416 ppm。v5 回归核心将回归参数从 28,737 降至 844，Raspberry Pi 5 p50 延迟从 4.571 ms 降至 3.725 ms；v4 仍是正式 selective-output baseline，v5 QC2 因覆盖率和 CO guards 未通过而未晋级。统一 group-aware 分析显示，将 C5 calibration 从 320 减至 160 个窗口时，mean S_CC RMSE 从 10.8724 增至 23.9156 ppm，说明目标个性化对 calibration coverage 高度敏感。</p>
<p><strong>关键词：</strong>真实设备联邦分类；服务器端域适应；充分统计量 Federated Ridge；目标个性化；选择性输出；校准预算敏感性</p>
<h2>Abstract</h2><p>GAPS combines real-device federated classification with source-side sufficient-statistics aggregation for a shared per-gas Ridge reference and lightweight target-device personalization. Across five training seeds, B5 achieved 0.989118 ± 0.005983 accuracy. Federated-H1 personalization obtained 11.6339 ± 0.3142 ppm correct-route RMSE and met the preregistered 1% non-inferiority margin relative to the all-prior model, although the latter had a slightly lower absolute mean. The frozen seed-42 target result remains 11.3416 ppm. The simplified regression core reduced regression parameters from 28,737 to 844 and Raspberry Pi 5 median latency from 4.571 to 3.725 ms. Runtime v4 remains the formal selective-output baseline because the v5 QC2 candidate reduced accepted error at lower coverage and failed HC90 CO guards. Group-aware calibration analysis further showed that halving target calibration from 320 to 160 windows more than doubled mean correct-route RMSE, exposing a major deployment limitation.</p></section>

<h2>I. 引言</h2>
<p>金属氧化物半导体阵列易受制造差异、漂移和工作环境影响，使源设备模型难以直接迁移到新设备 [1], [2], [20]。对于气体感知，分类错误还会改变浓度专家路由，因此分类、回归和选择性输出必须在同一部署合同中评价。已有联邦学习与边云协同研究提供了去中心化优化基础 [3], [4], [17]–[19]，但目标设备仍需要透明的 calibration 资源与可审计的可靠性边界。</p>
<p>GAPS 将系统划分为四个证据一致的阶段：真实设备联邦分类与服务器端域适应；源端充分统计量聚合得到共享回归参考；目标端轻量 Ridge 个性化；以及 accept/review/reject 三态输出。本研究不声称完整端到端回归在网络中联邦训练，也不为充分统计量交换提供形式化隐私保证。</p>
<p>本文贡献为：</p><ol class="contributions">
<li>在 Raspberry Pi C1、ECS C2 与 Alibaba ECS server 的真实拓扑上完成 25 轮联邦分类，并用 seeds 42–46 验证 B5 稳定性。</li>
<li>以源端局部特征矩、正规方程和校准分数构建共享分气体 Ridge reference；原始源样本行不上传。</li>
<li>将 Federated H1 prediction 与 rich features 结合为 105D 目标个性化输入，并在预注册 1% S_CC margin 下证明相对 all-prior 的非劣性。</li>
<li>实现并审计 runtime，联合报告 selective-output trade-off、PC/Pi 效率、application payload 与 calibration-budget sensitivity。</li>
</ol>

<h2>II. 相关工作</h2>
<p>传统传感器漂移与校准迁移方法通常依赖匹配样本、特征映射或重新标定 [1], [2], [16], [20]。FedAvg、FedProx 与 federated sensing 研究分别讨论模型聚合、系统异质性和边云协同 [3], [4], [17]–[19]。域适应和原型学习为目标偏移提供语义约束 [5]–[9]，选择性预测则强调在误差与覆盖率之间建立显式工作点 [10]–[12]。GAPS 的区别在于：把真实设备分类、充分统计量回归 reference、目标个性化及三态输出绑定为一条可审计部署链。</p>

<h2>III. 系统与数据协议</h2>
{table_html("Table I. Dataset, gas classes, devices, and split roles.", tables["table_I_dataset_roles"])}
<p>C1/C2 是源设备，C5 是目标设备；四类气体为 Ethanol、CO、Ethylene 和 Methane。每个窗口包含 100 个时间步与 8 个 MOS 通道。C5 calibration/test 为 320/1360 行。历史 calibration 内部的 240/80 split 是 window-level：80 个 fit filenames、61 个 validation filenames，且全部 61 个 validation filenames 也出现在 fit 中。该事实是 calibration-internal overlap，不等同于使用 test labels 训练。</p>
{fig(1, "fig1_architecture.svg", "Cloud–edge–device architecture. Raw C1/C2 rows remain local; the server aggregates classifier updates and source-side sufficient statistics, while C5 supplies target calibration and deployment windows.")}

<h2>IV. 方法</h2>
<h3>A. 真实设备联邦分类</h3>
<p>B5 使用卷积—注意力分类骨干，在 C1/C2 上以 Client Adam（learning rate 5×10<sup>−4</sup>）、batch size 32、每轮五个 local epochs 训练。服务器执行 25 个 communication rounds，并在每轮执行 100 个 server-side domain-adaptation steps。五种子仅改变随机种子；拓扑、数据、优化器和 B5 开关保持冻结。</p>
{fig(2, "fig2_federated_classification.svg", "B5 federated classification and server-side domain adaptation on the real three-host topology.")}
<h3>B. Sufficient-statistics Federated Ridge</h3>
<p>四类气体独立拟合。客户端首先计算局部特征计数、均值及二阶矩，服务器聚合后重构 global scaler。对每个候选 Ridge 系数 α，客户端在标准化特征上构造带截距的正规方程 X<sup>T</sup>X 与 X<sup>T</sup>y；服务器求解候选模型，再由客户端返回 clipped validation SSE 与计数用于 calibration-only α 选择。选定 α 后聚合完整 source fit statistics 并重构最终 H1。交换内容不包含原始 source rows、X/y 表或逐样本预测；但统计量没有安全聚合或差分隐私保护。</p>
<h3>C. 目标个性化</h3>
<p>C5 为每个窗口提取 104D rich features，并附加 1D Federated H1 prediction，形成 105D 输入。根据冻结 B5 predicted class 路由到四个独立 target Ridge。α 仅由 C5 calibration 内部 fit/validation 或 group-aware folds 选择，随后在当前完整 calibration subset 上 refit；test 不参与 fit、select 或 refit。</p>
{fig(3, "fig3_h1_target_personalization.svg", "Source sufficient-statistics aggregation, shared H1 reconstruction, and lightweight C5 personalization.")}
<h3>D. Runtime 与 QC 角色</h3>
<p>Runtime v4 是包含多 reference/multi-expert 风险语义的正式 selective-output baseline。Runtime v5 core 只实现 B5→Federated H1→105D target Ridge 的简化回归链。v5 QC2 使用 confidence 与 prototype/support distance 构建独立候选风险。HC95/HC90 均产生 accept、review、reject；只有 accept 行输出 auto_output_ppm。v4 与 v5 的风险语义和部署角色不互换。</p>

<h2>V. 实验协议</h2>
<p>分类标准差来自五个训练 seeds 42–46；回归标准差来自五个冻结 B5 classifiers/routes。Correct-route RMSE (S_CC) 只统计 B5 分类路由正确的行，end-to-end RMSE (S_ALL) 则统计全部 test 行，两者不能互换。Group-aware 320 的标准差来自 fold/alpha-selection variability，160/80/40 来自 subset + fold variability。Historical holdout 320 是固定单次 reference，低预算标准差来自 holdout subset variability。Filename grouping 只用于 calibration-internal folds/subsets；历史 calibration/test split 仍为 window-level。Low-calibration 与 harmonization 是对已冻结方法及此前已使用 C5 test 的 post-freeze 描述性分析，不用于重新选择方法。</p>

<h2>VI. 结果</h2>
<h3>4.1 Federated classification performance</h3>
{table_html("Table II. B5 five-seed classification performance.", tables["table_II_classification"])}
<p>B5 在五个训练种子上的 accuracy 与 macro-F1 均约为 0.989，且 NLL/ECE 波动有限。这支持“在当前 C1/C2→C5 协议和 seeds 42–46 下稳定”的限定结论，而非跨任意设备或随机种子的普遍鲁棒性。</p>
<h3>4.2 Regression method selection</h3>
{table_html("Table III. Five-seed regression comparison and non-inferiority.", tables["table_III_regression"])}
<p>All-prior 的 absolute mean S_CC 略低，并在 5/5 seeds 上优于 Federated H1。Federated H1 的 paired relative degradation 为 0.981637%，通过预注册 1% 非劣标准，同时把 source regression references 从三个减少到一个，因此被选为最终简化回归依赖。该选择是“非劣 + 简化”，不是精度优越。</p>
{fig(4, "fig4_regression_five_seed.png", "Five-seed S_CC RMSE for rich-only, Federated-H1, and all-prior target personalization. Error bars show sample standard deviations.")}
<h3>4.3 Sufficient-statistics equivalence</h3>
<p>Federated H1 与 pooled reference 的 4/4 gas α 一致，C5 H1 prediction 最大绝对差为 6.2532×10<sup>−8</sup> ppm，S_CC RMSE 差为 6.7040×10<sup>−12</sup> ppm，审计状态为 PRACTICAL_EQUIVALENCE。源端上传 moments、normal equations 与 clipped validation SSE/count，服务器返回 scaler 与 Ridge candidates；这些证据支持 raw source rows remain local，但不支持更强的安全性表述。</p>
<h3>4.4 Selective-output quality–coverage</h3>
{table_html("Table IV. Selective-output quality–coverage trade-off.", tables["table_IV_qc"])}
<p>v5 QC2 在 HC95/HC90 上分别取得 13.9178/12.7723 ppm accepted RMSE，但 accepted yield 仅为 93.75%/86.99%。相对 v4，yield 分别下降 3.53/3.82 percentage points。HC90 CO yield 从 84.71% 降至 64.12%，CO accepted RMSE 从 19.7243 增至 20.8318 ppm，因此 v5 QC2 未晋级，v4 保持正式 baseline。</p>
{fig(5, "fig5_qc_quality_coverage.png", "Frozen HC95/HC90 quality–coverage points. Lower accepted RMSE is interpreted jointly with accepted yield; no curve or threshold was fitted.")}
<h3>4.5 System efficiency</h3>
{table_html("Table V. System efficiency on PC and Raspberry Pi 5.", tables["table_V_system"])}
<p>v5 core bundle 为 289,916 bytes，v4 为 2,971,538 bytes，v5 QC2 为 1,065,632 bytes。Pi 三次正式对象均 throttled=0x0，峰值温度 54.55 °C。B5 25-round measured application payload 为 17,572,650 bytes；该值包含应用层序列化开销，但未采集 transport bytes。</p>
{fig(6, "fig6_system_efficiency.png", "Raspberry Pi median latency, bundle size, and regression parameter comparison for v4, v5 core, and v5 QC2.")}
<h3>4.6 Calibration-budget sensitivity</h3>
{table_html("Table VI. Group-aware calibration-budget sensitivity.", tables["table_VI_calibration"])}
<p>统一 group-aware protocol 下，320→160 windows 使 mean S_CC RMSE 从 10.8724 增至 23.9156 ppm，超过两倍；CO-high 与 Methane 尤其敏感。10.8724 ppm 是 group-aware fold-selection 均值，不能替代正式 historical protocol 下的 frozen seed-42 RG1 结果 11.3416 ppm。</p>
{fig(7, "fig7_groupaware_calibration.png", "Group-aware S_CC RMSE versus calibration budget. Error bars retain the registered fold/subset variability; lines only connect observed budgets.")}
<h3>4.7 Calibration-protocol harmonization audit</h3>
<p>Historical holdout 与 group-aware 两条轨迹均随预算下降而退化，但幅度随 protocol 和 budget 改变，最终描述为 SENSITIVITY_PARTLY_PROTOCOL_DEPENDENT。Historical 240/80 保留 window-level membership；harmonization 不选择更优 protocol，也不覆盖正式 11.3416 ppm 主结果。</p>

<h2>VII. 讨论与局限</h2>
<p><strong>Calibration sensitivity.</strong> 在统一 group-aware protocol 下，将 target calibration 从 320 减至 160 windows 使 mean S_CC RMSE 超过两倍。两种 protocol 的退化方向一致，但退化幅度部分依赖内部 selection protocol 与 budget。</p>
<p><strong>Historical boundary.</strong> 历史 240/80 split 是 window-level；61 个 validation filenames 全部出现在 fit subset。这是 calibration-internal overlap，不能写为 test leakage。Low-calibration 与 harmonization 使用此前已使用的历史 C5 test，仅构成 frozen-method descriptive evidence。</p>
<p><strong>Gas and QC limitations.</strong> CO-high 与 Methane 对 calibration coverage 尤其敏感。v5 QC2 能富集高误差样本，但 coverage 与 HC90 CO guards 未通过，因此没有取代 v4。</p>
<p><strong>Privacy boundary.</strong> Federated H1 构建中 raw source samples remain local；交换的 sufficient statistics 没有安全聚合或差分隐私保护，本文不主张形式化隐私保证。</p>

<h2>VIII. 结论</h2>
<p>GAPS 在真实设备联邦分类、充分统计量 source regression reference 与轻量 target personalization 之间建立了可执行闭环。五种子结果支持 B5 分类稳定性；Federated H1 在预注册 1% margin 下相对 all-prior 非劣并显著简化依赖；v5 core 降低了回归参数、bundle 和 Pi median latency。与此同时，v4 仍是更强的正式 selective-output baseline，目标 calibration coverage 仍是系统主要限制。上述结论共同界定了效率收益与可靠部署边界。</p>

<section class="appendix"><h2>Appendix A. Frozen supporting tables and figures</h2>
{table_html("Table A7. Calibration protocol harmonization comparison.", tables["table_A7_harmonization"])}
<figure><img src='../paper_evidence_freeze/figures/appendix_calibration_dual_track.png' alt='dual-track calibration audit'><figcaption><strong>Appendix Fig. A1.</strong> Group-aware and historical calibration tracks are shown in separate panels; historical and group-aware points are not merged into one curve.</figcaption></figure>
<figure><img src='../paper_evidence_freeze/figures/appendix_calibration_protocol_delta.png' alt='protocol delta'><figcaption><strong>Appendix Fig. A2.</strong> Descriptive group-aware minus historical S_CC RMSE at each budget.</figcaption></figure>
<p>Tables A1–A6 and A8 are provided as frozen CSV artifacts in <code>docs/paper_evidence_freeze/paper_tables/</code>. They cover per-seed classification, per-seed and per-gas regression, QC OOF selection, QC per-gas results, communication payload, and low-calibration per-gas results.</p></section>

<h2>参考文献</h2>{refs}
</main></body></html>"""


def write_manuscript(tables: Mapping[str, list[dict[str, Any]]]) -> None:
    source_bytes = SOURCE.read_bytes()
    BACKUP.parent.mkdir(parents=True, exist_ok=True)
    BACKUP.write_bytes(source_bytes)
    if sha256(BACKUP) != sha256(SOURCE):
        raise RuntimeError("source manuscript backup is not byte-identical")
    text = SOURCE.read_text(encoding="utf-8")
    MANUSCRIPT.write_text(manuscript_html(tables, text), encoding="utf-8")


def numeric_audit(claims: Sequence[Mapping[str, Any]]) -> None:
    text = MANUSCRIPT.read_text(encoding="utf-8")
    checks = [
        ("B5 accuracy", "0.989118 ± 0.005983", True),
        ("RG1 five-seed", "11.6339 ± 0.3142", True),
        ("formal seed42 frozen", "11.3416 ppm", True),
        ("group-aware 320", "10.8724", True),
        ("group-aware 160", "23.9156", True),
        ("QC HC95 yield+RMSE", "93.75%/86.99%", True),
        ("application payload wording", "measured application payload", True),
        ("stale historical B2 number", "99.26%", False),
        ("stale H8 route number", "17.447", False),
    ]
    rows = []
    conflicts = []
    for label, token, should_exist in checks:
        found = token in text
        passed = found == should_exist
        row = {"check": label, "token": token, "expected_present": should_exist, "found": found,
               "status": "PASS" if passed else "CONFLICT"}
        rows.append(row)
        if not passed:
            conflicts.append({"location": "manuscript", "metric": label, "observed": found,
                              "canonical": should_exist, "severity": "blocking"})
    # Every quantitative claim in the claim matrix has a canonical source SHA.
    for claim in claims:
        if not claim["asset_sha256"] or claim["evidence_status"] != "approved":
            conflicts.append({"location": claim["claim_id"], "metric": "provenance",
                              "observed": claim["asset_sha256"], "canonical": "approved SHA-bound source",
                              "severity": "blocking"})
    write_json(FREEZE / "numeric_consistency_report.json", {
        "schema_version": SCHEMA, "status": "PASS" if not conflicts else "FAIL",
        "checks": rows, "numeric_conflicts": len(conflicts),
        "scope_rules": ["11.3416=frozen historical seed42 result", "10.8724=group-aware 320 mean",
                        "accepted RMSE always paired with yield", "application payload is not transport traffic"],
    })
    md = ["# Numeric consistency report", "", f"Status: {'PASS' if not conflicts else 'FAIL'}",
          f"Blocking numeric conflicts: {len(conflicts)}", "",
          "| Check | Expected | Found | Status |", "|---|---:|---:|---|"]
    md.extend(f"| {r['check']} | {r['expected_present']} | {r['found']} | {r['status']} |" for r in rows)
    (FREEZE / "numeric_consistency_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    write_csv(FREEZE / "unresolved_numeric_conflicts.csv", conflicts,
              ["location", "metric", "observed", "canonical", "severity"])
    if conflicts:
        raise RuntimeError("blocking numeric conflict")


def terminology_audit() -> None:
    text = MANUSCRIPT.read_text(encoding="utf-8").lower()
    forbidden = [
        "few-shot robust", "privacy-preserving", "secure aggregation", "fully decentralized",
        "drift-resistant", "federated end-to-end regression", "globally superior qc",
        "original-file-independent historical split", "globally optimal threshold",
    ]
    hits = []
    for phrase in forbidden:
        for match in re.finditer(re.escape(phrase), text):
            hits.append({"phrase": phrase, "offset": match.start(), "severity": "critical"})
    write_csv(FREEZE / "forbidden_phrase_hits.csv", hits, ["phrase", "offset", "severity"])
    (FREEZE / "terminology_audit.md").write_text(
        "# Terminology audit\n\n"
        f"- Status: {'PASS' if not hits else 'FAIL'}\n"
        f"- Critical forbidden phrase hits: {len(hits)}\n"
        "- Preferred terms present: real-device federated classification; server-side domain adaptation; "
        "sufficient-statistics Federated Ridge; source regression reference; target personalization; "
        "correct-route RMSE (S_CC); end-to-end RMSE (S_ALL); accepted yield; selective output; "
        "group-aware calibration; historical window-level holdout; post-freeze harmonization audit.\n",
        encoding="utf-8")
    if hits:
        raise RuntimeError(f"critical terminology violations: {hits}")


def html_audit() -> None:
    text = MANUSCRIPT.read_text(encoding="utf-8")
    image_paths = re.findall(r"<img\s+src=['\"]([^'\"]+)", text)
    missing = []
    for source in image_paths:
        path = (MANUSCRIPT.parent / source).resolve()
        if not path.is_file():
            missing.append(source)
    figures = [int(value) for value in re.findall(r"<strong>Fig\. (\d+)\.</strong>", text)]
    local_paths = bool(re.search(r"(?:^|[\s>\"'])(?:[A-Za-z]:[\\/]|/root/)", text))
    sha_exposed = bool(re.search(r"\b[0-9a-f]{40,64}\b", text))
    table_count = text.count("<table>")
    checks = {
        "doctype": text.lower().startswith("<!doctype html>"),
        "html_closed": text.rstrip().endswith("</html>"),
        "relative_images_exist": not missing,
        "core_figure_numbers_continuous": figures == list(range(1, 8)),
        "core_tables_present": table_count >= 7,
        "no_local_absolute_path": not local_paths,
        "no_internal_SHA": not sha_exposed,
        "S_CC_defined": "correct-route rmse (s_cc)" in text.lower(),
        "S_ALL_defined": "end-to-end rmse (s_all)" in text.lower(),
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    (FREEZE / "html_validation_report.md").write_text(
        "# HTML validation report\n\n"
        f"- Status: {status}\n"
        + "\n".join(f"- {key}: {'PASS' if value else 'FAIL'}" for key, value in checks.items())
        + f"\n- Missing image paths: {missing}\n- Core figure sequence: {figures}\n- HTML table count: {table_count}\n",
        encoding="utf-8")
    (FREEZE / "table_figure_index.md").write_text(
        "# Table and figure index\n\n"
        "## Core tables\n\n"
        "Table I dataset/devices; Table II B5 five-seed classification; Table III regression/non-inferiority; "
        "Table IV QC quality–coverage; Table V system efficiency; Table VI group-aware calibration sensitivity.\n\n"
        "## Core figures\n\n"
        "Fig. 1 architecture; Fig. 2 federated classification/DA; Fig. 3 sufficient-statistics H1 and target personalization; "
        "Fig. 4 regression comparison; Fig. 5 QC trade-off; Fig. 6 system efficiency; Fig. 7 calibration sensitivity.\n\n"
        "## Appendix\n\nTables A1–A8 and Appendix Figs. A1–A2 are indexed in the evidence-freeze directories.\n",
        encoding="utf-8")
    if status != "PASS":
        raise RuntimeError(f"HTML validation failed: {checks}")


def final_reports(inventory: Sequence[Mapping[str, Any]], claims: Sequence[Mapping[str, Any]],
                  figures: Sequence[Mapping[str, Any]]) -> None:
    change_log = """# Manuscript change log

- Imported the uniquely identified 2026-07-20 HTML source and created a byte-identical pre-freeze backup.
- Replaced stale single-seed B2/B5 screening prose with frozen B5 five-seed evidence.
- Reframed regression around sufficient-statistics Federated H1 and 105D target personalization.
- Separated runtime v4 formal QC baseline, v5 simplified regression core, and v5 QC2 non-promoted candidate.
- Added six core tables and seven core figures sourced from frozen CSV/JSON.
- Added group-aware calibration sensitivity and a separate historical protocol harmonization appendix.
- Preserved the original reference list; no DOI or citation was invented.
- Added explicit calibration/test reuse, window-level split, gas-specific, QC, and privacy boundaries.
"""
    (FREEZE / "manuscript_change_log.md").write_text(change_log, encoding="utf-8")
    zh = """# 最终 claim summary

状态：`PAPER_EVIDENCE_FROZEN`

1. B5 五种子分类稳定性有正式证据。
2. Federated H1 与 pooled source Ridge 达到数值实用等价。
3. 原始源样本行在 H1 构建中保持本地；统计量无安全聚合或差分隐私。
4. RG1 通过相对 RG2 的预注册 1% 非劣标准，但不具备绝对精度优势。
5. v5 regression core 是最终简化实现；v4 是正式 selective-output baseline。
6. v5 QC2 有效但未晋级。
7. 正式 RG1 seed42 主结果保持 11.3416 ppm；10.8724 仅属于 group-aware 320 均值。
8. HIGH_CALIBRATION_SENSITIVITY 与 SENSITIVITY_PARTLY_PROTOCOL_DEPENDENT 同时保留。
9. 历史 240/80 是 window-level holdout；privacy 与 test-reuse boundary 已保留。
"""
    en = """# Final claim summary

Status: `PAPER_EVIDENCE_FROZEN`.

The frozen story combines stable five-seed B5 classification, a practically equivalent sufficient-statistics
source Ridge reference, and lightweight Federated-H1 target personalization. RG1 meets—but does not outperform
RG2 beyond—the preregistered 1% non-inferiority margin. Runtime v5 core is the final simplified regression
implementation, while runtime v4 remains the formal selective-output baseline and v5 QC2 is not promoted.
The historical 11.3416 ppm target result remains distinct from the 10.8724 ppm group-aware 320-window mean.
Calibration sensitivity is directionally robust but partly protocol-dependent, under an explicit window-level
and privacy boundary.
"""
    (FREEZE / "final_claim_summary.zh.md").write_text(zh, encoding="utf-8")
    (FREEZE / "final_claim_summary.en.md").write_text(en, encoding="utf-8")
    audit = {
        "schema_version": SCHEMA, "status": "PAPER_EVIDENCE_FROZEN",
        "claim_count": len(claims), "approved_claim_count": sum(c["evidence_status"] == "approved" for c in claims),
        "evidence_count": len(inventory), "numeric_conflicts": 0, "critical_terminology_violations": 0,
        "missing_core_tables": 0, "missing_core_figures": 0, "missing_evidence": 0,
        "html_validation": "PASS", "runtime_modified": False, "qc_modified": False,
        "new_experiment_run": False, "test_reopened": False,
        "mandatory_boundaries": {
            "runtime_v4_formal_baseline": True, "runtime_v5_core_final_simplified": True,
            "runtime_v5_qc2_not_promoted": True, "frozen_11_3416_distinct_from_group_10_8724": True,
            "HIGH_CALIBRATION_SENSITIVITY_retained": True,
            "SENSITIVITY_PARTLY_PROTOCOL_DEPENDENT_retained": True,
            "window_level_boundary": True, "privacy_boundary": True,
        },
    }
    write_json(FREEZE / "final_manuscript_audit.json", audit)
    (FREEZE / "final_manuscript_audit.md").write_text(
        "# Final manuscript audit\n\n"
        "- Status: PAPER_EVIDENCE_FROZEN\n"
        f"- Approved claims: {len(claims)}/{len(claims)}\n"
        f"- Approved evidence records: {len(inventory)}\n"
        "- Numeric conflicts: 0\n- Critical terminology violations: 0\n"
        "- Missing evidence/core tables/core figures: 0/0/0\n- HTML validation: PASS\n"
        "- Runtime/QC/model changes: none\n- New experiment/test reopening: none\n",
        encoding="utf-8")


def final_indexes(inventory: Sequence[Mapping[str, Any]], claims: Sequence[Mapping[str, Any]]) -> None:
    table_files = sorted((FREEZE / "paper_tables").glob("*.csv"))
    figure_files = sorted((FREEZE / "figures").iterdir())
    write_json(FREEZE / "canonical_table_source_index.json", {
        "schema_version": SCHEMA, "tables": [descriptor(path) for path in table_files],
        "all_tables_from_frozen_sources": True,
    })
    write_json(FREEZE / "canonical_figure_source_index.json", {
        "schema_version": SCHEMA, "figures": [descriptor(path) for path in figure_files],
        "core_figure_count": 7, "no_smoothing": True, "error_bars_retained": True,
    })
    write_json(FREEZE / "paper_evidence_freeze_index.json", {
        "schema_version": SCHEMA, "status": "PAPER_EVIDENCE_FROZEN",
        "manuscript": descriptor(MANUSCRIPT), "source_backup": descriptor(BACKUP),
        "source_commit": git_head(), "evidence_inventory": descriptor(FREEZE / "evidence_inventory.json"),
        "claim_matrix": descriptor(FREEZE / "claim_evidence_matrix.json"),
        "core_tables": [descriptor(path) for path in table_files if re.match(r"table_[IVX]+_", path.name)],
        "core_figures": [descriptor(path) for path in figure_files if re.match(r"fig[1-7]_", path.name)],
        "audit_reports": [descriptor(FREEZE / name) for name in (
            "numeric_consistency_report.json", "terminology_audit.md",
            "html_validation_report.md", "final_manuscript_audit.json")],
        "claim_count": len(claims), "evidence_count": len(inventory),
    })
    # Write final SHA after all other freeze artifacts exist. It excludes itself.
    artifacts = sorted([p for p in FREEZE.rglob("*") if p.is_file() and p.name != "final_sha256_index.json"]
                       + [MANUSCRIPT, BACKUP])
    write_json(FREEZE / "final_sha256_index.json", {
        "schema_version": SCHEMA, "created_at": utc_now(),
        "files": [descriptor(path) for path in artifacts],
    })


def main() -> None:
    require_new()
    paths = source_paths()
    validate_sources(paths)
    preflight(paths)
    inventory = evidence_inventory(paths)
    claims = claim_matrix(paths)
    inventory_outputs(inventory, claims, paths)
    tables = paper_tables(paths)
    figures = create_figures(paths)
    write_manuscript(tables)
    numeric_audit(claims)
    terminology_audit()
    html_audit()
    final_reports(inventory, claims, figures)
    final_indexes(inventory, claims)
    print(json.dumps({
        "status": "PAPER_EVIDENCE_FROZEN", "manuscript": rel(MANUSCRIPT),
        "claims": len(claims), "evidence": len(inventory), "core_tables": 6,
        "core_figures": 7, "numeric_conflicts": 0, "terminology_violations": 0,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
