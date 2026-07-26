#!/usr/bin/env python3
"""Create the protocol-closed IoT-J manuscript without changing frozen evidence.

The evidence-frozen HTML is treated as an immutable input. This script performs
deterministic prose/table substitutions only; it does not execute experiments,
load models, evaluate predictions, or alter any reported result.
"""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "docs" / "paper"
SOURCE = PAPER / "GAPS_IoTJ_evidence_frozen_20260726.zh.html"
OUTPUT = PAPER / "GAPS_IoTJ_protocol_closed_20260726.zh.html"
TABLE = PAPER / "tables" / "table_legacy_classification_ablation_protocol_closed_20260726.csv"
INDEX = PAPER / "GAPS_IoTJ_protocol_closeout_index_20260726.json"
EXPECTED_SOURCE_SHA256 = "b7b0aace15367c0ffd60fb3fed5bf93a4ca269f0bd6b74e07836aba4f63a96a4"
PROTOCOL_NAME = "calibrated-target held-out-window evaluation"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, found {count}")
    return text.replace(old, new, 1)


LEGACY_ROWS = [
    {
        "Configuration": "A0",
        "Target labels": "0",
        "Distribution alignment": "None",
        "Semantic constraint": "None",
        "Accuracy": "0.265441",
        "Evidence status": "LEGACY_SINGLE_SEED",
    },
    {
        "Configuration": "A0T",
        "Target labels": "320",
        "Distribution alignment": "None",
        "Semantic constraint": "Target supervised CE",
        "Accuracy": "0.982353",
        "Evidence status": "LEGACY_SINGLE_SEED",
    },
    {
        "Configuration": "A5",
        "Target labels": "320",
        "Distribution alignment": "Historical CORAL/MMD/stage/adversarial",
        "Semantic constraint": "Historical alignment + replay",
        "Accuracy": "0.730147",
        "Evidence status": "LEGACY_HISTORICAL_SEMANTICS",
    },
    {
        "Configuration": "A6",
        "Target labels": "320",
        "Distribution alignment": "None",
        "Semantic constraint": "Historical prototype/consistency/residual",
        "Accuracy": "0.980147",
        "Evidence status": "LEGACY_HISTORICAL_SEMANTICS",
    },
    {
        "Configuration": "B1",
        "Target labels": "320",
        "Distribution alignment": "Corrected class-conditional CORAL",
        "Semantic constraint": "Semantic core",
        "Accuracy": "0.987500",
        "Evidence status": "LEGACY_SINGLE_SEED_CORRECTED_SCREEN",
    },
    {
        "Configuration": "B2",
        "Target labels": "320",
        "Distribution alignment": "Corrected global/class MMD2",
        "Semantic constraint": "Semantic core",
        "Accuracy": "0.992647",
        "Evidence status": "LEGACY_SINGLE_SEED_CORRECTED_SCREEN",
    },
    {
        "Configuration": "B3",
        "Target labels": "320",
        "Distribution alignment": "Corrected cross-domain class/phase stage MMD2",
        "Semantic constraint": "Semantic core",
        "Accuracy": "0.988971",
        "Evidence status": "LEGACY_SINGLE_SEED_CORRECTED_SCREEN",
    },
    {
        "Configuration": "B4",
        "Target labels": "320",
        "Distribution alignment": "Corrected Wasserstein-min adversarial",
        "Semantic constraint": "Semantic core",
        "Accuracy": "0.989706",
        "Evidence status": "LEGACY_SINGLE_SEED_CORRECTED_SCREEN",
    },
    {
        "Configuration": "B5 (v3 screen)",
        "Target labels": "320",
        "Distribution alignment": "Corrected full distribution terms",
        "Semantic constraint": "Semantic core",
        "Accuracy": "0.988971",
        "Evidence status": "LEGACY_SINGLE_SEED_CORRECTED_SCREEN",
    },
    {
        "Configuration": "Final B5 (seed42)",
        "Target labels": "320",
        "Distribution alignment": "Corrected full distribution terms",
        "Semantic constraint": "Semantic core",
        "Accuracy": "0.980147",
        "Evidence status": "FINAL_FROZEN_CANONICAL",
    },
]


def table_html() -> str:
    cells = []
    for row in LEGACY_ROWS:
        cells.append(
            "<tr>"
            + "".join(
                f"<td>{row[key]}</td>"
                for key in (
                    "Configuration",
                    "Target labels",
                    "Distribution alignment",
                    "Semantic constraint",
                    "Accuracy",
                    "Evidence status",
                )
            )
            + "</tr>"
        )
    return (
        "<h3>4.0 Legacy classification ablation context</h3>\n"
        "<p class='table-title'>Table II-A. Legacy single-seed classification context; "
        "not a strict component-wise ablation of final B5.</p>"
        "<table><thead><tr><th>Configuration</th><th>Target labels</th>"
        "<th>Distribution alignment</th><th>Semantic constraint</th>"
        "<th>Accuracy</th><th>Evidence status</th></tr></thead><tbody>"
        + "".join(cells)
        + "</tbody></table>\n"
        "<p class='note'>A0/A0T/A5/A6 retain historical single-seed execution semantics; "
        "B1–B5 are corrected single-seed screening configurations. The historical A7 result "
        "is intentionally excluded, and the table must not be interpreted as a strict "
        "component-wise decomposition of the canonical final B5.</p>\n"
    )


def main() -> None:
    if sha256(SOURCE) != EXPECTED_SOURCE_SHA256:
        raise RuntimeError("evidence-frozen manuscript SHA256 mismatch")
    source = SOURCE.read_text(encoding="utf-8")
    text = source
    text = replace_once(
        text,
        "GAPS · IoTJ-STYLE EVIDENCE-FROZEN MANUSCRIPT",
        "GAPS · IoTJ-STYLE PROTOCOL-CLOSED MANUSCRIPT",
        "preprint label",
    )
    text = replace_once(
        text,
        (
            "<p>C1/C2 是源设备，C5 是目标设备；四类气体为 Ethanol、CO、Ethylene 和 Methane。"
            "每个窗口包含 100 个时间步与 8 个 MOS 通道。C5 calibration/test 为 320/1360 行。"
            "历史 calibration 内部的 240/80 split 是 window-level：80 个 fit filenames、"
            "61 个 validation filenames，且全部 61 个 validation filenames 也出现在 fit 中。"
            "该事实是 calibration-internal overlap，不等同于使用 test labels 训练。</p>"
        ),
        (
            f"<p><strong>Data protocol.</strong> 本文将目标设备协议统一称为 "
            f"<em>{PROTOCOL_NAME}</em>。先从原始 measurement files 构造 100×8 windows，"
            "再按 gas class 与 concentration 分层划分 C5 calibration/test，分别包含 320/1360 "
            "个 windows。同一个具体 window/sample row 只属于一个 subset，不跨 calibration 与 test "
            "重复；但同一原始文件生成的不同 windows 可以分别出现在两个 subsets。因此，该协议不强制"
            "原始文件互斥，也不表示对全新 measurement runs 的泛化。历史 calibration 内部 240/80 "
            "selection 仍为 window-level；filename overlap 描述的是 grouping dependence，不等同于"
            "使用 test labels 训练。</p>"
        ),
        "system/data protocol",
    )
    text = replace_once(
        text,
        (
            "<p>C5 为每个窗口提取 104D rich features，并附加 1D Federated H1 prediction，"
            "形成 105D 输入。根据冻结 B5 predicted class 路由到四个独立 target Ridge。α 仅由 "
            "C5 calibration 内部 fit/validation 或 group-aware folds 选择，随后在当前完整 calibration "
            "subset 上 refit；test 不参与 fit、select 或 refit。</p>"
        ),
        (
            "<p>C5 为每个窗口提取 104D rich features，并附加 1D Federated H1 prediction，"
            "形成 105D 输入。根据冻结 B5 predicted class 路由到四个独立 target Ridge。α 仅由 "
            "C5 calibration 内部 fit/validation 或 group-aware folds 选择，随后在当前完整 calibration "
            "subset 上 refit。Test labels 不用于 model fitting、hyperparameter selection、alpha "
            "selection、QC threshold selection 或 checkpoint selection；evidence freeze 之后也不再"
            "进行 method selection 或 reselection。</p>"
        ),
        "target personalization test boundary",
    )
    text = replace_once(
        text,
        (
            "<p>分类标准差来自五个训练 seeds 42–46；回归标准差来自五个冻结 B5 classifiers/routes。"
            "Correct-route RMSE (S_CC) 只统计 B5 分类路由正确的行，end-to-end RMSE (S_ALL) 则统计"
            "全部 test 行，两者不能互换。Group-aware 320 的标准差来自 fold/alpha-selection variability，"
            "160/80/40 来自 subset + fold variability。Historical holdout 320 是固定单次 reference，"
            "低预算标准差来自 holdout subset variability。Filename grouping 只用于 calibration-internal "
            "folds/subsets；历史 calibration/test split 仍为 window-level。Low-calibration 与 harmonization "
            "是对已冻结方法及此前已使用 C5 test 的 post-freeze 描述性分析，不用于重新选择方法。</p>"
        ),
        (
            f"<p><strong>Evaluation protocol.</strong> 正式目标设备评价采用 "
            f"<em>{PROTOCOL_NAME}</em>：window construction precedes splitting；C5 windows 按 gas "
            "class 与 concentration 分层为 320 calibration 和 1360 test。同一具体 window/sample row "
            "不跨 subset，但同一 original file 的不同 windows 可以同时出现在 calibration 与 test。"
            "Filename overlap 与 test-label leakage 是不同问题：前者限制 original-file/session-level "
            "外推，后者涉及是否使用 test labels 进行训练或选择。本文 test labels 不参与 fitting、"
            "hyperparameter/alpha/QC-threshold/checkpoint selection；evidence freeze 后不再选择方法。"
            "分类标准差来自 seeds 42–46；回归标准差来自五个冻结 B5 routes。S_CC 只统计分类路由正确行，"
            "S_ALL 统计全部 test 行。Low-calibration 与 harmonization 仅为 frozen-method post-freeze "
            "描述性分析，不用于方法重选。</p>"
        ),
        "experimental protocol",
    )
    text = replace_once(
        text,
        "<h2>VI. 结果</h2>\n",
        "<h2>VI. 结果</h2>\n" + table_html(),
        "legacy ablation insertion",
    )
    text = replace_once(
        text,
        (
            "<p><strong>Historical boundary.</strong> 历史 240/80 split 是 window-level；"
            "61 个 validation filenames 全部出现在 fit subset。这是 calibration-internal overlap，"
            "不能写为 test leakage。Low-calibration 与 harmonization 使用此前已使用的历史 C5 test，"
            "仅构成 frozen-method descriptive evidence。</p>"
        ),
        (
            "<p><strong>Evaluation boundary.</strong> The target-device evaluation uses a "
            "class- and concentration-stratified window-level split. Different windows from "
            "the same measurement run may occur in both calibration and test. The reported "
            "results therefore characterize post-calibration held-out-window performance "
            "rather than generalization to entirely unseen measurement runs.</p>\n"
            "<p>该边界反映原始文件层面的相关性，而不是 test-label leakage：同一个具体 window/sample "
            "row 不跨 subset，且 test labels 未进入 fitting 或冻结选择流程。历史 calibration 内部 "
            "240/80 selection 以及 post-freeze sensitivity 分析继续按各自已登记的 window-level "
            "边界解释。</p>"
        ),
        "limitation wording",
    )
    text = replace_once(
        text,
        (
            "<p><strong>Privacy boundary.</strong> Federated H1 构建中 raw source samples remain "
            "local；交换的 sufficient statistics 没有安全聚合或差分隐私保护，本文不主张形式化隐私保证。</p>"
        ),
        (
            "<p><strong>Privacy boundary.</strong> Federated H1 构建中 raw source samples remain "
            "local；交换的 sufficient statistics 没有安全聚合或差分隐私保护，本文不主张形式化隐私保证。</p>\n"
            "<p><strong>Future work.</strong> 后续工作仅包括 unseen-session 或 original-file-independent "
            "evaluation、更多 physical target devices，以及针对 sufficient statistics 的更强隐私保护。"
            "FedProx/FedAdam/SCAFFOLD 排名、多目标 reruns、新 QC 和新 regression heads 不属于当前投稿"
            "范围或待完成实验。</p>"
        ),
        "future work",
    )

    # Guard against affirmative overclaims. Required limitation/future-work phrases
    # may appear only in their explicitly bounded contexts.
    forbidden_affirmative = [
        "completely leakage-free",
        "zero-shot target generalization",
        "independent experimental file evaluation",
    ]
    hits = [term for term in forbidden_affirmative if term.lower() in text.lower()]
    if hits:
        raise RuntimeError(f"forbidden affirmative terminology: {hits}")
    required = [
        PROTOCOL_NAME,
        "window construction precedes splitting",
        "320 calibration",
        "1360 test",
        "Test labels 不用于 model fitting",
        "not a strict component-wise ablation of final B5",
        "The target-device evaluation uses a class- and concentration-stratified window-level split.",
        "NO_FURTHER_EXPERIMENTS_REQUIRED_FOR_CURRENT_SCOPE",
    ]
    # The terminal status is carried by the index/scope lock, not manuscript prose.
    for phrase in required[:-1]:
        if phrase not in text:
            raise RuntimeError(f"required manuscript phrase missing: {phrase}")

    OUTPUT.write_text(text, encoding="utf-8")
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    with TABLE.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(LEGACY_ROWS[0]))
        writer.writeheader()
        writer.writerows(LEGACY_ROWS)

    if sha256(SOURCE) != EXPECTED_SOURCE_SHA256:
        raise RuntimeError("frozen manuscript changed during closeout")
    payload = {
        "schema_version": "iotj.manuscript_protocol_closeout.index.v1",
        "status": "NO_FURTHER_EXPERIMENTS_REQUIRED_FOR_CURRENT_SCOPE",
        "protocol_name": PROTOCOL_NAME,
        "source": {
            "path": SOURCE.relative_to(ROOT).as_posix(),
            "sha256": sha256(SOURCE),
            "immutable": True,
        },
        "output": {
            "path": OUTPUT.relative_to(ROOT).as_posix(),
            "bytes": OUTPUT.stat().st_size,
            "sha256": sha256(OUTPUT),
        },
        "legacy_ablation_table": {
            "path": TABLE.relative_to(ROOT).as_posix(),
            "rows": len(LEGACY_ROWS),
            "bytes": TABLE.stat().st_size,
            "sha256": sha256(TABLE),
            "historical_a7_included": False,
        },
        "test_access_boundary": {
            "test_labels_used_for_model_fitting": False,
            "test_labels_used_for_hyperparameter_selection": False,
            "test_labels_used_for_alpha_selection": False,
            "test_labels_used_for_qc_threshold_selection": False,
            "test_labels_used_for_checkpoint_selection": False,
            "method_selection_after_evidence_freeze": False,
        },
        "current_scope_new_training_items": [],
        "cancelled_submission_experiments": [
            "original-file-level retraining",
            "FedProx",
            "FedAdam",
            "SCAFFOLD",
            "multi-target reruns",
            "new QC",
            "new regression heads",
        ],
        "future_work_only": [
            "unseen-session or original-file-independent evaluation",
            "additional physical target devices",
            "stronger privacy protection for sufficient statistics",
        ],
        "supporting_documents": [
            {
                "path": path.relative_to(ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in [
                ROOT
                / "docs"
                / "experiments"
                / "iotj_manuscript_protocol_scope_lock_20260726.zh.md",
                ROOT
                / "docs"
                / "experiments"
                / "iotj_protocol_closeout_number_consistency_audit_20260726.md",
            ]
        ],
        "experiments_run": 0,
        "frozen_results_modified": False,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    INDEX.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
