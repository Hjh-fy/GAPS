# 系统文档目录化索引

## 读文档顺序

建议按 `Evidence Freeze -> Method Story -> Teacher Briefing -> Submission Docs` 的顺序阅读。这样先看到冻结指标，再看到方法叙事，最后进入论文草稿和归档清单。

## T1-T7 / F1-F5 资产顺序

| item | kind | title | source | placement |
| --- | --- | --- | --- | --- |
| F1 | figure | System pipeline | results\current_base_teacher_briefing_pack_20260708\figures\F1_system_pipeline.svg | opening_system_story |
| T3 | table | P4 threshold guard deployment audit | threshold_guard_metrics.csv | main_result |
| F2 | figure | P4 threshold guard per-client gains | results\current_base_teacher_briefing_pack_20260708\figures\F2_threshold_guard_gains.svg | main_result_visual |
| F3 | figure | CO/nonCO safety panel | results\current_base_teacher_briefing_pack_20260708\figures\F3_co_nonco_safety.svg | safety |
| T4 | table | P4 selected thresholds | threshold_guard_selected_thresholds.csv | threshold_provenance |
| F4 | figure | Low-cal budget stability | results\current_base_teacher_briefing_pack_20260708\figures\F4_low_cal_stability.svg | stability |
| T1 | table | Real-route full main table | real_route_mainline_summary.csv | full_context |
| T2 | table | Real-route Accepted+Review main table | real_route_post_qc_summary.csv | qc_context |
| T5 | table | P3 low-cal stress | selector_low_cal_metric_summary.csv | stress_table |
| T6 | table | P5 route-gap appendix | light_route_gap_appendix_table.csv | route_gap_appendix |
| F5 | figure | Route-gap appendix | results\current_base_teacher_briefing_pack_20260708\figures\F5_route_gap_appendix.svg | route_gap_visual |
| T7 | table | Claim-evidence matrix | claim_evidence_matrix.csv | claim_evidence |

## 文档与资产索引

| id | category | path | purpose | stage | status |
| --- | --- | --- | --- | --- | --- |
| FZD | evidence freeze | results/current_base_evidence_freeze_20260708/current_base_evidence_freeze.zh.md | 冻结 P1-P5 current-base 证据与检查结果 | Freeze | source |
| MET | metrics | results/current_base_evidence_freeze_20260708/frozen_headline_metrics.csv | P4/P3/P5 主指标来源 | Freeze | source |
| TCK | paper tables | results/current_base_evidence_freeze_20260708/paper_table_checklist.csv | T1-T7 表格清单 | Freeze | source |
| FCK | paper figures | results/current_base_evidence_freeze_20260708/paper_figure_checklist.csv | F1-F5 图清单 | Freeze | source |
| MTH | method story | results/current_base_method_story_20260708/method_story_blueprint.zh.md | 模块-方法-证据叙事蓝图 | Story | source |
| SYS | system docs | results/current_base_method_story_20260708/system_documentation.zh.md | 实验命令、参数、清理建议初版 | Story | source |
| TBR | teacher briefing | results/current_base_teacher_briefing_pack_20260708/teacher_briefing.zh.md | 给老师看的汇报版 | Briefing | source |
| TFS | teacher briefing | results/current_base_teacher_briefing_pack_20260708/table_figure_sequence.csv | T1-T7 / F1-F5 汇报顺序 | Briefing | source |
| SLD | teacher briefing | results/current_base_teacher_briefing_pack_20260708/teacher_slide_outline.csv | 8 页汇报提纲 | Briefing | source |
| SRC1 | source document | CLS_ExpB_StrongDA_2080_Classification_Backbone_Freeze_Manual.md | 分类基座冻结、time-aware 20/80 协议、R3aK16 历史主线和 QC v2 边界 | SourceReview | source |
| SRC2 | source document | 面向气体传感器长期部署的单机模拟联邦持续学习系统技术报告草稿.docx | 单机模拟联邦持续学习系统、原型/漂移/QC 闭环的方法学底座 | SourceReview | source |
| F1 | paper figures | results/current_base_teacher_briefing_pack_20260708/figures/F1_system_pipeline.svg | F1 论文/汇报图 | Briefing | source |
| F2 | paper figures | results/current_base_teacher_briefing_pack_20260708/figures/F2_threshold_guard_gains.svg | F2 论文/汇报图 | Briefing | source |
| F3 | paper figures | results/current_base_teacher_briefing_pack_20260708/figures/F3_co_nonco_safety.svg | F3 论文/汇报图 | Briefing | source |
| F4 | paper figures | results/current_base_teacher_briefing_pack_20260708/figures/F4_low_cal_stability.svg | F4 论文/汇报图 | Briefing | source |
| F5 | paper figures | results/current_base_teacher_briefing_pack_20260708/figures/F5_route_gap_appendix.svg | F5 论文/汇报图 | Briefing | source |
| PMD | paper draft | results/current_base_submission_docs_pack_20260708/paper_method_chapter_draft.zh.md | 论文方法章节中文草稿 | SubmissionDocs | generated |
| SDX | system index | results/current_base_submission_docs_pack_20260708/system_documentation_index.zh.md | 系统文档目录化索引 | SubmissionDocs | generated |
| ARC | archive plan | results/current_base_submission_docs_pack_20260708/intermediate_archive_plan.zh.md | 中间文件归档建议 | SubmissionDocs | generated |

## source document synthesis

| id | source | usable takeaway | current-base update | paper role |
| --- | --- | --- | --- | --- |
| FCL | 面向气体传感器长期部署的单机模拟联邦持续学习系统技术报告草稿.docx | The system should be described as a single-machine simulated federated continual learning loop: client-local training uploads parameters/prototypes/statistics, the server aligns representation and drift statistics, and deployment adds online adaptation, calibration, and QC. | Use this as the method-level architecture layer before the current F6 -> H2.3+/H8+C4 -> threshold guard story; do not claim distributed RPC in the current local simulation. | System motivation and full pipeline formula. |
| CLS | CLS_ExpB_StrongDA_2080_Classification_Backbone_Freeze_Manual.md | CLS-FlowerExpB-TimeAware2080 freezes server_latest_adapted.pth + logits under C12->C345 time-aware 20/80 protocol; clean rerun weighted accuracy is 0.989444, NLL 0.107643, ECE 0.009875. | Treat this as the classification-backbone freeze principle. The current paper pack uses the latest F6 real-route base as the deployed route provider, so CLS/F6 should be written as lineage, not as two competing claims. | Classification route-contract and no-retrain justification. |
| REG | CLS_ExpB_StrongDA_2080_Classification_Backbone_Freeze_Manual.md | REG-R3aK16-AutoV2-TimeAware2080 was the earlier regression freeze: source C1/C2 FedAvg plus per-client auto_v2/QC; it located C5-CO as the weak cell. | Current P4 evidence supersedes this regression branch with H2.3+ anchor plus H8+C4 guarded rescue. r3ak16 remains historical evidence and an efficiency decision, not a retraining target. | Negative/legacy result and motivation for guarded CO rescue. |
| QCV2 | CLS_ExpB_StrongDA_2080_Classification_Backbone_Freeze_Manual.md | QC v2 can generate 40-D response descriptors and risk signals, but single composite/route thresholds did not beat old QC v1 guardrails. | Keep QC v2 as a candidate risk signal and diagnostic module. The current P4 threshold guard still reports validation-selected per-client thresholds and Accepted+Review metrics. | QC boundary and deployment-risk caveat. |

## 实验命令索引

| stage | purpose | command | outputs | notes |
| --- | --- | --- | --- | --- |
| P1 | C5 CO-bin and residual-bin rescue audit | python run_real_route_c5_rescue_audit.py | results/real_route_c5_rescue_audit_20260705/c5_rescue_bin_metrics.csv | No training; compares H2.3+ vs H8+C4 on paired real-route predictions. |
| P2 | C5 threshold selector validation | python run_real_route_c5_selector_validation.py | results/real_route_c5_selector_validation_20260705/c5_route_co_risk_threshold_sweep_calibration.csv | Uses calibration predictions to choose risk threshold; test only evaluates selected threshold. |
| P3 | C3/C4/C5 low-calibration threshold selector stress | python run_real_route_selector_low_cal_stress.py --target-clients C3,C4,C5 --budgets 12,24,48,80 --repeats 50 | results/real_route_selector_low_cal_stress_20260706/selector_low_cal_metric_summary.csv | No training; resamples validation rows and evaluates fixed test predictions. |
| P4 | Export threshold guard deployment candidate | python export_real_route_threshold_guard_deployment_candidate.py | results/real_route_threshold_guard_deployment_candidate_20260707/threshold_guard_policy.json | Freezes validation-selected thresholds and writes policy/metrics/test outputs. |
| P5 | Build light real-vs-oracle route-gap appendix | python run_light_route_gap_appendix.py | results/light_route_gap_appendix_20260708/light_route_gap_appendix_table.csv | Appendix context only; not the optimization target. |
| Story | Build current-base paper story pack | python run_current_base_paper_story_pack.py | docs/superpowers/reports/2026-07-05-current-base-paper-story-pack.zh.md | Aggregates real-route, P1-P5, oracle supplement, risks and next plan. |
| Freeze | Freeze current-base evidence package | python run_current_base_evidence_freeze.py | docs/superpowers/reports/2026-07-08-current-base-evidence-freeze.zh.md | Builds artifact inventory, headline metrics, checks and paper checklist. |
| Verify | Run focused regression tests | python -m pytest tests/test_current_base_evidence_freeze.py tests/test_current_base_paper_story_pack.py tests/test_real_route_selector_low_cal_stress.py tests/test_real_route_threshold_guard_deployment_candidate.py tests/test_light_route_gap_appendix.py -q | pytest pass/fail | Use workspace-local TEMP if pytest default temp dir is not writable. |
| Briefing | Build F1-F5 and teacher-facing briefing package | python run_current_base_teacher_briefing_pack.py | results/current_base_teacher_briefing_pack_20260708/teacher_briefing.zh.md | Consumes frozen evidence; no training. |
| SubmissionDocs | Build paper method draft, system index, and archive plan | python run_current_base_submission_docs_pack.py | results/current_base_submission_docs_pack_20260708/ | Documentation pack only; no file moves. |

## 参数索引

| name | value | used by | meaning |
| --- | --- | --- | --- |
| target_clients | C3,C4,C5 | P3/P4 | 目标域客户端集合。 |
| selector_rule | route_class=CO and qc_risk_value>=per_client_threshold -> H8+C4 else H2.3+ | P4 runtime | per-client threshold guard 的核心选择规则。 |
| primary_reporting_slice | qc_decision in {accept, review} | P4/reporting | 论文主指标报告的 QC 后样本集合。 |
| fallback_non_co_route | H2.3+ | P4 runtime | 非 CO route 默认回退到 H2.3+，保护 nonCO。 |
| low_cal_budgets | 12,24,48,80 | P3 | 低 calibration stress 的每客户端 validation 抽样预算。 |
| low_cal_repeats | 50 | P3 | 每个预算重复抽样次数。 |
| tau_C3 | 0.015903 | P4 runtime | C3 validation-selected risk threshold; selected from calibration/validation only. |
| tau_C4 | 0.000000 | P4 runtime | C4 validation-selected risk threshold; selected from calibration/validation only. |
| tau_C5 | 0.050000 | P4 runtime | C5 validation-selected risk threshold; selected from calibration/validation only. |

## 后续维护规则

- 新实验先写入独立 `results/<name>_<date>/`，再进入 freeze/story/briefing。
- 能进入论文主线的结果必须有 manifest、CSV 指标、报告 Markdown 和测试覆盖。
- 文档中的主指标以 `frozen_headline_metrics.csv` 为准，不直接从零散日志手动摘数。
- 清理中间文件时只移动、不删除，并保留归档 README。
