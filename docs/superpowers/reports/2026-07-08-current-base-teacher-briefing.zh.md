# 当前基座给老师汇报版

## 汇报主线

这次汇报不按“尝试过哪些模型”展开，而按系统逻辑展开：真实部署 route 下，F6 分类基座提供 route 与 risk context；H2.3+ 作为稳健目标域 calibration anchor；H8+C4 作为 CO-priority rescue stream；最后用 validation-selected per-client threshold guard 安全选择输出。

## 先讲的主结果

P4 threshold guard 在 Accepted+Review ALL 上达到 5.850 / 0.0339，相对 H2.3+ gain=1.199，H8 usage=20.3%。

| scope | RMSE/NRMSE | H2.3+ | H8-all | gain | H8 usage |
| --- | --- | --- | --- | --- | --- |
| ALL | 5.850 / 0.0339 | 7.049 | 6.509 | 1.199 | 20.3% |
| C3 | 4.882 / 0.0307 | 5.797 | 5.870 | 0.915 | 21.2% |
| C4 | 5.774 / 0.0333 | 6.336 | 6.567 | 0.562 | 21.6% |
| C5 | 7.558 / 0.0406 | 9.695 | 7.632 | 2.137 | 17.0% |

## F1-F5 图表资产

| figure | path |
| --- | --- |
| F1 | results\current_base_teacher_briefing_pack_20260708\figures\F1_system_pipeline.svg |
| F2 | results\current_base_teacher_briefing_pack_20260708\figures\F2_threshold_guard_gains.svg |
| F3 | results\current_base_teacher_briefing_pack_20260708\figures\F3_co_nonco_safety.svg |
| F4 | results\current_base_teacher_briefing_pack_20260708\figures\F4_low_cal_stability.svg |
| F5 | results\current_base_teacher_briefing_pack_20260708\figures\F5_route_gap_appendix.svg |

## T1-T7 / F1-F5 汇报顺序

| order | id | kind | title | source | talk track |
| --- | --- | --- | --- | --- | --- |
| 1 | F1 | figure | System pipeline | results\current_base_teacher_briefing_pack_20260708\figures\F1_system_pipeline.svg | 先讲系统为什么这样分层：real-route -> 双回归流 -> threshold guard -> QC report。 |
| 2 | T3 | table | P4 threshold guard deployment audit | threshold_guard_metrics.csv | 先给 P4 主结果，说明这是当前最可提交结果。 |
| 3 | F2 | figure | P4 threshold guard per-client gains | results\current_base_teacher_briefing_pack_20260708\figures\F2_threshold_guard_gains.svg | 用 per-client gain 图让老师快速看到 ALL/C3/C4/C5 都有收益。 |
| 4 | F3 | figure | CO/nonCO safety panel | results\current_base_teacher_briefing_pack_20260708\figures\F3_co_nonco_safety.svg | 说明 CO rescue 与 nonCO safety 同时成立。 |
| 5 | T4 | table | P4 selected thresholds | threshold_guard_selected_thresholds.csv | 强调 tau 来自 validation/calibration，不是 test tuning。 |
| 6 | F4 | figure | Low-cal budget stability | results\current_base_teacher_briefing_pack_20260708\figures\F4_low_cal_stability.svg | 解释低 calibration 下 selector 选择过程的稳定性。 |
| 7 | T1 | table | Real-route full main table | real_route_mainline_summary.csv | full-set 作为实际端到端难度背景。 |
| 8 | T2 | table | Real-route Accepted+Review main table | real_route_post_qc_summary.csv | Accepted+Review 是论文主报告切片。 |
| 9 | T5 | table | P3 low-cal stress | selector_low_cal_metric_summary.csv | 低 calibration stress 的表格证据。 |
| 10 | T6 | table | P5 route-gap appendix | light_route_gap_appendix_table.csv | route-gap 只作 appendix，不抢主线。 |
| 11 | F5 | figure | Route-gap appendix | results\current_base_teacher_briefing_pack_20260708\figures\F5_route_gap_appendix.svg | 说明 C5 full-set gap 的分类/route-noise 背景。 |
| 12 | T7 | table | Claim-evidence matrix | claim_evidence_matrix.csv | 最后用 claim-evidence matrix 收束贡献和边界。 |

## 8 页汇报提纲

| slide | title | content | assets |
| --- | --- | --- | --- |
| 1 | 问题与一句话主线 | 真实部署 route 下，目标域回归需要 profile calibration + guarded rescue。 | K1/K2 claims |
| 2 | 系统主线图 | F6 real-route -> H2.3+ anchor / H8+C4 rescue -> threshold guard -> QC Accepted+Review。 | F1 |
| 3 | 主结果 | P4 Accepted+Review ALL/C3/C4/C5 全部优于 H2.3+。 | T3 + F2 |
| 4 | 为什么 guard 安全 | CO 上允许 rescue，nonCO H8 usage=0。 | F3 |
| 5 | 阈值来源与无泄漏 | tau_C3/C4/C5 由 validation/calibration 选择，test 只做最终审计。 | T4 |
| 6 | 低 calibration 稳定性 | 预算 12/24/48/80 重采样；budget=80 时 C3/C4/C5 positive gain rate=100%。 | T5 + F4 |
| 7 | route-gap 作为附录解释 | C5 full-set gap 最大，解释 route-noise 背景，不替代 real-route 主线。 | T6 + F5 |
| 8 | 贡献、边界与下一步 | 贡献 K1-K5；边界是 current-base C12->C345；下一步 P6 跨 source-target 验证。 | T7 |

## 老师可能追问的点

- 为什么不用 oracle-route 当主结果：因为主线是实际部署 real-route；oracle/classification-correct 只作机制解释。
- 为什么 H8+C4 不全量替换：因为 H8 是 CO rescue stream，nonCO 由 H2.3+ 保护，P4 nonCO H8 usage=0。
- threshold 有没有 test 泄漏：没有，tau_C3/C4/C5 来自 validation/calibration；test 只做最终审计。
- 低 calibration 是否稳定：P3 budget=80 达到 C3/C4/C5 positive gain rate=100%，12/24/48 作为趋势证据。
- 下一步是什么：P6 跨 source-target 验证，用同一套 T/F 结构复现。
