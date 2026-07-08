# 论文方法章节草稿

> 面向当前基座 C12->C345 的论文方法章节初稿。可作为第 3/4 章的中文底稿，后续再翻译/压缩成正式论文语气。

## 章节主线

本文方法部分建议围绕一个核心句展开：在真实部署 route 下，目标域气体回归需要以 F6 分类基座提供的 route/risk context 为入口，以 H2.3+ 作为稳健 target calibration anchor，以 H8+C4 作为 CO-priority rescue stream，再通过 validation-selected per-client threshold guard 选择最终输出。

## 方法公式

对客户端 `c` 的样本 `i`，记 F6 输出的 route 为 `r_i`，风险分数为 `q_i`，客户端阈值为 `tau_c`：

```text
ŷ_i = H8+C4_i,  if r_i = CO and q_i >= tau_c
ŷ_i = H2.3+_i, otherwise
```

当前阈值：tau_C3=0.015903 / tau_C4=0.000000 / tau_C5=0.050000。阈值只来自 calibration/validation，test 仅用于最终 audit。

## 可直接改写的章节段落

### S1 问题定义：真实 route 下的目标域回归

目标域 C3/C4/C5 的气体浓度回归不是单纯换一个回归头的问题；真实部署时分类 route 可能含噪，CO 与 nonCO 的误差结构也不同，因此需要把分类上下文、目标域校准和回归输出选择放在同一个系统里报告。

证据位置：T1, T2。

写作提醒：开头不要写成模型堆叠竞赛，而要写成部署条件下的 profile calibration 问题。

### S2 系统总览：F6 real-route 到双回归流

系统由 Problem framing、F6 real-route classification base、H2.3+ target profile calibration、H8+C4 formal route rescue、QC Accepted+Review reporting、Per-client threshold guard 组成。F6 分类基座先给出 real-route/risk context，H2.3+ 作为稳健目标域 anchor，H8+C4 作为 CO-priority rescue stream，最后由 per-client threshold guard 做输出选择。

证据位置：F1。

写作提醒：这里放 F1，并明确 runtime 不能使用 true label、test label 或 oracle route。

### S3 F6 分类基座：可部署 route context

F6 的作用不是直接改善回归数值，而是在部署时提供 route_class 与风险上下文，决定后续是否允许进入 CO rescue。论文中应强调 real-route 是主线，oracle-route/classification-correct 只用于机制解释或附录对照。

证据位置：F1, T6, F5。

写作提醒：老师已强调重点报告分类正确下的回归性能，但主线仍要守住 real-route，可把 classification-correct 作为机制补充。

### S4 H2.3+ 与 H8+C4：anchor-rescue 双流设计

H2.3+ 是默认输出流，承担 balanced target calibration 与 nonCO safety；H8+C4 不是全量替代，而是针对 CO 高残差区域的 formal rescue stream。这个设计解释了为什么 r3ak16 一类额外回归头当前不应再重复训练：它没有进入 P4 可部署主线，也没有改变 anchor-rescue 的证据结构。

证据位置：T3, F3。

写作提醒：可以把 r3ak16 放进负结果/效率决策说明，不作为后续主线重复训练对象。

### S5 Per-client threshold guard：验证集选择的安全门控

对客户端 c 的样本 i，记 F6 给出的 route 为 r_i，风险分数为 q_i，验证集选择阈值为 tau_c，则 ŷ_i = H8+C4_i if r_i=CO and q_i>=tau_c; otherwise ŷ_i = H2.3+_i。当前阈值为 tau_C3=0.015903 / tau_C4=0.000000 / tau_C5=0.050000。这个公式把 H8+C4 限制在 CO rescue 条件内，避免 nonCO 被高风险回归头误伤。

证据位置：T4。

写作提醒：必须写清 tau_c 来自 calibration/validation，不来自 test tuning。

### S6 QC Accepted+Review：论文主报告切片

Accepted+Review 是当前论文主报告切片，因为它对应系统认为可报告或需人工复核但仍可纳入分析的样本；Reject 保留为风险暴露和部署边界，不作为主性能口径。

证据位置：T2, T3。

写作提醒：这一段回应老师要求：重点报告分类正确/可报告路径下的回归指标，同时保留 full-set 作为端到端难度背景。

### S7 低 calibration stress：阈值选择是否稳定

低 calibration stress test 通过 12/24/48/80 的 per-client validation budget 重采样，检查 threshold selector 在小校准集下是否还能保持正收益。当前 budget=80 时 C3/C4/C5 positive gain rate 达到 100%，可作为稳定性证据。

证据位置：T5, F4。

写作提醒：12/24/48 写趋势，不要夸大为所有低预算完全稳定。

### S8 主结果陈述：P4 deployment candidate

P4 threshold guard 在 Accepted+Review ALL 上达到 5.850 / 0.0339，相对 H2.3+ gain=1.199，H8 usage=20.3%。C5 达到 7.558 / 0.0406，gain=2.137，说明 guard 的主要收益来自高难度 C5 CO rescue。

证据位置：T3, F2, F3。

写作提醒：这段可以作为方法后接实验主结果的桥段。

### S9 写作边界与后续验证

当前证据边界是 current-base C12->C345，图表顺序为 F1、T3、F2、F3、T4、F4、T1、T2、T5、T6、F5、T7。P5 route-gap 是附录解释，不替代 real-route 主线。下一步 P6 应扩展不同 source-target 组合，用同一套 T/F 结构复现结论。

证据位置：T6, T7, F5。

写作提醒：结尾要主动讲边界，避免老师追问时显得是在回避泛化问题。

## 模块-方法对应表

| id | module | principle | evidence | paper section |
| --- | --- | --- | --- | --- |
| M0 | Problem framing | 把系统拆成 real-route 主线、profile calibration、CO rescue、QC reporting 和 validation-selected guard。 | P5 shows C5 full-set route gap is dominant; P4 shows guarded real-route Accepted+Review gain. | Introduction / Method overview |
| M1 | F6 real-route classification base | 使用 F6 分类基座的 predicted route_class/pred_class 作为回归 profile routing 条件。 | Story pack treats real-route as main line; oracle-route kept as mechanism appendix. | System pipeline / Experimental setting |
| M2 | H2.3+ target profile calibration | 使用目标域 direct-head / weak-blend profile 作为默认 ppm prediction stream。 | P4 nonCO rows keep H2.3+ and H8 usage 0. | Regression profile calibration |
| M3 | H8+C4 formal route rescue | 把 formal C4 route rescue 作为 CO-priority candidate stream，而不是全样本替换。 | C5-CO gain 5.690; C5-nonCO H8 usage 0.0%. | CO-priority rescue branch |
| M4 | QC Accepted+Review reporting | 主提交指标使用 qc_decision in {accept, review}，reject 保持为风险暴露而非主性能。 | P4 Accepted+Review ALL 5.850 / 0.0339. | Evaluation protocol |
| M5 | Per-client threshold guard | if client in C3/C4/C5 and route_class=CO and risk>=tau_client then H8+C4 else H2.3+. | thresholds: C3=0.015903, C4=0.000000, C5=0.050000; ALL gain 1.199, C5 gain 2.137. | Guarded profile selector |
| M6 | Low-calibration stress validation | 重复抽样 validation rows，按 budget 选择 per-client threshold，再固定 test 评估。 | P3 budget=80 gives 100% positive-gain rate for C3/C4/C5. | Robustness / Calibration stress |
| M7 | Evidence freeze and appendix | hash artifact inventory + freeze checks + table/figure checklist + route-gap appendix. | 14/14 freeze checks passed; P5 kept as appendix context only. | Reproducibility / Appendix |

## T/F 使用位置

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

## 方法章节边界

- real-route 是主线，oracle-route/classification-correct 只作为机制解释或附录。
- H8+C4 是受控 CO rescue，不是全量替换 H2.3+。
- r3ak16 当前不进入主线复训，除非后续 P6 发现新的 source-target 组合需要额外候选。
- P5 route-gap 用来解释 full-set 难度，不替代 Accepted+Review 主结果。
