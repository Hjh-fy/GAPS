# 实验室三气体准确率恢复：阶段实验审计

## Audit scope and intended claim

审计REC-A1、REC-A3、REC-A4、REC-A5能否支持P2→P3实验室三气体分类的阶段性
改进结论。输入结果保持只读；REC-A2失败attempt和retry单独保留。

## Compared experiments

| Experiment ID | Split | Model | Checkpoint | DA | Calibration | QC | Seeds | Provenance |
|---|---|---|---|---|---|---|---|---|
| REC-A1-CB2 | 全浓度time-purged，420 test | proto_replay | round 25 adapted | corrected B2 | P3 90窗口 | valid | 42 | formal summary |
| REC-A3-COND | 同A1 | proto_replay | round 25 adapted | corrected B2 | P3 90窗口 | valid | 42 | formal summary |
| REC-A4-STABLE150 | 稳定范围，360 test | proto_replay | round 25 adapted | corrected B2 | P3 90窗口 | valid | 42 | formal summary |
| REC-A5-NOCH2 | 同A1 | proto_replay，input_dim=5 | round 25 adapted | corrected B2 | P3 90窗口 | valid | 42 | formal summary |

## Findings

| Finding ID | Severity | Check | Evidence | Impact | Required action | Status |
|---|---|---|---|---|---|---|
| INT-01 | informational | checkpoint | 四项均固定round 25 | 不再受source calibration饱和选轮影响 | 保持last_round | closed |
| INT-02 | informational | 完整范围单变量 | A3和A5均低于A1 | 当前批次不支持电导或去CH2单独提升 | 不晋级为组合因素 | closed |
| INT-03 | major | A4样本范围 | A4仅360/420窗口 | 99.72%不能直接对比完整范围94.52% | 必须同时报告coverage/common-scope | mitigated |
| INT-04 | major | A4归因 | A4同时改变P2训练范围与归一化统计 | 不能把3.06个百分点完全归因于一个环节 | 后续增加“全训练、稳定测试”正式arm或保存预测 | open |
| INT-05 | blocking | 独立确认 | 仅seed 42且P3 test已查看 | 不能形成确认性或论文级泛化结论 | 新日期/新重复数据确认 | open |
| INT-06 | major | 时间边界 | 使用名义通气时间 | 0–150 s定位存在边界误差 | 用精确边界重建最终数据 | open |
| INT-07 | blocking | REC-A2完整性 | 初次attempt失败，retry未完成 | 目标监督效果未知 | 仅纳入完成且audit valid的retry | open |

## Leakage assessment

- calibration与test原始时间窗口通过purge避免重叠。
- 归一化统计只由P2 source train拟合。
- checkpoint固定为第25轮，P3 test不参与选轮。
- 但P3 test此前已被查看，后续组合选择存在post-hoc偏倚风险；当前结果保持探索性。

## Baseline, completeness, and reproducibility assessment

- A1/A3/A5具有相同420窗口测试范围，可进行当前seed下的描述性单变量比较。
- A4必须作为“稳定范围协议”单独报告，不能与完整范围排名。
- 四项均有25轮checkpoint、正式summary和valid postflight audit。
- 缺少多seed与独立采集批次，不能评价统计稳定性。

## Verdict: blocked

允许内部阶段性结论：

1. 完整范围下，电导和去CH2没有超过A1；
2. 早期0–150 s窗口是明显困难范围；
3. 稳定范围协议在共同360窗口上表现显著更好。

阻止将这些表述升级为确认性泛化结论，直到REC-A2完成、精确边界确认，并在新数据或
独立重复上验证。

## Unknowns and handoff

- 等待REC-A2 retry2完成并审计。
- 下一步不应立即组合电导或去CH2。
- 若目标是完整时间覆盖，应优先设计早期窗口专门处理，而不是只报告稳定段99.72%。
