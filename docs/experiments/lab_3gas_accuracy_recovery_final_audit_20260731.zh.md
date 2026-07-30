# 实验室三气体准确率恢复：完整实验审计

## Audit scope and intended claim

审计 REC-A1/A2/A3/A4/A5 是否完成、是否可公平比较，以及它们能否支持“实验室自测
P2→P3 三气体分类的提准结论”。现有结果资产只读，失败 attempt 不删除、不替换。

## Compared experiments

| Experiment ID | Split | Model | Checkpoint | DA | Calibration | QC | Seeds | Provenance |
|---|---|---|---|---|---|---|---|---|
| REC-A1-CB2 | all-concentration time-purged；420 test | proto_replay，6ch | round 25 adapted | corrected B2 | P3 90 窗口 | valid | 42 | formal summary + postflight |
| REC-A2-TCE-RETRY2 | 同 A1；420 test | 同 A1 | round 25 adapted | corrected B2 + target CE=1.0 | P3 90 窗口 | valid | 42 | formal summary + history + postflight |
| REC-A3-COND | 同 A1；420 test | exact conductance，6ch | round 25 adapted | corrected B2 | P3 90 窗口 | valid | 42 | formal summary + postflight |
| REC-A4-STABLE150 | stable scope；360 test | proto_replay，6ch | round 25 adapted | corrected B2 | P3 90 窗口 | valid | 42 | formal summary + postflight |
| REC-A5-NOCH2 | 同 A1；420 test | proto_replay，5ch | round 25 adapted | corrected B2 | P3 90 窗口 | valid | 42 | formal summary + postflight |

## Findings

| Finding ID | Severity | Check | Evidence | Impact | Required action | Status |
|---|---|---|---|---|---|---|
| FIN-01 | informational | REC-A2 完整性 | retry2 完成 25/25；每轮 fit/evaluate failures=0；postflight=`valid` | 原“REC-A2 未完成”阻塞已解除 | 保留 attempt0/retry1 失败记录 | closed |
| FIN-02 | informational | checkpoint 选择 | 所有纳入比较实验固定 round 25；source calibration 仅监控 | 无最早满分轮选择偏差 | 后续继续固定 final round | closed |
| FIN-03 | informational | A1 vs A2 单变量 | 同数据/模型/seed/轮数；target CE 0→1；397/420 vs 397/420 | 固定目标 CE 没有净提升 | 不组合晋级 | closed |
| FIN-04 | informational | runtime 差异 | 两个 archive 的源文件差异仅为 evaluate 日志和 failure details | 不改变算法比较，但 archive 身份必须保留 | 报告两个 SHA | closed |
| FIN-05 | major | A4 样本范围 | A4 为 360/420=85.71% coverage | 99.72% 不可与全范围 94.52%直接排名 | 同报 coverage 和 common-scope | mitigated |
| FIN-06 | major | A4 归因 | A4 同时改变 P2 train 窗口与 P2-only normalization | +3.06pp 不能归因于单一处理 | 增加全训练/稳定测试等拆分消融 | open |
| FIN-07 | blocking | 独立确认 | 仅 seed 42；P3 test 已用于多轮 post-hoc 分析 | 不支持确认性泛化或论文主结论 | 新采集/独立留出与多 seed 验证 | open |
| FIN-08 | major | 时间边界 | 仍使用名义通气时间 | 早期 0–150 s 定位可能受真实阀切换延迟影响 | 用精确日志重建边界 | open |
| FIN-09 | informational | 数据泄漏 | normalization 仅用 P2 train；calibration/test 有 purge；P3 test 不选轮 | 未发现直接重叠或选轮泄漏 | 保持协议 | closed |

## Leakage assessment

- P2 normalization statistics 只由 P2 source train 拟合。
- P3 calibration 与 P3 test 的重叠相邻窗口已 purge。
- checkpoint 固定为配置前确定的第 25 轮；P3 test 不参与停止或选轮。
- 但 P3 test 已在消融设计前后被多次查看，后续在同一 test 上继续选择组合方案会形成
  post-hoc 选择偏差。现有结果只能作为探索性证据。

## Baseline, completeness, and reproducibility assessment

- A1/A2/A3/A5 共享 420 窗口测试范围，可做 seed 42 下的描述性单变量比较。
- A4 只能在 360 窗口 stable common-scope 和 coverage 约束下报告。
- REC-A2 retry2 的 server run config 确认：
  `rounds=25`、`profile=proto_replay`、`input_dim=6`、
  `domain_adapt_steps=100`、`lambda_target_ce=1.0`、
  `use_adapted_as_global=true`。
- P2 客户端为 Cloud B，P3 calibration/DA 与 Flower server 为 Server A；树莓派 P1 在
  P2→P3 实验中按协议不参与。
- REC-A2 attempt0 与 retry1 是失败运行，不可并入性能统计；retry2 是唯一完成的 A2 结果。

## Verdict: blocked

实验执行层面已经完成，可以形成以下内部探索性结论：

1. 完整 420 窗口下，A2 固定目标 CE、A3 精确相对电导和 A5 去 CH2 均未超过 A1。
2. A4 在 85.71% 稳定段覆盖下达到 359/360=99.72%，是当前最强稳定段协议。
3. 完整覆盖的主要剩余困难仍是气体刚通入后的早期动态窗口，而非整次 exposure 分类。

证据批准仍为 `blocked`：单 seed、重复查看 P3 test、名义时间边界和缺少独立新数据，
使这些结果不能升级为确认性论文结论。

## Unknowns and handoff

- 优先级一：取得精确气路切换时间，重建或至少复核 0–150 s 标签边界。
- 优先级二：在不继续用当前 P3 test 调参的前提下，设计早期窗口专门处理，并保留
  完整覆盖率指标。
- 优先级三：在新采集或真正独立的 P3 holdout 上做多 seed 冻结验证。
- 不建议当前直接组合 conductance、no-CH2 或 target-CE，因为三个单变量均未带来
  完整范围净提升。
