# 实验室三气体 P1+P2→P3 Fold 1 完成审计

## Audit scope and intended claim

审计对象为 `lab3gas_nominal_P12_to_P3_fold1_s42_r25le3`。本审计分别回答：

1. 训练与 postflight 是否完整、可复现；
2. 该结果能否支持严格的“仅用 P1/P2 训练、P3 calibration 适配、P3 test
   独立评估”结论；
3. 该结果能否作为实验室三气体五折总体性能。

## Compared experiments

| Experiment ID | Split | Model | Checkpoint | DA | Calibration | QC | Seeds | Provenance |
|---|---|---|---|---|---|---|---|---|
| `LAB3GAS-P12-P3-F1-S42-R25LE3-BASE` | fold 1：train groups 3/4/5，calibration group 2，test group 1 | `strong_cls`，`(100,6)`，3 类 | round 8 base | none at selected checkpoint | source calibration 选 round | none | 42 | training source `4354e9f3…46e60` |
| `LAB3GAS-P12-P3-F1-S42-R25LE3-DA` | 同上 | 同上 | round 8 adapted | `fixed_da_strong`，100 steps/round | P3 calibration-assisted server DA | none | 42 | recovery evaluator `f3f674df…9c22a` |

## Findings

| Finding ID | Severity | Check | Evidence | Impact | Required action | Status |
|---|---|---|---|---|---|---|
| F-01 | informational | 训练完整性 | 25 个 base checkpoint、25 个 adapted checkpoint；history 25 rounds | 训练完整 | 无 | pass |
| F-02 | informational | 客户端参与 | 每轮 fit/evaluate 均为 2 clients，累计 fit/evaluate failures 均为 0 | 无缺失客户端 | 无 | pass |
| F-03 | informational | DA 配置 | 每轮 100 steps；所有 history 数值有限；adapted checkpoint 发生变化 | DA 实际执行 | 无 | pass |
| F-04 | informational | round 选择 | 25 个 round 仅以 P1/P2 calibration exposure Macro-F1 选 round；锁定 round 8 | 未使用 P3 test 选轮次 | 无 | pass |
| F-05 | informational | target-test 边界 | 仅 selected round 的 base/adapted 各生成一个 P3 test 结果；无 per-round target-test 文件 | 未发现 test tuning/selection | 无 | pass |
| F-06 | minor | postflight 恢复 | 原训练 runtime 漏打包 `train_centralized_baseline.py`；失败日志被保留；使用新内容寻址 evaluator 在新目录恢复 | 不改变 checkpoint 或选择规则，但必须保留双 runtime provenance | 后续 source freeze 加入依赖闭包测试 | closed-with-provenance |
| F-07 | blocking | 目标数据预处理边界 | `normalization_fit_scope=all_clients_train_only`；归一化统计包含 P3 train groups 3/4/5 | 不满足“目标板仅 calibration 可见”的严格跨平台协议；结果带有目标域无标签特征参与 | 按实验方向用 source-train-only 统计重建数据，再从头跑五折 | open |
| F-08 | major | 精确时间边界 | 数据集仍为 `nominal_schedule` | 只能作为名义边界 screening，不能作为最终实验室性能 | 精确边界确认后重建最终数据 | open |
| F-09 | blocking | 总体完整性 | 当前只有 fold 1、seed 42；P1+P2→P3 其余四折和 P2→P3 均缺失 | 不能报告五折均值、方差或两方向结论 | 修正 normalization 后完成锁定矩阵 | open |
| F-10 | major | server DA 数据本地性 | 云 A 持有 P1/P2 calibration arrays 作为 source DA loader | 不能宣称 source raw windows 全部不出端 | 报告中明确边界；若需严格 FL，另立 prototype/statistics-only DA | open |

## Leakage assessment

训练后的 checkpoint 选择和 P3 test 打开顺序通过审计，没有发现
target-test 用于训练、DA、停止或 round 选择。但是预处理阶段的 Z-score
统计由三个平台的 train groups 联合拟合，包含目标 P3 的 train features。
这是无标签目标特征参与，不是 test-label leakage，但与预期的
source-only cross-platform evaluation 不一致，因此阻塞严格结论。

## Baseline, completeness, and reproducibility assessment

运行配置、checkpoint、恢复 evaluator 和数据版本均有 SHA/路径记录，fold 1
本身可复现。base 与 adapted 使用同一 selected round、同一 P3 样本，比较是
配对的。当前只有一个 fold、一个 seed 和 6 个目标 test exposures，不能计算
可靠的跨 fold/seed 不确定性，也不能把 138 个重叠窗口视为 138 个独立重复。

## Verdict: blocked

执行完整性为 `valid`，fold-1 名义边界描述性 screening 可以分析；但由于
target-inclusive normalization、名义时间边界以及剩余 folds/direction 缺失，
该结果不能升级为严格跨平台或总体三分类 Evidence。

## Unknowns and handoff

- source-only normalization 后 fold 1 的性能变化：unknown。
- 精确通气边界数据上的性能：unknown。
- 五折均值、SD、CI：unknown。
- P2→P3 相对 P1+P2→P3 的差异：unknown。
