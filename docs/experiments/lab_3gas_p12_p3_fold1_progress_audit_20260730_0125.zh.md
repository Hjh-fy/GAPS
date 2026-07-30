# 实验室三气体 P1+P2→P3 Fold 1 运行中审计

## Audit scope and intended claim

本记录是 2026-07-30 01:25（Asia/Shanghai）的运行中快照，只判断
`lab3gas_nominal_P12_to_P3_fold1_s42_r25le3` 是否仍按锁定配置正常推进。
它不批准分类性能 Evidence，也不报告未完成实验的最终指标。

## Compared experiments

| Experiment ID | Split | Model | Checkpoint | DA | Calibration | QC | Seeds | Provenance |
|---|---|---|---|---|---|---|---|---|
| `LAB3GAS-P12-P3-F1-S42-R25LE3` | exposure-group-aware 5-fold；fold 1；train/calibration/test = 3/1/1 个浓度序号组 | `strong_cls`，输入 `(100,6)`，3 类，1 phase | 运行中；已生成 base round 1–21、adapted round 1–20 | `fixed_da_strong`，100 steps/round，P3 calibration-assisted | source/P3 calibration；P3 test 尚未打开 | none | 42 | source archive `4354e9f3…46e60`；dataset manifest `1c49a808…24dbb` |

## Findings

| Finding ID | Severity | Check | Evidence | Impact | Required action | Status |
|---|---|---|---|---|---|---|
| F-01 | blocking | 25-round completeness | 云 A 当前 base=21、adapted=20，round 21 正在执行 DA | 尚不能批准结果或运行 postflight | 等待 25 轮、评估和回收完成 | open |
| F-02 | informational | 三端进程身份 | server PID 1179907；C2 PID 98688；C1 PID 13212；run tag、client ID、fold、LE3 与协议一致 | 未发现错机或错客户端 | 继续监控 | pass |
| F-03 | informational | 客户端参与及失败 | round 20 聚合评估收到 2/2、0 failures；round 21 两客户端 fit 均完成 | 当前通信和客户端参与正常 | 完成后检查全部 25 轮 history | pass-so-far |
| F-04 | informational | 目标 test 边界 | 客户端 `eval_split=calibration`；尚无 `formal_evaluation` | 当前未发现训练期打开 P3 test | 完成后核对仅 selected round 产生 target-test 输出 | pass-so-far |
| F-05 | informational | Pi 健康 | 01:25 温度 46.1°C，`throttled=0x0`；此前监控约 45.5–49.9°C | 未发现热降频 | 继续监控至退出 | pass-so-far |
| F-06 | major | server DA 数据边界 | 云 A 命令使用 P1/P2 calibration 作为 `server_val_data`，P3 calibration 作为 `server_calib_data` | 当前实现不是“所有 source raw windows 严格不出端”的纯 FL 数据边界 | 结果叙述必须明确；若需严格数据本地化，应另立算法变体 | open |
| F-07 | informational | stderr | 本地 controller stderr 为 0 | 未发现控制器错误 | 完成后再次核对 | pass-so-far |

## Leakage assessment

当前客户端每轮评估明确使用 calibration。目标 P3 test 尚未打开；计划是在所有
25 个 round 仅按 source calibration exposure Macro-F1 锁定一个 round 后，再对
该 round 的 unadapted/adapted checkpoint 各读取一次 P3 test。当前没有发现
target-test 用于训练、DA、停止或 round 选择的证据。

## Baseline, completeness, and reproducibility assessment

配置身份与计划一致：P1/P2 为 source clients，P3 为 target；25 rounds、LE3、
batch 32、seed42、3 classes、6 channels、100 DA steps/round。当前仅 fold 1
且尚未完成，其他四折和 P2→P3 扩展尚未产生，因此既不能形成五折结果，也不能
支持两种 source 配置的公平比较。

## Verdict: blocked

这是“运行正常但尚未完成”的阻塞状态，不是实验失败。只有 25 个 base/adapted
checkpoint、完整 history、锁定后的 target-test 评估、结果回收和
`validate_three_node_run.py` 全部通过后，fold 1 才能进入 `audited`。

## Unknowns and handoff

- round 21–25 的最终完成状态：unknown。
- selected round 与 P3 test 指标：unknown。
- 五折均值、离散度和 P1+P2→P3 vs P2→P3 比较：unknown。
