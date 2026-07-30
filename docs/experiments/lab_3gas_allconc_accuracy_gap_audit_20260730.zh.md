# 实验室全浓度 P2→P3 与公共分类结果可比性审计

## Audit scope and intended claim

审计两个问题：

1. 是否可以把实验室 91.90% 与公共 98.90% 的约 6.99 pp 差解释为同一方法性能下降；
2. 当前全浓度运行是否已经满足正式审计通过条件。

## Compared experiments

| Experiment ID | Split | Model | Checkpoint | DA | Calibration | QC | Seeds | Provenance |
|---|---|---|---|---|---|---|---|---|
| `LAB3GAS-ALLCONC-P2P3-S42` | all-concentration, raw-time-purged within exposure | 3-class/6-channel `strong_cls` | source-selected round 2 adapted | legacy fixed strong | P3 90 windows/30 exposures | none | 42 | remote summary SHA fixed |
| `PUBLIC-C1-C5-B2-S42` | class×concentration window-level 20/80 | 4-class/8-channel `proto_replay` | final round 25 adapted | corrected B2 | C5 320 windows, same 80 files as test | none | 42 | canonical public summary |

## Findings

| Finding ID | Severity | Check | Evidence | Impact | Required action | Status |
|---|---|---|---|---|---|---|
| `GAP-01` | blocking | 相同任务/数据 | 3 类实验室气体与 4 类公共气体不同 | 不能把差值归因于模型退化 | 仅在同一实验室数据内做消融 | open |
| `GAP-02` | blocking | 相同 split 难度 | 公共 calibration/test 共享 80/80 原始文件；实验室原始时间交集为 0 | 公共任务明显更接近同文件窗口内插 | 分开命名任务和结果 | open |
| `GAP-03` | major | checkpoint 一致 | 实验室 round 2；公共 round 25 | 差值混入训练时长/选轮规则 | 预声明固定 round 25 对照 | open |
| `GAP-04` | major | DA 一致 | legacy strong 与 corrected B2 有 10 项关键参数不同 | 差值混入服务端目标函数 | matched corrected-B2 消融 | open |
| `GAP-05` | major | 预处理一致 | 实验室实现相对电阻；公共合同为相对电导 | 强响应被压缩且源/目标饱和比例不同 | 从原始 R 重建 matched 数据 | open |
| `GAP-06` | major | 时间覆盖一致 | 实验室使用 0–1200 s；公共为 60–170 s time-aware | 实验室包含输气瞬态和长时间漂移 | 报告固定延迟/onset 对照及覆盖率 | open |
| `GAP-07` | major | 数据预算一致 | 420 vs 2360 source windows；90 vs 320 target calibration | 实验室估计方差和域适配资源更少 | 报告数据量，不做直接上限比较 | open |
| `GAP-08` | blocking | postflight | 旧审计器硬编码 138 windows/6 exposures | 当前运行不能标记为 audited/approved | 使用数据目录动态推导 90/30 与 420/30，并生成新审计文件 | local fix ready |
| `GAP-09` | major | 测试独立性 | 当前 P3 test 已被查看并用于本轮诊断 | 后续调参结果属于 post-hoc 开发 | 冻结组合后用新日期/重复实验确认 | open |
| `GAP-10` | informational | 系统级结果 | adapted exposure Accuracy=30/30 | 气体暴露级识别已正确 | 同时报 window/exposure 指标 | resolved |

## Leakage assessment

- 实验室 P3 calibration/test 的原始时间点交集为 0；当前未发现直接窗口泄漏。
- calibration/test 仍共享物理 exposure，因此属于同暴露内泛化，不是独立重复实验泛化。
- 公共 C5 calibration/test 由合并后的窗口随机重分，80 个 calibration 文件全部也出现在 test；
  该协议可作为既有公共实验，但不能被描述为跨文件独立测试。
- 当前实验按 P2 calibration 选轮，P3 test 在选中 round 2 后才打开；未发现 target-test 选轮泄漏。

## Baseline, completeness, and reproducibility assessment

- 25 个基础 checkpoint 和 25 个 adapted checkpoint 完整，Flower fit/evaluate 无客户端失败。
- 正式 summary 已生成，样本数与新数据集合同一致。
- 只有 seed 42；稳定性为 `unknown`。
- 当前旧 postflight 失败源于数据合同硬编码，不是训练或评估缺失；但在新审计文件生成前仍不能批准。

## Verdict: blocked

`91.90%` 与 `98.90%` 可以并列作为两个不同协议的描述性结果，但不能支持
“同一方法在实验室数据上下降约 7 pp”的因果说法。

全浓度训练和评估已经计算完成，但正式 Evidence 状态仍为 `blocked`：
必须使用动态 target 数据合同产生新的、不可覆盖的 postflight audit，并保留旧失败审计记录。

## Unknowns and handoff

- round-25 matched 结果：unknown。
- corrected B2 matched 结果：unknown。
- 正确相对电导、去 CH2、固定 150 s 延迟的正式结果：unknown。
- 精确 onset 和独立新批次确认：unknown。
