# 实验室三气体准确率恢复：阶段结果分析

## Input contract and provenance

- 实验：REC-A1-CB2、REC-A3-COND、REC-A4-STABLE150、REC-A5-NOCH2。
- 方向：P2/C2 → P3/C3；全保留浓度；三分类；seed 42；25轮；本地3轮。
- checkpoint：固定 `server_round_025_adapted.pth`，不使用source calibration选轮。
- 正式指标均复制自各实验 `formal_evaluation_summary.json`，标记为 `reported`。
- A1模型在A4稳定测试范围上的补充评估标记为 `recomputed diagnostic`，SHA256：
  `25A0E1436F53D976E177921BAFA1B6DB1932938294815D3566CC1ADD13B833EA`。

## Descriptive statistics

| 实验/诊断 | 窗口数 | 正确数 | Accuracy | Macro-F1 | 暴露级Accuracy | 样本范围 |
|---|---:|---:|---:|---:|---:|---|
| REC-A1 corrected B2 | 420 | 397 | 94.52% | 94.50% | 100% | 完整test |
| REC-A3 精确相对电导 | 420 | 393 | 93.57% | 93.60% | 100% | 完整test |
| REC-A5 去除CH2 | 420 | 394 | 93.81% | 93.83% | 100% | 完整test |
| A1 checkpoint稳定范围诊断 | 360 | 348 | 96.67% | 96.67% | 100% | 与A4相同test |
| REC-A4 稳定范围训练协议 | 360 | 359 | 99.72% | 99.72% | 100% | 稳定test |

仅有一个seed，且同一暴露内窗口相关，因此不把360/420个窗口视为独立实验重复，
不计算跨seed SD、置信区间或显著性检验。

## Assumptions, comparisons, effect sizes, and corrections

### 完整420窗口上的同范围比较

- REC-A3相对REC-A1少正确4个窗口，Accuracy下降0.95个百分点。
- REC-A5相对REC-A1少正确3个窗口，Accuracy下降0.71个百分点。
- 当前批次不支持“精确相对电导单独提高GAPS准确率”。
- 当前批次不支持“单独去除CH2提高GAPS准确率”。
- 三个实验的暴露级Accuracy均为100%，说明差异集中在暴露内窗口。

### REC-A4稳定范围

REC-A4保留全部30次P3暴露、全部三气体、v1/v2和全部保留浓度，只改变每次暴露
内参与source train和target test的时间窗口：

- P2 train：420 → 360窗口；
- P3 calibration：保持90窗口；
- P3 test：420 → 360窗口；
- 每次P3暴露：14 → 12个test窗口；
- 每类：120个test窗口；每个版本：180个test窗口；
- 覆盖率：360/420 = 85.71%。

保留的基础索引与目标气体起点相对时间为：

| 索引 | 时间范围 |
|---:|---|
| 5–9 | 250–350、300–400、350–450、400–500、450–550 s |
| 13–17 | 650–750、700–800、750–850、800–900、850–950 s |
| 21–22 | 1050–1150、1100–1200 s |

被REC-A4额外排除的是索引0、1，即0–100 s和50–150 s。索引3、11、19继续用于
P3 calibration；与校准窗口重叠的相邻索引继续purge。

同一个A1模型在完整test上为94.52%，在稳定360窗口上为96.67%。因此仅改变评估
范围即可提高2.14个百分点；被排除的60个早期窗口中正确49个，准确率81.67%。

在相同360窗口上，REC-A4为99.72%，比A1稳定范围诊断高3.06个百分点，多正确11个
窗口。该差异对应完整“稳定范围训练+P2-only重新归一化”协议，不能仅归因于删除
测试窗口。

REC-A4混淆矩阵仅剩1个乙酸→乙醛错误：

```text
[[120, 0,   0],
 [  0, 120, 0],
 [  1, 0, 119]]
```

## Anomalies and sensitivity analysis

- REC-A2-TCE初次attempt在第1轮客户端evaluate failure后失去注册；第1轮适配
  checkpoint可独立加载并在P2 calibration达到90/90，故不支持“模型checkpoint
  损坏”的解释。该attempt保留为失败记录。
- REC-A2 retry1因旧SSH反向隧道遗留占用18080端口而在训练前失败，无checkpoint。
- REC-A2 retry2使用新run ID、新内容寻址runtime和fail-fast监控重新运行。

## Proposed paper tables and figures

- 表：完整420窗口单变量消融（A1/A3/A5），单列报告暴露级Accuracy。
- 表：A1完整范围、A1稳定common-scope、A4稳定协议，同时报告窗口数与coverage。
- 图：按目标气体起点相对时间绘制窗口错误率；需要正式保存逐窗口预测后再制作。

## Unknowns, conflicts, and audit handoff

- 只有seed 42，跨seed稳定性未知。
- 当前P3 test已在前期分析中查看，结果仅能作为post-hoc探索证据。
- 边界仍以名义时间为主，精确进气时间尚未确认。
- 尚无正式逐窗口预测流，不能把所有正式错误精确定位到某个时间索引。
- REC-A2尚未完成，不进入本阶段性能排序。
