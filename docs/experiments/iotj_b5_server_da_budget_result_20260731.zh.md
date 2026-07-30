# GAPS IoT-J B5 Server-DA 步数敏感性结果

## 实验身份

本实验固定 B5、seed 42、C1/C2→C5、25 个联邦轮次、本地训练
1 epoch/round、batch size 32、Client Adam 和学习率 5e-4。唯一变化为
服务器适配步数：100、80、50、30 steps/round，对应总步数 2500、2000、
1250、750。

DA100 复用既有 LE1 结果；DA80、DA50、DA30 按顺序运行。C5 test 只用于
固定配置评价，不参与拟合、早停、checkpoint 选择或步数选择。本实验是冻结后
seed 42 单种子敏感性分析，不替代 B5 五种子正式证据。

## 结果

| 配置 | Accuracy | Macro-F1 | NLL | ECE | 错误数 | 总耗时 | 相对 DA100 节省 | 0.5 pp 双指标门槛 | 证据身份 |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| DA100 | 98.9706% | 98.9712% | 0.076054 | 0.008967 | 14 | 1.338 h | 0% | PASS | existing canonical reference |
| DA80 | 98.5294% | 98.5303% | 0.105687 | 0.012612 | 20 | 1.127 h | 15.74% | PASS | canonical / postflight PASS |
| DA50 | 97.5000% | 97.5087% | 0.145886 | 0.019382 | 34 | 0.791 h | 40.84% | FAIL | canonical / postflight PASS |
| DA30† | 98.6029% | 98.6053% | 0.072129 | 0.007259 | 19 | 0.537 h | 59.83% | numeric PASS | non-canonical technical result |

† DA30 完成 25/25 轮、C1/C2 每轮参与、750 个 DA steps、round-25 adapted
checkpoint 严格加载和 1360 个唯一 row key 评估；但 C2 资源采样覆盖率为
0.948214，低于锁定 validator 下限 0.95，因此正式 postflight fail closed。

## 分析结论

1. DA80 是唯一同时通过正式观测审计并满足 Accuracy/Macro-F1 下降不超过
   0.5 个百分点的新增配置，可节省约 15.74% attempt wall time。
2. DA80 的 NLL 和 ECE 均劣于 DA100，因此不能概括为“完全无影响”。
3. DA50 的 Accuracy 和 Macro-F1 均下降约 1.46–1.47 个百分点，明显超出
   工程保持容差，不建议晋级。
4. DA30 数值上优于 DA50且落在 0.5 个百分点容差内，说明性能随步数并非
   单调变化；但其观测合同未通过，不能据此晋级或覆盖正式配置。
5. 当前仅为单种子结果，不进行显著性或统计非劣声明。

## 建议

保持 DA100 为冻结正式 B5 配置，不修改 runtime、QC 或论文冻结结果。DA80
可作为未来多种子确认的唯一正式候选；DA30 如需形成正式证据，只能在另行授权后
按相同锁定配置干净重跑，不能事后降低 0.95 覆盖率门槛。
