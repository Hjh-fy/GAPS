# IoT-J H1 sufficient-statistics federated equivalence 正式结果

## A. 正式运行与协议

- 正式代码 commit：`e12a4eb61180c8819b9f6e87dee71b103ac040a8`；
- 运行时 local HEAD 与 origin HEAD 完全一致；
- source 为 C1/C2，target 为 C5，classifier 为 frozen B5 seed42；
- source train/calibration 为 4720/640 行，C5 calibration/test 为
  320/1360 行；
- C5 test 不参与 source scaler、source alpha、source refit、target alpha
  或 target refit；
- runtime v4、row map、HC95/HC90 两套 parity report/runtime rows 六个冻结
  SHA256 在运行前后完全一致。

## B. 精确数学与实现

现有 H1 是 104D、四气体独立的自定义闭式 Ridge，而不是 sklearn Ridge。
pooled 路线在 C1+C2 train 上计算总体均值/标准差，显式添加常数列，并求：

`beta = pinv(D^T D + diag(0, alpha, ..., alpha)) D^T y`

截距不受惩罚。alpha 网格为
`[0, 0.01, 0.1, 1, 10, 100, 1000]`，以 pooled C1+C2 calibration
RMSE 选择；随后在 train+calibration 上 refit。验证和推理预测均裁剪到对应
fit 标签范围。

federated-stats 路线先由 C1/C2 分别产生
`n_i, sum_x_i, sum_x2_i`，服务器重构共同 scaler 并广播；客户端再产生
`A_i=D_i^T D_i, b_i=D_i^T y_i`，服务器只聚合这些统计量重构候选 Ridge。
alpha 选择时客户端仅返回 calibration SSE/count。最终 refit 在各客户端本地
train+calibration 上重复同一协议。服务器 API 不接受 raw rows、raw X 或 raw y。

## C. 等价性判定

正式结论为 **`PRACTICAL_EQUIVALENCE`**，不是 `EXACT_EQUIVALENCE`：

| 项目 | 最大误差 | 预注册 exact 门槛 | 结果 |
|---|---:|---:|---|
| scaler mean/std | `1.9539925233402755e-14` | `1e-10` | 通过 |
| coefficient/intercept | `1.7438189274798788e-06` | `1e-8` | 未通过 |
| C5 H1 prediction | `2.0809125089726876e-08 ppm` | `1e-6 ppm` | 通过 |
| Ridge+H1 S_ALL RMSE 差 | `1.1747758321689616e-10 ppm` | practical `0.01 ppm` | 通过 |
| Ridge+H1 S_CC RMSE 差 | `5.32196509084315e-12 ppm` | practical `0.01 ppm` | 通过 |

四气体 alpha 全部相同：Ethanol `0`、CO `0.01`、Ethylene `0`、Methane
`0`。最大系数差来自 Methane 的 alpha=0 伪逆解；分块充分统计量聚合造成极小
浮点扰动，但在病态/非唯一参数表示中被 `pinv` 放大。没有为获得
`EXACT_EQUIVALENCE` 放宽门槛，也没有修改 pooled baseline。参数表示未通过
exact 门，但输出和下游指标达到预注册 practical 门。

## D. C5 regression 结果

| Variant | Cal-val RMSE | Test S_ALL RMSE | MAE | NRMSE | S_CC RMSE |
|---|---:|---:|---:|---:|---:|
| Ridge rich-only | 17.222114 | 25.898477 | 10.980890 | 0.196839 | 14.201930 |
| Ridge + H1 pooled | 15.394324 | 25.648978 | 9.383748 | 0.204295 | 11.341599 |
| Ridge + H1 federated-stats | 15.394324 | 25.648978 | 9.383748 | 0.204295 | 11.341599 |

pooled 数值复现了前一阶段的正式 E2 Ridge+H1 reference（约
15.394/25.649/11.342）。

## E. 分气体结果

| Gas | Pooled RMSE | Federated-stats RMSE | 解释 |
|---|---:|---:|---|
| Ethanol | 26.065824 | 26.065824 | 数值一致 |
| CO | 22.665000 | 22.665000 | 数值一致 |
| Ethylene | 35.505157 | 35.505157 | 数值一致 |
| Methane | 13.331720 | 13.331720 | 参数表示差最大，但预测一致 |

CO-high 200–250 ppm RMSE 为 pooled `35.0212127843`、federated-stats
`35.0212127845`；差异不具实际意义。

## F. 隔离、泄漏与异常审计

- 相关测试 `27 passed`；
- B5 route parity 为 calibration-validation `80/80`、test `1360/1360`，
  route mismatch 均为 0；
- C1/C2 统计量拥有独立 provenance；修改 C1 原始输入只改变 C1 统计量，
  C2 同理；
- C5 calibration 未参与 source H1 训练；C1/C2 source test 未用于 source
  train/select；C5 test 未用于任何选择或 refit；
- 没有 NaN/Inf，没有覆盖已有结果，没有修改 runtime v4 或 QC；
- 唯一需要报告的数值异常是 alpha=0 下 `pinv` 参数表示对浮点聚合顺序敏感，
  因此不能写成 exact coefficient equivalence。

## G. Evidence boundary 与建议

本实验支持如下有限表述：

> source raw samples remain local and only aggregated sufficient statistics
> are used to reconstruct the global Ridge solution.

本实验不支持 secure aggregation、差分隐私、密码学隐私、充分统计量不泄漏，
也不等于真实 Flower regression network closure。

建议：保持 runtime v4 不变。H1 sufficient-statistics 路线可作为
**practically equivalent simplified source-reference candidate**，但由于
coefficient exact 门未通过且当前只有 seed42，不应直接提升为最终论文主方法。
若后续要强化该证据，只进行独立授权的 multi-seed numerical confirmation；
本阶段到此停止。

正式结果目录：
`results/iotj_h1_federated_ridge_equivalence_20260724/`。
