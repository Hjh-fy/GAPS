# IoT-J H1 pooled-to-federated sufficient-statistics equivalence audit

## 1. 审计范围

本审计只验证现有正式 H1 pooled-source Ridge 能否由 C1/C2 本地充分统计量
数值重构。runtime v4、HC95/HC90、B5 classifier、现有 H1/H2/H3 资产及 QC
均保持只读；不实现 Flower regression，也不开展 Pi、multi-seed 或
low-calibration。

预注册判定：

- `EXACT_EQUIVALENCE`：四气体 alpha 相同，scaler 最大误差不超过
  `1e-10`，系数/截距最大误差不超过 `1e-8`，C5 H1 prediction 最大误差
  不超过 `1e-6 ppm`。
- `PRACTICAL_EQUIVALENCE`：H1 prediction 最大误差不超过 `1e-3 ppm`，
  且 Ridge+H1 的 S_ALL/S_CC RMSE 差均不超过 `0.01 ppm`。
- 否则为 `NOT_EQUIVALENT`。

## 2. 现有 H1 的精确定义

代码事实来自：

- `run_regression_head_ablation.py`
- `run_source_lightweight_regression_head_ablation.py`
- `run_source_augmented_target_ridge_eval.py`
- `scripts/evaluate_iotj_source_prior_target_head_factorial.py`

H1 是四个按真实气体类别独立训练的 source Ridge：

1. 输入为 `rich_feature_dict` 的排序后 104D 特征；
2. source fit 使用 C1+C2 train，source validation 使用 C1+C2 calibration；
3. 每一气体、每一特征在 pooled train 上计算均值与总体标准差
   （NumPy `nanstd` 默认 `ddof=0`）；非有限输入以均值替代，标准差绝对值
   小于 `1e-9` 时置为 1；
4. 标准化后显式添加常数列，目标函数为
   `pinv(D^T D + diag(0, alpha, ..., alpha)) D^T y`，因此截距不惩罚；
5. alpha 网格固定为 `0, 0.01, 0.1, 1, 10, 100, 1000`，以 C1+C2
   calibration 全局 RMSE 最小者为准；验证预测裁剪到 train 标签
   `[min(y), max(y)]`；
6. 选定 alpha 后在 C1+C2 train+calibration 上重新计算 scaler 并 refit；
   推理继续裁剪到 refit 标签范围；
7. 四种气体完全独立。

这里不是 sklearn `Ridge` 或 sklearn `StandardScaler`；正式语义是上述自定义
NumPy 闭式解。

## 3. 联邦充分统计量协议

每气体独立执行：

1. C1/C2 本地发送 `n_i, sum_x_i, sum_x2_i`；服务器得到
   `mu = sum_i(sum_x_i)/N`，
   `sigma = sqrt(sum_i(sum_x2_i)/N - mu^2)`，再按既有规则处理近零尺度。
2. 服务器广播共同 scaler。客户端在标准化设计矩阵上计算
   `A_i=D_i^T D_i, b_i=D_i^T y_i`，并发送标签范围 extrema。
3. 服务器仅由 `sum_i A_i, sum_i b_i` 和 alpha 重构 Ridge。
4. alpha 选择时，服务器广播候选参数；每个客户端只返回 calibration
   `SSE_i` 与 `n_i`，服务器以 `sqrt(sum SSE_i / sum n_i)` 选择 alpha。
5. final refit 对各客户端本地 train+calibration 重复同一两阶段协议。

服务器聚合 API 不接受 raw row list、raw X 或 raw y。充分统计量分别绑定
C1/C2 独立 provenance；隔离测试验证修改某一客户端原始输入只改变该客户端
统计量。

## 4. C5 比较与泄漏边界

统一 frozen B5 seed42 route 和 C5 320/1360 split：

- A：正式冻结 `Ridge + 104D rich`；
- B：`Ridge + 104D rich + H1_POOLED`；
- C：`Ridge + 104D rich + H1_FEDERATED_STATS`。

B/C 的 C5 calibration fit/validation、alpha 网格、refit、clipping 和气体路由
完全相同。C5 test 只在模型与判定门冻结后打开；C5 calibration 不进入 source
H1 训练；C1/C2 source test 不进入 source fit 或选择。B5 route 必须通过
80/80 calibration-validation 与 1360/1360 test parity。

## 5. Evidence boundary

允许的准确表述仅为：

> source raw samples remain local and only aggregated sufficient statistics
> are used to reconstruct the global Ridge solution.

本实验不声称 secure aggregation、差分隐私、密码学隐私，也不声称充分统计量
本身不泄漏信息。

## 6. 运行前状态

- local HEAD = origin HEAD =
  `48a0b13ab49af17a841b867078b40040c084b862`；
- runtime v4、row-map、HC95/HC90 两套 parity report/runtime rows 六个
  SHA256 与正式审计完全一致；
- 正式结果将在唯一新目录
  `results/iotj_h1_federated_ridge_equivalence_20260724/` 生成。
