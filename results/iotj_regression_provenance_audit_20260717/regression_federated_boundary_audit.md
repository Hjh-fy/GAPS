# Regression Federated Boundary Audit

审计日期：2026-07-17
状态：只读 provenance audit；没有新回归训练。

## 1. 可安全称为 federated 的组件

### 1.1 真实 Flower 分类主线

C1/C2 classifier local training、模型更新聚合及服务器侧 DA 属于当前真实 ECS + Pi + PC Flower 分类系统。这是论文“federated”主张的核心边界。分类器输出 logits/probabilities/predicted route，随后进入回归链路。

### 1.2 R0 的离线 FedAvg source regression reference

R0 使用的 `R3aK16_source_regression.pt` 按代码设计由 C1/C2 本地 regression checkpoint 做 sample-count-weighted parameter averaging。它可称为“offline FedAvg source regression reference”。

限制：`gaps_flower/regression_server.py` 明确描述单机/文件式 checkpoint 聚合，审计未发现该 R0 producer 通过真实 Flower application messages 完成训练。因此不能把它写成“已审计的真实 Flower 回归训练”，也不能把它外推为 R4/H8 的训练方式。

## 2. 只能称为 multi-source regression reference 的组件

H1 per-gas Ridge、H2 per-gas MLP、H3 shared MLP 的正式实现均：

1. 在同一 Python 进程读取 C1 与 C2 的预构建 window tensors、classification/regression labels、phase 与 metadata；
2. 把两个客户端的数据 append 到同一个 `source_train`/`source_val` list；
3. 对 Ridge/MLP 直接调用集中式 fit；
4. 不发生 Flower message、client update、FedAvg 或参数聚合。

因此安全命名是：

- centrally pooled C1/C2 per-gas Ridge reference；
- centrally pooled C1/C2 per-gas MLP reference；
- centrally pooled C1/C2 shared MLP reference with gas one-hot；
- collectively: pooled multi-source regression references。

它们不能称为 federated Ridge、federated MLP 或 privacy-preserving source heads。

## 3. Target-personalized 边界

- R1：C5 calibration 上的 per-gas rich-feature Ridge。
- R2：C5 calibration 上的 per-gas H2.3 MLP anchor。
- R3：C5 calibration-validation 选择的 H2.3+/Ridge blend。
- R4/H8：C5 calibration 上拟合的 per-gas Ridge，输入为 104 个 C5 rich features 加三个 pooled source prediction。
- QC：C5 calibration-fit/calibration-validation 拟合 feature reference、risk calibrator 与 FULL/HC95/HC90 workpoints。

这些组件是 target-local/target-personalized central calibration，不是 federated training。C5 test 只用于冻结后的 prediction/evaluation；R7 例外地读取 test truth，因此必须保持 oracle diagnostic 标签。

## 4. 论文误导风险

| 风险表述 | 风险等级 | 原因 | 安全替代 |
|---|---|---|---|
| “GAPS 全流程均为联邦学习” | 高 | H1/H2/H3 与 C5 heads/QC 均为集中式拟合 | “GAPS combines federated classification with pooled multi-source regression references and target-personalized calibration.” |
| “整个系统不集中任何 source window” | 高 | H8 source heads 在同一进程读取 C1+C2 processed window arrays | 只把“不集中原始源窗口”限定于真实 Flower classifier training；回归段明确披露 pooled source windows |
| “H8 使用 federated Ridge/MLP” | 高 | H1/H2/H3 无消息交换或参数聚合 | “H8 uses centrally pooled C1/C2 source regression references.” |
| “R0 证明正式回归也是 Flower” | 高 | R0 是独立 baseline；producer 是离线 checkpoint FedAvg | “R0 is an offline FedAvg source regression reference; R4 uses different pooled heads.” |
| “formal R4 runtime 已冻结并完成 parity” | 高 | final C5 bundle/1360-row parity 尚未完成；现有 runtime candidate 是旧 CO-gated/H8+C4 系谱 | “R4 is the current formal analysis point; its final C5 runtime bundle remains pending.” |
| “source predictions 只是 diagnostic” | 中 | H1/H2/H3 是 R4 feature 和 QC source-spread 的显式输入 | 将三者列为 current formal R4/QC dependencies；在 ablation 后再决定能否移除 |

## 5. 推荐论文措辞

中文：

> GAPS 的分类骨干通过真实物理客户端 Flower 训练并在服务器执行目标域适配。浓度回归与分类训练解耦：C1/C2 的预构建 source window arrays 在 ECS 上集中拟合 per-gas Ridge、per-gas MLP 和带 gas one-hot 的 shared MLP，作为 multi-source prediction references；随后仅用 C5 calibration 拟合 target-personalized per-gas Ridge，并在 calibration-validation 上冻结 QC。因而本文的 federated claim 限于分类协同训练，当前 H8 source heads 不宣称为 federated regression。

English:

> GAPS combines real-device federated classification with a decoupled calibration stage. The current H8 regressor uses centrally pooled C1/C2 per-gas Ridge, per-gas MLP, and gas-conditioned shared-MLP prediction references, followed by C5-personalized per-gas Ridge fitting and calibration-only quality-control selection. Accordingly, the federated claim applies to collaborative classification, whereas the current H8 source heads are multi-source centralized references rather than federated regressors.

若需要保留 R0：

> R0 is retained as an offline sample-weighted FedAvg source-regression reference. It is not the training mechanism used by the H8 source Ridge/MLP heads and is not the final C5 regression dependency.

## 6. 当前结论边界

- 可以确定 H1/H2/H3 的训练方式，因为 executable code 与 formal B2 manifest 一致：source clients 为 C1/C2，source features 为三列，104 -> 107 features。
- 可以确定 formal R4/H8 结构依赖全部三种 source prediction。
- 不能声称 final deployment bundle 已冻结；现有旧 runtime candidate 与 formal C5 fixed-R4 的全路由语义仍需 bundle/parity 阶段统一。
- 不建议现在实现 distributed Ridge。先等待 final classifier/prediction stream，再做 source-head dependency ablation；只有 H1 Ridge 被证明具有不可替代的增益时，distributed sufficient-statistics Ridge 才有足够论文价值。
