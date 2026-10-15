# 面向传感器漂移的校准辅助云边协同气体感知系统

## 副标题

联邦语义分类、目标个性化定量与可靠输出

## English Title

A Calibration-Assisted Cloud-Edge Gas Sensing System Under Sensor Drift: Federated Semantic Classification, Target-Personalized Quantification, and Reliable Output

> [!STATUS]
> 本稿为导师汇报与实验闭环用中文完整初稿，结果更新至 2026-07-15。seed-42 的 v2r1/v3 分类、A6/B5/B2 正式 C5 回归、FULL/HC95/HC90 QC、oracle-route 和三方向 B2/B5 六运行均已完成；种子 43-46、低校准压力、最终 runtime parity 和端侧开销仍需补齐。文中所有单种子与 post-screen 证据均明确标注，不应直接复制为投稿定稿结论。

## 摘要

长期部署的电子鼻和气体感知节点同时受到传感器漂移、设备异质性、目标设备标注稀缺以及错误输出风险的影响。仅在源设备上训练分类器，或只报告分类正确条件下的浓度误差，均不足以描述一个可部署物联网感知系统。本文提出校准辅助的云边协同气体感知框架 GAPS。系统以阿里云 ECS 为 Flower 服务器，以树莓派和 PC 为物理源域客户端，在不集中原始源窗口的条件下训练轻量时序分类器；服务器使用目标设备 calibration 执行语义与分布适配；目标端再训练按预测类别路由的 Ridge/MLP 浓度专家，并由可靠性 QC 将输出分为 accept、review 和 reject。真实 C1/C2 -> C5 seed-42 实验中，source-only FedAvg accuracy 仅为 26.54%，相同标签预算 target-CE 为 98.24%；修正版轻量 B2 达到 99.26%，预声明完整 B5 为 98.90%。固定 H8 在 A6/B5/B2 上的实际路由 RMSE 分别为 28.014、17.447 和 14.656 ppm，而分类正确条件 RMSE 均约 11.3 ppm；对全部 1360 个窗口强制真类别路由后，三组 RMSE/NRMSE 均为 11.908/0.069，说明分类错路由主导端到端尾部。B2-HC95 在 95.66% 自动收益率下将 accepted RMSE 降至 12.673 ppm，并识别 7/10 个分类错误。三方向验证进一步显示 B2 在 C1 -> C5 数值更优，B5 在 C5 -> C1 显著更优并在 C4+C5 -> C1 数值更优，表明完整适配栈的收益依赖迁移方向和源域异质性。所有性能排名仍限于 seed 42，需通过多种子和端侧开销实验完成最终证据闭环。

**关键词:** 气体传感器漂移；联邦学习；云边协同；校准辅助域适应；语义原型；个性化回归；选择性输出

## Abstract

Long-term electronic-nose deployments face sensor drift, device heterogeneity, limited target calibration labels, and the operational risk of unreliable concentration outputs. This paper presents GAPS, a calibration-assisted cloud-edge framework that connects physical-client federated classification, server-side target adaptation, target-personalized Ridge/MLP quantification, and accept/review/reject quality control. In real C1/C2-to-C5 seed-42 experiments, source-only FedAvg achieved only 26.54% accuracy and an equal-label-budget target-CE baseline achieved 98.24%; the corrected compact B2 configuration reached 99.26%, while the predeclared full B5 reached 98.90%. With the fixed H8 regressor, actual-route RMSE for A6, B5, and exploratory B2 was 28.014, 17.447, and 14.656 ppm, respectively, whereas correct-class RMSE remained near 11.3 ppm. Forcing the true route on all 1360 test windows produced the identical 11.908 RMSE and 0.069 NRMSE for all three streams, isolating route errors as the main source of the end-to-end tail. At the B2-HC95 operating point, 95.66% of windows were automatically accepted with 12.673 ppm RMSE, while 7 of 10 route errors were flagged. Three cross-direction experiments further showed that B2 was descriptively better for C1-to-C5, whereas B5 was significantly better for C5-to-C1 and descriptively better for C4+C5-to-C1. The full adaptation stack is therefore direction- and heterogeneity-dependent rather than uniformly additive. All rankings remain seed-42 evidence and require multi-seed and edge-efficiency confirmation.

**Index Terms:** sensor drift, federated learning, cloud-edge collaboration, calibration-assisted domain adaptation, semantic prototypes, personalized regression, selective prediction

[[PAGEBREAK]]

## 1. 引言

金属氧化物气体传感器阵列具有成本低、响应快和易集成等优势，但其统计特性会随设备、批次、环境和时间发生变化。传感器漂移会使训练分布与部署分布分离，进而同时破坏气体类别识别和浓度定量。经典漂移补偿通常在集中数据上建立分类器或集成模型 [1], [2]，难以直接覆盖多设备持续部署时的数据归属、边缘算力和通信约束。

联邦学习通过交换模型更新而非集中原始数据，为分布式 IoT 节点提供了协同学习途径 [3]。然而，FedAvg 并不自动消除设备间的统计偏移。本文的核心实验正说明这一点：在 C1/C2 上进行 source-only 联邦训练后，C5 测试准确率仅为 26.54%，且平均置信度高达 99.29%，呈现出严重的跨域过置信失效。客户端侧增加原型对齐、上一轮特征蒸馏或选择性聚合，也未在当前设置下独立弥合这一差距。

目标设备校准是现实传感系统中常见且可解释的操作。与将少量目标标签隐藏在“无监督适配”叙述中不同，本文把 C5 calibration 明确建模为系统资源，并设置相同目标标签预算的 target-CE 基线。由此，研究问题不再是“能否在完全无标签目标域上迁移”，而是“如何在有限且透明的目标校准预算下，使云边模型获得可靠的分类路由，并将其连接到浓度定量和风险控制”。这一定位更符合真实 IoT 试验台、传感器管理和部署维护的要求，也避免了不公平的标签预算比较。

另一个常被忽略的问题是分类与回归之间的路由耦合。气体浓度专家通常按类别训练；部署时只能按预测类别调用相应专家。因此，分类正确条件下的回归误差 S_CC 描述的是数值模型能力上限，真实预测路由下的 S_ALL 才描述端到端系统行为。若只报告 S_CC，错误分类造成的大误差会被系统性隐藏；若只报告 QC 后低误差而不报告覆盖率，则会把“少输出”误写成“模型更准确”。

基于上述问题，本文构建 GAPS 云边协同气体感知系统。其主要工作可概括为：

1. 构建真实 ECS、树莓派和 PC 组成的 Flower 云边试验台，在冻结的 C1/C2 source 到 C5 target 协议下完成可恢复、可审计的分类训练与产物回收。
2. 设计类别-阶段语义记忆路径，将客户端原型对齐、上一轮特征约束、服务器语义原型和设备残差建模纳入同一分类框架，并通过 A0/A0T/A2-A7 因果矩阵区分客户端、聚合和服务器适配因素。
3. 将联邦分类与目标个性化浓度回归解耦，在 C5 calibration 上比较每气体 Ridge、MLP、H2.3+ 和 source-prediction-augmented H8，并规定所有超参数和专家选择仅使用 calibration-validation。
4. 建立能力线与系统线两条结果合同：能力线报告分类性能和 QC 前 S_CC 回归性能；系统线报告真实路由 S_ALL、accept/review/reject 的误差、样本数和覆盖率，并要求离线评估与部署 runtime 数值一致。

本文当前是一份结果驱动的完整初稿。已完成的 seed-42 核心筛选支持“目标校准必要、语义方法族有效、联合配置值得确认”的判断，但不支持多种子显著性、差分隐私、完全无监督域适应或闭环部署已经完成等表述。后续结果将按本文预注册的证据门槛补齐。

## 2. 相关工作

### 2.1 气体传感器漂移与电子鼻学习

Vergara 等构建了跨 36 个月的气体传感器阵列数据，并通过分类器集成研究漂移补偿 [1], [2]。该类工作证明了长期漂移会显著降低固定模型的识别能力，也说明跨时间或跨批次验证比同分布随机验证更接近部署问题。现有许多电子鼻研究仍将类别识别和浓度回归作为彼此独立的离线任务，较少同时处理分布式训练、目标校准、预测路由和拒识策略。

本文不改变导师确认的窗口级、类别和浓度分层 calibration/test 划分，而是在此协议上增加严格的数据角色和行级对齐合同。为避免把窗口随机性解释成更广泛的跨文件泛化，论文最终版本还应在附录提供按文件或 repeat 聚类的 bootstrap 置信区间，但该分析不替代主协议。

### 2.2 IoT 中的联邦与异质学习

FedAvg 以本地样本数加权模型更新，是联邦深度学习的基础方法 [3]。FedProx 等工作进一步讨论了设备系统异质性和非独立同分布数据带来的优化困难 [4]。在 IoT 场景中，联邦学习的价值不仅是一个优化算法，还包括端侧数据保留、掉线恢复、通信预算和模型部署合同。IEEE Internet of Things Journal 的范围也强调 IoT 系统架构、传感技术、试验台和应用集成 [15]。

GAPS 使用 FedAvg 等价参数聚合作为可解释基线，并在真实物理客户端上执行训练。需要强调的是，参数交换并不自动形成差分隐私保证；本文只主张原始源窗口不集中上传，不主张形式化隐私预算。

### 2.3 域适应、语义原型与持续约束

MMD 通过核均值嵌入衡量分布差异 [5]，CORAL 对齐源域和目标域的二阶统计 [6]，DANN 利用梯度反转学习域不可分特征 [7]。这些方法大多以无标签目标域为典型设置。本文的 A5-A7 会使用 C5 calibration 的类别或阶段信息，因此应称为 calibration-assisted 或 label-assisted adaptation，而非 UDA。

原型网络通过类中心表示实现低样本度量学习 [8]；经验回放或蒸馏可减缓模型在连续更新中的表征漂移 [9]。GAPS 将原型细化到 gas class 与 response phase 的组合键，并通过服务器指数滑动平均维护长期语义记忆。与少样本新类别识别不同，本文原型用于固定四类气体在跨设备条件下的语义约束。

### 2.4 置信度校准与选择性输出

神经网络可能在错误预测上保持高置信度，ECE 和 NLL 因而是准确率之外的重要指标 [10]。最大 softmax 概率是误分类和分布外检测的基础信号 [11]，SelectiveNet 等工作进一步以风险-覆盖曲线评价拒识系统 [12]。本文的 QC 不以“拒绝更多样本”作为性能改进，而是同时报告误差、样本数和 coverage，并设置随机拒绝对照和固定覆盖率比较。

## 3. 系统架构与问题定义

### 3.1 云边系统角色

[[FIGURE:architecture]]

系统由四类角色组成。C1 在树莓派上运行 Flower client，C2 在 PC 上运行第二个 Flower client；阿里云 ECS 运行 Flower server、聚合策略和目标校准适配；C5 作为目标设备提供固定 calibration 和 test 分区。分类训练期间，C1/C2 原始窗口保留在客户端，服务器接收模型参数和启用配置所需的语义统计。C5 calibration 可在服务器侧参与适配、回归头拟合和 QC 阈值选择，C5 test 始终只用于冻结后的最终评估。

### 3.2 数据协议

本文数据来自公开气体传感器漂移资源 [16] 的 time-aware 窗口化版本。每个分类样本表示为 x_i ∈ R^(100×8)，对应 100 个时间步和 8 个响应通道。分类标签 y_i^cls 属于四类气体：乙醇、一氧化碳、乙烯和甲烷；浓度标签 y_i^ppm 为 ppm；阶段标签 p_i 属于 early、middle 和 late。主协议如下。

| 项目 | 冻结设置 | 作用 |
|---|---|---|
| Source clients | C1、C2 | 真实联邦分类客户端与 source regression reference |
| Target client | C5 only | 目标校准、测试和个性化回归 |
| C5 calibration/test | 320/1360 windows | 20%/80% 类别与浓度分层切分 |
| Calibration inner split | 75% fit / 25% validation | 选择回归超参数、融合和 QC 阈值 |
| Classification rounds | 25 | 所有主消融一致 |
| Training seeds | 42-46 | seed 42 核心筛选；其余用于确认 |
| Primary key | client, split, sample_index | 对齐分类、回归、QC 与 runtime 流 |

C3 和 C4 不作为目标域，也不得进入 C5 主指标。数据目录名称和历史元数据中仍存在早期 `c1234src` 或 `c345tgt` 字样，但冻结 manifest 和实际 loader 只激活 C1、C2 和 C5。所有流必须通过唯一键连接；重复键、缺失行、calibration/test 交叉或使用 test 标签选择超参数均判为实验失败。

### 3.3 任务与输出合同

分类器 f_θ 输出 logits o_i、预测类别 ŷ_i^cls、置信度和 64 维特征。回归器 h_c 以预测类别 c=ŷ_i^cls 路由并输出浓度 ŷ_i^ppm。QC 策略 q 根据分类不确定性、响应偏离和浓度偏离给出 accept、review 或 reject。仅 accept 行产生自动输出 `auto_output_ppm`；review/reject 保留审计信息但不应被当作自动交付值。

## 4. 校准辅助联邦语义分类

### 4.1 轻量时序分类骨干

分类路径实例化 `FedGasBaseModel`。输入转置为 B×8×100 后，依次通过三个深度可分离 TCN 残差块。通道数为 8→32→48→48，卷积核为 3，dilation 分别为 1、2 和 4；每块使用 affine InstanceNorm1d、残差投影和 ReLU。残差连接借鉴深层网络的稳定优化思想 [13]。TCN 输出经 4 头自注意力 [14] 和可学习注意力池化，再映射为 64 维分类特征。训练阶段加入标准差 0.01 的特征噪声和 0.1 dropout，分类特征在进入线性头前进行 L2 归一化。

| 模块 | 结构 | 主要参数 |
|---|---|---|
| Input | 100×8 window | 不使用共享 `norm_stats.npz` |
| DS-TCN 1 | 8→32, k=3, d=1 | InstanceNorm + residual |
| DS-TCN 2 | 32→48, k=3, d=2 | InstanceNorm + residual |
| DS-TCN 3 | 48→48, k=3, d=4 | InstanceNorm + residual |
| Attention | 4-head self-attention | learned temporal pooling |
| Feature/head | 48→64→4 | L2 norm, dropout=0.1 |

当前 checkpoint 序列化 22,765 个参数，其中 19,557 个参与实际分类前向；历史兼容的 `channel_attn` 和未启用 `feat_proj` 共 3,208 个参数仍进入通信。这相当于约 14.1% 的序列化参数冗余，是模型冻结后的轻量化目标，而不是当前 v2 运行中途可修改的结构。

### 4.2 客户端目标函数

客户端基础目标为标准分类交叉熵。含语义对齐的 profile 使用 class-phase 原型作为 InfoNCE 正负样本；含 replay 的 profile 从第 2 轮开始冻结上一轮服务器模型，并约束当前归一化特征接近上一轮特征。客户端总损失为：

$$L_client^(k) = L_CE^(k) + λ_align L_align^(k) + λ_rep L_rep^(k).$$

其中 λ_align=0.05、温度 τ=0.1，λ_rep=2.0。第 1 轮没有服务器原型和上一轮教师，因此只执行 CE。Flower 分类路径显式设置 `USE_REG_LOSS=False`，客户端不训练任何神经回归头；浓度估计由后续目标端回归阶段完成。

客户端在训练后以未归一化 64 维 `reg_feat` 计算 gas class c 与 phase p 的局部均值 μ_(k,c,p)、对角方差和样本数。CE-only 和 replay-only 配置不执行无用的原型遍历，也不上传空 JSON；设备残差统计仅在 semantic DA 组启用。

### 4.3 参数聚合、语义记忆与选择性聚合

基础参数聚合使用样本数加权 FedAvg：

$$θ^(t+1) = Σ_k [n_k / Σ_j n_j] θ_k^(t+1).$$

当选择性聚合和服务器 DA 均关闭时，GAPS 与 FedAvg 在 1e-7 绝对误差内数值等价，因此 A1 只作为合约测试，不重复完整训练。服务器对 class-phase 原型执行 EMA：

$$μ_bar_(c,p)^t = ρ μ_bar_(c,p)^(t-1) + (1-ρ) μ_(c,p)^(t,weighted),  ρ=0.8.$$

选择性聚合以客户端原型和服务器原型的平均余弦一致性 s_k 调整 FedAvg 权重，并设置最小比例 s_min=0.3。真实 warmup 为 3 轮，因此第 1-3 轮保持 FedAvg，第 4 轮起才启用语义权重缩放。

### 4.4 服务器校准适配

每轮聚合后，ECS 使用 C1/C2 calibration 作为 source rehearsal，并使用 C5 calibration 作为 target adaptation 数据，执行 100 个服务器优化步，学习率 5×10^-4。所有 DA 组都保留 source CE；A0T 额外设置 target CE 权重 1.0，A5-A7 的 target CE 权重为 0，但仍会在类条件或阶段条件项中读取目标标签，因此属于 label-assisted adaptation。

服务器候选项包括类条件 CORAL、固定 σ=1 高斯核分布差异、类别中心原型锚定、source-to-prototype consistency、语义原型与设备残差拟合、跨客户端原型一致性、阶段正则和类条件对抗项。当前 v2 的完整 A7 权重如下。

| 损失项 | 权重 | 当前解释边界 |
|---|---:|---|
| CORAL | 0.5 | 类条件二阶统计对齐 |
| Global/Class kernel discrepancy | 0.5 / 0.5 | 代码对 biased MMD-squared 再平方 |
| Prototype anchor | 0.3 | 源/目标类中心靠近语义中心 |
| Adversarial | 0.5 | 旧符号组合仍需独立修正版验证 |
| Target CE | 0.0 | A0T 单独使用 1.0 |
| Semantic/residual fit | 0.05 | class-phase 语义与设备残差分解 |
| Prototype consistency | 2.0 | source feature 到语义原型 |
| Residual matching | 0.1 | 服务器残差匹配上传估计 |
| Prototype pair L2 | 0.2 | detached 常数诊断，无参数梯度 |
| Current stage term | 0.2 | 域内跨阶段不变性，而非同阶段跨域对齐 |

为保证实验忠实于可执行代码，本文不把上述遗留项改写成更理想的数学形式。`compute_mmd` 已返回 biased empirical MMD-squared，调用端再次平方；当前 stage 项比较同一域内不同阶段；`proto_mmd` 实为 detached prototype pair-L2 且无训练梯度；WGAN critic 与 GRL 的符号方向尚不能支持“缩小 Wasserstein 距离”的结论。这些因素保留在 v2 以维持矩阵一致性，但正式创新主张应集中于已被数据支持的语义方法族，并以新组名验证修正版。

### 4.5 分类消融矩阵

| ID | 客户端机制 | 聚合 | 服务器机制 | 因果问题 |
|---|---|---|---|---|
| A0 | CE | FedAvg | 无 | source-only 基线 |
| A0T | CE | FedAvg 等价 | source CE + target CE | 相同标签预算强监督基线 |
| A1 | CE | GAPS, no selective | 无 | 数值等价合约，不训练 |
| A2 | CE + alignment | FedAvg 等价 | 无 | 客户端原型对齐 |
| A3 | CE + replay | FedAvg 等价 | 无 | 上一轮特征约束 |
| A4 | alignment + replay | FedAvg 等价 | 无 | 客户端联合机制 |
| A4S | A4 | selective GAPS | 无 | 选择性聚合 |
| A5 | A4S | selective GAPS | distribution DA | 分布统计方法族 |
| A6 | A4S + residual stats | selective GAPS | semantic DA | 语义/残差方法族 |
| A7 | A6 client | selective GAPS | semantic + distribution + stage | 联合配置 |

A5 与 A6 是替换式方法族对照，并不是 A5 上逐项添加语义项。A7 才把 distribution、semantic 和当前 stage 项组合在一起。这一区分对结果解释十分重要。

## 5. C5 目标个性化浓度定量

### 5.1 分类与回归解耦

分类骨干只输出预测路由、置信度、`cls_feat` 和 `reg_feat`。C5 的浓度模型在目标 calibration 上独立训练。该设计避免因少量浓度标签反复重训联邦骨干，允许不同气体拥有不同回归复杂度，也使分类错误与数值回归错误可以分别定位。

R3aK16 仅作为 C1/C2 source regression reference。它训练过 source-domain 神经回归结构和 Ridge 参考，但不是 C5 目标域最终输出。论文不得将 R3aK16 的源域结果包装成目标域主方法。

### 5.2 部署可见特征与基础专家

每个 100×8 窗口提取 104 维丰富响应描述，包括逐通道均值、标准差、极值、幅值、斜率、绝对差分统计，以及全局响应幅值、波动、通道排序比例、窗口时间、onset 相对位置、插值质量和阶段 one-hot。分类骨干另导出 64 维 `reg_feat`。

每种气体独立拟合 Ridge。输入标准化后，截距不受惩罚：

$$β_hat_α = (X^T X + α I_0)^† X^T y.$$

α∈{0, 0.01, 0.1, 1, 10, 100, 1000}，只在 calibration-validation 上按 RMSE 选择，然后使用完整 calibration 重拟合。预测裁剪到该气体 calibration 浓度范围。

C5 浅层 MLP 使用 ReLU 和 LBFGS，`max_iter=800`；hidden∈{(16),(32),(64),(32,16)}，α∈{0.001,0.01,0.1,1}。每个气体头在真实类别对应的 calibration 行上训练，部署时按预测类别路由。该 104 维 per-gas MLP 是 H2.3 anchor。

### 5.3 H2.3+ 与 H8

H2.3+ 将 104 维 MLP anchor 与 168 维 rich+`reg_feat` Ridge 进行受约束融合：

$$y_hat(w) = (1-w) a + w r,  w∈{0,0.1,0.25,0.5,0.75,1}.$$

只有当 validation 整体 RMSE 改善且 non-CO RMSE 相对 anchor 的退化不超过 1 ppm 时，才允许 w>0；否则回退 w=0。该规则防止复杂融合被强制保留。

H8 首先在 C1/C2 上训练 per-gas Ridge、per-gas MLP 和 shared MLP 三个 source reference，再将三者的 ppm 预测与 C5 的 104 维响应描述拼接，输入 C5 per-gas Ridge。该结构的意义不是让源模型直接控制 C5 输出，而是把源域可迁移的低维预测作为目标端线性校准的辅助基函数。主协议显式关闭 C4 rescue。

### 5.4 专家选择与泄漏防线

预注册候选包括固定 H2.3+、固定 H8、预测 CO 即使用 H8 的简单 gate、P4 风险 gate 和可靠性约束 selector。最终方法必须由 calibration-validation 选择并冻结，再打开 test。

历史 P4 使用的上游 `composite_response_risk` 会按 `true_class` 选择 ppm 量程，再被重命名为 `risk_score`。这使错路由测试行间接使用真值类别，属于部署不可见信息。故历史 P4 只能作为泄漏诊断，不能进入正式主表。新 selector risk 必须只依赖预测类别、logits、窗口响应和 calibration reference，并通过“修改 true class 不改变风险”的自动测试。

## 6. 可靠性 QC 与评价合同

### 6.1 风险与三态输出

部署可见风险包括 1-max softmax probability、top-1/top-2 margin、预测浓度到最近 calibration 浓度的归一化距离、响应签名距离、类别响应排序风险和响应边缘风险。对选中的风险 q_(i,j) 和 calibration 阈值 η_j，定义归一化风险：

$$R_i = max_j q_(i,j) / η_j.$$

若 R_i≤r_low 则 accept，若 R_i>r_high 则 reject，其余为 review。当前 0.9/1.1 只是代码默认值，论文阈值必须由 calibration-validation 冻结。最终部署 bundle 必须 fail closed：缺失 QC policy、calibration reference 或必需风险字段时应拒绝启动，而不是默认 accept。

### 6.2 两条结果主线

定义 S_ALL 为全部 C5 test 行，S_CC 为分类正确行，S_AR 为 QC 判定 accept 或 review 的行。论文主表必须同时回答两个问题。

1. 能力线：分类 accuracy、macro-F1、NLL、ECE 和每类 recall；随后报告 QC 前 S_CC 的 RMSE、MAE、NRMSE 和 R2，回答“路由正确时浓度模型能做到什么”。
2. 系统线：真实预测路由下 S_ALL 的回归指标；随后报告 accept-only 与 S_AR 的误差、N 和 coverage，回答“系统实际自动交付或进入复核的性能如何”。

S_AR∩S_CC 只用于诊断 QC 是否优先保留分类正确样本，不能被标成纯 S_CC。任何 QC 结果都必须给出父集合 N、保留 N 和 coverage，并与随机拒绝和固定覆盖率基线比较。

[[FIGURE:evaluation_contract]]

## 7. 实验设置

### 7.1 真实云边训练环境

Flower server 运行于阿里云 ECS。C1 客户端运行于物理树莓派 `gaps-pi`，C2 客户端运行于同一局域网内 PC。controller 在每组实验前检查 ECS 空闲、同步精确代码和 manifest、启动服务器和两个客户端，并在 25 轮完成后回收 checkpoint、history、client statistics、prototype statistics 和日志。2026-07-11 的 v2r1 核心队列已完成全部九个 seed-42 训练组，不使用本地模拟替代论文训练。

### 7.2 训练超参数

| 类别 | 参数 | 设置 |
|---|---|---|
| Client training | rounds / local epochs | 25 / 5 |
| Client optimizer | Adam LR / batch | 5×10^-4 / 32 |
| Stabilization | gradient clip | 5 |
| Alignment | λ / τ | 0.05 / 0.1 |
| Replay | λ / start | 2.0 / round 2 |
| Prototype memory | EMA ρ | 0.8 |
| Selective aggregation | warmup / min scale | 3 / 0.3 |
| Server adaptation | steps / LR | 100 per round / 5×10^-4 |
| Critic | updates / LR / GP | 3 / 0.001 / 10 |
| Evaluation | checkpoint | final round 25 |

所有分类结果使用统一 evaluator 从 round-25 checkpoint 重新推理 C5 calibration/test 行流。最终分类指标包括 accuracy、macro-F1、四类 recall、15-bin ECE、NLL 和混淆矩阵。回归指标包括 RMSE、MAE、NRMSE、R2、N 和 coverage。

## 8. 结果与分析

### 8.1 Seed-42 分类核心筛选

[[FIGURE:classification]]

| 组别 | 主要机制 | Accuracy | Macro-F1 | NLL | ECE |
|---|---|---:|---:|---:|---:|
| A0 | Source-only FedAvg | 26.54% | 17.93% | 11.2816 | 0.7275 |
| A0T | Equal-label target CE | 98.24% | 98.24% | 0.1742 | 0.0158 |
| A2 | Client alignment | 29.71% | 27.28% | 3.9258 | 0.6508 |
| A3 | Client replay | 29.12% | 23.12% | 9.3266 | 0.6953 |
| A4 | Alignment + replay | 29.34% | 22.15% | 6.3050 | 0.6847 |
| A4S | A4 + selective aggregation | 31.62% | 22.64% | 5.9823 | 0.6653 |
| A5 | Distribution DA | 73.01% | 74.15% | 1.0907 | 0.2001 |
| A6 | Semantic DA | 98.01% | 98.02% | 0.1384 | 0.0178 |
| A7 | Combined semantic/distribution/stage | 98.60% | 98.60% | 0.1132 | 0.0118 |

结果首先表明跨设备偏移极强。A0 对甲烷的 recall 为 95.00%，但乙醇、CO 和乙烯 recall 仅为 4.71%、0.88% 和 5.59%；同时平均置信度为 99.29%。因此 A0 并非“随机猜测”，而是过置信地塌缩到少数目标类别。只看 source validation 无法发现这一部署风险。

A2、A3、A4 和 A4S 的准确率均低于 32%，说明客户端原型对齐、上一轮特征约束和选择性聚合在没有目标校准适配时不能独立解决 C1/C2 到 C5 的偏移。A4S 相对 A4 的 accuracy 增加 2.28 个百分点，但 macro-F1 只增加约 0.50 个百分点，当前单种子证据不足以把 selective aggregation 写成稳定增益。

A5 的 accuracy 提升到 73.01%，四类 recall 分别为 62.35%、73.82%、82.35% 和 73.53%，说明分布统计方法族可以部分恢复类别边界，但 NLL 和 ECE 仍明显高于 A0T/A6/A7。A6 用 semantic DA 替换 A5 的 distribution DA 后达到 98.01%，表明 class-phase semantic anchor、consistency 和 residual family 是当前最有价值的方法方向。由于 A5 与 A6 不是逐项累加，二者差值只能解释为方法族对比，不能解释为某一个单项的边际贡献。

A7 在 A6 语义配置上联合 distribution 和当前 stage 项，达到最高 accuracy 98.60%、macro-F1 98.60%、最低 NLL 0.1132 和最低 ECE 0.0118。相对 A6，accuracy 增益为 0.59 个百分点；相对相同标签预算的 A0T，增益仅为 0.37 个百分点。该差距具有方向性价值，但没有种子 43-46 的均值、标准差和配对置信区间之前，不能宣称 A7 稳定显著优于简单 target CE。

### 8.2 v3 修正版分类与模块作用

针对 legacy MMD 再平方、域内跨阶段对齐和对抗方向问题，v3 使用 conventional MMD2、跨域同类别同阶段对齐和修正的 Wasserstein feature objective。`prototype pair-L2` 的两个端点仍为 detached 统计量，因此 B1-B5 将其权重固定为 0。

| 组别 | 主要机制 | Accuracy | Macro-F1 | NLL | ECE |
|---|---|---:|---:|---:|---:|
| B1 | Corrected CORAL baseline | 98.7500% | 98.7534% | 0.0988 | 0.0120 |
| B2 | Corrected global/class MMD2 | **99.2647%** | **99.2657%** | **0.0690** | **0.0067** |
| B3 | B2 + same-class/same-phase stage | 98.8971% | 98.8980% | 0.1022 | 0.0108 |
| B4 | B2 + Wasserstein adversarial | 98.9706% | 98.9714% | 0.0835 | 0.0081 |
| B5 | B2 + CORAL + stage + adversarial | 98.8971% | 98.8990% | 0.0704 | 0.0093 |

B2 在 seed 42 上同时取得最高 accuracy、macro-F1 和最好的 NLL/ECE。B3、B4 和 B5 均未超过 B2，说明额外 CORAL、stage 和 adversarial 项没有在主方向产生简单叠加增益。B2 是打开 B1-B5 test 排名后选出的性能候选，因此其下游回归/QC属于 post-screen exploratory evidence；B5 仍是预声明完整修正版，不能因单次排名而删除其机制价值。

### 8.3 正式 C5 回归阶梯

所有 A6/B5/B2 回归模型和策略拟合均在阿里云 ECS 上完成。R3 为 calibration-validation 选择的 H2.3+；R4 为固定 H8；R7 根据 test 真值逐样本选择 R3/R4，只作为不可部署 oracle。

| 分类器 | 分类错误 | R3 S_ALL/S_CC RMSE | R4 S_ALL RMSE/NRMSE | R4 S_CC N/RMSE | R7 S_ALL RMSE |
|---|---:|---:|---:|---:|---:|
| A6 | 27 | 29.6730 / 11.9446 | 28.0144 / 0.2276 | 1333 / 11.3890 | 25.5184 |
| B5 | 15 | 20.1082 / 12.4435 | 17.4473 / 0.1352 | 1345 / 11.3890 | 15.7940 |
| B2 | 10 | 19.7959 / 12.3415 | **14.6564 / 0.1059** | 1350 / **11.3288** | 12.6393 |

B5 和 B2 的 H2.3+ 验证选择权重为 0，因此 R3 实际退化为 R2，没有产生融合增益。固定 H8 R4 在三个分类器上均优于可部署的 R1-R3/R5-R6。B2 相对 B5 的 S_CC R4 差异只有 0.0602 ppm，但 S_ALL RMSE 低 2.7910 ppm，说明收益主要来自更少且破坏性更低的分类错路由，而不是正确路由回归器发生显著改变。B2 的选择发生在分类 test 排名打开之后，必须保持探索性标签。

### 8.4 QC、Nonreject 与 oracle-route

HC95 是主工作点，HC90 是次工作点，FULL 是 coverage-1 对照。Yield 只统计 accept；Nonreject 合并 accept 和 review。所有阈值只在 calibration-validation 上选择，测试标签不进入部署风险。

| 分类器 | Workpoint | Accept/Review/Reject | Yield | Nonreject | Accepted RMSE/NRMSE | Nonreject RMSE/NRMSE | 错路由召回 |
|---|---|---:|---:|---:|---:|---:|---:|
| A6 | FULL | 1360/0/0 | 100.00% | 100.00% | 28.0144 / 0.2276 | 28.0144 / 0.2276 | 0.00% |
| A6 | HC95 | 1233/86/41 | 90.66% | 96.99% | 21.1455 / 0.1731 | 22.3047 / 0.1839 | 66.67% |
| B5 | FULL | 1360/0/0 | 100.00% | 100.00% | 17.4473 / 0.1352 | 17.4473 / 0.1352 | 0.00% |
| B5 | HC95 | 1309/33/18 | 96.25% | 98.68% | 15.9075 / 0.1197 | 16.0983 / 0.1213 | 46.67% |
| B2 | FULL | 1360/0/0 | 100.00% | 100.00% | 14.6564 / 0.1059 | 14.6564 / 0.1059 | 0.00% |
| B2 | HC95 | 1301/35/24 | 95.66% | 98.24% | **12.6729 / 0.0857** | **12.8614 / 0.0858** | **70.00%** |

B2-HC95 只 reject 24 个窗口并把 35 个窗口转入 review，却识别 7/10 个分类错误；匹配随机拒绝的错路由平均召回约为 4.02%。该结果支持 QC 作为高覆盖率风险分流层，但不表示 QC 修复了被复核或拒绝的预测。

为分离路由误差与数值回归误差，扩展实验保留全部 1360 个测试窗口和原 QC mask，只强制 `route_class=true_class`。A6/B5/B2 的 FULL forced-true-route RMSE/NRMSE 均为 `11.9082/0.0690`。这一上界保留了原本误分类的困难窗口，因此不同于删除错误窗口的 S_CC，也不能作为部署指标。

### 8.5 B2/B5 跨方向验证

| 源域 -> 目标域 | 模型 | Accuracy | Macro-F1 | NLL | ECE |
|---|---|---:|---:|---:|---:|
| C1 -> C5 | B2 | **98.8971%** | **98.8954%** | **0.1008** | **0.0107** |
| C1 -> C5 | B5 | 98.3088% | 98.3131% | 0.1322 | 0.0150 |
| C5 -> C1 | B2 | 97.6493% | 97.6525% | 0.2769 | 0.0237 |
| C5 -> C1 | B5 | **98.3582%** | **98.3605%** | **0.1718** | **0.0156** |
| C4+C5 -> C1 | B2 | 98.9552% | 98.9565% | 0.1059 | 0.0096 |
| C4+C5 -> C1 | B5 | **99.1418%** | **99.1436%** | **0.0894** | **0.0082** |

C1 -> C5 中 B2-B5 为 `+0.5882 pp`，McNemar `p=0.0963`；C5 -> C1 中为 `-0.7090 pp`，`p=0.0043`；C4+C5 -> C1 中为 `-0.1866 pp`，`p=0.3323`。最后一组因 B2 最差类别召回低 `0.7463 pp`，预声明三指标 0.5 pp 规则判为 `B5_favored`，但准确率差异不显著。三方向结果说明 B5 的完整适配栈在困难反向迁移和异构多源场景更有价值，而 B2 在主方向具有更好的性能/复杂度比。

### 8.6 当前系统证据与缺口

| 证据模块 | 已完成 | 尚需完成 | 最终论文输出 |
|---|---|---|---|
| Classification | v2r1/v3 seed-42 与三方向六运行 | B2/B5 seeds 43-46 | mean+/-std、paired seed difference、per-class recall |
| Regression | A6/B5/B2 R0-R7 正式 ECS 闭环 | 多种子下游稳定性 | S_ALL、S_CC、oracle-route 主表 |
| Selector/QC | FULL/HC95/HC90、随机拒绝、forced-true-route | 低校准压力与 runtime policy parity | risk-coverage 与三态输出表 |
| IoT runtime | 真实训练拓扑、恢复和安全加固 | Pi/PC latency、RSS、实际通信、掉线恢复 | 系统开销与可用性表 |
| Evidence archive | 轻量 CSV/JSON/manifest 跟踪 | 大型原始产物独立归档 | claim-to-evidence map |

按 float32 参数计，22,765 个序列化分类参数对应约 91.1 kB 的单向裸参数负载，其中有效前向参数约 78.2 kB。该数值只是通信下界，不包含 Flower 协议、张量序列化和语义 JSON。正式系统表必须使用实际网络字节、每轮耗时和端侧峰值内存，不应以理论估计替代测量。

## 9. 讨论

### 9.1 为什么目标校准是系统资源而不是协议缺陷

在传感器部署中，目标设备上线校准本来就是维护流程的一部分。A0 与 A0T 的巨大差异说明，完全 source-only 的跨设备分类在当前数据上不可用。透明地报告 C5 calibration 的 320 个窗口、其标签用途和每轮 100 个服务器适配步，比把目标标签隐藏在复杂损失中更严谨。A0T 还给出了强公平基线：任何语义适配贡献都必须超过或至少在校准、开销和稳定性上优于这一简单方法。

### 9.2 语义方法族的合理创新边界

A6 接近 A0T/A7，而 A5 明显较弱，支持把 class-phase semantics 作为主要研究方向。v3 进一步表明，修正的 global/class MMD2 核心 B2 在主方向已经足够强，完整 B5 没有出现简单叠加增益；但跨方向实验中 B5 在 C5 -> C1 显著更优，并在 C4+C5 -> C1 改善点估计、校准误差和最差类别召回。因此，论文不应把所有附加损失都包装成普遍有效，而应将 B2 定位为紧凑核心，将 B5 定位为面向困难迁移和源域异质性的鲁棒增强。

A7 的 legacy 小幅增益不能证明其中每个联合项有效。v3 已修正 MMD2、跨域同类别同阶段对齐和 Wasserstein feature objective，并将无训练梯度的 prototype pair-L2 权重固定为 0。当前更稳健的论文叙述是：A6/A7 说明校准辅助语义方法族有效，B2/B5 进一步区分紧凑核心和完整鲁棒增强；最终贡献必须由 seeds 43-46 与系统开销共同决定。

### 9.3 为什么目标域回归以 Ridge 为重要主线

目标 C5 calibration 只有 320 个窗口，并需按四个气体分头训练。此时目标端 Ridge 具有参数少、闭式可解释、易于 calibration-validation 选择和便于部署的优势。源域可以训练 Ridge、per-gas MLP 和 shared MLP 参考，但它们在 H8 中作为辅助预测特征，而不是直接替代 C5 头。正式结果中，H2.3+ 在 B5/B2 上均退化为 H2.3 anchor，而固定 H8 取得更低 S_ALL RMSE，说明低维源预测增强比直接混合更复杂目标 MLP/Ridge 更稳定。因而“源域训练神经与线性参考，目标域以轻量 Ridge 为最终个性化头”最符合当前代码和结果。

### 9.4 局限性

当前研究仍有六项边界。第一，正式分类与回归排名只有 seed 42，多种子不确定性待补。第二，主目标只有 C5，跨方向结果是 appendix 验证，尚不能主张任意新设备泛化。第三，窗口级随机分层切分是导师确认的主协议，但相邻窗口可能相关，需补充按文件/repeat 聚类 bootstrap 作为敏感性分析。第四，C5 calibration 标签同时服务分类适配、回归和 QC，必须报告标签预算和低校准压力测试。第五，系统没有形式化差分隐私或安全聚合，不能将联邦训练等同于隐私保证。第六，部署代码已经 fail-closed 加固，但正式 C5 bundle 的 1360 行 runtime parity、Pi/PC 延迟、RSS 和实际通信仍未冻结。

## 10. 结论

本文围绕真实 IoT 气体感知部署构建了 GAPS 校准辅助云边协同系统，将物理客户端联邦分类、目标 calibration 适配、C5 个性化 H8 定量和三态可靠输出连接为一条可审计流程。seed-42 结果表明，source-only FedAvg 在目标设备上严重失效；修正版 B2 在主方向提供很强的性能/复杂度比，完整 B5 在困难反向迁移中表现出更好的鲁棒性；固定 H8 优于 H2.3+ 和简单选择器，端到端回归尾部主要由错路由驱动；deployment-visible QC 能在约 95% 自动收益率下集中分流明显高风险窗口。当前系统故事已经形成分类、回归和可靠输出闭环，但最终投稿结论仍取决于 seeds 43-46、低校准压力、正式 runtime parity 和端侧开销。

## 附录 A. 建议的下一阶段实验

1. 在相同 ECS/Pi/PC 拓扑补齐 B2/B5 seeds 43-46，报告 mean、sample standard deviation、配对 seed 差、方向例外和 bootstrap 置信区间。
2. 完成回归/QC 低校准压力实验，并明确它不代表重新训练分类器后的端到端标签预算。
3. 生成正式 C5 deployment bundle，完成全部 1360 个 test 窗口的 offline/runtime 逐值一致性验证。
4. 在树莓派和 PC 测量分类、回归、QC 延迟、RSS、模型字节、实际每轮通信和掉线恢复时间。
5. 对窗口相关性补充按文件/repeat 聚类 bootstrap，但保留导师确认的窗口级分层主协议。
6. 冻结论文图表、claim-to-evidence map 和原始大产物归档位置，再形成投稿版本。

## 附录 B. 可主张与不可主张

| 当前可主张 | 当前不可主张 |
|---|---|
| 真实 ECS/Pi/PC 完成 seed-42 v2r1/v3 与三方向六运行 | 五种子稳定显著优于 A0T/B2/B5 |
| C1/C2 source 到 C5-only target 协议 | C3/C4 属于目标主结果 |
| Flower 分类不训练回归头 | 目标域只使用 Ridge、从未训练 MLP |
| A5-A7 是 calibration-assisted | 完全无监督目标域适配 |
| A6 semantic family 值得确认 | A5→A6 是单项增量因果贡献 |
| B2 是主方向紧凑性能候选，B5 是鲁棒增强候选 | 当前每个 CORAL/MMD/ADV/stage 项均普遍有效 |
| 正式固定 H8 是当前最佳可部署 coverage-1 回归器 | R7 oracle 或 forced-true-route 是部署性能 |
| HC95 可在高收益率下集中分流风险 | QC 修复了 review/reject 的预测 |
| 原始源窗口不集中到 ECS | 差分隐私或安全聚合保证 |
| QC 合同和风险审计已建立 | 新 C5 bundle 已完成 runtime 闭环 |

## 参考文献

1. A. Vergara, S. Vembu, T. Ayhan, M. A. Ryan, M. L. Homer, and R. Huerta, “Chemical gas sensor drift compensation using classifier ensembles,” Sensors and Actuators B: Chemical, vol. 166-167, pp. 320-329, 2012, doi: 10.1016/j.snb.2012.01.074.
2. J. Fonollosa, I. Rodríguez-Luján, and R. Huerta, “Chemical gas sensor array dataset,” Data in Brief, vol. 3, pp. 85-89, 2015, doi: 10.1016/j.dib.2015.01.003.
3. B. McMahan, E. Moore, D. Ramage, S. Hampson, and B. A. y. Arcas, “Communication-efficient learning of deep networks from decentralized data,” in Proc. AISTATS, PMLR 54, pp. 1273-1282, 2017.
4. T. Li, A. K. Sahu, M. Zaheer, M. Sanjabi, A. Talwalkar, and V. Smith, “Federated optimization in heterogeneous networks,” in Proc. MLSys, vol. 2, 2020.
5. A. Gretton, K. M. Borgwardt, M. J. Rasch, B. Schölkopf, and A. Smola, “A kernel two-sample test,” Journal of Machine Learning Research, vol. 13, pp. 723-773, 2012.
6. B. Sun and K. Saenko, “Deep CORAL: Correlation alignment for deep domain adaptation,” ECCV Workshops, 2016, arXiv:1607.01719.
7. Y. Ganin et al., “Domain-adversarial training of neural networks,” Journal of Machine Learning Research, vol. 17, no. 59, pp. 1-35, 2016.
8. J. Snell, K. Swersky, and R. S. Zemel, “Prototypical networks for few-shot learning,” in Proc. NeurIPS, pp. 4077-4087, 2017.
9. D. Rolnick, A. Ahuja, J. Schwarz, T. P. Lillicrap, and G. Wayne, “Experience replay for continual learning,” in Proc. NeurIPS, 2019.
10. C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, “On calibration of modern neural networks,” in Proc. ICML, PMLR 70, pp. 1321-1330, 2017.
11. D. Hendrycks and K. Gimpel, “A baseline for detecting misclassified and out-of-distribution examples in neural networks,” in Proc. ICLR, 2017.
12. Y. Geifman and R. El-Yaniv, “SelectiveNet: A deep neural network with an integrated reject option,” in Proc. ICML, PMLR 97, pp. 2151-2159, 2019.
13. K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in Proc. CVPR, pp. 770-778, 2016.
14. A. Vaswani et al., “Attention is all you need,” in Proc. NeurIPS, 2017.
15. IEEE Internet of Things Journal, “Guidelines for Authors,” 2026. [Online]. Available: https://ieee-iotj.org/guidelines-for-authors/
16. A. Vergara, “Gas Sensor Array Drift at Different Concentrations,” UCI Machine Learning Repository, 2012, doi: 10.24432/C5MK6M.
