# FedOnGas Phase3：研究成果收敛与联邦公平验证准备

2026-11-04，依据冻结v5及Phase0/0.5/1/1.5/2A，Phase2B固定675124ab6985cf32199e0b270e24b62d0e6f1621。本轮新模型训练0、模型推理0、Flower运行0、Pi部署/测量0。继续使用独立managed worktree。

## 研究进度与本轮发现

浓度研究已收敛：统一144条raw/clip主指标（72个结果视图，含明确重复引用）、模型/协议/标签预算与评价口径，保留负迁移和336场景故障回退。共享rho方向停止扩展，Source-only不被提升成无标定系统。Source-only运行clip使用已定义的gas物理量程而非目标test，但Oracle类别/量程条件及F回退cal依赖不能省略。

分类模块工程已存在，最有希望的论文机制是目标监督适配后的模型跨轮反馈。旧C5匹配消融有单seed窗口条件正向证据；Class-MMD和原型权重并非一致正贡献。B编号存在多套含义，不能将旧strictB0/B1或JulyB1-B5与新P3矩阵混同。

当前最大阻塞：共同Day7合法分类起点S*、原始输入/祖先身份及目标标签/实际步数/有效开关闭合。仅有接口实现不能说明TAC已经做过公平比较。旧严格C5权重是不同源协议的条件候选，未生成新预测路由结果。

## 阅读顺序

- [浓度成果固定与Source-only裁剪审计](REGRESSION_FREEZE.md)、REGRESSION_CONSOLIDATED.csv。
- [联邦已有证据审计](FEDERATED_FAIRNESS_AUDIT.md)、[最小参数与成本计划](FEDERATED_EXPERIMENT_PLAN.md)、FEDERATED_MINIMUM_MATRIX.csv（六行draft计划）。
- [端到端模型兼容性](END_TO_END_COMPATIBILITY.md)、[PC/Pi验收合同](PI_ACCEPTANCE_CONTRACT.md)。
- [毕业论文贡献证据矩阵](CONTRIBUTION_EVIDENCE.md)、[继续/停止结论](NEXT_DECISION.md)。

evidence包含79项v5来源SHA/生成版本（缺失保持unknown）、Flower源码门控行、7个旧严格分类端点配置、3个匹配消融、57个ONNX候选字节清单、7checkpoint来源/时序判断、各阶段指标和负结果来源。onnx/onnxruntime当前未安装，本轮不安装、不猜图形状。0.413ms仍不能作为新SVR或完整链路性能。

下一步是合法分类资产/参数合同预检，通过后才六组C5 A4 pilot，不立即重训整个矩阵。C3→C5后续接入才符合新cal时间顺序；C4保留回顾性支线。所有未来端点固定，不用test逐轮调参/选best round。本文阈值和资源数字属于计划/估计，不是实测验收结果。

所有历史结果保持原字节，统一表只做带出处的合并引用，不替换原指标。audit.py与reports.py为本轮入口；verification.json和ARTIFACT_SHA256.csv提供校验。公开审阅副本仅报告、参数、摘要和脚本文本，原始信号、模型权重/ONNX和特征矩阵保留本地。
