# FedOnGas C5 5%/20% 三种子研究：执行规格

授权来源：本对话2026-10-14用户给出的27任务完整规格，以及要求重试Pi并沿用既有节点。
研究ID：FEDONGAS-C5-BUDGET-MULTISEED-20261014。
输出：results/fedongas_c5_budget_multiseed_20261014/。

## 假设与比较

- H1：FT−TO评价对应seed的source训练初始化作用。
- H2：FULL−FT评价完整interleaved commissioning相对纯监督目标微调的表现。
- H3：FULL−CLASS评价源域class-phase知识组织对C5 late目标的作用。
- 两档预算分别计算每seed差值，再报告均值和sample SD（ddof=1）；不预设正收益、不自动检验显著性。

## 严格范围

SRC 3次（42/43/44）；TO、FT、CLASS、FULL各2预算×3seed=6次，总27次。
所有新任务独立输出，不复用旧SRC/target checkpoint；同seed FT两预算共用本轮SRC。
CLASS/FULL每预算从同seed共同随机初始化完整重跑，适配参数逐轮返回源client。
5%/20%沿用既有80/320窗口CSV，40个class×concentration strata每格2/8窗口；不重新抽样。
C5测试固定1360；输入50×8；C1/C2 source；原TCN-Attention四分类器。
25 source FL rounds，每client每round 1 epoch，Adam5e-4，batch32；目标累计2500更新。
TO/FT采用25 blocks×100updates，每block重建Adam，全分类器目标CE，source CE关闭。
CLASS/FULL每round100updates，target CE=0，source CE=1。
global MMD²、class MMD、class-conditional CORAL、原ADV各0.5；anchor0.3、fit0.05、consistency2.0。
保持已核实A4的prototype MMD0.2；FULL stage MMD0.2，CLASS stage MMD0。
EMA0.8old+0.2new；ce_stats；client align/replay关闭；selective aggregation关闭。
device residual保持原A4 unavailable，legacy align regularizer disabled。
源侧phase统计的真实信息只用于FULL；CLASS在统计前按class汇总、忽略真实phase，并按class查询。
CLASS的兼容序列化key(c,0)中0仅为固定占位，库只有4项，不代表真实phase0。
CLASS输入view的phase数组由行数生成固定0，不从真实phase文件转换；concentration生成0占位。
FULL输入view的phase沿用原文件；目标concentration也不进入适配API。
TO/FT输入API只有x/class；TO读取随机初始化资产，不读取训练后SRC模型；FT只额外读同seed SRC。

## 初始化与随机性

每seed生成一个共同random state asset，SRC/TO/CLASS/FULL严格加载相同ordered fingerprint。
FT初始化为对应SRC round25状态。服务器DA模型重建要使用当前seed，修复旧辅助函数隐含42的调用默认值。
保留原DA source calibration的500-row上限和RandomState(42)选择规则；这是固定源样本支持，不随seed重新抽样。
target calibration全部使用，预算身份不随seed变；DataLoader和训练随机性使用运行seed。

## 运行资源

原三机拓扑：server root@121.40.139.213 /root/gaps_env/bin/python；
C1 gaps@192.168.137.172 /home/gaps/GAPS/gaps_rpi_env/bin/python；
C2 root@114.55.171.63 /root/gaps_c2_cpu_env/bin/python。
均CPU；保留专用环境并记录实际版本。PyTorch当前分别2.12.0+cu130/2.12.1+cpu/2.12.0+cpu，Flower1.23.0。
全部方法使用同一各节点环境；该新研究不声称与历史2.5.1执行位级一致。
部署由freeze commit生成的隔离runtime；现有未提交server_app/strategy/target_information不写入本次runtime。

## 测试、保存及停止

运行前验证预算hash、nested身份、strata、测试身份、形状、共同初始化、目标CE梯度、CLASS统计和查询phase不变性。
保存每次配置、代码commit、数据hash、日志、checkpoint/hash、更新数、耗时和失败记录。
已完成端点不得覆盖或重跑；部分失败立即停队列并保留证据，不修改loss/steps/split/seed矩阵。
27训练端点完成后才统一打开target test；输出24行target结果、8行均值/SD、逐类/混淆矩阵和18行seed配对差值。
SRC评价使用round25聚合；CLASS/FULL round25 adapted；TO/FT update2500。
结论只限于现有window-level protocol；3seed反映训练随机性，不等于未见exposure泛化或统计显著。
完成测试、分析、审计及GitHub同步后停止。
