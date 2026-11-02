# Phase 1 预执行合同（2026-11-02）

依据：用户本轮明确授权及Phase0.5固定提交a6aa42c91e2e2b562371517f3f512f6a251206b9。复用当前独立worktree，新增文件仅在research/fedongas_phase1_20261102；v5、原数据、旧结果/权重/代码均只读。

1. 原样冻结Phase0.5 ROLES_A2/A4.csv。合法源仅C1R1/Day4、C2R1/R2/Day1/7，共120暴露。目标cal使用预定F010/F100或嵌套F010/F040/F070/F100，固定[60,70)、[150,160)两窗；开发不参与任何目标选参或早停。C3 Day8/14/17；C4 Day3/3/9、C5 Day18/18/22，后两者仅两日期替代，C4只作回顾性source-before-test解释。
2. 原字节导入v5 strict_data/canonical_quantitative_features/rich_feature_dict、svr_core/frozen_target_numerics。71D名称/顺序完全一致；metadata为空、phase=-1，不把标签/日期/文件名加入sensor71。训练标准化仅用本折或标定输入，ddof0，std<1e-9置1；两模型均使用固定(y-LOW)/RANGE标签尺度，LOW=12.5/25/12.5/25，RANGE=112.5/225/112.5/225。Ridge截距不惩罚，复用原数值函数，验证与直接ppm拟合的仿射等价性。
3. 源侧leave-client-out2折。Ridge alpha=[0,.01,.1,1,10,100,1000]；SVR C=[1,10,100]、gamma_factor=[.1,1,10]，gamma=gamma_factor/71，epsilon=.01，原solver。每气体按原函数未裁剪验证RMSE/确定性首候选tie选择。源文件均21窗时，此选择亦为暴露等权；不更改CV准则来追逐目标分数。锁定源参数后才fit目标。
4. 冒烟在真实矩阵前：负向泄漏/哈希/预算/指标测试；固定seed42合成8×71输入的Ridge(alpha1)/SVR(C10,gamma_factor1)各一头，有限预测、同训练scaler、序列化精确往返、Ridge标签仿射等价。冒烟不读取目标test、不选择参数；失败即停。正式拟合128次source-CV+48目标头=176；加2合成smoke=178。无额外source bank、候选或神经网络训练。
5. 全48头及数据/参数/程序哈希封存后才生成目标test数值特征和模型预测。目标身份/真值元数据在审计时已知，不虚称未接触标签；本轮封存顺序约束的是模型选择和预测访问。development仅封存身份/原质量掩码，不生成开发预测用于选择。
6. 主结果固定为clip到本气体标定支持（两预算均端点齐全）。同时完整报告unclipped。任务是每个10秒窗输出一次浓度。主NRMSE=100*sqrt(mean_exposure(mean_window((error/RANGE_g)^2)))；主RMSE/MAE同样先窗内统计再暴露等权。全窗pool指标补充。暴露均值预测的RMSE/NRMSE另列auxiliary_mean_fusion，禁止替代单窗主指标。
7. 保存每窗原始/裁剪预测、误差、时间、暴露ID、gas/浓度和seen标签；保存每暴露的单窗MSE/MAE及均值融合误差。每气体/浓度/seen-unseen/early-stable切片共享样本。仅Oracle_ALL，不执行分类checkpoint、预测路由或QC。
8. 同test暴露配对，按4气体各10暴露分层bootstrap2000、seed42；分别报告6个设备×预算比较，不汇总三日期/两日期为同协议。CI仅为给定模型和单个test日的条件性描述，不代表跨日/跨种子稳定。以SVR-Ridge的NRMSE差为负表示改善，不通过CI选择或重训模型。
   额外描述性诊断：按已拟合SVR的训练scaler/固定gamma计算test到标定样本的最大RBF相似度，记录median及低于1e-3比例，仅用于解释核覆盖，不改变参数、样本或结果。
9. 数据/身份/时间/维度/哈希冲突停止对应实验；所有负结果保留。结果仅回顾性评估，历史test已有观察。新候选仅提出继续/停止建议，不执行KRR或残差迁移。
10. 完成后发布精简GitHub审阅副本、更新统一索引并附固定commit；不上传原始信号、特征NPZ、权重、ZIP或凭据。
