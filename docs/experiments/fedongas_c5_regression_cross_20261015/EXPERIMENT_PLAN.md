# TO/FT/CLASS × 目标Ridge/源参考增强Ridge：36组流水线评价

授权：2026-10-15本对话确认设计后用户“可以运行”。仅C5；不重训任何分类器或源FedRidge，不扩展FULL/QC/其他target。
输出根：`results/fedongas_c5_regression_cross_20261015/`。既有结果只读，不覆盖。

## 假设和控制

H1：同一classifier/预算/seed下R84相对83D降低S_ALL浓度误差；允许负、零收益，按gas报告差异。
H2：同一regressor/预算/seed下FT、CLASS相对TO改善最终任务；分类器只影响预测路由，不能称其改善回归映射。
H3：FT+R84、CLASS+R84相对TO+83D的联合复用流程是否值得采用；不预设优越性，不把CLASS叫作FULL。

3classifier × 2regressor × 2budgets × 3seeds=36条评价记录；分类端点为20261014研究已完成的18个TO/FT/CLASS模型及缓存。
预算5%=原80窗口、20%=原320窗口，40strata每格2/8；5%严格nested于20%；所有test固定1360及原身份顺序。
canonical-v1 HZ5_MEAN_W10S，5 Hz/50×8。83D/H1特征从原canonical/budget数组重新提取，不使用legacy缓存。
目标浓度必须来自原budget数据集真实regression_labels并逐行核对metadata；20261014分类兼容view里的zero regression占位禁止使用。
H1是现有104D固定5Hz描述符，包括21项协议/phase元数据，不是classifier embedding，不声称采样率不变。

## 共享拟合与参数规则

目标83D Ridge：每gas用真实calibration类别分组，传感统计83D。
R84_CONCAT：同样83D加该gas冻结source FedRidge的1D预测；source H1、alpha、scaler、coefficients均复用已PASS的R0-v2 model_lock，不重训。
source prior对三个classifier及三个seed完全相同。feature/selection不依赖classifier，因此每预算/方式只拟合一个4gas bank，共4banks/16最终gas fits。
采用既有canonical_r1_v1数值函数：float64、校准内standardization、未正则化intercept、numpy.linalg.pinv；raw预测用于CV，最终预测按calibration gas min/max截断。
alpha候选仅`0,.01,.1,1,10,100,1000`；deterministic 5-fold raw-filename-grouped calibration CV，pooled RMSE最小、first tie。
5%每gas20窗口/20文件组，20%每gas80窗口/20文件组；每方法使用同一组折分，不用classifier route或test指标调参。
这是已授权的固定参数选择规则，不追加grid、模型选择规则或调参范围。禁止复用旧20% target Ridge系数/alpha到5%。

## 评价、配对和停止

先锁定18个classifier checkpoint/hash、原预测缓存hash、source prior和4个target banks/alpha/folds；随后读取test并完成正式regression评价。
test分类路由直接从缓存window_index取出并核对真实类/总数；无classifier forward或训练。
主表36行S_ALL（预测类别路由，全部1360）；总体和按真实gas切片输出RMSE/MAE ppm、NRMSE_range（无量纲）。
固定gas量程：Ethanol/Ethylene=112.5 ppm，CO/Methane=225 ppm；总体sqrt(mean((error/range_true_gas)^2))。
诊断包含S_CC、Oracle_ALL、Oracle_CC；同预算/方式Oracle_ALL应在classifier/seed间严格相同，明确共享，不把复制值当独立回归重复。
保存全部per-window实际浓度/路由/四条回归预测，gas和scope汇总、12组mean/sample SD(ddof1)、按seed配对差值。
配对共66行：3项R84−83D、每regressor下FT−TO/CLASS−TO/CLASS−FT、FT+R84−TO+83D及CLASS+R84−TO+83D，均分别跨两预算×三seed。负误差差值表示改善。
不做显著性声明，不把test窗口当独立设备；既定window/raw-time邻域重叠局限保留。源参考同时编码冻结H1元数据，不能将其收益单独归因于纯source浓度先验。
执行基础测试、compileall、固定输入/checkpoint/预测/模型hash核验，保存失败证据，精确commit+push后停止；不自动训练或优化。
