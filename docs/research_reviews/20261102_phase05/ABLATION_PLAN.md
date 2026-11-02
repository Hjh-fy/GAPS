# Ablation Plan

|Hypothesis ID|Factor|Levels|Held constants|Primary metric|Expected Evidence|Confound check|Stopping rule|
|---|---|---|---|---|---|---|---|
|P05-H1|回归器|TargetRidge71 / TargetSVR71|同71D/角色/标定窗/source-only HP/标签成本/test|暴露等权NRMSE/MAE|直接强基线|不得同时换路由或QC|预算/样本/参数来源不匹配即停|
|P05-H2|暴露预算|2 / 4每气体|嵌套固定档位/共同dev与test/同模型|成对暴露误差及未见档位|低预算失效边界|不换浓度、不给某方法更多dev标签|不按test回选预算或档位|

必要基线为上述两个，不添加新的ResidualSVR、rho、联合或维护网络。执行顺序、176/180小模型fit预算和条件性72D停止规则见EXPERIMENT_PLAN.md。Target-as-Client/TargetCE反馈属于后续静态配置缺口，未登记为本轮训练行。
