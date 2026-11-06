# Phase3.8 六组冻结分类器联合离线评价：BLOCKED

依据Phase3.7固定版本 `ba7a143677404155e93083d0c43e15e9e72b4658`。用户批准的唯一一次 `OFFLINE_JOINT_EVAL_C5_DAY22_SIX_FIXED_VIEWS` 已消费并执行，没有重试。

计划5040视图窗（6×840，同40次物理暴露）；实际1376条记录。B2完成840窗，其余五组在首次logits超容差时停止，剩余3664次未运行。六组没有全部锁定，因此**未执行评分程序，不发布部分组联合性能或赢家**。

|视图|已执行/计划窗|状态|首次超容差row（0起）|
|---|---:|---|---:|
|B0|1/840|BLOCKED：logits漂移|0|
|B1|13/840|BLOCKED：logits漂移|12|
|B2|840/840|视图PASS，整体BLOCKED|—|
|B3|507/840|BLOCKED：logits漂移|506|
|B4|9/840|BLOCKED：logits漂移|8|
|B5|6/840|BLOCKED：logits漂移|5|

840窗strict信号及71D特征与历史缓存逐位一致；六分类端点、四F-SVR71头及scaler身份核验通过，运行前后state/scaler不变。已观察1376窗argmax与Phase3.5全部一致；五条事件是 `NUMERIC_LOGIT_DRIFT`，不是已观察到的类别变化。不能外推未运行窗口。

已观察1376次原桥接均返回OK，raw/clip有限、预测类别唯一选头、每窗一次回归调用；这不表示五个失败视图通过验收。未运行窗口不能伪记为UNAVAILABLE，完整流量不可用率和条件回归覆盖率尚未获得。

阅读顺序：[执行审计](EXECUTION_AUDIT.md) → [数值证据](NUMERIC_CONSISTENCY.md) → [评价交付状态](JOINT_EVALUATION_REPORT.md) → [部署前条件](PC_PI_READINESS.md) → [下一步](NEXT_DECISION.md)。

代码：[唯一入口](code_review/execute.py.txt)、[透明调用观测](code_review/inference.py.txt)、[权限绑定](code_review/authorized_assets.py.txt)、[保存结果审计](code_review/audit_saved_results.py.txt)、[尚未运行的评分程序](code_review/score.py.txt)。原Phase3.7 bridge/assets数值实现未改；新授权仅扩展本次Day22读取范围。

完整1376条逐窗结果、六组部分/完整JSON和原始模型均留本地。公开包只含审计、哈希、五条漂移logits及接口代码文本，不含原始信号、71D数值、完整逐窗输出或权重。训练0、scaler拟合0、Pi0。本轮是历史已观察Test的回顾性系统验收，未形成完整联合性能结论。
