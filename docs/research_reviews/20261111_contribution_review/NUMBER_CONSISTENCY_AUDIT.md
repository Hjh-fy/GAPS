# 数字一致性审计

## 正式来源

原始指标只转录，不重新计算预测、评分或置信区间。准确率百分数显式由fraction转为%，显示精度不改变原CSV。六行配置计数为静态表的逻辑计数。

|发现|指标与范围|正式值/事实|潜在错误写法|分类/严重度|处置与精确来源|
|---|---|---|---|---|---|
|N1|旧pipeline P95|0.4127459ms，B20/20%标定|ThreePass block65说5%|scope mismatch/major|保留稿；引用历史LATENCY_SUMMARY.csv及MANIFEST.json；SHA在HISTORICAL_SOURCE_SUMMARY.csv|
|N2|旧与新联合延迟|旧ONNX/Ridge4线程；新Torch/SVR1线程|当作同任务加速倍数|scope mismatch/blocking|SYSTEM_VERSION_BOUNDARIES.md，P312 PERFORMANCE.md|
|N3|残差赢强基线|5/6胜F；2/6胜SourceOnly；1/6同时胜|5/6代表全面最优|comparison mismatch/blocking|P2B README固定六行；不是新评分|
|N4|反馈/原型Accuracy差|−1.548/−0.476个百分点|正增益或相对百分比|sign/unit mismatch/blocking|P35 RESULT_ANALYSIS.md，原CI含0|
|N5|后期风险不确定|64条中55条开发变化超零带|后期开发风险都近零|label scope mismatch/blocking|postmortem HISTORICAL_CLARIFICATION.md|
|N6|条件回归NRMSE|每gas，分类正确且回归有效的不同子集|跨气体统一总NRMSE或据其排六系统|metric/scope mismatch/blocking|P310 RESULTS.md|
|N7|Pi状态|P311严格BLOCKED；P312独立性能PASS|性能通过等于数值通过|contract mismatch/blocking|P311 NUMERICAL_AND_RECOVERY.md；P312 PERFORMANCE.md|
|N8|模型文件数量|996条清单引用含重复|996次独立模型训练|count scope mismatch/major|PRIVATE_ASSET_AVAILABILITY.csv|

## 已匹配的显示转换

P35目标0.819048显示81.9048%，Macro-F1保持0–1；P2B NRMSE保持百分数；旧Ridge相对降低9.08/4.65/3.55%不作百分点。Pi联合P99 3.936–3.992ms与持续期3.999–4.023ms对应不同测量阶段，不择低合并。

## 判定与交接

窄范围数字转录可用；上述跨范围或优势主张阻断。原稿不修改，实际提交文件身份unknown。交由后续论文修订按正式来源改写叙事，不改变原指标、负结果或历史文件。
