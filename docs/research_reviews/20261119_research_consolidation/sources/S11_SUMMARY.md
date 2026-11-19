# Sprint11A — STOP_LOCAL_DISTANCE_REFERENCE

## 实际完成

复用Sprint9的128D特征缓存、Batch1冻结scaler、源/完整Bias逐记录预测及相同标定ID。完成全部8步Cal2→Eval3至Cal9→Eval10：24442行预算视图的距离/权重诊断、122210行五方法预测视图、900行raw/clip指标。没有重新训练、调用Ridge.predict或重新fit scaler。

仅测试一个规则：在原source scaler空间用RMS欧氏最近邻距离；尺度s为源训练同气体留一记录最近邻距离的中位数；`w=exp(-0.5*(d_cal/s)^2)`，再插值Source与当前Bias预测。s只使用Batch1；不使用未来浓度、历史未来误差或目标全批特征拟合尺度。d_source仅作诊断，不进入权重；Simple LOEO完全保持原全局决定。

## 五方法对照

主指标为固定共同五气体、40个气体—时间案例的raw宏平均NRMSE，各气体按原Batch1浓度跨度归一化：

|每气体标定记录|Source|Full Bias|Simple LOEO|Fixed Half|Simple Local|
|---|---:|---:|---:|---:|---:|
|2|42.645%|32.686%|33.052%|33.540%|42.588%|
|4|42.645%|31.892%|31.051%|33.075%|42.421%|

全部可评分六气体45案例/预算的raw风险与有益校正抑制：

|预算/方法|有害案例|累计额外NMSE|最严重负迁移pp|抑制有益校正代价NMSE|
|---|---:|---:|---:|---:|
|B2 Full Bias|12|0.948884|66.869|约0|
|B2 Simple LOEO|9|0.865103|66.869|0.757289|
|B2 Half|10|0.207189|27.492|2.524062|
|B2 Local|7|0.001242|1.458|8.415120|
|B4 Full Bias|13|0.750811|47.229|约0|
|B4 Simple LOEO|7|0.646594|47.229|0.012127|
|B4 Half|12|0.166203|18.141|2.649994|
|B4 Local|7|0.000641|0.790|8.757908|

额外损失=sum_case max(NMSE_method−NMSE_source,0)。抑制代价是对完整Bias原本有益的记录，计算方法相对完整Bias新增的正向归一化平方损失，先案例内记录等权、再案例求和；也计部分更新不足，**与Sprint10“完全拒绝更新”的漏更代价不同，不能直接混比**。Source抑制代价8.531167/8.912470。不同气体ppm未混池；记录数不是独立统计样本数。

Local仅在B2 Ammonia相对Simple有微小改善（24.592% vs24.755%），其余逐气体比较更差。全部逐气体及逐批次raw/clip结果保留，不能用单个成功案例宣称整体有效。

## 局部有效性证据

- 每预算44个合法候选且可评分的气体—时间案例，`Spearman(d_cal,full_gain)`中位数B2=−0.101（24负/20正），B4=0.0077（22负/22正）。full_gain为Source平方误差减Full平方误差，正数才表示有益。没有稳定、跨预算一致的距离辨别证据。
- d_source与full_gain的案例内相关中位数为+0.319/+0.287，不能将“偏离Source”直接等同“校正危险”。Bias完整校正幅度在单个气体—时间案例内恒定，相关性44/44未定义，不能虚构其逐样本辨别力。
- 案例间“真实浓度落在标定范围外的比例”与中位d_cal的相关为0.272/0.420，支持距离反映部分覆盖变化；它与校正收益是不同问题。所有浓度分层只在权重封存后作描述，从未用于预测区域定义。
- 完整Bias在B2记录视图中有2835次方向反转、1229次同向幅度过冲；B4为2851/1222。这些记录存在同批依赖、两预算重复，不能据此扩大统计有效样本量。

**Ethylene Cal7→Eval8**：Cal高浓度、Eval全部10ppm的30条记录，d_cal/s中位数25.216/9.643，平均w约9.14e−81/1.17e−11；Local几乎退回Source，raw NRMSE保持12.375%，完整Bias为79.244%/59.603%。确实挡住了该次过冲，但不能代表其余案例。

总体44个案例的“案例内权重中位数”的中位数仅5.07e−7/3.17e−4，说明固定参照严重衰减。源NN尺度来自较密集源训练集，没有针对2/4点稀疏标定参照做预算匹配，这是本参照的限制；本轮不据结果换尺度，也不能据它否定所有可能的局部校正方法。

## 成本、访问与核验

当前标定仍90/180条，Local新增浓度标签0；B2嵌套B4，两预算唯一标定记录180。共同源训练445条另列；Source自身无需新增标定。首次读取了12221条未来评价记录的128D缓存用于距离分析，float64特征约11.93MiB。加载器共解码13910条全数据缓存（约13.58MiB特征数组），未选当前记录不参与尺度或权重拟合；这不是原始传感器特征提取计时。

源几何计算约0.006秒；特征拼接、距离及逐行制表约3.90秒，完整入口约26.93秒。计时包含Python开销，诊断d_source带来额外开销，不是新模型部署延迟。原始特征获取/提取时延UNMEASURED。数值正权重不等于float64预测发生变化，实际变化计数见WEIGHT_ACTIVITY.csv。

SOURCE_GEOMETRY_LOCK封存scaler、源尺度及source身份；每步先保存不含未来浓度的WEIGHTS_AND_PREDICTIONS_NO_LABELS，再评分并生成LOCAL_DIAGNOSTICS。仍使用Oracle气体元数据，未研究分类路由。加载器会验证缓存中已有浓度字段，但浓度不传给权重函数；这是代码角色隔离，不是OS级标签隔离。6项定向单元测试通过；独立几何/权重/成本/指标重算见FINAL_VERIFICATION.json。

## GO / STOP

**STOP：不开展11B，不调距离、温度或阈值，不增加local+LOEO门控。** 原因：Local两个预算均显著差于Simple的描述性平均误差，且B4距离排序不具一致方向。局部距离避免了一次严重失效，却没有提供可靠的有益校正适用范围识别，不能宣称方法创新有效。

保留强基线和本次负结果。当前没有足够依据为该固定距离参照投入独立数据集确认；若将来另立问题，应先用合法源侧数据论证证据量与尺度的关系，不能继续拿本轮8步误差选函数。

## 运行与文件

```powershell
[Local machine path/command omitted in public snapshot; original SHA retained.]
$s11 = 'development/fast_sprint11a_20261119'
& $py -B -X utf8 "$s11/test_local.py"
& $py -B -X utf8 "$s11/run_local.py" --out "$s11/results/new_replay"
```

新输出目录必须不存在。当前完整结果在results/run1；analyze.py、describe_activity.py、plot_results.py默认分析该封存目录，拒绝覆盖已有分析/图目录。LOCAL_DIAGNOSTICS与SAMPLE_EFFECTS为逐记录覆盖/收益，DISTANCE_GAIN_ASSOCIATIONS保留每气体每批次相关，CONCENTRATION_COVERAGE仅作事后浓度分析，SUMMARY/PER_GAS/DYNAMIC为比较表，三张图同时提供PNG/SVG/PDF。DELIVERY_SHA256锁文件。

全部8步早已被观察，仅为回顾性开发；2/4条记录不称独立物理暴露。无CI、显著性或独立泛化主张。没有访问Twin Day22、连接Pi、修改历史冻结资产或合并main。
