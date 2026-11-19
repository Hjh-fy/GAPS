# Sprint 9 — 跨时间适配强基线

## DONE

复用Sprint8的UCI270 ZIP及128D加载器、Sprint1A Bias校准器。完成Batch2标定→Batch3评价至Batch9标定→Batch10评价的8个滚动步骤。先跑单步pilot，扩展时直接复用源银行和首步结果。仅6次Batch1源Ridge拟合、360次标量Bias拟合（含LOO），无网格搜索。

三方法使用同一固定Batch1源银行，当前批次每气体B2/B4记录拟合唯一Bias候选。Simple LOEO实际是**record-LOO**，只依当前标定均值损失差决定是否接受；拒绝时回到固定Source，不保留上一步Bias。每步决策封存后才评分下一批次。

### 实际结果（raw）

主动态曲线固定五气体组成；NRMSE按各气体Batch1浓度跨度归一化，再对40个气体—时间案例宏平均，不混合不同气体ppm。

|每气体记录预算|Always Source NRMSE%|Always Adapt NRMSE%|Simple LOEO NRMSE%|Adapt / Simple有害更新|Simple漏掉有益更新|Simple接受率|
|---|---:|---:|---:|---:|---:|---:|
|2|42.645|32.686|33.052|12 / 9|7|35/45（77.8%）|
|4|42.645|31.892|31.051|13 / 7|2|37/45（82.2%）|

有害更新及遗漏计数覆盖全部可评分六气体案例：每预算45例，其中44例有候选。接受率分母是45次有候选的决策，包含一次下一批次没有Toluene、不能评分的已接受更新；因此Simple有害更新分别为9/34、7/36次可评分已接受更新。完整全可用气体宏平均见`analysis/SUMMARY.csv`：Source45.949%，B2 Adapt33.772% / Simple34.097%，B4 Adapt33.016% / Simple32.269%。raw与clipped分别保留，以上未用clip掩盖负收益。

Simple并未稳定优于Always Adapt：B2略差，B4略好；减少部分错误更新，同时拒绝了部分有益更新。逐气体、逐批次、后悔值及版本链均已保存。共享协议累计消耗B2=90、B4=180条标定浓度标签；Always Source本身新增标签需求为0。B2嵌套B4，两预算合计唯一标定记录180，不是270个独立样本。

### 值得解决的具体问题

Ethylene Batch7→8：B2标定LOO预测改善（ΔNMSE=−0.579），但NRMSE由Source12.375%变为Bias79.244%；B4同样接受，变为59.603%。标定Bias分别为−213.780、−160.899 ppm，下一批次Source平均残差仅−6.265 ppm。

对固定源预测器，已封存结果精确满足：`MSE(Bias)-MSE(Source)=b²−2b·mean(y_next−p_source)`。该事后分解支持“标定偏置不能稳定迁移至下一批次、可能过度校正”这一局部解释；未来残差只用于分析，从未进入选择。不能将其直接归因为物理老化或宣称可靠性算法已经成立。

**结论：保留Simple LOEO强基线；不开发EBSU v1。存在研究价值的缺口是当前标定收益向下一时间批次的可迁移性，而非继续调整硬门槛。** 若开展下一轮，应另行固定跨时间机制和验证协议，不能从这8步反向选参数。这是单次采样轨迹、固定Ridge+Bias候选下的描述性证据。

## 可复现入口

在仓库根目录执行（输出目录必须为全新目录；现有`pilot1/full1`只读保留）：

```powershell
[Local machine path/command omitted in public snapshot; original SHA retained.]
$s9 = 'development/fast_sprint9_20261118'
& $py -B -X utf8 "$s9/test_rolling.py"
& $py -B -X utf8 "$s9/run_rolling.py" --out "$s9/results/pilot_new" --through 2
& $py -B -X utf8 "$s9/run_rolling.py" --out "$s9/results/full_new" --through 9 --pilot "$s9/results/pilot_new"
```

本次原始输出在`results/pilot1`、`results/full1`。`PREDICTIONS.csv`73326行、`METRICS.csv`540行、`DECISIONS.csv`288行；不是73326条独立采集样本，而是相同下一批次记录在预算/方法视图下的展开。`analyze.py`、`diagnose_bias.py`和`plot_curves.py`只读取封存CSV，默认指向`full1`，使用独立全新分析/图目录且拒绝覆盖。核验入口为`verify_delivery.py`；`FINAL_VERIFICATION.json`与`DELIVERY_SHA256.json`记录结果及来源。

## BLOCKED / 限制

Toluene在Batch3–5缺失：不填样本或指标。Cal5→Eval6无合法Toluene候选，三方法均回退Source。记录不等于Twin式独立物理暴露；只使用Oracle气体元数据做回归，不包含分类任务。公开历史批次回顾性模拟，不是首次盲测；完整评分标签在程序角色上隔离于后续拟合，非操作系统级标签隔离。无独立重复、显著性或置信区间主张。

## DEFERRED

EBSU v1、新选择阈值、更多候选、多种子、新外部数据实验及任何Pi/Flower部署均未执行。GAPS历史数据、Day22、冻结模型、正式bundle、main均未修改。
