# Sprint 8：EBSU v0 负结果与下一步

**EBSU v0没有优于简单LOEO选择，不应进入生产更新集成。** 代码与合法源侧真实运行均完成；不因负结果改阈值或扩大候选。

## 固定实验

P1：C2 Day1源→C1 Day4伪目标；P2：C1 Day4源→C2 Day7伪目标。沿用冻结物理暴露身份、71D、每气体2/4次标定、每暴露60/150秒双窗；每气体验证6个互斥物理文件126窗。共16个气体/预算案例。候选仅为原Sprint1A Bias-only，Source为Phase2B P1/P2 SourceRidge71，无新Source模型或分类器训练。

原始数值函数、暴露等权损失与clip口径复用旧实现；AlwaysSource/AlwaysAdapt与Sprint1A逐气体raw/clipped指标逐项精确一致。这里是Oracle气体隔离回归，不是新的分类—回归端到端结果。

## 主结果

下表为**16个气体/预算案例的NRMSE算术宏平均**，单位%；不是把所有窗口混池后开根号的整体NRMSE。主口径为单窗误差、暴露等权、clipped；raw完整保留。

|方法|宏平均NRMSE %|接受更新|负迁移案例|有益更新被拒|事后Oracle平均后悔值 pp|
|---|---:|---:|---:|---:|---:|
|Always Source|11.0287|0/16|0|10|2.7955|
|Always Adapt|8.3896|16/16|6|0|0.1564|
|Simple LOEO|8.5345|5/16|0|5|0.3013|
|EBSU v0|11.0287|0/16|0|10|2.7955|

EBSU相对Simple LOEO退步2.4942个百分点，相对Always Adapt退步2.6391个百分点。其“零负迁移”来自全部不更新，不能作为保护机制有效的证据。只看A4也未改善：EBSU 11.0287%，Simple LOEO 8.0784%；A2 EBSU全部返回证据不足，没有虚构置信区间。

Simple LOEO接受了P1 A2的CO/Ethylene和P1 A4的Ethanol/CO/Ethylene，当前开发数据上均改善。P2全部拒绝，在避免错误更新的同时也错过Ethylene等小幅收益。Always Adapt的6个退步案例为：P1 A2 Methane；P2 A2 Ethanol/CO/Methane；P2 A4 Ethanol/CO。最大一次退步为P2 A4 Ethanol的0.9787个百分点。逐例数值见`analysis/CASE_COMPARISON.csv`及`NEGATIVE_TRANSFERS.csv`。

## 消融与机制判断

- 删除覆盖门槛：接受P1 A4 CO/Ethylene两次有益更新，宏平均降至9.8799%，但仍不及Simple LOEO。
- 删除残差或最差LOEO稳定性门槛：接受率仍为0，结果不变。覆盖门槛阻塞下，不能据此断言另两项在一般情况下无用。
- 固定95%源几何覆盖门槛在14/16案例不满足；它测量几何距离，不保证校准映射风险。跨设备需要校正的偏移可能同时触发“不覆盖”。这是当前病例中的支持性观察，非因果证明。

**停止EBSU v0固定硬门槛方向，不用当前误差反向放宽1%/95%或一致性要求。** Simple LOEO可作为下一轮跨时间最小强基线，但它也未超过Always Adapt的平均误差；需同时保留收益、错更新、错拒与后悔值，不能只报零负迁移。

## 成本、可复现性与边界

没有额外标定暴露或目标Development标签；A2/A4分别为每头4/8窗。16个完整Bias拟合＋48个LOEO标量拟合，共64次，方法间共享。记录的标量拟合约0.0077秒，选择阶段约0.1283秒（含这些拟合），进程总墙钟约3.81秒；全进程峰值约344 MiB包含原数值依赖导入，不是选择器增量内存。

本轮保守规则在验证预测之前封存，未改动。初次运行在汇总CSV序列化时遇到重复budget字段；只修复输出字段后另开source_run2。两次已生成的基础预测和指标表SHA完全一致，失败现场保留。主基线复现、规则回放及哈希检查见`analysis/VERIFICATION.json`。

P1/P2验证结果曾在旧Sprint被观察，本轮是回顾性开发；预算共享验证暴露，跨任务源数据也有复用，不能将16案例当独立统计重复。无显著性检验/可靠置信区间，无C3/C4/C5标签访问、Day22访问或新目标性能结论。源侧覆盖还使用源暴露特征中心，不能宣称纯充分统计量联邦实现。

## 外部准备

已从官方来源获取UCI浓度版与2025 Long-Term数据的元数据及真实加载小样本，两个接口通过，外部训练为0。优先UCI做“早源批→较晚标定→更晚评价”的新协议；预算称测量记录，不能伪称Twin式独立暴露。Long-Term只对明确标有ppm的Diacetyl/Phenylethanol准备文件/日期级回归；EtOH浓度未知，日期格式及缺Day14先核对。见`EXTERNAL_DATA_NEXT.md`。

## 文件与命令

- 方法：`ebsu.py`；可运行源侧入口：`run_source.py`。
- 注册、参数、逐暴露LOEO和决策锁：`results/source_run2/REGISTRATION_LOCK.json`、`LOEO_FOLDS.csv`、`DECISIONS_LOCKED.json`。
- 对照、逐窗基础预测与成本：同目录`METHOD_COMPARISON.csv`、`BASE_WINDOW_PREDICTIONS.csv`、`SUMMARY.csv`、`COST_AND_VERIFICATION.json`。
- 逐气体、消融、原因日志及实验配置：`analysis/`。
- 外部加载器：`external_loaders.py`；官方原始REST响应及真实样本：`external/fetch1/`。

```powershell
[Local machine path/command omitted in public snapshot; original SHA retained.]
[Local machine path/command omitted in public snapshot; original SHA retained.]
[Local machine path/command omitted in public snapshot; original SHA retained.]
```

11项新增定向测试通过。未修改main或冻结资产，没有把无可靠增益的方法接入Qt/Pi/模型切换器。下一步仅建议另行预注册小规模跨时间基线验证，不自动启动。
