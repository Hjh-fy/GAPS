# 审阅范围与保留失败

本包结论BLOCKED，不是完整六组联合性能报告。一次真实执行1376/5040条：B0=1、B1=13、B2=840、B3=507、B4=9、B5=6。五组有限logits逐分量超原容差，1376条argmax全部与旧P35相同；没有修改batch/阈值或重跑。无全六视图锁，score未运行，不发布B2单独成绩或部分排名。

README→EXECUTION_AUDIT→NUMERIC_CONSISTENCY→JOINT_EVALUATION_REPORT→NEXT_DECISION。NUMERIC_DRIFT_EVIDENCE公开5条事件的4维logits、逐分量阈值和窗口哈希，可独立重算；不含信号、71D值或模型权重。完整1376条预测、840窗身份、模型与原始输入仅本地。公开ARTIFACT清单列其来源SHA并不意味着文件已公开。OBSERVED_PREFIX_CALLS是调用可用性诊断，不是完整流量/条件性能报告；未执行3664条不冒记UNAVAILABLE。

FINAL授权唯一有效且已消费；旧预执行注册/授权已supersede保留，不能当第二次运行许可。预执行严格logits门禁修复、一次事后验证工具误用unittest导致0测试及修正pytest记录均保留；数值和算法代码在执行后未改。14纯合成/假对象测试通过。独立审计0模型调用。

原P37 bridge/assets内容不改，新权限回调仅批准Day22数据白名单；应用层读前检查不声称OS隔离或隐私。历史P37/P35/P36/P15共2566文件复核原SHA，v5未写、未额外全量重hash。公开路径脱敏，源SHA与发布SHA分别列在PUBLICATION_MANIFEST。

PC数值验收与Pi部署继续BLOCKED。底层数值差异原因未确认，不能认定某后端/模型层是根因；下一阶段仅建议单独注册合成/Calibration诊断。本轮无训练、refit、额外seed、调参、Pi或attempt2。
