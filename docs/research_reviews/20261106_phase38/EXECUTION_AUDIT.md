# 唯一执行审计

FINAL注册/授权绑定Phase3.7合同、容差、模型来源、10个数值/执行代码SHA、6个输入文件SHA、40个raw白名单及独立run1。原授权模板不是执行授权；预执行审查修复无效logits停止条件后，旧未消费注册被显式supersede，最终授权只消费一次。旧文件保留，`execute.py`只读取FINAL路径。

`AUTHORIZATION_CONSUMED.json`在首次资产加载前以独占写入创建；`run1`和消费标志任一存在即拒绝再运行。消费时间及授权/注册SHA均保存。权限扩展通过新回调绑定原loader，原P37文件内容未改；这是应用层审计白名单，不是OS安全或隐私隔离证明。

预检先完成40次Day22原始暴露、840有效窗、50×8 float32信号、71D float64特征、原始SHA、时间/标签/ppm和三份身份索引对齐；与历史输入缓存逐位一致，scaler匹配原Calibration证据。真实gas不进入`predict_window(window)`，只在调用后加入记录用于评分。

|视图|已执行/计划窗|状态|首次超容差row（0起）|
|---|---:|---|---:|
|B0|1/840|BLOCKED：logits漂移|0|
|B1|13/840|BLOCKED：logits漂移|12|
|B2|840/840|视图PASS，整体BLOCKED|—|
|B3|507/840|BLOCKED：logits漂移|506|
|B4|9/840|BLOCKED：logits漂移|8|
|B5|6/840|BLOCKED：logits漂移|5|

实际分类1376次、71D提取1376次、回归1376次；独立特征预检还提取840次，不混为联合预测记录。每条预测的head等于预测gas，错误路由也不调用Oracle补救。部分记录包括状态、raw/clip、单位、支持、模型/数据SHA及Phase3.5对照logits。已保存的五条漂移均为各视图末条，随后停止该视图，其他预定视图按相同合同执行。

`run1/MATRIX_BLOCKED.json`已写入，无`ALL_SIX_VIEWS_LOCKED.json`，无evaluation目录，未运行score。B2完整不解除全矩阵评分门禁。总计1371条logits MATCH、5条NUMERIC_LOGIT_DRIFT，类别argmax漂移0；1376条推理OK、UNAVAILABLE 0，范围仅限已执行前缀。

六视图计时合计4.505804秒，含视图调用/记录、部分加载首次开销；不含完整raw预检/导入和总进程墙钟，不能解释为单窗部署延迟。总进程墙钟与峰值内存未形成独立receipt，控制台输出未完整归档；结构化逐窗、事件及锁均在。没有训练、额外模型前向验证或Pi测试。

后续仅对保存的JSON/哈希做审计，无新模型调用；14项纯假对象测试在执行前通过。最终独立检查及历史制品哈希复核另列。冻结v5与旧Phase资产没有被写入。
