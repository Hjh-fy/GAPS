# Phase3.11 实机离线验收

继承 Phase3.10 固定提交 ae2a614249443edaeae78d421851a3bdc57a92d6 及 PC-F32-CPU-SINGLE-v1。复制 6 个原分类端点、4 个原 joblib 回归头、原工厂、适配器、71D 特征和 JointBridge。便携加载器只负责路径重定位、严格加载、完整性及调用记录。

| 顺序 | 证据与停止条件 |
|---|---|
| 环境锁 | 只接受 Linux aarch64 的 Raspberry Pi 5。记录实际依赖、线程与算子配置。依赖不匹配即 BLOCKED |
| Gate A | 32 Calibration 与 4 固定合成窗，六组同 PC golden 对照。包括同进程重复、完整模型重载和独立进程恢复，以及异常与旧快照检查 |
| Gate B | Gate A 通过后才在 PC 制备无标签 Day22 载荷。用原身份及逐窗哈希核对 840 窗，与封存 PC 5040 条预测逐行比较 |
| 性能 | Gate B 通过后独立执行。记录冷进程首窗、加载、分类、71D、SVR、整段以及持续运行资源原始记录 |

数值判定规则和性能目标先固化于 PI_PROTOCOL.json。跨平台逐分量比较使用固定 PC 参考和 atol+rtol×abs(reference)。同 Pi 重复、重载及进程重启要求保存输出一致。错误信号和故障探针的预期 UNAVAILABLE 单独验收。

没有实机时只交付准备证据，状态 PENDING_HARDWARE。PC 测试只检验包可搬运和入口正确。Stage B 输入不进入初始传输包，避免在 Gate A 前打开 Day22 数值缓存。
