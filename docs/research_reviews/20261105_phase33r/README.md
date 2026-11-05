# FedOnGas Phase3.3R：Flower原生兼容与合成网络验收

**Native Flower调度PASS、受控localhost通信PASS、算法/数据合同保持不变。SOURCE attempt2具备重新申请授权条件，但未授权、未启动；六组pilot仍全部BLOCKED_NO_SSTAR。** 基于Phase3.2固定b0c7a5cabfec3a41b40cdf67fc0802b67103a3ed及Phase3.3失败固定41bcbba60a4b0b167182efc9e14873fdecd29f33。

独立研究版本新增runtime.py/contract.py，不覆盖旧文件。仅把configure_fit/aggregate_fit公开参数改回Flower规范名称，并在内部使用原局部别名；另把canonical源输入/初始锁指向只读Phase32，授权与输出留在新版本。REPAIR_DIFF及AST回归证明训练/聚合/EMA/DA/损失/预算函数保持不变。

|验收项|结果|
|---|---|
|签名/Signature.bind|修复前2失败，修复后3测试通过；全部调度回调矩阵已核对|
|原生网络两轮|真正server+两个client，Server.fit_round原生keyword configure、FitIns/合成fit/FitRes/aggregate/下一轮全state接收均通过|
|模型/初始state|原TCN50×8、4类、64D；同注册随机初始SHA，未重生成|
|预算|固定两轮合成网络316优化步/5040呈现；另轻量unit53合成步；不是正式SOURCE25轮|
|真实输入|真实源数值/目标数据读取0，进程审计拒绝真实npy/npz/raw；无SOURCE授权|
|网络权限|默认沙箱connect10013仍在；批准的普通用户宿主localhost8080成功；未改防火墙/端口，无管理员令牌|
|退出/资源|3进程exit0，正常断开，8080可重新bind|
|attempt1与未来|失败现场42制品及旧代码未覆盖；attempt2仅预注册，独立输出/新代码及环境SHA|

阅读：[修复与回归](REPAIR_AND_REGRESSION.md) → [网络验收](NATIVE_NETWORK_REPORT.md) → [权限环境](NETWORK_PERMISSION.md) → [SOURCE恢复计划](SOURCE_ATTEMPT2_PLAN.md) → [门禁](NEXT_DECISION.md)。完整参数SOURCE_ATTEMPT2_REGISTRATION及ENVIRONMENT_LOCK，指纹/成本NETWORK_VERSION_CHAIN和FINAL_VERIFICATION。
