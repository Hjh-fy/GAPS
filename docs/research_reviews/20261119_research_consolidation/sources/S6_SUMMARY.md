# Sprint 6 — REPLAY PASS / LIVE BLOCKED

2026-11-18。已交付采集入口、真实历史数据重放、Pi分类、Qt显示和逐行记录。**未完成真实在线气体闭环，不宣称LIVE通过。**

## 实际完成

- `pi_collect.py`：Pi系统Python直接运行原Qt `SerialWorker`及43字节/20字段解析器，保留原时间戳、16路ADC、温湿度和辅助UART字段；PC可使用原SerialWorker本地采集。Qt Connect/Disconnect可重新建立采集线程，断线不退出应用。
- `run_ui.py`：复用原MainWindow缓存/曲线及Sprint5 SSH PipeClient。区分LIVE、STM32-REPLAY和LAB-REPLAY；显示来源、设备、类别、ppm状态、模型版本和保存路径。实验室数据没有填充到16路ADC曲线。
- `server.py`：复用原实验室 `EdgeAIRuntime`、TorchScript和冻结P2归一化参数。原始ADC输入无条件拒绝，不能靠 `_model_input_precomputed` 标记绕过。仅有来源锁的六通道实验室REPLAY可执行分类。
- 记录复用`ExperimentSession`：raw.csv、meta.json、事件日志和edge_ai_predictions.csv；逐行flush，避免断线丢失尾部记录。完整原始电阻文件另保存在测试包`inputs/lab_original.csv`，会话中的lab raw.csv是**模型预处理后逐点输入**，不冒称原始电阻。

## 已执行结果

|项目|结果|证据|
|---|---|---|
|新增兼容性/解析/记录测试6项|PASS|TEST_LOG.txt|
|Qt断线状态信号测试1项|PASS|QT_TEST_LOG.txt|
|PC真实数据重放|PASS|results/pc_lab2、pc_stm1|
|Pi STM32历史重放62行|PASS，62行完整保存，推理全部拒绝|results/pi_stm_final|
|Pi实验室原始电阻→100×6窗口→分类|PASS，输出乙醛/class0|results/pi_lab_final|
|浓度|UNAVAILABLE，未加载任何浓度头|分类CSV的三个ppm字段为空|
|坏候选保持健康实例、合法重载、非法输入恢复、断线|PASS_PI5|results/pi_checks|
|真实串口LIVE|BLOCKED_HARDWARE，0行|results/pi_live_probe|

实机测试使用`[PI_HOST]`，原CPython3.12/Torch2.5.1环境运行分类，Pi系统Qt5.15.15/pyserial3.5运行原串口采集。已有USB/ACM/rfcomm传感器设备未发现；实际打开`/dev/rfcomm0`返回端口不存在，Qt未崩溃。未打开用途未知的ttyAMA10。

实验室模型要求CH1/2/4/6/8/9的电阻（ohm），300秒暴露前中位数基线，原训练协议的1Hz重采样、相对电阻与5秒中心平滑，100点窗口；冻结归一化仅来自P2训练数据。预处理直接调用已有`build_exposure_records`，选第一份P3原始文件的E2、暴露后250秒固定窗口，无选参、训练或评分。中心平滑依赖未来2秒，当前仅用于明确标记的离线REPLAY。

原STM32帧只有3路已有电阻推导，尚无上述6路ADC→电阻及通道映射的已验证合同，且公开模型要求不同的8通道/50点输入。**即使串口重新连接，目前也只能合法采集、保存并显示不兼容，不能解除在线分类阻塞。**实验室模型标签是乙醛/甲烷/乙酸，不复用公开四气体映射。没有兼容浓度模型，ppm始终UNAVAILABLE。

## 启动

仓库根目录执行，每次建立新的本地/Pi输出目录：

```powershell
./development/fast_sprint6_20261118/START_APP.ps1 -Mode lab-replay
./development/fast_sprint6_20261118/START_APP.ps1 -Mode stm32-replay
# 接回实际串口后可采集；推理仍受电气合同阻塞，不自动解除：
./development/fast_sprint6_20261118/START_APP.ps1 -Mode live -Port /dev/rfcomm0
```

PC离线入口（去掉--offscreen显示窗口，目录须新）：

```powershell
[Local machine path/command omitted in public snapshot; original SHA retained.]
[Local machine path/command omitted in public snapshot; original SHA retained.]
```

Pi测试目录`/home/gaps/gaps_sprint6_20261118_test1/`，入口`payload/server.py`、`payload/pi_collect.py`，全部历史模型/正式部署包未修改。原始来源、参数和各文件SHA见`build_final/payload/SOURCE_LOCK.json`；`FINAL_VERIFICATION.json`核对62/100行时间戳、数值及实际分类CSV，`DELIVERY_SHA256.json`为交付哈希。

## 安全切换与边界

Sprint5安全管理器固定为四分类/71D/四SVR合同，不能把三分类实验室包伪装成该接口。本轮不修改其算法或放宽验证。实验室分支仅增加薄重载适配：调用已有验证加载器在独立实例中加载同一个登记包，通过后才替换引用，故障时保留旧实例；Pi接口测试已验证。GUI不开放任意候选或未登记模型选择。

首次PC重放发现旧记录器会自动写入两列实验索引，已补入新lab记录列；失败目录pc_lab1保留。后续build1/2/3是中间制品，最终使用build_final。历史Day22访问、模型调整、全量验收重跑和云端训练均为0；Phase3.11保持BLOCKED。没有自动合并main。

下一步仅需补齐硬件连接及六路通道/电气转换证据，再决定是否授权真实在线分类验证；本轮不通过补零、拼接或随意插值绕过。
