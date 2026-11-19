# Sprint 5 — DONE / PI5 APPLICATION PASS

2026-11-18。先完成PC进程模拟，再通过已授权SSH在真实Pi 5运行；不是WAITING_FOR_PI。仅Calibration第0窗重复调用，无Day22推理/评分、训练或正式包写入。Phase3.11仍BLOCKED，Phase3.12历史结论不变。

## 可运行功能

- `endpoint.py`：SSH标准输入/输出JSON请求，复用Phase3.11/3.12 portable FP32 runtime、71D和四SVR头，以及原Sprint2 `safe_update.py`、Sprint1B `joint_output.py`。无监听端口、MQTT、Registry或新GUI框架。
- `transport.py`：顺序请求、超时、EOF、设备身份和JSON校验；计算发生在Pi，不在Qt侧偷偷推理。仅开放登记候选initial/candidate与CAL0/S_zero/非法输入测试，不能请求任意数据文件。
- `run_qt.py`：继承Sprint4现有窗口及QThread信号链，显示设备、气体、class_id、ppm、head_id、状态和版本。旧STM32串口未强制接入。
- 独立包复用原数值模块和全部六分类端点，当前应用固定B3，非事后选择。v1/v2模型字节相同，仅包版本不同，验证软件切换能力，不代表新训练权重。

## 实际结果

|检查|PC模拟|Pi 5→Qt实机|
|---|---|---|
|单窗真实冻结模型输出与显示|PASS|PASS|
|合法候选v1→v2|PASS|PASS|
|错误SHA拒绝，健康实例身份保持|PASS|PASS|
|非法输入及恢复|PASS|PASS|
|断开通信后GUI继续响应|PASS|PASS|

新增5项单元测试通过，见`TEST_LOG_FINAL.txt`。PC结果为`results/pc_sim1/RESULTS.json`；最终实机为`results/pi2/RESULTS.json`（六场景、显示字段与后台相等），Pi日志已取回`results/pi2/PI_ENDPOINT/`。`pi1`为中间通过记录，pi2补充设备身份核对与Pi模式显示后通过。

实机：aarch64 Raspberry Pi 5；既有CPython3.12环境，torch2.5.1 / NumPy1.26.4 / SciPy1.13.1 / sklearn1.8.0 / joblib1.4.2，intra/inter-op均1。显示CO、class1、32.51097652175592 ppm、VALID_OUTPUT、pi-app-v2，头ID `026f10778722ed314397260738635cc1b275e800ebfeb3f70e075260db6c15dd`。原只读健康窗口PyTorch警告保留，未改数值实现。

断线测试主动终止本次SSH传输会话，不是拔网线实验。Qt显示COMM DISCONNECTED、INFERENCE_ERROR及**最后已知**版本，清空当前气体/ppm；不把缓存结果当新预测。手动模式界面继续运行，自动演示模式保存证据后关闭。当前不实现自动重连。

## 最小启动

仓库根目录：

```powershell
./development/fast_sprint5_20261118/START_PI_APP.ps1
```

每次生成新的PC及Pi测试输出目录，不覆盖既有记录。窗口自动启动Pi端点、加载健康包并重放CAL0。顶部按钮可重放和加载登记候选；下方`Reject bad SHA (demo)`演示失败保留。仅依赖原SSH密钥及现有环境，无管理员权限、系统服务或防火墙修改。

PC模拟（新目录；需Qt时去掉`--offscreen`）：

```powershell
[Local machine path/command omitted in public snapshot; original SHA retained.]
[Local machine path/command omitted in public snapshot; original SHA retained.]
```

直接Pi端入口（stdio JSON，输出目录须新建；通常由Qt自动启动）：

```bash
/home/gaps/fedongas_phase311_20261109/venv/bin/python -u -B /home/gaps/gaps_sprint5_20261118_test1/payload/endpoint.py --registration-sha 908aa3e9590d13f8aeb36b8d145eac65346a86599843c128fa1176aa0219e2d2 --out /home/gaps/gaps_sprint5_20261118_test1/direct_NEW
# 逐行输入：
# {"op":"activate","package":"initial"}
# {"op":"predict","case":"CAL0"}
# {"op":"activate","package":"candidate"}
# {"op":"activate","package":"candidate","fault_sha":true}
# {"op":"predict","case":"S_zero"}
# {"op":"close"}
```

Pi包位于`/home/gaps/gaps_sprint5_20261118_test1/payload/`，运行证据`run1/`、`run2/`；原环境及历史目录只读复用。本地部署构建记录见`build1/BUILD.json`。`build_payload.py`可在新输出目录重新打包，校验每个来源SHA；只增加隔离导入命名空间，未重写模型、回归或模型管理器。

本轮没有未完成核心阻塞。通过的是Calibration单窗应用链路和安全切换；不是跨平台全量数值验收，也不是在线采集闭环或云端模型下发验收。未自动合并main或启动后续研究。
