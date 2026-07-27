# v2.0 修复与验证报告

## 已修复

- HC-04 默认端口改为 `/dev/rfcomm0`，附带绑定脚本。
- 串口线程增加主动 `cancel_read/close`、3 秒退出保护和 finished 清理。
- 断线重连不再清空显示和未保存片段。
- 断线时自动暂停连续记录。
- 增加跨连接连续 stream 序号、连接内序号和实验内序号。
- 新实验创建失败不会静默丢失当前文件；新实验成功后建立干净边界。
- Save Segment 在尚未创建实验时可保留当前片段并创建目录后保存。
- 大片段达到内存上限时记录截断元信息。
- CSV 写失败后保持 ERROR 状态，不再被 LIVE 文本覆盖。
- 磁盘检查使用当前实验真实路径，极低空间时自动停写。
- UI 事件同步持久化到 `event_log.csv`。
- 增加 Baseline / Exposure / Recovery / Note 标记。
- 增加 ADC 12-bit 合理性软检查。
- YAML/JSON 配置文件实际生效。
- 顶栏精简，端口/波特率移动到 Status 页面；Status 与 Edge AI 页面可滚动。

## 新增边缘 AI 能力

- 独立 AI 后台线程，不阻塞 PyQt 绘图和串口接收。
- 部署包 schema 校验。
- 模型输入通道映射、基线、滑窗和 mean/std 归一化。
- `raw_adc`、`relative_adc`、`relative_conductance` 三种模式。
- 输入频率一致性 guard。
- TorchScript 分类、浓度回归和可选风险分数输出。
- per-class `none/bias/affine/piecewise_affine` 校准。
- `accept/review/reject` 两阈值 QC。
- AI 输出写入 `edge_ai_predictions.csv`。

## 已执行验证

```text
python -m py_compile *.py                    PASS
python parser_self_test.py                   PASS
python ui_logic_self_test.py                 PASS
python edge_ai_package_self_test.py          PASS
build_edge_ai_package.py schema validation   PASS
```

## 当前环境无法执行的验证

当前构建容器未安装 PyQt5，因此没有运行真实窗口的 offscreen GUI smoke test；代码已通过 Python 编译，但仍应在树莓派上执行：

1. 800×480 布局检查；
2. `/dev/rfcomm0` 连接与退出；
3. 30 分钟和 6 小时稳定性；
4. 真正的 TorchScript 模型推理；
5. 断线、重连、磁盘不足和大文件保存故障注入。

## 不应过度声称的部分

- 包中没有用户真实训练好的模型；
- 真实 `auto_v2` neural specialist 尚未自动转换为 TorchScript；
- 复杂 QC v2 风险融合需由模型/后处理导出真实 risk score；
- Flower Client 常驻服务与 OTA 尚未接入 UI 主进程。
