# GAPS Edge AI UI v2.2 响应式重构记录

日期：2026-07-27

## 目标

本轮只调整上位机视觉层和布局行为，不改变 STM32 帧解析、采集保存、
模型输入、Runtime v5 推理、QC fail-closed 语义或审计写入逻辑。

主要目标：

- 让 800×480 树莓派屏幕不再横向溢出；
- 让实验人员先看到气体、浓度、置信度和 QC，而不是内部状态代码；
- 在 1920×1080 等大屏上利用剩余空间展示推理趋势；
- 保留完整模型身份和内部状态，供 tooltip、事件日志和结果文件审计。

## 界面变化

- 顶部小屏标题缩短为 `GAPS AI`，隐藏重复的帧统计；
- 底部小屏只保留新建实验、开始记录和数据目录；
- 六张 AI 卡片使用可收缩宽度，QC 显示映射为
  `Unavailable / Accepted / Review / Rejected`；
- 原始 QC 值（例如 `disabled_pending_dependency_audit`）仍保存在
  `last_ai_result`、预测审计 CSV 和 QC 卡 tooltip 中；
- 增加 AI 状态、模型身份、实验阶段、基线、窗口和输入频率；
- 推理摘要移动到状态详情之前；
- 大屏显示最近 120 次浓度预测曲线，小屏隐藏曲线以保证首屏信息密度；
- Edge AI 和状态页明确关闭横向滚动；
- Qt 截图 smoke test 增加六张 AI 卡片横向可见性检查。

## 验证口径

验证使用树莓派真实 Wayland 会话、正式 Runtime v5 公共数据部署包和公共
calibration 首个 100×8 窗口。截图测试同时检查：

- Runtime v5 包身份与指纹；
- 分类路由、H1、最终浓度、置信度和 QC 映射；
- `QC disabled` 时自动浓度输出仍为空；
- 六张 AI 卡片均处于水平可视范围内；
- 800×480 与最大化大屏均能生成真实窗口截图。

截图与机器可读结果保存于：

```text
results/edge_ai_public_runtime_v5_calibration_replay_20260727/
```

本验证是部署一致性与界面验收，不构成新的模型精度或性能 benchmark。
