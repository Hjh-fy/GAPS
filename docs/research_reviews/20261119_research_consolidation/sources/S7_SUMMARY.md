# Sprint 7 — 系统整合交付

2026-11-18。三个统一演示模式均在原Pi5链路各启动一次并通过核心检查。仅新增包装入口、进程内模式标识、启动记录与材料索引；Sprint4/5/6原程序、模型与正式包未修改。

## 启动

```powershell
./development/fast_sprint7_20261118/START_DEMO.ps1 -Mode PUBLIC_REPLAY
./development/fast_sprint7_20261118/START_DEMO.ps1 -Mode LAB_REPLAY
./development/fast_sprint7_20261118/START_DEMO.ps1 -Mode STM32_RAW_REPLAY
```

默认使用已授权Pi SSH及原环境。`-Smoke`自动演示后关闭；`-Device pc-sim`明确为PC模拟。每次自动创建唯一的本地/Pi输出目录。完整参数、依赖与保存位置见`DEMO_MODES.md`。

## 交付与实测

|模式|一次启动检查|输出与能力|
|---|---|---|
|PUBLIC_REPLAY|PASS / Pi5|封存标定单窗；分类—71D/SVR浓度；复用切换、拒绝及断线演示|
|LAB_REPLAY|PASS / Pi5|100×6真实历史电阻窗口；实验室分类；ppm UNAVAILABLE|
|STM32_RAW_REPLAY|PASS / PC Qt＋Pi记录|62行历史ADC曲线与保存；推理拒绝，ppm UNAVAILABLE|

新增启动包装器4项定向测试通过。`acceptance/<MODE>/`保存实际命令、源程序SHA、控制台、引擎结果和带明确模式标识的截图。`MINIMAL_ACCEPTANCE.json/csv`为本轮最小验收；没有全量回归或历史指标重算。

论文材料：

- `SYSTEM_IMPLEMENTATION_MATERIAL.md`：系统实现章节技术草稿，含架构、联邦分类与回归关系、Pi—Qt流程、切换/恢复、保存及能力边界。
- `materials/FIGURE_TABLE_INDEX.md`：三份可编辑Mermaid图与表格索引。
- `materials/SCREENSHOT_INDEX.md/csv`：8幅真实截图的原始来源、图注、限制和SHA；按字节复制，既有结果未改变。
- `materials/EVIDENCE_INDEX.csv`：技术主张与已有工程记录/冻结合同的精确文件依据。新增工程证据没有新Git固定提交，使用本地SHA；未将其提升为独立科研审计或算法性能证据。

## 边界与下一步

LIVE仍因硬件与六路ADC→电阻合同阻塞，未新增LIVE模式。Pi5应用PASS不改变Phase3.11跨平台完整数值验收BLOCKED。三种模式不共享类别ID语义，不向实验室/ADC模式填入公共模型的ppm；v1/v2切换演示不代表新模型训练。未访问Day22、训练、改变容差、改正式包或合并main。

当前交付依赖原workspace和已准备Pi测试目录；不是便携安装器。私有模型及原始数据不复制进论文素材。可直接据本材料整理硕士论文系统章节，正式引用前应保留REPLAY、UNAVAILABLE及未完成能力的限定。没有自动启动下一轮实验。
