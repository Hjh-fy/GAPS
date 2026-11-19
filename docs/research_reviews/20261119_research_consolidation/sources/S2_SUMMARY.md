# SPRINT 2 — 2026-11-17

**DONE**：独立目录实现最小安全更新，未修改冻结模型、旧runtime或正式部署包。

- `safe_update.py`复用Phase3.9 `Runtime.load`的候选加载/失败保留事务、现有manifest JSON/文件检查、Sprint1 SHA与统一输出。新增部分只绑定兼容合同、候选健康检查与活动引用切换；无Registry/MLflow。
- 加载器接口为`loader(package_root, verified_manifest)`，返回已有runtime的classifiers、extractor、heads、bridge；`adapt_loaded_assets`可包装新加载的现有资产实例。调用者须提供已批准manifest SHA，不能由候选自批准。
- 检查包内文件、SHA、标签顺序、50×8输入、71D顺序/版本、六分类端点、四回归头、scaler维度/摘要和ppm支持。健康检查使用合成输入，执行全部分类器及全部头，包含未被路由选中的头。
- 候选使用独立实例；拒绝与健康实例共享组件/scaler内存。检查及日志持久化全部通过后，仅切换内存活动引用。失败保留原对象、版本和预测；加载期间健康实例仍可推理。不覆盖模型文件。
- 定向测试22项通过，覆盖正常切换、缺文件、SHA错误、接口/标签/特征不兼容、缺头、scaler失配、加载/健康失败、日志写入失败、共享实例及并发可用性。版本和原因逐条写入`UPDATE_EVENTS.jsonl`。
- `results/run2/SWITCH_EXAMPLES.json`展示healthy-v1→candidate-v2成功；candidate-v3缺文件、candidate-v4非有限输出均拒绝，活动版本和输出保持candidate-v2。冷启动无健康实例时返回MODEL_NOT_READY。

**BLOCKED**：无Sprint2工程阻塞。验收仅为合成包及故障测试，没有宣称真实候选权重已完成更新或跨平台数值验收。

**DEFERRED**：Sprint3、Pi连接和远程更新均未执行。训练、评分、真实模型推理和Day22访问均为0；Phase3.11严格数值BLOCKED保持原结论和容差。

**运行命令**（项目根目录，输出路径须为全新目录，使用现有依赖）：

```powershell
[Local machine path/command omitted in public snapshot; original SHA retained.]
[Local machine path/command omitted in public snapshot; original SHA retained.]
```

下一步止于本轮代码交付，不自动推进Sprint3或合并main。
