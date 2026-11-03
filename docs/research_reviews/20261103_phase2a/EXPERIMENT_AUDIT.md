# 审计结论与门禁

审计声称仅覆盖：历史方法语义与字节来源、批准71D源输入/时间角色、合成接口等价及未来可执行计划。对新迁移有效性没有证据批准。

AUDIT_FINDINGS.csv按blocking/major/minor/informational列出12项发现。A01阻止旧权重直接复用；A02/A03/A04阻止当前目标迁移训练；它们不妨碍完成只读审计和发布。数据输入合法与合成数值可复用已经确认，不能提升成真实新模型或端到端系统已验收。

泄漏审计：源数据120暴露全部来自Day7白名单，与所有目标cal/dev/test文件互斥；输入scaler计划只访问合法源，目标残差scaler只看相同4/8cal窗；源alpha/F-SVR参数来自已锁源选择。新rho尚未选择。没有打开目标数值矩阵或加载历史checkpoint作推理；旧模型JSON仅结构/来源校验。C4的时间例外和全部回顾性属性不隐藏。

公平比较计划：F-SVR是主基线和rho0的同一权重，N-SVR/KRR只读参考；所有新候选同71D、A2/A4身份/窗口、Oracle、支持和单窗暴露等权主评价；Source-only不消耗targetcal标签。旧83/104与历史窗口指标不拼入新矩阵。Source-only跨预算视图不当独立训练重复。

验证：7项接口检查通过，24次合成拟合（8个Ridge数值重建/参考，16个SVR直接/rho0训练路径），0真实源拟合、0真实目标拟合、0历史pickle/joblib推理。四gas充分统计量与pooled参考达原1e-6ppm门槛；8个rho0探针raw/clip逐元素相同。源文件120个与v5读取资产28个SHA核对；Phase1 125项、Phase1.5 185项封存负载未变。完整记录见checks与evidence_v2。

Verdict：只读事实及合成接口证据可在限定范围内approved；71D新真实源模型和目标迁移性能仍blocked/pending。Phase2B可以进入源侧构建与合同验收，不能直接进入目标迁移训练。历史谱系不完整、确认集缺失、模型尚未建立不靠改命名或目标test调参修复。
