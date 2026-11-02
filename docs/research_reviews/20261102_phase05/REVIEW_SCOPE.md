# 外部GPT agent阅读说明

本包用于审阅研究协议、工作量、证据边界和下一轮设计。主入口README.md，随后读EXPERIMENT_PLAN.md、EXPERIMENT_MATRIX.csv、FLOWER_AUDIT.md、REUSE_INVENTORY.md及evidence/。

- 本轮640个raw文件重哈希；320个目标暴露可用；142行71D黄金对齐；0模型fit/0checkpoint推理/0部署。
- C3三日期可成立；C4/C5仅两个日期，cal/dev同日但暴露互斥。C4共同双源队列晚于最早标定日，仅回顾性协议成立。
- acquisition角色清单含A4下全部376个活动文件（120源+256目标）及对应A2角色，未列出的264个原始文件均unused；完整640份身份审计保留本地。
- 2/4每气体标定=每目标8/16次暴露；如果development用于选参，C3总目标标签48/56，C4/C5总量32/40。主矩阵不使用目标dev选择。
- 新矩阵12配置仍draft，无新性能结果。Source-only参数/新源缓存尚待闭合；现有7个C5分类权重仅为原协议条件复用候选。
- 所有结果是已观察历史数据上的回顾性协议审计，不是从未观察过的前瞻确认；旧window-level结果不能变成独立暴露泛化证据。

符号路径：FROZEN_BASELINE_V5为本地2026-11-02只读v5；PHASE05_WORKTREE为本轮独立工作区；ORIGINAL_WORKSPACE为原工程；USER_HOME为旧工作树路径前缀。它们不是GitHub内现成目录。

未上传raw信号、特征数值NPZ、模型、checkpoint、备份ZIP、凭据或远程拓扑。审阅包不能独立重新执行数据审计；可依据保留的SHA、日程、角色和配置证据核查其逻辑。DATA/QA记录里的旧checkpoint路径仅是来源索引，不表示该文件已上传。

报告中的本地审计HEAD=e9c64ff5是代码上下文，不代表GitHub文档分支包含全部v5代码。PUBLICATION_MANIFEST.json分别记录本地来源文件SHA与发布副本SHA；路径替换后的副本不冒充原字节冻结资产。

建议审阅重点：是否准确区分文件互斥/日期互斥/即时因果；标签预算是否公平；少组数CV是否泄漏；旧权重是否含未来暴露；三者能否同协议汇总；最小矩阵及停止条件是否足够。
