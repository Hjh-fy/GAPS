# GAPS 项目级科研 Skill 第一版设计

## 1. 目标与范围

在 GAPS 仓库中建立 7 个项目级科研 Skill：

1. `experiment-planner`
2. `experiment-registry`
3. `result-analysis`
4. `experiment-audit`
5. `claim-evidence`
6. `number-consistency-audit`
7. `gaps-research-orchestrator`

第一版采用“工作流说明 + 固定模板 + 共享数据契约 + 验证用例”。所有 Skill 默认只读现有实验资产，不自动扫描并解释历史实验，不自动生成或覆盖 registry、CSV、审计报告及其他现有成果，也不运行实验。

## 2. 设计原则

- 每个 Skill 聚焦一种科研动作，description 明确写出触发条件和职责边界。
- 共享字段语义只有一个权威来源，避免 7 份定义逐渐漂移。
- 未经明确证据，不从目录名推断 split、DA、checkpoint、calibration 或 QC。
- 缺失字段使用 `unknown`；来源冲突使用 `conflict`，并记录冲突来源。
- 模板是新成果文件的起点，不代表 Skill 可以覆盖同名现有文件。
- `gaps-research-orchestrator` 只判断阶段、Evidence Gap 和下一步 Skill，不重复子 Skill 的分析逻辑。
- 第一版为后续脚本化预留接口，但不实现复杂自动化。

## 3. 目录结构

```text
.agents/skills/
├── experiment-planner/
│   ├── SKILL.md
│   ├── assets/
│   ├── references/
│   └── scripts/
├── experiment-registry/
├── result-analysis/
├── experiment-audit/
├── claim-evidence/
├── number-consistency-audit/
├── gaps-research-orchestrator/
└── _shared/
    ├── contracts/
    │   ├── experiment-record.md
    │   ├── metric-record.md
    │   ├── evidence-record.md
    │   └── handoff-protocol.md
    └── references/
        ├── gaps-taxonomy.md
        ├── read-only-policy.md
        └── skill-boundaries.md
```

每个 Skill 的 `assets/` 保存固定输出模板，`references/` 保存该 Skill 特有规则，`scripts/` 只保存接口说明或占位说明。共享契约放在 `_shared/`，不作为可直接触发的 Skill。

## 4. Skill 职责和输入输出

### 4.1 experiment-planner

- 触发：把研究问题或假设转化为 baseline、ablation、metrics 和 Expected Evidence。
- 输入：`RESEARCH_BRIEF.md`、研究假设、已有基线与资源约束。
- 输出：`EXPERIMENT_PLAN.md`、`EXPERIMENT_MATRIX.csv`、`ABLATION_PLAN.md`。
- 不处理：不运行实验，不分析结果，不登记已完成实验，不声称假设成立。

### 4.2 experiment-registry

- 触发：登记、核对或规划实验 ID 与可追溯元数据。
- 输入：用户确认的实验信息、config、checkpoint、结果路径及代码版本。
- 输出：`experiment_registry.csv` 的新草稿、候选记录或更新建议。
- 不处理：不根据模糊目录名猜测实验口径，不分析指标，不评判实验公平性。

### 4.3 result-analysis

- 触发：对口径已确认的 CSV/JSON 指标做描述统计、效应分析、异常检查及论文表图规划。
- 输入：已确认口径的结果文件、registry 记录和指标定义。
- 输出：`RESULT_ANALYSIS.md`，以及建议表格或图表清单。
- 不处理：不判断实验设计是否公平，不补造缺失 seed，不把报告值冒充重新计算值。

### 4.4 experiment-audit

- 触发：检查比较是否完整、公平、可复现，识别 split、checkpoint、seed、baseline 和泄漏风险。
- 输入：实验计划、registry、结果、配置与数据切分元数据。
- 输出：`EXPERIMENT_AUDIT.md`。
- 不处理：不重新执行实验，不修改结果，不代替统计分析，不批准没有证据的 claim。

### 4.5 claim-evidence

- 触发：把经过审计的实验或文献 Evidence 映射到论文 Claim，并维护支持强度和来源。
- 输入：已审计结果、已核验文献证据、拟写声明。
- 输出：`CLAIMS_EVIDENCE.md`。
- 不处理：不创造数字，不把未审计结果标为已确认，不撰写整篇论文，不做引用真实性审计。

### 4.6 number-consistency-audit

- 触发：核对稿件正文、摘要、结论、表格、图注与 Evidence 表中的数字一致性。
- 输入：稿件、批准的表格和图、`CLAIMS_EVIDENCE.md` 及指标口径。
- 输出：`NUMBER_AUDIT.md`。
- 不处理：不判断引用语境，不判断实验公平性，不以格式化或舍入为由静默改数。

### 4.7 gaps-research-orchestrator

- 触发：判断 GAPS 当前研究阶段、最大 Evidence Gap 和下一步应调用的 Skill。
- 输入：项目状态文件以及上述标准产物的存在性、完整性和审计状态。
- 输出：`PROJECT_STATUS.md`、`NEXT_ACTIONS.md`，或明确的下一 Skill 调用建议。
- 不处理：不执行子 Skill 的规划、分析、审计或写作，不自行选择冲突口径。

## 5. 共享数据契约

### 5.1 实验记录必需字段

```text
experiment_id
source_clients
target_clients
split_protocol
model
checkpoint
DA
calibration
QC
seed
result_path
metrics
status
notes
```

建议追溯字段：

```text
code_commit
config_path
dataset_path
created_at
evidence_status
provenance
```

字段值必须来源于用户确认、配置文件、manifest、registry 或可追溯结果元数据。仅由目录名推测的值不能升级为事实。

### 5.2 Metric 记录

Metric 至少记录：名称、数值、单位、方向、数据切片、聚合方式、seed 集合、来源文件、计算状态和备注。Accuracy、ECE、RMSE、NRMSE、Coverage、Coverage+Review、Accepted RMSE、Route-correct RMSE、Latency 与 Payload 必须保留完整指标名称和适用样本范围。

### 5.3 Evidence 记录

Evidence 至少记录：`evidence_id`、关联实验、指标与比较、来源、审计状态、支持的 claim、限制和 provenance。未通过 experiment audit 的 Evidence 不得标为 `approved`。

### 5.4 状态约定

- `draft`：信息尚未完整。
- `registered`：记录已建立，但结果未必存在。
- `completed`：实验执行完成，尚未完成审计。
- `audited`：已完成审计并记录结论。
- `approved`：可进入 Claim–Evidence 表或论文。
- `blocked`：缺少必要信息或资产。
- `conflict`：来源之间存在未解决冲突。
- `unknown`：字段值无法可靠确定；这是字段值而非工作流状态。

## 6. Skill 调用关系

```text
RESEARCH_BRIEF.md
  → experiment-planner
  → experiment-registry
  → 实验执行（7 个 Skill 之外）
  → result-analysis
  → experiment-audit
  → claim-evidence
  → research-writing-skill（现有个人 Skill）
  → number-consistency-audit
```

`gaps-research-orchestrator` 可在任一阶段读取标准产物的状态，但只能给出路由建议。例如 registry 缺少 split 时，它应调用或建议 `experiment-registry`；比较口径冲突时建议 `experiment-audit`；已有审计结果但缺少论文声明映射时建议 `claim-evidence`。

## 7. 只读与错误处理

- 读取现有 `results/`、Markdown、CSV、JSON、配置和 checkpoint 路径时不修改源文件。
- checkpoint 第一版只记录路径及可安全读取的元数据，不加载模型执行推理。
- 生成模板实例或报告前要求用户明确目标路径。
- 若目标已存在，停止并提供差异、候选更新或新文件名，不直接覆盖。
- 无法确认的字段写为 `unknown` 并记录原因。
- 多来源不一致时保留各来源，标记 `conflict`，转交 `experiment-audit`。
- Result Analyst 必须区分已有报告值和本次重新计算值。
- 任一 Skill 发现职责外问题时只记录 handoff，不越界完成。

## 8. 最小验证

### 8.1 结构验证

- 7 个目录均存在合法 `SKILL.md`。
- YAML frontmatter 至少包含匹配目录的 `name` 和明确的 `description`。
- 共享引用路径和 Skill 间交叉引用有效。
- 模板包含其契约规定的必需字段。
- `scripts/` 明确标注第一版接口，不包含复杂自动化。

### 8.2 触发验证

每个 Skill 至少包含 2–3 个应触发场景和 1 个不应触发反例。验证清单记录输入、预期 Skill、禁止调用的 Skill 和判断理由。

代表性场景：

- “为 Strong DA 与 CORAL 设计消融矩阵”触发 `experiment-planner`。
- “登记 EXP-031 的 split、checkpoint 和结果路径”触发 `experiment-registry`。
- “比较 seed 42/43/44 的均值和标准差”触发 `result-analysis`。
- “检查这些实验是否使用同一 split 和 checkpoint”触发 `experiment-audit`。
- “将 C5 Accuracy 改善映射到论文结论”触发 `claim-evidence`。
- “核对摘要与 Table II 的 98.98% 是否一致”触发 `number-consistency-audit`。
- “当前最大的 Evidence Gap 是什么”触发 `gaps-research-orchestrator`。
- “润色 Introduction”不应触发上述 7 个，应交给 `research-writing-skill`。

### 8.3 交接验证

使用不接触真实实验结果的虚拟 GAPS 记录验证：Planner 输出可被 Registry 接收；Result Analyst 能识别 Metric 契约；Audit 能发现 split/checkpoint 冲突；Claim–Evidence 拒绝未审计数字；Number Audit 报告稿件数字不一致；Orchestrator 选择正确的下一 Skill。

## 9. 第二阶段脚本化方向

优先顺序：

1. `experiment-registry`：只读采集 manifest/config/git/path 元数据，生成候选记录而不是直接写 registry。
2. `result-analysis`：针对已确认的一种稳定 JSON/CSV schema 做汇总、均值、标准差和表格草稿。
3. `experiment-audit`：检查 registry 中的 split、checkpoint、seed、baseline 和 provenance 完整性。

可选的 `list-results-candidates` 工具仅列出 `results/` 下候选目录及 `.csv`、`.json`、`.md`、`.pth` 文件，不解释实验口径，不写 registry，不修改结果。第一版只预留接口，不实现工具。

## 10. 验收标准

- 7 个 Skill 均能被项目级 Skill 发现机制识别。
- 每个 Skill 的职责、触发条件、输入、输出和非职责明确。
- 所有 Skill 使用同一套字段和状态语义。
- 所有模板能支持 GAPS 当前常见实验维度，但不依赖历史目录命名猜测。
- 最小触发、反例、结构与交接验证全部通过。
- 实现不修改或覆盖任何已有实验资产。
