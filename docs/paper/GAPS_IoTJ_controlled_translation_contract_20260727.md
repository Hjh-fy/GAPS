# GAPS IoT-J 受控英文翻译合同

## 1. 合同状态

正式状态：

`CHINESE_CONTENT_FROZEN_FOR_CONTROLLED_ENGLISH_TRANSLATION`

唯一中文科学内容源：

`docs/paper/GAPS_IoTJ_chinese_content_frozen_20260727.zh.html`

冻结 SHA256：

`5fdf5f5b57ca72c969c3684dc2fda795b3727dd8caaff3dbee5c4c1fc4ad5f3e`

该冻结稿与
`docs/paper/GAPS_IoTJ_submission_candidate_v3_1.zh.html`
逐字节一致。源稿与冻结稿均不得覆盖或修改。

## 2. 人工批准范围

以下事项已获得作者人工确认：

1. 标题；
2. 三项贡献及其顺序；
3. 目标 calibration 与 calibrated-target held-out-window evaluation 边界；
4. Appendix A/B 的内容和证据身份；
5. 作者、单位及公开信息作为独立 submission metadata 管理。

上述事项不再通过英文翻译、LaTeX 排版或图形重绘重新选择。

## 3. 每个派生稿的强制入口检查

任何英文、LaTeX、图表或 supplementary 派生物开始前必须：

1. 验证冻结稿 SHA256 等于本合同记录值；
2. 将冻结稿路径、SHA256 和当前 Git commit 写入派生物 provenance；
3. 记录派生操作类型、负责人、时间和输出路径；
4. 以冻结稿为唯一中文科学内容源，不从 v1/v2/v3、protocol-closed 或 evidence-frozen 稿混合拼接正文；
5. 若发现需要改变科学内容，立即停止派生，不得静默修改冻结稿或英文稿。

## 4. 允许的派生操作

允许：

- 在不改变含义的前提下进行受控英文翻译和学术语言润色；
- 转换为 IEEEtran/IEEE IoT-J LaTeX 结构并进行双栏排版；
- 调整换行、段落长度、表图位置和交叉引用；
- 在不删除证据的前提下，将已经批准的附录明细移入 supplementary material；
- 使用相同冻结数据重绘英文图，保持数值、趋势、方法身份和 caption message；
- 修正冻结 Fig. 1 英文重绘中的 broad locality label，使其明确区分本地分类训练窗口、H1 统计交换和服务器端源侧适配参考集；
- 对参考文献执行经过来源核验的元数据或 IEEE/BibTeX 格式修正；
- 独立维护作者、单位、通信作者、致谢、基金、数据/代码公开范围和 cover letter；
- 创建字节不变的归档副本、SHA 索引和受控派生记录。

## 5. 禁止的科学内容变化

禁止：

- 修改或覆盖冻结中文稿及其唯一源稿；
- 改变标题含义、三项贡献及顺序、研究问题或系统主线；
- 改变模型、损失、超参数、数据集角色、calibration/test 协议、test-access 边界或评价身份；
- 改变任何数字、单位、精度、比较方向、不确定性、表图数据或结果解释；
- 将描述性结果改写为统计显著性或形式化非劣结论；
- 扩大源数据本地性、隐私、联邦范围、部署泛化或可移植性 claim；
- 删除不利结果、文件级相关性、校准依赖、隐私限制、运行对象差异或 legacy evidence 身份；
- 将正式选择性输出基线、简化回归核心与独立 QC 候选合并为同一运行对象；
- 通过新训练、推理、评估、benchmark、test 访问或 post-freeze method selection 改写论文结论；
- 使用其他中文稿替代本冻结稿作为英文翻译科学来源。

## 6. 数字与 claim 审计

每个达到 advisor-review 或 submission-candidate 状态的英文派生稿必须执行：

- 跨摘要、正文、表格、图注、讨论和结论的数字一致性审计；
- 方法—实现语义对齐审计；
- claim–evidence 与 limitation 保留审计；
- 术语一致性审计；
- 冻结稿 SHA 和派生 lineage 审计。

允许语言重组和篇幅压缩，但不得以压缩为由删除影响结论解释的证据边界。

## 7. 变更升级规则

若英文翻译或导师审阅提出必须改变科学内容的请求：

1. 停止当前派生流程；
2. 将请求记录为显式 amendment proposal；
3. 标明受影响 claim、数字、证据、章节和冻结边界；
4. 在获得新的人工批准前，不得修改冻结稿或把修改写入正式英文候选稿；
5. 任何批准的科学 amendment 必须生成新的版本、SHA 和 provenance，不能覆盖本冻结版本。

## 8. 冻结证据

- 中文源提交：`c4342e8a009f8bc8f3e38af56ccca4f504ce2fc8`
- 论文证据冻结提交：`b78e8bec989cc8a925698d682aff05efe859fcd2`
- 系统/runtime 冻结提交：`36ee62339c025064cb415bfd13c5e7139a954edc`
- 数字审计：`docs/paper/candidate_v3_1_number_audit.json`
- 数字审计 SHA256：`052e833b74a9424b00ac7363b4bbfa50c8bd61a0a0e7fe4684dbeb7f601014be`
- 中文冻结索引：`docs/paper/GAPS_IoTJ_chinese_content_freeze_index_20260727.json`

本合同不授权重新打开实验、test 或工程开发。
