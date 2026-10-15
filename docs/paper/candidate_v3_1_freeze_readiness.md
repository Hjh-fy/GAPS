# GAPS IoT-J 中文内容冻结就绪记录（v3.1）

## 当前状态

`READY_TO_FREEZE_CHINESE_CONTENT_FOR_ENGLISH_TRANSLATION`

不可变冻结状态：`PENDING_HUMAN_SIGNOFF`

## 已完成

- 六项中文冻结前必改问题均已修正。
- v3.1 数字冲突为 0。
- v1/v2/v3、protocol-closed、evidence-frozen 稿未覆盖。
- 未运行训练、推理、评估或 benchmark。
- 未访问正式 test 资产。
- 模型、runtime、QC、threshold、图表数据和参考文献列表均未修改。

## 冻结前人工确认

请作者或导师确认以下五项：

1. 标题：`GAPS：面向跨设备气体感知的联邦分类与轻量回归个性化`。
2. 三项贡献的强度与顺序不再调整。
3. 目标校准依赖和 held-out-window 边界可按当前措辞进入英文稿。
4. Appendix A/B 的内容保留；最终 IEEE 排版可在不删除证据的前提下将部分明细移入 supplementary。
5. 作者、单位、通信作者、数据/代码公开范围在冻结稿中如何呈现。

## 确认后的受控动作

人工确认后只执行以下冻结动作，不再改写正文：

1. 从 `GAPS_IoTJ_submission_candidate_v3_1.zh.html` 复制生成新的不可变中文冻结稿；
2. 记录源稿 SHA256、冻结稿 SHA256、Git commit 和生成时间；
3. 生成中文内容冻结索引；
4. 将英文稿的唯一中文源绑定到该冻结稿；
5. 后续只允许翻译、图形英文重绘、IEEE LaTeX 排版和经批准的参考文献格式修正，不再打开算法、实验或系统工程阶段。

在完成上述人工确认与 SHA 锁定前，不使用
`CHINESE_CONTENT_FROZEN_FOR_CONTROLLED_ENGLISH_TRANSLATION`。
