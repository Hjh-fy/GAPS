# 证据访问和独立数据边界

本轮将可取得文件与独立验证范围分开。22阶段的模型清单共996条模型文件引用全部存在、字节SHA匹配，0缺失、0不匹配。它们含同一资产重复引用，并非996个独立训练模型。本轮不反序列化模型；恢复/数值有效性引用各阶段固定验收，不能把本轮哈希检查写成重新推理验收。细节在PRIVATE_ASSET_AVAILABILITY.csv。旧Pi审计72项及历史26来源均可取得。源输入数值和原始信号没有重新加载，数据身份依据封存元数据，不冒称现场独立重哈希。

UNAVAILABLE明确包括ISCAS平台上传回执/精确提交文件身份、审计范围内的STM32固件源码、尚未闭合的完整历史选参用途及新C5未来确认队列。某个未知文件名缺失不代表整个GUI未实现；实际edge_ui_app、edge_ai_runtime、worker和解析器均取得。远程ECS/Pi现状本轮未重连，不以公开摘要推断运行中服务已独立核验。

既有资产审计表明C1Day10/15、C2Day11没有进入当前S*，但曾进入旧源训练；C3Day8/14曾进入旧缓存。它们可另立回顾性开发，不能重命名首次盲测。C5开发为同设备同日独立物理暴露，不是跨日期。C3Day14早于C5Day18接入，接入后评价只能称历史留出保持，不叫未来在线。来源 [POSTMORTEM/ASSET_AUDIT.md](https://github.com/Hjh-fy/GAPS/blob/a41b93fc2db2c8de76310af8256e990443e0be76/docs/research_reviews/20261111_gate3_postmortem/ASSET_AUDIT.md)、[GATE3/C3_EXTENSION_PROPOSAL.md](https://github.com/Hjh-fy/GAPS/blob/93c2f9fe445b0c8079cba8d52bae8f6279dd089e/docs/research_reviews/20261110_gate3/C3_EXTENSION_PROPOSAL.md)。

现有CE曲线能支持非仿射观察，但缺g_dev，无法识别`R_dev-r_cal=(r_dev-r_cal)+(R_dev-r_dev)`两分量。若不允许新增访问，结论维持INSUFFICIENT_EVIDENCE_STOP。新增梯度提案是独立选项，当前没有执行，更不以开发梯度生成策略或选参数。来源 [POSTMORTEM/IDENTIFIABILITY.md](https://github.com/Hjh-fy/GAPS/blob/a41b93fc2db2c8de76310af8256e990443e0be76/docs/research_reviews/20261111_gate3_postmortem/IDENTIFIABILITY.md)、[POSTMORTEM/NEW_ACCESS_PROPOSAL.md](https://github.com/Hjh-fy/GAPS/blob/a41b93fc2db2c8de76310af8256e990443e0be76/docs/research_reviews/20261111_gate3_postmortem/NEW_ACCESS_PROPOSAL.md)。
