# 恢复、入口与制品说明

主代码runtime.py，契约contract.py；源码复用清单V5_IMPORT_PROVENANCE。本轮不修改已有root代码。build.py是一次性准备入口，已运行后不可在原目录重跑；现有数据、随机初始state、spec与锁按SHA恢复，不重新生成随机文件替代注册版本。

完整恢复需保留本阶段目录、Phase31及其Phase1/2只读helper、v5来源目录。GitHub只有审阅代码和文档，无原始信号/18npy/权重，所以不能仅拉公开报告就重训。跨机先恢复相同v5位置或明确映射prior helper的本地基路径，重新校验SHA；不能静默跳过。Source预训练只挂data/source；C5包在另一路径，Test/Development没有运行数据包。

已实际安全加载保存的完整TCN随机初始state、严格克隆到另一模型，合成zeros50×8前向为4类/64D，无梯度。非法S*、错误schema/dtype、未知输入均拒绝；真实S*的正向祖先/恢复需未来已授权源训练生成后验证。不能把Tiny合成checkpoint用作S*。

不支持中途round/optimizer精确续跑：所有Adam按原设计每轮新建，不保存其跨轮状态。失败后保留失败目录，用同一初始state/spec/输入在新输出目录从round1重跑；是否重跑仍需登记授权。不能覆盖同名结果或跳至最后一轮补齐假链。研究S*恢复需整个源run25轮来源包；最终weights-only恢复通过不等于训练身份合法。

本地保留preflight与preflight_v2（合成）、integration（合成）、失败日志、所有逐批索引、随机初始化及数据流记录。未生成真实SSTAR_LOCKED、未创建TRAINING_AUTHORIZATION、未部署任何模型。
