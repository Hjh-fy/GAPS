# 社交与社区调研

社区用于发现问题和实现线索；方法有效性仍回到官网、固定源码和受控实验核验。没有把匿名经验升格为实验结论。

|来源|实际取得的信息|对项目的启发|证据限制|
|---|---|---|---|
|[Reddit作者介绍scikit-activeml](https://www.reddit.com/r/MachineLearning/comments/1ea8kc8/p_scikitactiveml_an_active_learning_library_in/)|作者解释选样策略与学习循环分离，并讨论分类/回归|复用query接口，保留我们自己的标签访问与暴露账本|是2024时期发帖；不能据其旧评论判断2026年其他库维护状态|
|[Bosch开发板换单传感器的求助](https://community.bosch-sensortec.com/mems-sensors-forum-jrmujtaw/post/issues-creating-algorithm-using-adafruit-bme688-in-conjunction-with-OoSYqnmGuIxGGqF)|用户报告八传感器开发板训练、单传感器测试及暴露布置变化后的问题|跨设备适配与场景代表性是实际需求|设备与暴露方式同时变化，不能归因为某一因素或直接证明我们的机制|
|[Flower官方社区Pi问答](https://discuss.flower.ai/t/how-can-i-run-federated-learning-on-a-raspberry-pi/50)|官方账号链接嵌入式联邦示例|实际设备客户端运行/运维方式可参考|当前已修复Flower1.23原生路径，不因此升级版本或重做客户端|
|[PyTorch issue124025](https://github.com/pytorch/pytorch/issues/124025)|用户提交单样本与batch输出差异案例|复用固定执行合同、环境收据和分层比较思路|相似现象不是本项目30条历史超差完整根因|
|[OpenSmell作者IoT帖子](https://www.reddit.com/r/IOT/comments/1w8x6u1/im_building_a_deviceindependent_protocol_for/)|作者征求设备无关协议反馈，指向仓库|可参考化学身份与数据来源字段|仅取得检索摘要和仓库交叉核验；完整帖子未作为深读依据，项目是pre-alpha|

X/Twitter公开检索使用了scikit-activeml、OpenSmell、electronic nose+GitHub等组合，没有取得足够相关且可核验的原帖作为技术证据。未登录私人群、Discord或封闭社交账号，不能声称覆盖所有社区。Reddit旧帖与当前仓库分开记录日期，不以热度或点赞证明成熟度。

[PyTorch官方数值说明](https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html)支持批量/切片与跨平台可有浮点差异；它没有替我们的注册合同取消门禁。旧Phase3.8和Phase3.11严格失败继续保留，不能据论坛建议放宽容差或改后端后重命名PASS。
