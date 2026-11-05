# 停止、恢复与下一轮门禁

本轮停在BLOCKED。SOURCE唯一启动已经使用，无论0步还是25轮，不自动把失败启动重算成“还可再跑一次”。保留SOURCE授权文件、ONCE_STARTED、3个失败输出目录和全部日志；不覆盖、不清理、不改冻结runtime，不自动开始第二次SOURCE或任何pilot。

下一阶段可审阅的最小工作：新注册版本修复configure_fit公共关键字名称，校准aggregate_fit公共签名；原模型/数据/spec超参数保持不变；补原生Flower dispatcher与本机gRPC无真实数据冒烟，必须明确实际运行环境/套接字权限。通过后请求单独批准恢复SOURCE，指定新输出位置并保留本次attempt1。批准SOURCE仍不授权pilot，不能自动扩72分支。

在新合法S*生成及25轮state/receipt/来源/恢复一致性验收前，六组全部BLOCKED。B1缺S*；B0/B2/B3/B4/B5同时缺S*和可运行的原生callback/网络条件。B1–B5原2500/40000合同未改变，B3/B4仅反馈、B4/B5仅参数加权的预注册定义保留，但不等于本轮完成以S*初始化的最终核验。

恢复证据可从FINAL_VERIFICATION、SOURCE_RUN_FILE_MANIFEST、原授权/命令/启动前SHA、进程日志和observer读取；没有可恢复的训练中间状态/optimizer或最终S*。当前原命令会因已有输出目录而拒绝覆盖；不要手工换输出目录绕过一次性边界。后续真实训练必须先新登记并得到明确授权。

本轮没有目标Test评价、分类准确率、预测类别路由或Pi部署。C3Day17仍只能作为C5Day18后历史保持集，C4保留回顾性，数据协议和未来固定端点评价不变。
