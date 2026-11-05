# 实际配置、标签访问和执行命令

正式spec为contracts/{SOURCE,B0..B5}.json，validator仅接受完整注册字典；最终effective为cli_plans/*.json（逐个通过真实runtime.py plan命令），指纹见EXECUTION_REGISTRY_LOCK。COMMANDS.json保留真实argv数组，含空格的路径不拼接执行；可用PowerShell `&`配合逐参数引号或Python subprocess.run(argv)。本轮只调用plan和故意验证未授权server拒绝，不调用真实训练。

通用入口：`PYTHON_EXECUTABLE research/fedongas_phase32_20261105/runtime.py plan --spec <contracts/B4.json绝对路径> --output <新effective.json绝对路径>`。真实action为server/client/target，参数只有--spec、--checkpoint、--checkpoint-lock、--client-id、--output、--address，均真实parser实现，没有伪造旧开关。

SOURCE/B0启动server及C1/C2；B1仅target；B2 server+C1/C2/C5；B3/B4/B5 server+C1/C2，服务器读C5校准。SOURCE随机初始权重已保存SOURCE_RANDOM_INITIAL.pth及锁；六组checkpoint参数指向尚不存在的future_SSTAR/SSTAR.pth/LOCK，不使用旧协议权重顶替。未来S*需要将整个源run目录（25轮checkpoint/receipt/effective）作为来源包恢复，不只拷一个pth。

所有实际训练action在读取数据/模型前要求单独批准的TRAINING_AUTHORIZATION.json，值精确为{"approved_spec_sha256":所批准specSHA,"purpose":"SOURCE"}；本轮未创建。获明确授权后才可登记该文件及执行一次源预训练，批准SOURCE不授权B0–B5。命令实际运行已验证因缺授权退出；这不是仅写文档禁止训练。

Source专用目录data/source仅C1Day4、C2Day1/7三文件/客户端；Target目录独立，SOURCE input_pools只枚举1/2，不加载C5。C5仅Day18 A4，16暴露32窗，目标训练索引复用Phase31固定2500×16批表并按实际yield登记。Development/Test未导出、未挂载，任何额外文件都会在np.load之前拒绝。合法真实数据只做no-grad数组与loader检查；prototype统计只在合成模型测试中实际执行。

B1/B2/服务器每轮都是100次目标更新：目标loader在LOCAL_EPOCHS1内显式重复固定100批，而不是把dataset长度膨胀；num_examples仍32，完整25轮都是40000呈现。每轮/块Adam重建、lr5e-4与原循环一致。服务器初始与回传SHA、客户端序号、模型全state/encoder、原型内容和计数、上下行state/protobuf字节都写receipt；真实wire/TLS重试需执行时另测，不宣称隐私隔离。
