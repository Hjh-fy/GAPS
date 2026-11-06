# 运行、复核与恢复

当前环境沿用Phase33R ENVIRONMENT_LOCK，Python及依赖版本记录在MODEL_SOURCE_LOCK。先将本阶段目录加入Python sys.path（或以该目录为工作目录），本地加载冻结资产后：`from assets import FrozenAssets; bundle=FrozenAssets(); engine=bundle.bridge('B4'); out=engine.predict_window(window)`；window必须由原strict预处理生成的float32[50,8]。不能把任意8通道串口片段或64Dencoder向量直接传入。

生产桥接代码bridge.py不读文件，assets.py负责冻结模型/特征适配及当前Calibration白名单。所有源码/权重/输入需要本地归档；公开审阅包没有原始信号、特征值、scaler数组或权重，因此不能仅凭GitHub包独立运行模型。

新接口测试命令：`python -m pytest research/fedongas_phase37_20261106/test_bridge.py -q -p no:cacheprovider`。只有合成模型与异常，不训练、不加载Test。数值验收脚本`validate.py --final`已执行并封存，不可再次在同目录覆盖；复核应先核对SHA。若要重新运行，另建有登记的输出目录并维持同模型/输入/容差，不删除既有目录。

Calibraton_final为最终验收，calibration_attempt2为版本边界补强前成功数值记录，根VALIDATION_BLOCKED与attempt1源码保留日志构造失败。没有训练attempt或Test重跑。最终部署/Pi和Day22入口不在当前授权中。
