# 运行接口合同

`FrozenAssets().bridge('B0'...'B5').predict_window(window)`。当前assets加载器仅允许C5Day18A4文件，当前入口只有Calibration验证，无Day22运行开关。核心JointBridge不读取文件或任何真实gas，不接收gas/truth/oracle参数。

输入：有限float32 numpy[50,8]、原strict10秒5Hz、8通道电导响应窗；预处理不在核心路由器里隐式重做。ExtractorAdapter独立调用原canonical83的固定71索引，metadata={}、phase=-1、client=unknown；不是分类encoder64D。

正常顺序：模型/特征版本检查→原TCN logits→有限4logits与有效整数argmax→按预测gas查头→固定71名/序/shape/dtype/finite检查→原SVR.predict_head(clip=False)→raw finite→选中头Calibration支持clip一次。不softmax重调、不学习门控、不附加loss、不fit。

输出为JSON可序列化字段：status、reason、predicted_gas_id、predicted_gas、concentration_raw_ppm、concentration_clipped_ppm、unit='ppm'、support_ppm、classifier_group、classifier_version（文件SHA）、classifier_state_version、regressor_version（文件SHA）、feature_version（合同SHA）。浓度表示“预测气体的ppm”；错分时不能当作真实气体浓度。

异常：SIGNAL_INVALID、CLASSIFICATION_FAILED/INVALID、FEATURE_SCHEMA_MISMATCH、REGRESSOR_MISSING、MODEL_VERSION_MISMATCH、REGRESSION_FAILED/INVALID/NONFINITE。全部status=UNAVAILABLE、浓度字段null；分类失败时预测gas亦null，不借真实gas补救。若气体已预测而回归失败，可保留预测gas并明确浓度不可用。冻结SVR内部也会先拒绝nonfinite，适配器异常被转为REGRESSION_FAILED；核心收到非有限raw则REGRESSION_NONFINITE，均不clip挽救。

版本必须是已登记非空字符串；None/空白不能因相等而获准。真实loader先校验模型字节SHA、完整state schema/finite及stateSHA、gas/scaler/support/参数；同一加载实例推理前后state/scalerSHA不变。字段身份按加载时锁定，不宣称抵抗恶意内存篡改或OS级隔离。切换模型须重新经hash验证加载，不能手工改version字符串进行热切换。

支持：Ethanol/Ethylene12.5–125ppm；CO/Methane25–250ppm；由所选气体冻结头提供，不按真实气体改支持。raw允许超支持和负值，保留诊断，有限后再clip。
