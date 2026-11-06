# FedOnGas Phase3.7 严格分类—回归数值桥接

依据Phase3.6固定4bf929fbbaa456e0a21599ad39fd01e8145a0406，复用Phase3.5六个合法固定分类端点和Phase1.5 C5 A4四个F-SVR71头。最终Calibration门禁：**PASS**。本轮未训练、未refit scaler、未执行Day22 Test或Pi。

- 16个Day18独立标定暴露、32个固定窗口：重新按原strict流程生成的float32[50,8]与保存输入逐位一致，71D与保存特征逐位一致。
- 四个回归头各核对全部32窗（128个head/window对），原wrapper/适配器、批量/单窗raw输出最大差均0ppm。4条既有CO低/高端Calibration参考的raw/clip差均0。
- 六分类端点各32窗，批量/单窗logits最大差2.62260437e-05，满足预锁float32容差；argmax全部相同，同batch重新加载输出逐位相同。形成192条标定路由记录，不等于所有额外parity/故障推理调用总数。
- 35项合成接口测试通过，加载真实模型适配器的8项故障代理测试通过；非法类别、非有限输入/特征/logits/raw、缺头、版本及schema异常返回UNAVAILABLE，不Oracle补救，raw先有限检查再clip。

边界：历史并无32窗分类logits；本次分类输出验收是冻结state/原实现/当前Calibration重载一致性，不能冒称完整历史logits回放。全部128个当前head/window对与原冻结实现对照，其中4条另有历史CO Calibration golden；不把新生成值称旧golden。

主要入口：[bridge.py](code_review/bridge.py.txt)的JointBridge.predict_window；真实冻结适配器为[assets.py](code_review/assets.py.txt)。来源锁MODEL_SOURCE_LOCK、容差锁TOLERANCE_LOCK；数值表在calibration_final。API_CONTRACT与FUTURE_OFFLINE_EVALUATION_CONTRACT定义运行/评价边界。FINAL_VERIFICATION明确下一阶段门禁。

B0名称仅表示分类器Source-only；其联合系统与其他五组共用目标标定F-SVR71头及支持，不能称整体无目标标定。

先前两个运行现场保留：attempt1为记录构造重复字段异常；attempt2数值通过后，独立审查补强未知版本None/空值拒绝，最终实现再通过calibration_final。原模型/数值函数/容差始终未改，未覆盖旧结果。
