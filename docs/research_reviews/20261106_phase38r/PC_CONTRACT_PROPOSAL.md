# PC数值合同提案：独立审阅前不得启用

PROPOSED_PC_CONTRACT.json为v0提案，生产批准=false、Day22=false、Pi=false。保持现有六模型、四回归头、scaler与科学算法；沿用原单窗口部署任务，CPU float32、固定batch1、eval/inference_mode、线程1、锁定本机软件及原后端开关。这里保留原模式，不因诊断中某模式差异更小而换后端。

同合同数值验收继续原atol1e-4/rtol1e-5；同batch保存恢复要求dtype/shape/字节完全一致，原raw有限值、气体路由、71D schema及支持裁剪门禁照旧。需要单独冻结Calibration golden、模型/输入/layout/后端指纹、fail-closed读取审计、异常恢复和运行版本；跨硬件不能默认字节一致。现有Calibration证据可作为审阅材料，但不是生产部署批准或Pi一致性证明。

必须区分两个问题：原P38要求新单窗与旧no_grad/batch128参考通过，现仍失败；若继续这一合同，当前没有理由批准Day22 attempt2。另一种可能是经明确审阅后，将canonical单窗执行建立独立版本参考，旧跨执行形态差异单列为legacy comparison，不改变/重命名旧失败，不用argmax替代原logit标准。后者改变的是比较合同，必须显式审批，不能当成原门禁已经修复。

下一轮PC验收建议先只用合成和Calibration，独立日志记录全部文件打开/真实调用、同合同restore、边界输入/UNAVAILABLE。不能只凭这32窗保证Day22；若将来允许attempt2，必须新的授权/输出、六端点固定、历史已观察Test属性、全部先锁定再评分；原P38失败证据和参考永不覆盖。此处不给Day22可执行入口或有效授权。
