# 反馈、原型与训练轨迹

75服务器轮、150客户端收据、72个下一轮接收链接重新核对：B3 returned=aggregated，B4/B5 returned=adapted；每轮客户端收到上一轮returned，prototype encoder SHA与本地上传encoder一致。全部三组仍执行相同100步服务器适配，最终端点均adapted25。未发现版本错配；来源为ROUND_VERSION_CHAIN及Phase35原始native receipt。

每源每类phase0计数固定210/420，4类64D，phase0只是中性统计键。B3/B4/B5上传规则、EMA.8、warmup3、初始空原型一致。参数原型加权、EMA统计维护与原型损失不能混称一种机制。选择性相似度来自源客户端原型相对EMA参考；不宣称直接用目标原型加权。

B5：round4–25均激活，22/25=88%，warmup后100%。C1权重范围[0.333142334,0.333536049]，基准1/3；最大绝对偏移0.000202716（约0.02027个百分点），均值约0.00608个百分点。非“没有启用”，但权重非常接近普通样本权重。原型相似度约接近1，不等于已证明目标映射可信。

同源Sstar下B4/B5前三轮状态完全相同，round4加权后轨迹开始分开（Phase35 PAIRED_MECHANISM_PREFIX_PARITY及本轮SHA链）。后续Calibration在线批指标在B4/B5趋近100%；这与最终Test无优势并存，不足以归因为原型失效或梯度冲突。所有轨迹只含训练/Calibration诊断，不与每轮Test作相关或择轮。

Calibration_online_batch_accuracy是每轮100个优化步骤中批准确率的平均，非训练结束后对32窗独立评估，更非Development。round25 B3约74.625%、B4/B5均100%；TargetCE分别约2.893、6.63e-9、5.36e-9。标定批拟合饱和不能替代独立暴露泛化证据。

raw proto_loss/consist_loss可非零，但所有加权辅助项和相关lambda均0；TargetCE lambda1，SourceCE保留，未发现非预期损失。梯度有限性依据Phase35在每次Adam前检查及成功进程/25轮门禁，本轮不重新回放梯度。日志里val_loss是服务器合法源训练池CE，不是独立验证集。

规范轨迹见MODEL_WEIGHT_TRAJECTORY。早期派生PROTOTYPE_WEIGHT_TRAJECTORY中classification_head_state_changed一列实际只比较完整state SHA，不能支持分类头单独变化；规范表重命名为full_model_state_changed，原中间文件保留且不作为发布主表。CALIBRATION表source_model_change_max也指完整checkpoint的最大参数变化，不是分类头变化，也不是源保持性能变化。没有据此作头级因果解释。
