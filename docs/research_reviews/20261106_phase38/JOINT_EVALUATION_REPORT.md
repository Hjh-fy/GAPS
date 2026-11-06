# 分类—浓度联合评价交付状态

总体BLOCKED：六固定视图未全部完成和锁定，不能执行正式评分。用户要求的Accuracy、Macro-F1、逐类Recall、4×4混淆、完整流量不可用率、条件浓度误差/覆盖率、错误路由分布均暂不生成。B2完整与其他组按数值失败截断的前缀不能作为公平的六组联合比较。

已交付六组逐窗JSON（1/13/840/507/9/6条），本地合并`PRIVATE_OBSERVED1376_WINDOW_RESULTS.csv`。包括窗口/暴露身份，真实气体仅作记录，预测气体、实际head、raw/clip ppm、支持、版本及运行状态。剩余3664条从未执行，不造空预测、不记UNAVAILABLE、不补Oracle。

`OBSERVED_PREFIX_CALLS.csv`只记录已观察前缀各预测gas的实际请求/调用/有效数，0请求的比率为null；它不是840窗完整流量报告。五条logits漂移的bridge状态仍为OK，但视图验收失败，独立保留数值事件，不将其吞为普通正确预测。

`EVALUATION_DELIVERY_STATUS.json`逐项标明WITHHELD/NOT_CREATED。现有score.py保留预定实现和SHA，需全部六组锁且逐窗MATCH才可运行。本轮没有评分调用，也不拿已知Phase3.5准确率替代桥接评价。

未来条件回归只在分类正确且回归有效子集按真实gas计算，raw/clip并列，分母和独立暴露数一并给出；NRMSE归一化为对应冻结Calibration支持宽度（Ethanol/Ethylene112.5ppm，CO/Methane225ppm）。暴露等权误差在该子集内统计，浓度/暴露分组单列。不同分类器正确子集不同，不能据条件NRMSE给整体排名。空子集必须undefined，不能0误差。跨错分气体混合NRMSE和未定义业务容差的联合成功率均不计算。

本轮不说明B2优于其他算法，也不据失败位置推断某种气体、机制或物理老化。完整联合评价尚未获得。B0只表示分类器Source-only，整个联合系统仍依赖C5 A4回归训练和Calibration输出支持，不是无需标定系统。
