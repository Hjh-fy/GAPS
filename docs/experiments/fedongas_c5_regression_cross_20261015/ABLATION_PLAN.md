# 因子与范围

因子一：classifier=TO/FT/CLASS（既有checkpoint）；因子二：regressor=TARGET_RIDGE_83D/R84_CONCAT；预算5/20；训练seed42/43/44。
主因变量：S_ALL总体与per-gas RMSE、MAE、NRMSE_range；Oracle_ALL是共享模型诊断。
除classifier路由与已定义source-reference列外，目标身份、source reference、数值解法、候选alpha、folds、clipping和指标定义一致。
source参考包含既有H1协议/phase元数据，比较范围为这套已冻结增强流水线，不拆解其内部成分。
不增加额外因子、bootstrap、QC阈值或神经网络训练。无收益或负收益直接保留并结束。
