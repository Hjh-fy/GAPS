# C5四档预算真实类别回归

10%/15%新增4bank/16最终gas fits；5%/20%只复用；无分类/source回归重训。Oracle_ALL全部1360，逐gas340。
回归器与classifier seed无关，不对复制的Oracle计算seed SD或显著性。

|预算|回归|RMSE ppm|NRMSE_range|来源|
|---|---|---:|---:|---|
|5%|R84_CONCAT|26.713984|0.129234|reported_reused_5_20_no_refit|
|5%|TARGET_RIDGE_83D|27.034946|0.130546|reported_reused_5_20_no_refit|
|10%|R84_CONCAT|19.632778|0.097110|recomputed_new_fixed_bank|
|10%|TARGET_RIDGE_83D|20.370477|0.099242|recomputed_new_fixed_bank|
|15%|R84_CONCAT|25.644474|0.117935|recomputed_new_fixed_bank|
|15%|TARGET_RIDGE_83D|26.120114|0.119029|recomputed_new_fixed_bank|
|20%|R84_CONCAT|18.507090|0.086269|reported_reused_5_20_no_refit|
|20%|TARGET_RIDGE_83D|19.773050|0.091439|reported_reused_5_20_no_refit|

## 源增强差值（R84−83D，负值更好）

- 5%：ΔRMSE=-0.320963 ppm；ΔNRMSE=-0.001312。
- 10%：ΔRMSE=-0.737700 ppm；ΔNRMSE=-0.002132。
- 15%：ΔRMSE=-0.475640 ppm；ΔNRMSE=-0.001094。
- 20%：ΔRMSE=-1.265960 ppm；ΔNRMSE=-0.005170。

逐gas负收益和非单调趋势均保留；固定源参考含H1协议/phase信息及原有clipping，已知饱和未在本轮修复。
C3/C4主标定回归直接复用既有R1，不将旧A4分类路由当作CLASS。结果不支持对独立设备/不同划分的普适声明。
