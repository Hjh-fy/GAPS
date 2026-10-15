# 已完成：36组固定流水线，勿重复运行

日期2026-10-15；pre-run commit：`04f4f3883e248b9f23d61d01bd6a0aa30e68dd24`。
结果：`results/fedongas_c5_regression_cross_20261015/`。

## 结果摘要

以下为预测类别路由、全部1360 test、3分类训练seed的RMSE均值（ppm）；逐seed、sample SD、MAE、NRMSE与逐gas结果均已保存。

|分类器|5%目标83D|5%源增强R84|20%目标83D|20%源增强R84|
|---|---:|---:|---:|---:|
|TO|39.92289|39.73196|22.50146|21.44501|
|FT|37.55650|37.38278|26.40426|25.55294|
|CLASS|37.70607|37.53504|23.03229|22.51283|

源参考增强有小幅平均总体RMSE收益，但存在气体/seed负收益。5%下FT/CLASS优于TO，20%下TO更好。
不能由本轮声称FULL/类别–阶段机制优越，不能把共享Oracle重复视作独立随机回归实验。
详见结果根`RESULT_INTERPRETATION.md`、`CLASSIFIER_REGRESSION_REPORT.md`及`EXPERIMENT_AUDIT.md`。

## 复核而不是重跑

```powershell
D:\anaconda3\python.exe scripts/run_fedongas_regression_cross.py verify
```

该命令只核对冻结输入与保存预测/指标，不拟合、不访问测试原数组或重训分类器。
`run --freeze-commit 04f4f3883e248b9f23d61d01bd6a0aa30e68dd24`已执行一次；输出存在时会拒绝覆盖。不要删除输出根后重跑。
所有36评价结束；不自动增加FULL、C3/C4、QC、多seed或任何训练/搜索。
