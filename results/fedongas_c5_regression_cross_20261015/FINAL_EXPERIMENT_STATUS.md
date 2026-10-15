# 最终实验状态

- 研究：C5 TO/FT/CLASS × 83D目标Ridge/R84源参考增强Ridge × 5%/20% × seed42/43/44。
- **36/36 EVALUATED，原固定端点全部保存。** 0次新增神经网络训练、0次源回归训练，4个共享目标bank/16个最终gas Ridge fits。
- 运行前冻结：`04f4f3883e248b9f23d61d01bd6a0aa30e68dd24`。
- 输入/hash/总体scope复算PASS；29项相关测试PASS，1个历史目录存在性测试已记录并显式排除；compileall PASS。
- 科学结果：源增强有小幅平均总体RMSE收益但存在逐气体/seed负收益；分类知识复用的收益随预算变化。5% FT/CLASS相对TO改善，20% TO更好，CLASS+R84相对TO+83D近似持平。
- 未新增FULL/C3/C4、QC、多seed、神经训练或算法搜索。研究已停止。
- 中文解释：`RESULT_INTERPRETATION.md`；完整表和自动报告：`CLASSIFIER_REGRESSION_REPORT.md`。
- GitHub发布由本对话完成后汇报实际result commit及push验证；本文件不提前声称远端已更新。
