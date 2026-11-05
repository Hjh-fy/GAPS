# 数据身份、预处理及标签访问

evidence/TRAINING_WINDOW_IDENTITIES.csv逐行2552个训练窗，含原暴露文件、设备、日期、气体、浓度、起止秒、rawSHA、物理窗口身份、export row。源仅C1Day4(840窗)及C2Day1/7(1680窗)，共120次独立暴露；目标仅C5Day18 A4每气体4次/两固定响应窗(60、150s)，16暴露32窗。不重新选浓度，不随机窗口混合角色。源21窗/暴露有重叠，2520窗不是2520次独立采集。

真实原始文件仅打开136个：120源+16目标Calibration，RAW_ACCESS_TRACE逐文件SHA及操作。直接复用归档process_raw/有效mask，匹配Phase1锁定窗口身份，数据形状50×8、float32，labels0..3。没有回归浓度标签数组；GasSensorWindowDataset按原逻辑填零占位，USE_REG_LOSS=False。身份表浓度元数据为审计字段，不是额外监督任务。

本地data含18个npy：三客户端各train/calibration的features、class labels、phase labels。source calibration_是同一合法训练池复制，target calibration_是同32条train复制，不增加任何独立标签，不冒充Development。phase全0仅ordinary-class原型中性索引，不是实测动力学phase。模型schema保留num_phases3，其它phase槽未利用；若日后研究真实阶段必须另立合同。

EXCLUDED_ROLE_IDENTITIES保留Development/Test/unused/reserved身份元数据和历史SHA，不导出其数值。C5Day22 Test未打开；本轮没有任何Test数组推理、评分、模型选择或新预测类别路由。旧阶段文件仅哈希验证，其历史结果不构成本轮训练信号。

|组|训练标签访问|Development/Test|独立目标标签与重复呈现|
|---|---|---|---|
|B0|源两个客户端CE|均禁入训练/回调|0|
|B1|S*后仅C5 A4 CE|均禁入|16暴露32窗；预定40000呈现|
|B2|源CE+C5普通第三客户端CE|均禁入|相同A4；num_examples=32，不按50epochs膨胀|
|B3/B4/B5|源本地CE+服务器sourceCE/targetCE；服务器可见标定输入/标签|均禁入|相同A4；原生服务器实际80000，因而BLOCKED|

这些是角色预算/访问合同，不是声称真实六组已经运行。服务器看到target校准标签是集中监督适配，不宣传标签仅端侧或隐私隔离。

## 最终评价与时间边界

目标C5Day22 40暴露840窗只在全部端点及配置SHA锁定后由独立离线评价程序释放。不能default test、server-val指向目标Development/Test、best-test-round、事后选择设备最佳策略。旧数据已被观察，后续仍标回顾性，不能重称首次盲测。

全部组固定端点round25（B1 block25/update2500）；B3同时保存aggregated与adapted诊断，B3–B5主要适配端点统一用adapted，下一轮分发模型身份另列；禁止按两者分数择优。报告Accuracy、Macro-F1（固定4类、zero_division0）、逐类Recall/混淆矩阵、按暴露等权错误率：每暴露错窗比例后对暴露平均。单窗指标与暴露多数投票/均值融合另列；不将窗口bootstrap当独立重复。

源C1/C2 R4作为全部模型锁定后的历史保持评价，提前冻结身份与valid mask，不能用它们source预训练或选参；其原通用角色表为unused不意味着可随意转作训练。保持量同时报更新前S*与更新后指标差、逐类/逐暴露变化；source与target不同数据域分别汇报。C3Day8接入→C5Day18接入的模型可做顺序更新，但C3Day17 Test早于C5接入；接入后评价它只能叫历史留出暴露保持，不叫未来在线新采集。未来未接入节点的cal/dev/test不得反向进入此前训练，transfer只能在端点锁定后离线审计。C4Day3早于共同源Day7，只保留回顾性支线。
