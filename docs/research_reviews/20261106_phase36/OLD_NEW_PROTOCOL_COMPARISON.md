# 旧反馈正向证据与新协议负结果

旧资料经v5 SOURCE_INVENTORY的SHA核对，仅打开最终解释及合同元数据，未读旧Test逐轮曲线。旧M0=有反馈J0、M1=关闭反馈，原报告固定25轮C5为97.353% vs93.897%（+3.456pp）；本轮B4−B3为−1.548pp。两者都须保留，不能挑选一个口径作普遍结论。

|项|旧20261025 M0/M1|新Phase35 B4/B3|
|---|---|---|
|源队列|C1/C2各2360训练窗；origin指向fedongas_mmd_only_20261019，完整暴露/日期身份在已检索role/effective资料中unknown|C1Day4 840窗、C2Day1/7 1680窗；120独立暴露，源截至Day7|
|初始化|effective_config initial SHA97cc719d...；TRAINING_QA initial_fingerprint297af6fa...口径不同，未当同一SHA；无法证明等于新Sstar或合法Day7祖先|合法Sstar全state431fa41e...，完整25轮源祖先，六组相同|
|目标标定|C5名义5%，80窗；独立暴露预算/与Test互斥未由该window合同闭合|C5Day18 A4，16互斥暴露32窗；60/150s；无Dev/Test训练|
|Test|C5 1360窗，源C1/C2各680；原窗口协议，暴露重叠风险未排除|C5Day22 40暴露840窗，源C1Day21/C2Day16各40暴露840窗|
|损失|SourceCE+TargetCE+Class-MMD2 weight.5，normalized64D，原型加权开启|SourceCE+TargetCE，其他加权损失0，B3/B4参数原型加权关闭|
|客户端更新|25轮、local1epoch、batch_size字段32；源实际梯度/呈现完整账本未本轮重建|25轮、batch16、每源1epoch，实际3950源客户端更新/63000呈现|
|服务器|100steps/round；source reference640窗；effective batch_size32不能单独证明真实DA目标batch/呈现次数|100优化步/轮，Source10批+Target16窗，实际目标2500步40000呈现；source池2520|
|反馈与评价|M1无反馈仍适配，下一轮aggregated，评价adapted25；M0返回adapted|同语义，对应B3/B4|
|选择与独立性|旧Test历史已观察，解释还列后5轮均值，不能当5独立seed|只固定最终端点；本轮不读任何Test逐轮值；单seed回顾性|

旧严格20261020的R1–R3队列是另一套实验，不能因同名C5把它补到此M0/M1源日期。旧日期/暴露祖先缺口保持unknown；不是在本轮证明泄漏。批字段、目标窗口80、优化步2500不能推断旧目标呈现恰等于新40000。

损失、源/目标队列、起点、样本粒度、客户端预算及目标统计同时变化，不能把符号变化单独归因“严格划分”“取消MMD”“源模型变弱”或某一机制。只可说旧条件下正向观察未在新条件复制，当前不存在反馈普遍有效证据。
