# 模型来源与冻结范围

分类：Phase35六个锁定端点，继承Source attempt2的合法Day7 Sstar（C1Day4/C2Day1/7，120源暴露2520窗）。loader调用原Source checkpoint祖先验证，确认25轮源receipt/参数、初始state及源身份；六端点核对文件SHA、完整state schema/finite、stateSHA并strictload。本轮Calibration不参与训练或选端点。

回归：Phase1.5 FONG-P15-C5-A4-F-SVR-20261102 g0–g3，合法源fewshot选参、每气体4暴露8Calibration窗，原冻结数值代码、scaler、C/gamma/epsilon和量程。MODEL_SOURCE_LOCK保留原模型index条目、路径/SHA/训练文件/身份SHA；与Phase36已确认的32标定身份一致。头文件SHA及gas/scaler/support/参数在load时再检查；不fit、不替换成Source-only。

特征：FEATURE_CONTRACT完整71名/顺序/索引及extractor/rich formula路径，原函数由v5库存逐文件SHA校验后加载。分类64D是内部encoder，与71D独立；生产接口没有真实类别或文件名输入。

冻结核验：成功验收逐文件重哈希Phase35的2275制品、Phase36的51制品和Phase15的185制品，不覆盖模型、日志或指标；字节SHA检查不等于Test数值读取/评价。完整v5原文件不写入，本轮只读原数值函数和16标定原始文件。
