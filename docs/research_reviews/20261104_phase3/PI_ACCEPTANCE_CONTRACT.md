# PC/Pi和云边端验收准备（均待执行）

复用gaps_deploy/package_contract.py、inference.py等已有manifest/schema/权重匹配与旧UI/串口日志实现；本轮未安装运行库、未连接Pi、未部署新模型。57个ONNX含诊断图，不能每个文件都算部署分类器。PC/Pi版本、CPU/温度/调度/线程/ORT provider/库版本尚未测量，应写入验收记录。

## 包合同

一个不可分割manifest绑定：classifier graph与祖先checkpoint SHA、class-map、axis/dtype/opset、preprocessing代码/参数/采样/电阻单位、baseline版本、71D名字顺序/计算dtype、每gas回归头与scaler/支持、独立F回退、更新策略版本、训练身份与禁止暴露清单。分类encoder与原型的版本也必须配对；源失败不能换用相同文件名但不同特征版本的模型。

上位机输出每帧frame_id/time、gas_pred/置信度、raw/clipped浓度与ppm、clip支持、classification/regression/baseline/schema版本、fallback原因/头ID、status及分段延迟。原始电阻50×8不是已归一化窗，预处理依赖20–50s基线，不能二次归一化或把首次基线准备时间藏入神经推理时间。

## 拟定验收闸门（不是已达指标）

|项|执行方法|预定接受条件/阻塞|
|---|---|---|
|输入与数值|固定合成/源侧合法原始fixture，PC/Pi各自全预处理；逐窗identity/71D/scaler比较|身份及mask严格一致；float32特征rtol1e-5/atol1e-6；阈值执行前锁定，边界不合即停止|
|FP32分类|同fixture比logits/prob/argmax，先PC源checkpoint→ONNX再PC→Pi|proposed atol/rtol1e-4；argmax全一致，near-tie单列；失败不通过放宽容差修复|
|回归|相同预测gas和71D，比较F/N/KRR/Source/固定残差及F回退|FP64路径raw/clip<=1e-6ppm；若运行dtype变化须另注册容差而非混称通过|
|INT8|只使用合法source/cal做量化；FP32/INT8独立模型版本|数值/分类质量门槛执行前另定；不能沿用FP32等价结论，不能用Test定量化策略|
|延迟|采集/传输/分箱基线/71D/分类/路由/回归/clip/UI分段；warmup100、正式1000单窗，p50/p95/p99/max和冷启动另报|每5秒一窗时compute p95<5s为拟定可持续闸门；不是精度/整机既成性能|
|版本切换|staging验证全部SHA和schema，再原子切manifest指针；在途窗绑定旧完整版本|不得混用新classifier和旧regressor；失败保留last-good，日志可追溯|
|异常恢复|None/NaN/Inf、源头缺失/身份错、类输出非法、串口断连、时间倒退、进程重启、切包中断注入|非零残差用独立F头；rho0零源访问；分类失败拒给浓度；重连不复用错误baseline|
|长时运行|24h、5Hz输入/stride5s，约17280输出窗；掉线/重启脚本固定|0静默版本混用/非有限浓度；计数可解释；warmup后末2h与首稳态2h RSS中位数增幅<=10%为拟定门槛；日志有界轮转|

“0.413ms”本轮未追溯到足以核验的新SVR或完整链路测量合同，保留为历史范围待核的旧指标，不引用为新SVR、特征+分类+回归或整机端到端延迟。10秒采集窗本身与计算时间不同；全系统时延须说明从首样本还是末样本开始，是否包含初次baseline、网络和UI。训练/更新通信另计，不能用模型MAC或文件大小替代实际测量。

下一步先做PC包元数据/schema/故障预检，通过后才单Pi一致性与分段benchmark，最后24h稳定性；均不自动部署。要输出的是实际版本和实测证据，不能把本表拟定阈值当验收PASS。
