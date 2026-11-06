# Phase3.8R 分类数值诊断与PC验收决策

依据Phase3.8固定提交`4dbf732df9ed693c88cb6486099930a428acc90b`，原Phase3.7容差不变。**Calibration诊断完成；历史Phase3.8严格验收仍BLOCKED，PC生产合同尚待独立批准，Day22 attempt2未授权。**

主诊断3020/3200次真实模型forward；预先独立登记的上下文补充304/400次；合计3324次，全为Calibration/合成推理，无训练/refit/回归调用/Pi或Test输入。Calibration只有16暴露32窗；batch128是同32窗重复4次，不是128独立暴露。

原后端、六种batch/排列相对同模式batch1的Calibration结果：

|模型|跨形态最大logits绝对差|最大阈值比|最小Top1/2间隔|原容差窗通过率|
|---|---:|---:|---:|---:|
|B0|2.67028809e-05|0.200126|0.216975|100%|
|B1|5.24520874e-06|0.035881|18.069722|100%|
|B2|2.2649765e-05|0.211118|0.610975|100%|
|B3|1.8119812e-05|0.145757|1.379525|100%|
|B4|6.67572021e-06|0.041341|18.816394|100%|
|B5|6.67572021e-06|0.040424|18.896821|100%|

主矩阵288行（六模型×两输入域×四后端×六形态）全部通过原逐分量容差与argmax检查；逐字节并不全相同。重复位置3072行全部在容差内，其中141行非逐字节相同，最大差2.86102295e-6。所有负的逐字节一致性证据保留。

定位到两类差异：原后端跨batch首次出现在`tcn1.conv.pointwise`；关闭MKLDNN后首次出现在`self_attn:0`。相同batch32切换no_grad/inference_mode，首差在Attention；仅改变输入张量创建上下文不改变结果，关闭MHA fastpath后上下文差异消失。B0 profiler记录Attention内部bmm/mm路径变化。不能据此将某层称为故障，也未证明Day22五次超容差的唯一根因。

阅读：[batch诊断](BATCH_DIAGNOSIS.md) → [接口/序列化](INTERFACE_AND_SERIALIZATION.md) → [后端/逐层](BACKEND_AND_LAYER_EVIDENCE.md) → [根因等级](ROOT_CAUSE_CONFIDENCE.json) → [PC合同提案](PC_CONTRACT_PROPOSAL.md) → [继续/停止](NEXT_DECISION.md)。运行与证据限制见[执行审计](EXECUTION_AUDIT.md)。完整Calibration/合成logits、层比较与权重留本地，公开精简矩阵、哈希和代码文本。
