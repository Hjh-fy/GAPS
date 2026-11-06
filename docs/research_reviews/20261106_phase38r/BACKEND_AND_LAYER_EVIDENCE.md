# 计算环境、后端干预与逐层定位

Windows11，Intel Core i5-8400 CPU @2.80GHz，Python3.12.4、PyTorch2.5.1、numpy1.26.4；float32 CPU、torch线程1、interop线程和MKL/OpenMP配置见ENVIRONMENT。原MKLDNN及MHA fastpath均开启。四模式预先固定：original、仅关MKLDNN、仅关MHA fastpath、两者关；原始结果保留，各模式后最终恢复原开关，固定b32恢复数值逐字节一致。

基于同模型、同输入、同batch的216条backend对比，关闭MKLDNN最大logits差3.2901763916015625e-5；仅关MHA最大3.814697265625e-6；全部仍在原tol内。这是后端开关对数值有因果作用的受控证据，不证明某后端有bug，不构成生产模式优化选择。

只读hook记录实际执行顺序/输出，所有输入在batch轴对齐。原后端六模型、两域b1/b32首个非逐字节差均为tcn1.conv.pointwise；关MKLDNN后均移至self_attn:0，即Attention输出。关闭MHA不能消除MKLDNN开启时TCN先出现的差异；两者关闭仍有Attention的跨batch舍入差异。主b32各层未越描述性阈值，隐藏层阈值不是新增科学性能门禁。

同batch32上下文FF对II的首个差异，原后端六模型均在self_attn:0。输入创建因素不改变这一点，执行上下文因素改变；MHA fastpath关闭后消失，支持Attention fastpath与执行上下文的交互。B0 profiler两上下文都显示_native_multi_head_attention，但FF内部出现bmm([32,50,48],[32,48,144])，II出现mm([1600,48],[48,144])及clone/copy路径差异。维度对应Attention的QKV投影这一解释属于结合模型代码的推断；没有单独干预ATen内部每个矩阵操作，不能宣称已唯一证明某个kernel的错误。

B0主profiler原batch1/32的MKLDNN convolution计数3/8，关闭MKLDNN时消失；原native MHA两者1/1，关闭fastpath时消失。只对B0做算子profile，不能冒称六模型各自都profile。其余模型的层输出及控制独立检查。完整operator条目留本地，公开OPERATOR_ROUTE_SUMMARY给路由计数。

主诊断96个新增去hook控制（每域/每模式/模型的batch32和batch1首窗）逐字节一致；original Calibration的来源等价/重新加载还覆盖全部形态。context补充96个batch32去hook控制也一致。没有把hook首差解释成训练故障、物理老化或气体特性。
