# 原始实现、适配器、上下文及序列化

Phase3.5原推理位于main内部，main会访问Test，因此不调用。静态核验原forward表达式`m(torch.from_numpy(x[i:i+128]))[0].cpu().numpy()`及`json.dumps(z[i].tolist())`，在新诊断中按固定形态来源等价执行。P37 ClassifierAdapter和P38 ObservedClassifier原API只接受window，真实32次单窗对照；较大batch的诊断shim不是旧adapter新能力。

原后端下，保持inference_mode一致的来源等价forward、P37/P38实际单窗、batch诊断shim及strict reload均逐字节一致。输入/输出float32，所有模块eval，state SHA相同。主INTERFACE_MATRIX240行中仅六条P35_no_grad_only_control与inference模式非逐字节一致，最大1.0967254638671875e-5，但仍通过原容差；不能把“同上下文一致”误写成原P35真实上下文完全相同。

补充诊断在运行前另立CONTEXT_REGISTRATION，固定六模型×四后端×batch1首窗/batch32×四种创建/执行上下文：FF=普通tensor+no_grad执行；FI=普通tensor+inference执行；IF=inference tensor+no_grad执行；II=inference tensor+inference执行。304次forward，无新搜索。原后端batch32 FF=IF、FI=II逐字节相同；FI/II与FF的六模型输出不同、最大1.0967254638671875e-5。batch1首窗四组合相同。关闭MHA fastpath后，batch32四组合逐字节相同；单独关闭MKLDNN仍保留上下文差异，最大4.291534423828125e-6。全部288条context/去hook比较仍在原容差内。

原保存代码没有对logits设置小数位；Python float列表→JSON保留这些float32值。五条封存故障摘要两端logits均可精确转float32再回float64，无数值截断证据；当前288组输出JSON往返float32逐字节相同。仅静态读取这五条差异和公开摘要，没有打开完整历史Test CSV/1376条记录，不能把局部往返证明说成完整CSV行顺序/匹配再次核验。Phase3.8原身份检查结论保留其历史来源。
