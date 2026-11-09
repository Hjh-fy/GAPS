# Phase3.10 Day22 单窗联合评价 attempt2

PC-F32-CPU-SINGLE-v1 下六个冻结视图全部通过。C5 Day22 的 40 次物理暴露各 21 窗，六视图各 840 条，共 5040 条。唯一一次授权在独立预检与本次进程 Calibration 恢复均通过后消费。[独立预检](PRECHECK_PASS.json) · [授权消费](attempt2/AUTHORIZATION_CONSUMED.json) · [六视图封存](attempt2/ALL_SIX_VIEWS_LOCKED.json)

冻结 Phase3.9 的 float32 CPU 单窗路径，保留原 MKLDNN、MHA fastpath、分类和 ppm 容差。B3–B5 使用 adapted25，四个 C5 A4 F-SVR71 头与 scaler 来自原 PC 锁。B0 的 Source-only 口径限分类端，浓度端同样使用 C5 标定模型。[注册](REGISTRATION.json) · [PC 合同来源](https://github.com/Hjh-fy/GAPS/blob/cb850dadfbef986702e3155372fef225a818f2b1/docs/research_reviews/20261107_phase39/PC_CONTRACT_R5.json)

Phase3.8 历史裁决继续 BLOCKED。5040 条 Legacy Comparison 中，61 条超过旧 logits 容差，argmax 变化 0 条。旧比较在全部预测封存后由评分进程计算。本轮为既往已观察 Test 的回顾性评价。[分类与浓度报告](RESULTS.md)

预检、推理和评分分别保留 Python open 审计。预检与推理调用账本按 PID、请求和 SHA 交叉核对。登记时检查了历史身份表，该表原有气体和浓度列。新清单只保留窗口身份和哈希。输入重建检查 840 窗信号和 71D 哈希，正式标签表及旧预测只在评分读取。原始文件名仍编码气体与档位，分类器及回归路由只接收窗口。审计范围是钩子安装后的 Python open 及包装 API 可见结果。

本地封存包含六组 PRIVATE_WINDOWS.jsonl、合并评分 PRIVATE_ALL5040_SCORED.jsonl、文件访问日志与模型调用账本。公开包提供摘要、哈希、评分表和独立审阅。[独立审阅](INDEPENDENT_REVIEW.json) · [制品哈希](ARTIFACT_SHA256.csv) · [Pi 决定](DECISION.md)
