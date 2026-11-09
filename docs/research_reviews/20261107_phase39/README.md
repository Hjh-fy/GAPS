# Phase3.9 PC 单窗推理验收

PC-F32-CPU-SINGLE-v1 通过本机 Calibration 和合成验收。Phase3.8 历史 batch128 裁决继续为 BLOCKED。生效合同采用 Phase3.8R 的原始 MKLDNN 和 MHA fastpath 配置，在 CPU 上以 float32[1,50,8]、eval、inference_mode 和线程 1 执行冻结 Phase3.7 适配器和桥接。[生效合同](PC_CONTRACT_R5.json)

| 比较 | 窗口 | logits 向量 | 71D | 四头 raw/clip 对 | 路由 raw/clip 对 | 最大 logits / ppm 绝对差 |
|---|---:|---:|---:|---:|---:|---:|
| 独立进程 | 36 | 216 | 36 | 144 | 216 | 0 / 0 |
| 同进程重复 | 36 | 216 | 36 | 144 | 216 | 0 / 0 |
| 完整模型重载 | 36 | 216 | 36 | 144 | 216 | 0 / 0 |

输入为 C5 Day18 A4 的 16 次暴露、32 窗及 4 个预注册合成窗。golden 由 PID 40324 生成并在退出后封存，PID 34920 核验封存后独立复验。逐分量分类沿用 atol 1e-4、rtol 1e-5。三轮 logits、argmax 和 71D 精确一致，raw/clip 满足原 ppm 容差。模型、scaler、支持范围和输出版本均匹配锁。[Golden 封存](attempt5/golden/GOLDEN_SEAL.json) · [复验与环境](attempt5/verify/RESULT.json) · [独立审阅](INDEPENDENT_REVIEW.json)

| 工作者 | Python open 事件 | 允许 | 拒绝 | 真实分类 | 特征 | SVR |
|---|---:|---:|---:|---:|---:|---:|
| Golden | 6094 | 5732 | 362 | 216 | 252 | 360 |
| 复验 | 6662 | 6285 | 377 | 650 | 758 | 1082 |

每个数值工作者持久化了 Python open 决策和包装 API 结果，调用账本通过 PID、请求序号、输入 SHA 与 open 日志关联。两工作者各有两个预定范围拒绝。其余拒绝属于已登记的库探测，均维持拒绝。复验另外记录两次注入的重载文件缺失。覆盖范围从工作者注册审计钩子起，限 Python 审计事件和包装 API 可见结果，原生系统调用及解释器启动不在该审计内。完整本地日志位于两个工作者的 FILE_OPEN.jsonl 和 MODEL_CALLS.jsonl，公开包保留统计、SHA 和核验结果。[访问与调用核对](attempt5/GATE.json)

前四次尝试均保留 BLOCKED。第一、二次分别停在 NUL 与临时目录初始化，真实模型前向为 0。第三、四次各完成 216 次 golden 分类前向，因库探测分类门禁未封存，均未启动独立复验。修订限于精确空设备、工作者私有临时目录和库探测的审计处理。科学阈值、模型、输入、后端及每次预算沿用原锁。[第一次](ATTEMPT1_NOTE.md) · [第二次](ATTEMPT2_NOTE.md) · [第三次](ATTEMPT3_NOTE.md) · [第四次](ATTEMPT4_NOTE.md)

异常与恢复见 [恢复报告](RECOVERY_REPORT.md)。本机裁决与 Day22、Pi 的继续条件分列于 [决定矩阵](DECISION.md)。历史冻结资产和旧授权消费记录的复核见 [最终验证](FINAL_VERIFICATION.json)。完整逐窗 golden、复验输出和权重留本地，公开清单列出来源哈希。
