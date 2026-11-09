# ARM64 执行、恢复与下一阶段门禁

实机为 Raspberry Pi 5 Model B Rev 1.1，aarch64、Debian 13.5、内核 6.18.34+rpt-rpi-2712，内核可见内存 8,062 MiB。独立 CPython 3.12.11 通过 uv 安装到研究目录，保留系统 Python。PyTorch 2.5.1、NumPy 1.26.4、SciPy 1.13.1、scikit-learn 1.8.0、joblib 1.4.2 及依赖按哈希锁安装，25 个包的兼容性检查通过。[PyTorch 2.5.1 ARM64 wheel](https://pypi.org/project/torch/2.5.1/#files) · [scikit-learn 模型持久化要求](https://scikit-learn.org/stable/model_persistence.html)

设备使用 ATen/OpenMP CPU 后端，MKLDNN 可用且开启，MHA fastpath 开启，intra/inter-op 均为 1，已加载的 OpenBLAS 线程池为 1，float32 matmul 为 highest。硬件探查记录了温度与 CPU 时钟传感接口，正式温频与 RSS 采样属于性能门禁。没有算子级 trace，不能将 logits 漂移归因于某个层或内核。

第一次启动在文件审计日志初始化时因重复的 PID 参数退出，发生在模型加载和第一次推理之前。保留初始包与日志。仅修正重复 PID 字段后，在独立目录启动首次数值运行，模型、输入、参考、环境和协议哈希保持相同。此后没有第二次 Day22 推理。

## 私有恢复与核验

- 恢复原验收环境时使用封存的 PI_REQUIREMENTS.lock 与 CPython 3.12.11。采用 uv pip sync --require-hashes --no-build，并检查依赖一致性。
- 恢复归档前比对 FINAL_VERIFICATION.json 中的摘要。首轮包在新目录解压，以 BUNDLE_LOCK.json 校验全部 32 个成员。运行入口 run_pi.py 根据 --bundle-sha 校验合同与输入，再验证分类全 state、SVR、scaler 和 feature SHA。
- 已封存的证据可离线审计。先验证 PI_RUN_EVIDENCE.tar.gz 并解压到新目录，再以 audit_evidence.py 读取证据，不需要执行模型。该脚本对照私有 PC 包与输出证据，检查身份、数值和调用次数。
- 日志、权重、输入、逐窗差异和恢复向量留在本地私有制品与 Pi。GitHub 审阅包只含聚合指标、合同、环境摘要和哈希。

## 继续条件

完整数值门禁失败，因此性能门禁未运行。预先设定每组 20 窗预热、200 窗计时、60 秒持续运行和 1 秒资源采样。运行目标为单窗联合推理 P99 ≤100 ms、冷启动至首个联合输出 ≤60 s、峰值 RSS ≤1024 MiB、温度 <80 °C。当前没有该合同下的分段实测。历史 0.413 ms 不能用于该门禁。

若另立诊断，先以独立的合成和标定样本定位 CPU 计算路径差异并审阅业务数值要求。任何新合同应先固定并独立验证，再决定是否授权新一轮完整重放。在线传感器联调还需采样时序、通信、端到端延迟与故障恢复的实机证据。
