# B2-s42/a006 手动回收与系统指标诊断

## 结论边界

`c12_to_c5__b2__s42__a006` 已真实完成 25/25 轮，但 Controller 在本地证据回收阶段失败，状态链已经不可变地记录为 `failed / process_failure`。因此本页只能作为 failed-attempt/system diagnostic evidence，不能将 a006 改写为 canonical，也不能纳入正式 confirmation 统计或正式论文主表。

三端原始证据已手动回收到原 attempt 的 `raw/ecs`、`raw/pi`、`raw/ecs_c2`，远端原件保留。关键 events/resource/checkpoint SHA-256 与远端一致。使用正确 attempt basename 的短 junction 完成结构验证，结果为 `valid`，审计 SHA-256 为 `be0e1a1bd394e7e90f472842b0b026aa4eb6a84690486141739f2e31bb368893`：25 rounds、50 FitIns、50 FitRes，C1/C2 资源覆盖率分别为 97.27%/97.46%。

## 初步系统指标

- 25 轮 serialized Flower application messages：17,574,807 bytes（16.7606 MiB），平均每轮 702,992 bytes；transport bytes 未采集。
- round wall：mean 241.65 s，p50 228.06 s，p95 269.62 s，25 轮合计 6,041.36 s（约 100.69 min）。
- Pi C1 local train mean：42.09 s/round；ECS C2 local train mean：76.61 s/round。
- server DA mean：164.15 s/round，p95 189.22 s，占 mean round wall 约 67.93%；server non-DA mean 仅 0.068 s。
- Pi training-overlap RSS mean/peak：514.23/518.41 MiB；host CPU mean/peak：84.74%/91.29%；温度 mean/peak：57.69/62.25 °C；未观测到 throttling。
- ECS-C2 training-overlap RSS mean/peak：522.27/523.92 MiB；host CPU mean/peak：49.95%/54.74%。该 ECS 不提供 CPU 温度。
- Observer 全 producer 累计测量开销约 6.086 s，占 25 轮 round wall 合计约 0.1007%。

当前结果说明，在该 ECS server + Pi C1 + ECS C2 拓扑中，server DA 仍是最大的单阶段耗时；ECS-C2 local training 是第二大阶段，Pi C1 不是关键路径。该表述仍是 a006 failed-attempt diagnostic observation，而不是 canonical system claim。

## 失败根因与下一步

实际失败发生在训练完成之后：ECS-C2 拓扑没有本地 PC sampler 预先创建 `attempt/raw`，而 Controller recovery 在第一次 `scp` 前没有显式创建该父目录。手动验证使用过长的输出文件名时又独立触发 Windows 路径长度限制；短 junction 验证成功。

B5 未启动，三端目前没有残留 Flower 或资源采样进程。下一步必须先为 recovery parent creation 增加最小 RED→GREEN 回归测试与修复，并用短输出路径验证 Windows recovery/validator；不能仅凭 a006 的 raw evidence 把 failed 状态提升为 canonical。
