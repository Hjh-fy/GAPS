# FULL 20% seed43 第一次失败尝试与恢复

2026-10-15（Asia/Shanghai）。正式冻结commit `dc73318636335e0d2317613dff00ef73e6c798f1` 不变。

已完成25项的checkpoint、ordered fingerprint、协议和输入hash核验PASS；不重跑任何完成端点。
当前未完成项 FONG-FULL-B20-S43 于05:02:02发生Pi gRPC `StatusCode.UNAVAILABLE: Stream removed (Socket closed)`，控制器05:02:08 fail closed。
Pi第4轮local fit已返回；服务器日志停于round4 configure_fit；远程仅有round1–3聚合/适配checkpoint，无round25正式端点。
SSH恢复可用、Pi无重启、服务器该时段kernel journal无OOM记录。精确网络/SSH连接关闭触发因素未能由现存日志确定；不能声称已证明是Wi-Fi或模型错误。

归因层级：通信/进程基础设施中断，不是已验证的算法数值失败。
原隧道已配置ServerAliveInterval=30/CountMax=20；本次不修改已冻结源代码、协议或超参数，也不改变隧道拓扑。
已按cmdline及`/proc/<pid>/cwd`验证残留server PID1922181、C2 PID369807属于本任务的冻结runtime `/root/GAPS/iotj_final_runtime/4100c2532831`，仅对这两项执行SIGTERM。

原局部run目录移至此唯一attempt01目录，控制器stdout/stderr、pid和FAILED_CLOSED进度快照在此保存；远程1–3轮及其他输出完整复制在remote_server/。
原服务器目录另保留为 `/root/GAPS/iotj_final_classification_le1_20260804/FONG-FULL-B20-S43_failed_attempt01`。
FAILURES.jsonl仍保留原事件。

恢复策略：仅该未完成endpoint从共同random-S43原始初始化重新开始，完整25轮、LE1、每轮100-step适配、固定round25；不续训失败round3模型，避免改变随机性或更新预算。
控制器继续同一freeze版本，自动校验并跳过原25项完成端点，然后运行该项及FULL20 seed44。正式evaluation仍等全部27端点通过后仅执行一次。
如再发生异常，继续保留版本化证据、排查基础设施；绝不变更科学协议来强行跑通。
