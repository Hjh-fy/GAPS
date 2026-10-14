# 第二次启动失败：C2反向隧道端口占用

2026-10-15 Asia/Shanghai。freeze commit及协议仍为 `dc73318636335e0d2317613dff00ef73e6c798f1`。

第一次恢复控制器PID2816在连接客户端前fail closed：`controller-to-ecs-c2 SSH tunnel failed to start`。
C2 sshd journal证实05:18:01 `bind [127.0.0.1]:18080: Address already in use`，随后该SSH会话退出。
此次未开始任何client local training，也没有round1或round25 checkpoint，仅server run_config；不能计为完成或进行test评价。
后续只读检查无18080 listener/connection，未对其他sshd或非本任务进程发出终止。
已确认本次server PID1923050的cmdline及冻结runtime cwd后仅SIGTERM该残留进程。

本地attempt02目录保留全部启动日志、pid及进度；远程启动目录保留为 `FONG-FULL-B20-S43_failed_attempt02`，其资产复制到此remote_server。
重新用原 `_start_ecs_c2_tunnels` 建立全部3条原始拓扑隧道：ALL_3_TUNNELS_PASS；完成后清理测试隧道。
不改端口、不改SSH/训练配置、不改任何科学协议或完成端点。仅恢复同一个未完成FULL20 seed43完整25轮，然后seed44。
