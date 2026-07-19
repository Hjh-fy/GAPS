# a003 与 B2 两轮 pilot 的时间诊断

## 证据边界

- 输入为失败 attempt `c12_to_c5__b2__s42__a003` 已回收的 ECS、Pi、PC 事件与资源 JSONL；a003 保持 failed，不进入任何 confirmation 统计。
- 仅使用其完整的 round 1--22；不读取 C5 test、checkpoint 指标或模型输出。
- `client_waiting_or_sync_residual_s = round wall - max(C1 fit callback, C2 fit callback) - server aggregate`，是 Flower 调度、消息传输、控制器等待和未被上述计时覆盖工作的合并残差，不能单独标记为网络延迟。

## 与两轮真实 B2 pilot 的比较

| 指标 | a003 mean (s) | pilot mean (s) | 倍数 |
|---|---:|---:|---:|
| round wall | 803.56 | 193.07 | 4.16x |
| PC C2 local train core | 647.97 | 20.44 | 31.71x |
| Pi C1 local train core | 41.82 | 11.11 | 3.77x |
| ECS server DA | 152.68 | 168.53 | 0.91x |

## 结论

1. **主 slowdown 是 A：PC C2 local training。** 它平均占 round wall 的 80.6%，并相对 pilot 增长 31.71x。
2. **ECS server DA 仍是第二大绝对耗时，但不是此次变慢的来源。** a003 为 152.68 s，低于 pilot 的 168.53 s（0.91x）。
3. Pi local training 增长到 41.82 s，但与 PC 并行运行，且远小于 PC critical path；它不是主瓶颈。
4. 每轮可得的 waiting/synchronization 合并残差均值为 2.82 s；它不能支持“网络是主因”的结论。不存在随 round 持续增长的单调证据；逐轮 CSV 用于后续复核。
5. 按 a003 mean，真实三机 B2 的 25-round training 下界约为 5.58 h，另加恢复和 validator 时间。该估计只适用于当前 C2 PC 状态，不能外推为 B5 或其它 host placement。
