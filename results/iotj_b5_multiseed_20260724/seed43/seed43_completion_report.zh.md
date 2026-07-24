# B5 final multi-seed：seed43 正式 canary 完成报告

## 结论

`c12_to_c5__b5__s43__a001` 已在正式三机拓扑上完成，最终状态为 `canonical / validator_accepted`，postflight verdict 为 `PASS`。本次仅训练并评估 seed43；没有启动 seed44–46，没有重训 seed42，没有运行回归、Pi inference benchmark 或 low-calibration，也没有修改 runtime v4、QC、HC95/HC90。

## A. 三机 preflight

- 结果：`PASS`。
- 本地控制端 HEAD 与 origin 均为冻结 M0 commit `d3ac8b90c87d50a7a0be1fd985d883db15456703`。
- ECS server/DA、Pi C1、ECS C2 均可通过正式 SSH 链路访问；探测时钟偏差为秒级。
- 未发现旧 Flower/GAPS 训练进程，也未发现 8080/18080 端口冲突。
- Pi 探测温度为 48.8°C、`throttled=0x0`；三机 RAM、磁盘与输出写权限均通过。
- 冻结控制器的训练前与训练后 `--preflight-only` 均返回 `preflighted`。

实际内容寻址代码路径：

- server/DA：`/root/GAPS/confirmation_runtime/52bdbf96568014cc363f0ce3c666026be29f5f0279c7a130b41458d42a0d0c68/src`
- Pi C1：`/home/gaps/GAPS/confirmation_runtime/52bdbf96568014cc363f0ce3c666026be29f5f0279c7a130b41458d42a0d0c68/src`
- ECS C2：`/root/GAPS/confirmation_runtime_c2/52bdbf96568014cc363f0ce3c666026be29f5f0279c7a130b41458d42a0d0c68/src`

三机均绑定训练 commit `2ef7aea77b9dfabdd09da4f38742907a37c58c30`、source archive SHA256 `52bdbf...d0c68`、dataset manifest SHA256 `fb8946...430` 和 seed43 algorithm SHA256 `2fa304...70fe`。

## B. 训练完整性

- 25/25 rounds 完整。
- `FitIns=50`、`FitRes=50`。
- 25 轮均由 C1 与 C2 两个客户端参与，fit/evaluate failure 均为 0。
- 每个客户端每轮 `local_epochs=5`、`batch_size=32`、seed=43。
- 服务端每轮执行 100 DA optimizer steps，合计 2500 steps。
- 25 份 client stats、25 份 DA 记录、25 份普通 checkpoint 和 25 份 adapted checkpoint 连续存在。
- history round 序列严格为 1–25；没有静默跳轮、NaN/Inf、异常恢复或其他 seed checkpoint 恢复。

## C. 数据与配置合同

- source clients：C1、C2；target：C5。
- C1/C2 train 均为 2360 行；C5 calibration/test 为 320/1360。
- 服务端 `strict_calibration_split=true`，源域 DA 仅使用 C1/C2 calibration，各 320 行；目标域仅使用 C5 calibration 320 行，C5 test 未用于 fit/select/stop。
- ECS C2 客户端继续复用 seed42 正式九文件子集，客户端 phase 按 `-1` 处理；seed42 审计也记录相同语义。这是冻结拓扑合同，不是 seed43 漂移，且未擅自补入 phase 文件。
- rounds、local epochs、batch size、Adam 5e-4、B5 开关、DA 100 steps/round 和设备配置均未改变。
- runtime v4 合同绑定的 bundle、C5 features、metadata、phase labels、HC95 reference、HC90 reference 六个 SHA256 在 postflight 中全部一致。

## D. Checkpoint

最终 latest adapted checkpoint：

- 路径：`results/iotj_b5_multiseed_20260724/seed43/raw/c12_to_c5__b5__s43/c12_to_c5__b5__s43__a001/raw/ecs/training/server_latest_adapted.pth`
- SHA256：`8483f70cbf50bddcfce61440de89880deaa410745bda2c19197393f433b37f42`

round-25 adapted checkpoint：

- 路径：`results/iotj_b5_multiseed_20260724/seed43/raw/c12_to_c5__b5__s43/c12_to_c5__b5__s43__a001/raw/ecs/training/server_round_025_adapted.pth`
- SHA256：`4a2174e85f069fa04a02bbbf8e0467dc42f8d25d015b58eadc57fe4a98784ab6`

两者都是 seed43 round 25 adapted 模型，但因 checkpoint 容器字段/序列化对象不同，文件 SHA 不要求相同。Git 只记录路径与哈希，不提交 checkpoint 本体。

## E. 时间

- attempt start：2026-07-24 12:30:35.607976 +08:00
- preflight passed：2026-07-24 12:31:01.116727 +08:00
- canonical end：2026-07-24 14:13:06.643985 +08:00
- attempt 总墙钟：6151.036 s，约 1 小时 42 分 31 秒
- preflight 后墙钟：6125.527 s，约 1 小时 42 分 6 秒

25 轮 observer timing，均值 ± sample SD：

| 项目 | 秒 |
|---|---:|
| 每轮 fit wall | 239.146 ± 1.751 |
| Pi C1 本地 fit | 41.575 ± 1.674 |
| ECS C2 本地 fit | 73.445 ± 2.688 |
| server DA | 164.673 ± 1.249 |
| server non-DA aggregation | 0.0704 ± 0.0047 |

## F. 资源与告警

- Pi C1 peak RSS：543,571,968 bytes（约 518.4 MiB）。
- Pi 最高温度：61.7°C。
- Pi throttled nonzero samples：0；没有发生降频。
- Pi resource sample：5926，sample errors：0。
- ECS C2 peak RSS：531,951,616 bytes（约 507.3 MiB）；resource sample 5952，sample errors 0。
- server/DA 正式 sampler 未纳入当前 confirmation contract；运行中只读 spot check 约 1.19 GB RSS，仅作 informational 记录，不冒充正式 peak。
- 日志中有 11 个 warning 文本：Flower 旧 API deprecation 与 CPU `pin_memory` 提示；error=0、traceback=0、NaN/Inf=0。

## G. seed43 C5 classification

使用 round-25 adapted checkpoint、冻结 C5 test 1360 行、15-bin ECE：

| 指标 | 值 |
|---|---:|
| Accuracy | 0.9860294118 |
| Macro-F1 | 0.9860150907 |
| NLL | 0.1256369339 |
| ECE | 0.0133756879 |

Confusion matrix（true row × predicted column）：

```text
[[327,   3,  10,   0],
 [  1, 337,   2,   0],
 [  1,   1, 337,   1],
 [  0,   0,   0, 340]]
```

Per-class recall：C0 `0.9617647059`、C1 `0.9911764706`、C2 `0.9911764706`、C3 `1.0`。prediction stream 有 1360 行和 1360 个唯一 row key，没有缺失预测，可供后续 RG0/RG1/RG2 使用；本次没有启动目标回归。

seed43 指标无论高低均保留为正式 multi-seed 样本，不据此重选 B2、修改 B5、删除或重跑 seed43。

## H. 协议异常

没有 blocking、major 或 numerical anomaly。唯一需要持续保留的审计说明是 ECS C2 phase=`-1` 的既有 seed42 拓扑语义，以及 server RSS 没有正式 sampler peak；二者均已显式记录，不影响本次 canonical 判定。

## I. 后续建议

建议在获得新授权后，按 seed44 → seed45 → seed46 串行推进，并对每个 seed 重复相同 preflight、canonical validator、postflight 和最小分类评估。seed43 证明当前三机执行链路可用，但单个新增 seed 不能建立五种子稳定性或回归 gate 结论。

## J. seed44–46 剩余时间

按 seed43 1.71 小时/seed 的实测 attempt wall 估算，三个训练约 5.13 小时。加上三次 preflight、回收、postflight 和分类评估，建议预计 5.5–6.0 小时，保守预留 7.5 小时。不得并行启动。
