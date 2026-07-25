# IoT-J Runtime v5 独立 QC 协议（amendment v2）

## 边界

本协议只为冻结的 B5 seed42、real-topology Federated H1 和 C5 105D per-gas
Ridge 建立独立 QC。唯一允许的证据表述是：Runtime v5 使用由 calibration
导出的分类置信度、表示距离和 source-to-target 回归一致性信号进行选择性输出。
不修改 runtime v4，不复用 v4 风险数值、分量分布或阈值，也不引入 H2、H3、
H2.3、all-prior、test truth 或 residual。

## Protocol amendment v2

初始协议假设每个 filename 在 320 行 calibration 中恰有 4 行。正式 test 打开前
的只读审计否定了该假设：calibration 有 80 个 filename，每组实际为 1–7 行；
calibration 与 test 合并后每个 filename 有 21 行。修正规则为：同一 filename
当前落入 calibration 的全部行必须整体进入同一个 fold。

修正发生时：test 尚未打开，`qc_selection_lock.json` 尚不存在，runtime v4/HC
六个冻结 SHA 均未变化。除组大小假设外，其余预注册 QC 规则不变。

## 五折 OOF

- 行数 320，filename 数 80，fold 数 5，固定 seed `20260725`。
- 使用 deterministic largest-group-first greedy assignment。
- 按 total rows、gas rows、gas-concentration rows、group count、fold id 的顺序
  进行确定性平衡；禁止把同一 filename 拆到不同 fold。
- 每个 held-out fold 的 target Ridge、B5 prototype/support reference、ECDF 和
  per-predicted-gas MAD scale 只能使用其他四 folds。

filename grouping 只适用于 calibration OOF。历史 calibration/test split 本身仍为
window-level，因此不得声称 calibration 与 test 在 original-file level 完全独立。

## 候选与锁

QC1、QC2、QC3 继续按已批准的“组件内均值，再风险组间均值”计算。选择仅使用
OOF；复杂候选只有在 HC95 或 HC90 accepted RMSE 至少改善 0.25 ppm、两个工作点
均不退化且 yield 最大下降不超过 0.01 时才能替代更简单候选。QC3 未实质优于
QC2 时固定选择 QC2。

选定后使用全部 320 calibration 行重建 reference、ECDF、MAD scale 和工作点阈值，
先持久化并哈希锁定 `qc_selection_lock.json`。只有锁验证通过后，独立 test 阶段才
能一次性打开 1360 行，且不得重选 candidate、component、scale、ECDF、threshold
或 decision rule。

HC95 为 accept `q95`、reject `>q98.75`；HC90 为 accept `q90`、reject `>q97.5`；
中间为 review。只有 accept 行产生 `auto_output_ppm`。
