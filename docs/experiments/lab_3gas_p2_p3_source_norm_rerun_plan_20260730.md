# P2→P3 Source-only Normalization Rerun Plan

## Research brief and scope

- Brief source: 修正旧数据集中目标 P3 train features 参与 Z-score 拟合的问题。
- Target venue/audience: 实验室三气体分类内部验证；当前仍为 nominal-boundary screening。
- Resource budget: 先执行 fold 1、seed 42、25 Flower rounds、3 local epochs；
  postflight 审计通过后再执行 folds 2–5。

## Hypotheses

| ID | Falsifiable hypothesis | Baseline | Intervention | Primary metric | Expected Evidence | Acceptance criterion |
|---|---|---|---|---|---|---|
| `H-LAB-NORM-01` | 使用 P2 train 独立拟合 Z-score 后，P2→P3 可在不读取 P1/P3 train features 的条件下完成训练、source-calibration 选轮次和 P3 test | 旧 all-client normalization fold-1 screening（不作为公平性能 baseline） | normalization clients=`[2]` | P3 exposure Macro-F1 | 无目标 train 特征参与的可审计 fold result | 数据审计、25-round postflight 和 test-boundary 全部通过；不以性能高低决定重跑 |

## Fixed protocol

- Source clients: P2 / logical C2 / cloud B.
- Target clients: P3 / cloud A server-side calibration and held-out test.
- Split protocol: exposure-group-aware five-fold；每折 train/calibration/test groups
  为 3/1/1；fold 1 测试最低保留浓度组。
- Model/checkpoint policy: `strong_cls`；source calibration exposure Macro-F1
  选 round，window Macro-F1 次级，最早 round 再次级。
- Seeds: 42。
- DA / calibration / QC controls: `fixed_da_strong`，100 steps/round；
  P3 calibration-assisted；无 QC；P3 test 仅在 round 锁定后打开。
- Held constants: 25 rounds、LE3、batch 32、6 channels、3 classes、1 phase、
  nominal boundaries、窗口和滤波参数不变。
- Sole intended change: Z-score mean/std 仅由 P2 train exposures 拟合。

## Risks, unknowns, conflicts, and stopping rules

- Source-only normalization 是公平性修复，不保证提高准确率。
- 当前名义通气边界仍阻止最终性能 claim。
- Fold 1 只有 6 个独立 P3 test exposures，窗口不是独立重复。
- 任一数据 manifest/hash、25-round、client identity、NaN/Inf、checkpoint、
  postflight 或 target-test boundary 失败即 fail closed。
- 性能较低本身不触发重跑或调参；必须保持五折配置一致。
