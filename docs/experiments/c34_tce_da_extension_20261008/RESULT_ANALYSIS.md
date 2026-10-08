# C3/C4 +TCE DA 补充结果与三目标汇总

日期：2026-10-08。Study：`C34-TCE-DA-EXTENSION-20261008`。
结果目录：`results/c34_tce_da_extension_20261008/`。

## 执行和数据口径

新增六组：C3/C4 × CORAL+TCE、MMD+TCE、DANN+TCE。
每组从同一个 P0A pure-FedAvg round25、LE1、seed42 source checkpoint 独立加载，
然后执行100次参数更新。无需重训25轮Flower，也未加载已经适配C5的模型作为起点。
Adam lr=5e-4，batch=32，alignment weight=0.5，target CE weight=1.0，seed42。
损失：source CE + 0.5 global alignment + 1.0 target CE。
DANN的global alignment为domain BCE/GRL。

source batch沿用C1/C2 merged calibration loader；target adaptation只接收x/class。
C3/C4的train split没有使用。六个step100端点全部保存后才打开target test。
沿用legacy数据既有划分：C3 calibration/test=320/680；C4=160/320；
C5=320/1360，C5三项是旧结果直接复用（reported_read_only）。
不能将C3/C4旧划分描述为canonical-v1的约20/80三目标协议。

## 分类结果

C3/C4为新评价（recomputed）；C5为既有报告值（reported_read_only）。
Accuracy/Macro-F1为百分数；NLL为自然对数损失；ECE沿用15个等宽confidence bins。

| Target | Method | N_test | Accuracy (%) | Macro-F1 (%) | NLL | ECE |
|---|---|---:|---:|---:|---:|---:|
| C3 | CORAL+TCE | 680 | 99.5588 | 99.5584 | 0.0228 | 0.0045 |
| C3 | MMD+TCE | 680 | 99.4118 | 99.4118 | 0.0220 | 0.0028 |
| C3 | DANN+TCE | 680 | 99.2647 | 99.2625 | 0.0241 | 0.0060 |
| C4 | CORAL+TCE | 320 | 99.0625 | 99.0624 | 0.0422 | 0.0050 |
| C4 | MMD+TCE | 320 | 99.3750 | 99.3769 | 0.0286 | 0.0039 |
| C4 | DANN+TCE | 320 | 99.3750 | 99.3769 | 0.0294 | 0.0030 |
| C5 | CORAL+TCE | 1360 | 96.8382 | 96.8400 | 0.1437 | 0.0181 |
| C5 | MMD+TCE | 1360 | 96.4706 | 96.4800 | 0.1744 | 0.0192 |
| C5 | DANN+TCE | 1360 | 96.7647 | 96.7730 | 0.1556 | 0.0171 |

## 相对对应x-only方法的增量

Δ为+TCE减去相同legacy测试population的x-only E2结果，单位percentage points。
这些增量从CSV未舍入数值计算（recomputed）；两组使用相同source起点和100-step协议。

| Target | Method | ΔAccuracy (pp) | ΔMacro-F1 (pp) |
|---|---|---:|---:|
| C3 | CORAL+TCE | +6.471 | +6.523 |
| C3 | MMD+TCE | +1.618 | +1.625 |
| C3 | DANN+TCE | +4.118 | +4.152 |
| C4 | CORAL+TCE | +13.125 | +13.923 |
| C4 | MMD+TCE | +3.438 | +3.469 |
| C4 | DANN+TCE | +10.313 | +10.748 |
| C5 | CORAL+TCE | +58.162 | +69.053 |
| C5 | MMD+TCE | +39.779 | +41.364 |
| C5 | DANN+TCE | +44.926 | +55.114 |

这支持“目标类别监督提高了点估计分类性能”的描述；未做显著性检验。
不能据此单独识别alignment比target-CE-only提供了多少净增量，
因为本次没有新增C3/C4 target-CE-only control。

## 与历史GAPS行的上下文比较

相同legacy test population上的旧GAPS Accuracy为C3 98.9706%、C4 99.0625%、C5 98.4559%。
本次+TCE三组在C3均更高；C4 CORAL+TCE与GAPS的Accuracy相同，MMD/DANN+TCE高0.3125 pp；
C5三组仍低于GAPS，差值为−1.6176、−1.9853、−1.6912 pp。
这是post-hoc baseline与旧interleaved GAPS的描述性比较，训练时机/预算存在差异。
因此论文不能宣称GAPS在三目标上统一优于所有有标签DA方法。
可报告不同target的表现，并保留GAPS在C5困难目标上的历史优势。
该表不替代canonical-v1主表，也不拼入其不同test population的数据。

## 源域保留性能

源域为C1/C2合并test，N=1360。新六组source Macro-F1分别为：

| Target adaptation | CORAL+TCE | MMD+TCE | DANN+TCE |
|---|---:|---:|---:|
| C3 | 99.6322% | 99.1915% | 99.6322% |
| C4 | 99.5588% | 99.5588% | 99.5588% |

逐方法source-target F1 gap、source Accuracy和adaptation seconds保存在比较CSV中。

## 验证与文件

- 原适配实现测试：15 passed。
- compileall：PASS。
- 新六组均100 steps，总600 steps；diagnostic loss均finite。
- target CE active、global alignment active；class-conditional分支未运行。
- 3000条新逐窗口预测；9行三目标比较。
- 33个执行资产SHA256索引：0 mismatch；所有记录输入hash未变化。
- 从逐窗口概率重算Accuracy、Macro-F1、NLL、ECE，与6行正式CSV最大差为0.0。
- target test不用于调参、选点或early stopping。

```text
scripts/run_iotj_c34_tce_da_20261008.py
docs/experiments/c34_tce_da_extension_20261008/PROTOCOL.md
results/c34_tce_da_extension_20261008/protocol_manifest.json
results/c34_tce_da_extension_20261008/EXPERIMENT_MATRIX.csv
results/c34_tce_da_extension_20261008/c34_tce_comparison.csv
results/c34_tce_da_extension_20261008/c345_tce_comparison.csv
results/c34_tce_da_extension_20261008/tce_vs_xonly_deltas.csv
results/c34_tce_da_extension_20261008/per_window_predictions.csv
results/c34_tce_da_extension_20261008/EXPERIMENT_AUDIT.json
results/c34_tce_da_extension_20261008/sha256_index.json
```

checkpoint和100-step诊断位于`C3|C4/<method_key>/`，文件名分别为
`adapted_step_100.pth`和`adaptation_diagnostics.csv`。
GitHub提交包含代码、协议、比较表、概率预测、诊断和审计；checkpoint保留本地，hash供追溯。
全部六组完成后停止，不扩展条件对齐、seed、budget或canonical重跑。
