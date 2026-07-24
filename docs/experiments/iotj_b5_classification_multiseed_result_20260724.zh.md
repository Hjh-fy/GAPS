# IoT-J B5 classification five-seed 正式结果

五个 seed（42–46）均为 canonical；seed42 复用正式 checkpoint，seed43–46 在相同三机拓扑和冻结协议下顺序训练。所有新 seed 均通过 25/25 rounds、C1/C2 每轮参与、2500 DA steps、严格 checkpoint 加载和 1360 行唯一 predicted route 的 postflight。

| Seed | Accuracy | Macro-F1 | NLL | ECE | Errors |
|---:|---:|---:|---:|---:|---:|
| 42 | 0.980147 | 0.980220 | 0.150294 | 0.019368 | 27 |
| 43 | 0.986029 | 0.986015 | 0.125637 | 0.013376 | 19 |
| 44 | 0.994853 | 0.994852 | 0.054088 | 0.005050 | 7 |
| 45 | 0.991912 | 0.991924 | 0.082802 | 0.007931 | 11 |
| 46 | 0.992647 | 0.992661 | 0.088364 | 0.007308 | 10 |

## 五种子描述统计

- Accuracy：0.989118 ± 0.005983，range [0.980147, 0.994853]
- Macro-F1：0.989134 ± 0.005960，range [0.980220, 0.994852]
- NLL：0.100237 ± 0.037833，range [0.054088, 0.150294]
- ECE：0.010607 ± 0.005774，range [0.005050, 0.019368]
- Error count：14.80 ± 8.14，range [7, 27]

## 结论与边界

B5 在 seeds42–46 上保持稳定：最差 Accuracy 为 0.980147，最好为 0.994853；未出现训练、拓扑、checkpoint 或 row-map 异常。该结论只支持 B5 classification five-seed stability，不支持回归方法、QC、runtime v5、Pi 性能或 low-calibration 结论。

五种子分类 route 和 provenance 已齐备，因此具备在获得下一阶段授权后启动 B5 regression multi-seed 的输入条件；本次没有启动该阶段。
