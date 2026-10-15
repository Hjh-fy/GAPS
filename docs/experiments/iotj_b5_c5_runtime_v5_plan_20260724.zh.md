# IoT-J B5/C5 Federated-H1 Runtime v5 Candidate 正式计划

## 冻结决策

- final classifier：B5；
- runtime classifier：canonical seed42，不按五种子结果挑选 seed；
- classifier checkpoint SHA256：
  `9b268f659c60a1d3b9bb789d89e82b5cedae56b92173daca616caef247371e5c`；
- final regression：`SELECT_B5_FEDERATED_H1`；
- 决策证据 commit：
  `31ce3fbd7005f0088a96d8f12946599e40cf4b71`；
- RG1 的选择依据是预注册 1% 非劣 gate 和依赖简化，不代表其绝对精度
  优于 RG2。

## v5 相对 v4 的唯一算法变化

1. 删除 H2 per-gas MLP 与 H3 shared MLP source dependencies；
2. H1 使用真实三机传输并由 sufficient statistics 重建的四气体 global
   Ridge；
3. C5 target Ridge 输入从 104D+H1/H2/H3 改为 104D+H1；
4. 只有 QC dependency audit 判定风险语义兼容时，才允许用 v5
   calibration 重新校准 risk 与 HC95/HC90。

保持不变：B5、preprocessing、C1/C2→C5 角色、320/1360 split、row
map、predicted-class routing、基础输入/输出字段及全部 v4 冻结资产。

## 阶段和 fail-closed gate

1. V0：冻结 lineage、输入 SHA 与协议；
2. V1：Pi C1、ECS C2 分别只读自己的 source train/calibration；server
   仅接收 feature moments、normal equations、validation statistics；
3. V1 gate：4/4 alpha 相同、H1 prediction max diff ≤1e-6 ppm、
   Ridge+H1 S_CC/S_ALL diff 均 ≤0.01 ppm；
4. V2：只用 seed42 calibration 240/80 锁定 105D target Ridge，持久化
   lock 后才打开 test；
5. V3：新建独立 runtime 与 bundle，禁止依赖 H2/H3、C3/C4、H8+C4、
   R3aK16、P4 或 test label；
6. V4：320/1360 class、feature、H1、ppm、row key parity；
7. V5：审计 v4 risk/QC 是否依赖 v4-specific regression 语义；
8. V6：仅 PATH A 才重新校准 v5 HC95/HC90；
9. V7：应用 promotion gate；V8 停止。

任何关键 gate 失败均保留证据并停止，不临时更换 checkpoint、模型、
threshold 或 risk 定义。

## 隐私表述边界

允许表述：source raw samples remain local and only aggregated sufficient
statistics are used to reconstruct the global source Ridge reference.

禁止声称 sufficient statistics 不泄露信息，或已经实现 secure
aggregation、differential privacy、cryptographic privacy。

