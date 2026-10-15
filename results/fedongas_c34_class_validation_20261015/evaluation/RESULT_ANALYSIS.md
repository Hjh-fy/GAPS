# C3/C4最小跨设备分类验证

CLASS为主候选；TO/FT/CLASS均seed42、固定2500目标更新，CLASS另有25轮LE1源联邦训练。C5主标定seed42只复用。
每目标原canonical主标定split；C3 calibration678/test2677，C4/C5 calibration320/test1360。单seed无SD/CI/显著性。

|目标|方法|Accuracy %|Macro-F1 %|来源|
|---|---|---:|---:|---|
|C3|TO|99.92529|99.92529|new_fixed_endpoint|
|C3|FT|99.70116|99.70110|new_fixed_endpoint|
|C3|CLASS|99.96264|99.96269|new_fixed_endpoint|
|C4|TO|99.85294|99.85283|new_fixed_endpoint|
|C4|FT|99.77941|99.77941|new_fixed_endpoint|
|C4|CLASS|100.00000|100.00000|new_fixed_endpoint|
|C5|TO|99.63235|99.63235|reported_reused_C5_no_retraining|
|C5|FT|99.11765|99.12009|reported_reused_C5_no_retraining|
|C5|CLASS|99.41176|99.41218|reported_reused_C5_no_retraining|

## 方法差值（百分点）

- C3 FT-TO: ΔAcc=-0.22413, ΔMacro-F1=-0.22419
- C3 CLASS-TO: ΔAcc=+0.03736, ΔMacro-F1=+0.03740
- C3 CLASS-FT: ΔAcc=+0.26149, ΔMacro-F1=+0.26158
- C4 FT-TO: ΔAcc=-0.07353, ΔMacro-F1=-0.07342
- C4 CLASS-TO: ΔAcc=+0.14706, ΔMacro-F1=+0.14717
- C4 CLASS-FT: ΔAcc=+0.22059, ΔMacro-F1=+0.22059
- C5 FT-TO: ΔAcc=-0.51471, ΔMacro-F1=-0.51226
- C5 CLASS-TO: ΔAcc=-0.22059, ΔMacro-F1=-0.22017
- C5 CLASS-FT: ΔAcc=+0.29412, ΔMacro-F1=+0.29209

所有正/零/负收益保留；不会因test结果改变CLASS主候选、增加FULL或改训练预算。
匹配的CLASS75个逐轮点另存class_round_curves；旧A4曲线保持历史归属。回归主指标为独立的Oracle_ALL，不能将分类差异归因于共享Ridge映射改变。
