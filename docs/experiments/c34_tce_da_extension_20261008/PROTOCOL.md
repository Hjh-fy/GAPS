# C3/C4 有标签全局 DA 补充实验（2026-10-08）

用户确认后执行六组：C3/C4 × CORAL+TCE、MMD+TCE、DANN+TCE。
每组独立加载原 C5 研究的 P0A pure-FedAvg round25 source checkpoint。
SHA256：`4313c375a8fa2e929de9d65637a2196f6c0f0752c2dc78112020b8727351751c`。
Ordered state fingerprint：`c0379777c63c7a5b0910e5d2ab8a1c37869692db14fd940e0a86778c9f5a1769`。

输入沿用 legacy `client_data_c1234src_c5tgt_2080_timeaware_60_170_window_fullgrid`。
source 仅 C1/C2，source batch 沿用原 merged calibration loader；C3/C4 train split 不使用。
现有 C3 calibration/test=320/680；C4=160/320；C5=320/1360（直接报告旧结果）。
输入为既有100×8窗口；不建立新划分、不更换 canonical 数据。

固定优化器 Adam、lr=5e-4、batch=32、seed42、100 steps、gradient clip=5。
损失为 source CE + 0.5 global alignment + 1.0 target CE。
CORAL 为 covariance alignment；MMD 为 MMD²；DANN 为 domain BCE/GRL。
目标 adaptation API 只接受 x/class；不传入 phase、concentration、pseudo labels 或 test。
六个 step100 completion markers 全部完成后才加载 test 做正式评价。
不做 early stopping、超参数搜索、checkpoint selection 或新增 class-conditional 分支。

输出 Accuracy、Macro-F1、NLL（自然对数）、ECE（沿用既有15-bin实现）、source F1、
source-target gap、adaptation seconds、逐窗口预测、checkpoint、诊断及 SHA256 index。
将C3/C4新结果与C5已报告结果组成9行表；另报告相同测试population下相对x-only的增量。
单seed结果只作描述性分析，属于legacy补充机制实验。

运行：

```powershell
python scripts/run_iotj_c34_tce_da_20261008.py inspect
python scripts/run_iotj_c34_tce_da_20261008.py run
```

结果目录：`results/c34_tce_da_extension_20261008/`。
运行前将冻结配置、代码/数据 hashes 写入 protocol_manifest.json。
已有目录会拒绝重跑；失败保留证据，不自动搜索或覆盖完成端点。
