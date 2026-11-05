# 可复现证据及执行边界

本地入口依次为export_and_census.py、config_audit.py、checkpoint_audit.py、budget_checks.py、version_chain.py、final_probes.py、report.py、verify.py。依赖common.py及只读Phase2A/audit_common、Phase1/phase1_common、v5原函数和已锁身份。命令为`PYTHON_EXECUTABLE research/fedongas_phase31_20261105/<script>.py`，环境PYTHONDONTWRITEBYTECODE=1/PYTHONIOENCODING=utf-8；版本见interfaces/ENVIRONMENT。本轮没有安装包或网络训练。

这些入口使用exist_ok=False保留旧制品，**不应在本已封存目录反复运行**。重新验证只执行verify.py（只读校验及写独立verification记录）；若需重放合成测试，复制脚本到新的输出目录，保留同SHA的v5/历史依赖，不覆盖checks/chain。common.py AST隔离原函数，V5_*_PROVENANCE按原始源码SHA可核对未改数值实现。版本链仅数值训练用double，不包装成TCN真实训练。

本地保留真实源/Calibration tensor、完整2552窗身份、136raw读取日志、配置命令、2500步预定索引、7权重元数据、原生预算反例、6次适配链和接收SHA、局部失败记录及合成checkpoint。没有真实Test数据包、SSTAR_LOCKED或六组研究checkpoint。ARTIFACT_SHA256封存本轮全部有效与失败制品（不含manifest自身）；后续发布receipt放独立publication目录，不修改封存内容。

GitHub只发精简文档、配置、来源/身份/成本摘要、测试收据、脚本文本与源/发布SHA；不发npy、任何pth/权重、raw信号、71D输入、v5或ZIP。PUBLICATION_MANIFEST区分本地原文件SHA与路径脱敏/LF转换后的发布SHA；外部agent可审阅，不能仅靠精简包重训。测试/评价未进行，未来正式精度仍缺证据。
