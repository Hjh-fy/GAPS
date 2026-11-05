# 本地资产与恢复

正式源run：Phase3.3R/source_attempt2/{server,client_1,client_2}。server内有25个server_round_NNN.pth、25个phase32_receipt_NNN.json、最终SSTAR.pth/SSTAR_LOCKED.json、effective_config/history及本轮接受审计。Phase3.4/observer保留50个client trained snapshot、每轮batch/index/optimizer/cost与每进程访问记录；COMPLETE_SOURCE_RUN_MANIFEST和MODEL_AND_CHECKPOINT_INDEX逐文件SHA。

恢复S*时应连同整个server目录的25轮来源文件、注册初始checkpoint/lock、Source身份、固定Phase3.3R runtime/contract/ENV及批准收据保留，不能仅拷一个pth声称来源完整。私有输入仍在只读Phase32源包；v5与Phase1/2/31 helper保持原版本。不要重生成随机初始文件，也不要修改旧Phase33失败现场。

strict load请使用Phase3.3R runtime.load_checkpoint(model(),SSTAR路径,SSTAR_LOCKED字典,'sstar')，它会检查全25轮实际文件及源码/身份SHA。新环境先clean_environment并核验模块路径、Python包与模型schema；不得从旧PYTHONPATH缓存Phase32 contract。本轮恢复验证已将原最终轮与封装及重新保存模型在12合法源窗上比较一致。

原Adam按轮新建，未保存中途精确重跑的optimizer状态；本轮已完整结束，不做额外训练续跑。保存/加载权重一致不替代源身份审计，二者本轮都已通过。完整权重、source tensor和private恢复输出不上传公开GitHub，审阅包只含代码、配置、SHA与摘要。
