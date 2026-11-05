# 合法源分类起点S*审计

## 结论和搜索边界

没有证据充分的合法S*，不是断言整台机器不存在任何未知权重。v5 SOURCE_INVENTORY中2203条.pth/.pt/.ckpt路径（可能含非分类资产）均进入清单；7个严格分类候选逐文件SHA及weights_only=True安全加载已完成。活动仓库额外路径清单见ACTIVE_CHECKPOINT_PATHS及ACTIVE_SEARCH_LIMITS：原仓库遍历exit2为部分清单，另外两个worktree路径遍历完整；未把数千条路径冒充逐模型祖先审查。registered worktree清单同样不是其所有资产的完整取证。

7个候选属于旧source R1–R3/目标微调或服务器适配协议；其祖先训练身份未闭合为C1Day4、C2Day1/7。旧source-only A0也不例外。SAFE_CHECKPOINT_METADATA逐个保留checkpoint SHA、80个state键/形状、分类头、保存round、model_config、prototype键、optimizer和ancestry字段是否存在；缺项保持unknown，不能从路径或最终权重反推“没用过Test”。容器中没有完整原始暴露训练名单和祖先SHA链，优化器状态通常未保存，prototype/encoder版本也不能完整追溯。**不产生SSTAR_LOCKED。**

## 可复用结构、输入与预处理

原task.create_model→utils.create_model_by_config(with_reg_head=False)→FedGasBaseModel；TCN+attention池化、64D特征、4类。原代码合成(2,50,8)前向输出logits(2,4)、cls/reg特征均(2,64)，无梯度；旧A0结构strict load成功仅证明schema兼容。默认配置SEQ_LEN100仍被generic DA验证器当成输入长度，因此这是元数据/入口冲突，不能说TCN数学上不能处理50。

标签0/1/2/3=Ethanol/CO/Ethylene/Methane。复用20–50s基线g0、原process_raw、5Hz、50×8有效响应窗口、固定valid mask，loader normalize=False，不添加未验证归一化。训练期FEATURE_NOISE_STD=0.01、既有网络dropout保留为共同结构条件；ce_only关闭alignment/replay/regression/DP/MixStyle/传感器增强。遗留USE_CONTRASTIVE_CONSISTENCY=True等标志不等于实际进入客户端loss，须核对调用点及系数；不通过改profile引入其他损失。

## 唯一一次源预训练固定方案（PLANNED，未执行）

ID P31-SSTAR-DAY7-S42-PLAN；random seed42全新随机初始化，无外部祖先，encoder/分类头同一state版本；optimizer新建，语义原型空。仅C1Day4 40暴露840窗、C2Day1/7 80暴露1680窗，120暴露2520窗；不读任何C3/C4/C5或源R4。纯Source FedAvg、25轮、源每轮1epoch、batch16、Adam5e-4、无辅助损失/DA/选择性加权，普通样本量权重1/3、2/3；默认原Client每轮重建Adam必须明记，不携带旧优化器状态。固定round25，不按验证/测试选轮；SourceCE的calibration_文件是同一源训练池别名，不是独立验证集。

源本地每轮53+105=158步，总3950步、63000次源窗呈现。需要记录随机初始state/encoder/head/schema/参数序列SHA，代码与配置SHA，全部源暴露/窗口身份和输入SHA，各轮上/下行SHA及最终S*的祖先链。optimizer从S*分叉时按统一规则重置；如不保存optimizer，必须明确模型checkpoint不支持精确中途训练续跑，而不虚称全状态恢复。

执行命令的真实解析模板见EFFECTIVE_CONFIG_AND_COMMANDS.md。当前命令**不能直接生成已批准的S***：generic SEQ元数据100且续训无initial-state开关；历史strict runner只接受旧ID/batch32，另有未闭合依赖。下一轮仅做薄入口/新spec，复用现有模型、训练循环、普通FedAvg、独立stats上传与哈希记录组件；不改数值算法。薄入口未审阅和复检前不执行本预训练，更不串联六组pilot。

本机CPU可用、CUDA不可用。预算：源特征单份4032000B；模型参数91060B、完整Flower state载荷144692B；两客户端25轮双向张量载荷约14.4692MB，不含protobuf/原型/重试。CPU墙钟暂无合法实测，采用25×(53t_C1batch+105t_C2batch+t_round通信/聚合)登记，不用Tiny测试时长估计TCN训练。一次固定预训练获准后记录首轮耗时外推，仅用于进度，不改实验端点。
