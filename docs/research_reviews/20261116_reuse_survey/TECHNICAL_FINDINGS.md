# 技术与场景借鉴

本轮公开检索覆盖GitHub、官方文档、作者Reddit发帖、Bosch/Flower社区与PyTorch问题区。核验11个仓库的固定commit及README，进一步只读3个数值/选样模块、4份许可证和1份依赖表。没有安装依赖、克隆后执行、导入新库、训练、推理、重评分或连接Pi。来源类型与读取深度见SOURCE_REGISTER.csv，不能将README宣称当作独立复现实测。

## 1　最接近的外部场景是分析仪器的校准迁移

光谱仪同样面对源仪器模型可用、目标仪器只有少量标准样本的问题。[chemotools官方流程](https://chemotools.org/explore/calibration_transfer.html)明确要求配对标准样本，再把目标输入映射到源空间；[pynir](https://github.com/JinZhangLab/pynir/blob/0455b61bf9e4fbd4f20c9b8146228ab04c0ba692/README.md)同时提供输入空间迁移与预测层校正示例。这比泛泛搜索更大网络更贴近现有浓度问题。

对本项目最值得先审计的是低自由度预测校准：源模型保持冻结，检查少量标定能否校正预测偏置/尺度，同时保留其浓度排序。它应作为Source-only与F-SVR之间的强参照，不能提前称新方法。pynir的BS原实现拟合主/从仪器预测之间的映射；将源预测直接拟合目标真实ppm属于另一种监督校准，必须明确区分，不能只更名BS。

不能直接应用PDS。PDS的局部窗口是相邻波长；71D包含通道统计和全局描述，当前列顺序不具备同样的物理邻接含义。完整71x71无约束DS在每气体4/8训练窗、2/4独立暴露下也缺乏足够约束。相同标称浓度的两次采集不自动成为严格配对标准。以上是针对本项目的数据/维度判断，不是对光谱方法整体有效性的否定。

## 2　主动学习可直接复用选样接口，物理暴露约束需要我们提供

[scikit-activeml固定源码](https://github.com/scikit-activeml/scikit-activeml/blob/b6937613c4f8808c9588dacceb382b2896a8ff34/skactiveml/pool/_greedy_sampling.py#L19)中的GreedySamplingX按特征距离选择候选，query接收候选索引和批大小。它能作为现有71D上的代表性选窗基线，不要求把SVR换成可输出预测方差的GP。[Kennard–Stone](https://github.com/yu9824/kennard_stone/blob/5158f4f69349804c61806082c92dcdb62c56708f/README.md)适合提供简单空间覆盖参照。

我们仍需按原始暴露分组、限制每暴露窗口配额，并使用只由合法Calibration组成的候选池。库里的一个样本是一行向量，不会自动知道21个窗来自同次暴露。它们不能自动解决分类和浓度任务共享标定的问题。若仅在已知标签的Calibration暴露内选窗，应叫标定窗口选择；不能夸大为完成了真实主动气体采集。

旧A2/A4身份继续冻结。若未来要从其他浓度暴露挑选标定，应另立协议，重建公平基线并披露那些暴露的历史开发使用。不能直接把原Development改作Calibration后仍宣称沿用原比较。

## 3　现成源先验迁移可以补强基线，但先核对参数合同

[ADAPT RegularTransferLR](https://github.com/adapt-python/adapt/blob/fd67cb7c13d0dd743f272710489a68bde2cb5e75/adapt/parameter_based/_regular.py#L24)提供向源参数正则化的线性回归。适合审查SourceRidge71能否作参数先验。它与已实现源函数残差/AdaptiveSourcePrior存在概念重叠，先完成目标函数和坐标差分；若等价，直接复用既有数值核，不另计新候选。

源码使用SciPy LSQR的damp参数。[SciPy明确目标含damp平方](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.lsqr.html)，不能直接把文档lambda与现有Ridge alpha按名称对应。源码还把截距并入正则中心。源/目标scaler若不同，系数必须先变到同一坐标，不能直接相减。ADAPT依赖表含TensorFlow、cvxopt、scikeras，本轮不把整库装入已冻结Torch/sklearn环境。

[SKADA](https://github.com/scikit-adaptation/skada/blob/31f2ff8b6492bba0d8a28cd8296952b33123563b/README.md)可参考域身份、管线和验证组织。许多数据级适配方法需要源/目标特征共同可用，需逐项核对联邦假设。教程中的无监督目标输入不能解释成本项目允许读取Day22 Test做变换。

## 4　动态维护应分别记录输入变化、性能证据和版本晋升

[River ADWIN](https://riverml.xyz/latest/api/drift/ADWIN/)可作为低成本标量流监控组件。监控71D统计量得到的是输入变化线索；监控真实误差需要合法标签。不能把特征报警直接写成分类/浓度性能恶化，也不能把高度重叠窗口作为独立证据直接调用其统计保证。

[MLflow Registry](https://mlflow.org/docs/latest/registry/)可借鉴版本、在用/候选别名、来源和验收标签。它不自动保证Pi上六分类器与四回归头的一致切换，也不替我们决定候选是否可以晋升。建议云端登记元数据，Pi继续使用现有冻结bundle和轻量索引；完整包先校验/加载、候选失败保留旧快照。这是未来工程方案，不是本轮已实现功能。

另审查了[NannyML DLE的官方限制](https://nannyml.readthedocs.io/en/stable/how_it_works/performance_estimation.html)：其无标签回归性能估计依赖无concept drift和参考覆盖。当前恰在研究跨日浓度条件关系变化，不能把DLE作为现成真实风险裁判，更不应立即训练第二个风险网络。

## 5　行业电子鼻流程值得借鉴，硬件权重不能混用

[Bosch官方手册](https://www.bosch-sensortec.com/media/boschsensortec/downloads/application_notes_1/bst-bme688-an001.pdf)把采集、样本管理、训练、部署组织为完整流程，并在PDF pp44–45提示环境通道相关性及自热造成的误导。可借鉴设备/采集/样本身份、环境条件记录和独立测试流程。BME688的加热扫描与当前八通道输入不同，不能复用其模型权重、输出ppm定义或把其工具效果当作我们的实验证据。

[UAMSense](https://github.com/GNB-UAM/UAMSense/blob/00b945a332ce925c7119b9a6657d9e7ee6eed475/README.md)提供实验性多模态电子鼻硬件和UART-BME680设计。[八通道采集项目](https://github.com/RonishNadar/electronic-nose/blob/847227987f641100344d1118b7557f709022bc09/README.md)描述了WARMUP/BASELINE/EXPOSE/RECOVER、串口、环境字段和SNR报告。前者代码GPLv3、硬件CC-BY-SA4.0；后者未取得明确许可证，只作设计线索，不直接复制。现有STM32/UART/GUI已完成的能力优先复用。

[OpenSmell](https://github.com/YAAASSS/OpenSmell/blob/275d6392541ad3da204d8c3d1c22975f15454116/README.md)的schema、来源与设备能力语义可以作接口参考，但其README明确pre-alpha，当前物理演示范围为ESP32/LED控制路径，不代表已实现气体浓度计量。当前没有必要为本项目重写整套通信协议。
