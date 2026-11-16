# 固定源码静态检查

只读3个源码模块，没有导入、安装或执行。下面是可复用边界检查，不是已复现数值结果。

|对象|固定文件|核实内容|进入项目之前需要的检查|
|---|---|---|---|
|GreedySamplingX|[源码](https://github.com/scikit-activeml/scikit-activeml/blob/b6937613c4f8808c9588dacceb382b2896a8ff34/skactiveml/pool/_greedy_sampling.py#L19)|类19行，query60行；候选索引/批大小/距离策略|暴露分组配额、候选池角色、scaler仅合法训练输入；旧固定锚窗参照不覆盖|
|RegularTransferLR|[源码](https://github.com/adapt-python/adapt/blob/fd67cb7c13d0dd743f272710489a68bde2cb5e75/adapt/parameter_based/_regular.py#L133)|133行取目标输入；141–151行将截距加入源参数和X；178行调用lsqr(damp=lambda_,x0=beta_src)|源码实际正则系数为damp平方；截距惩罚/损失sum-vs-mean/scaler坐标；与旧prior去重|
|pynir BS|[源码](https://github.com/JinZhangLab/pynir/blob/0455b61bf9e4fbd4f20c9b8146228ab04c0ba692/src/pynir/CalibrationTransfer.py#L173)|190–206行用从仪器预测拟合主仪器预测的线性回归|主从方向、单位、标定专用输入、是否真的存在配对预测；不能将其偷偷改成真实标签监督却仍称同算法|

另一个具体协议风险来自[pynir README校准示例](https://github.com/JinZhangLab/pynir/blob/0455b61bf9e4fbd4f20c9b8146228ab04c0ba692/README.md)。示例先从idxTest产生yhat1/yhat2，随后在BS.fit中使用这两组预测。因此该演示在我们禁止Test参与拟合的协议下不能直接复制，即使没有使用真实Test浓度标签也不允许。此处只说明示例与本项目协议不同，不推断该项目所有研究均发生泄漏。

ADAPT固定requirements.txt包含numpy、scipy、tensorflow、scikit-learn、cvxopt、scikeras。文档和源代码参数含义有需要核对的映射，不能仅因名称一致就移植旧alpha。参考[SciPy LSQR目标](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.lsqr.html)。本轮只读依赖表，无环境变更。

固定仓库许可证元数据与4份实际许可证在UPSTREAM_LOCK.json登记。没有开展完整传递依赖许可证审计，也没有把未声明许可证的仓库作为可直接复制资产。
