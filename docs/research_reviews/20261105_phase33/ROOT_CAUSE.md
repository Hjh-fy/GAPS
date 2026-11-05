# 原因、验证与范围纠正

## 已确认的代码阻塞

原runtime.py:87的ReceiptMixin.configure_fit把Flower公共参数重命名成rnd/params/manager。已安装Flower1.23 server.Server.fit_round在实际原生调度中以server_round、parameters、client_manager关键字调用。Python在进入函数体前抛TypeError；没有配置完成FitIns，没有调用fit_clients，没有本地优化。server.log保存完整栈；CALLBACK_CONTRACT_AUDIT保存SDK文件SHA、实际函数源文本、冻结方法signature及Signature.bind复现，无训练。

aggregate_fit(self,rnd,results,failures)也不兼容公共server_round关键字，但本机Flower1.23实际对aggregate_fit采用位置参数。因此这是应一并校准的公共接口问题，**不是本轮第二个实际触发的服务器错误**，也不能断言修好configure后它必然在本机失败。

最小未来改动：在新版本入口使用标准configure_fit(self,server_round,parameters,client_manager)，内部别名rnd/params/manager继续调用原body；aggregate_fit参数名按公共接口恢复。算法、数据/seed/25轮/batch16/lr/初始权重不变。必须新代码版本与注册SHA，不覆盖Phase32，也不能在观察器中偷偷适配而继续声称原代码通过。

## 独立传输限制

两个client原日志均出现gRPC ConnectEx10013。无模型、无Flower、无训练的一字节localhost测试在同默认沙箱同样10013，而经批准的沙箱外测试成功，隔离出执行环境连接权限限制。TCP bind可成功不足以证明TCP connect或Flower握手可用。未将此写成模型/算法问题；不修改防火墙、不换端口猜测，也未重跑SOURCE。

## Phase3.2遗漏

preflight.py/integration.py直接位置调用configure_fit与aggregate_fit，成功验证了计数/EMA/回传等局部机制，却未走Server.fit_round的原生关键字调度；--help只验证parser/import。真实启动暴露这一差异。后续必须无真实数据地调用原生server调度，覆盖keyword签名、客户端连接/握手、初始模型广播和关闭流程后，再申请训练。以前“接口预检PASS”需限定为当时测试范围，不能外推完整Flower可运行。
