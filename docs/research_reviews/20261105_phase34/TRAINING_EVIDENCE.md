# 单次真实SOURCE训练与成本

三条已注册原生server/client argv保持不变。新授权在Phase3.3R目录新签发，purpose=SOURCE；未复用Phase3.3旧授权。科学spec/runtime/contract/environment/source身份/初始模型SHA先核验。使用注册clean_environment移除旧PYTHONPATH/FEDONGAS变量；随后只加入Phase3.4新哈希的观察bootstrap路径，逐进程assert contract确为Phase3.3R。这个只读观察扩展记录计数/索引/快照/成本，未改变loss/参数/seed/采样顺序，不适配或修补训练逻辑。

普通用户宿主环境、无管理员token，127.0.0.1:8080预先bind/connect通过，未修改防火墙或端口。目标数值白名单之外npy/npz会被Python审计拒绝；正式三个进程均没有拒绝事件/目标输入。guard自测仅在open前拒绝目标文件，没有读取其内容。训练用C1/C2规范源包，phase0仅中性类别键；Source纯CE、无DA/辅助loss/选择性权重、4类TCN50×8/64D、seed42、25轮各1epoch、batch16、Adam5e-4每轮重建。

逐client/round真实optimizer.step：C1每轮53，总1325；C2每轮105，总2625，共3950。每轮随机采样索引均恰好覆盖0..839或0..1679一次，63k真实训练呈现；classcounts及日期身份见SOURCE_ROUND_LEDGER，不把窗口重复当新增独立暴露。梯度每步检查有限，round后全state有限；每轮权重1/3与2/3以独立float64加权均值核对，最大差2.38418579102e-07为float32舍入容差内。

SOURCE_ROUND_LEDGER/ROUND_COSTS列50个client-round与25个server-round SHA/时间/载荷。模型完整state144692B、参数22765；上下行模型tensor共14,469,200B，FitIns下行7,765,250B、FitRes上行7,803,750B，合计15,569,000B。这是实际序列化ByteSize，不包含HTTP2/gRPC framing/控制握手/重试，不冒充wire抓包。

总墙钟72.990秒含启动/导入/通信/检查与状态快照。三进程RSS相加的0.2s采样峰值1,292,713,984B；共享页可能被重复计入，非系统唯一物理内存。每进程峰值、client本地fit时间均保存，不能把本地宿主结果外推Pi。本次Flower在localhost真实执行，source逻辑客户端位于同机并可加载两个合法源包，不声称物理数据隔离或隐私保护。

启动前已锁定round25，无Test/Development选轮。Flower初始evaluate返回None，fraction_evaluate0，25轮客户端评估均跳过，不是隐含目标验证。没有目标精度/端到端路由/部署。
