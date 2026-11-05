# 科学与执行合同

源C1Day4、C2Day1/7，120独立暴露2520窗；目标C5Day18A4，16独立暴露32窗。50×8、四类Ethanol/CO/Ethylene/Methane、seed42、batch16、Adam5e-4，统一每轮/块fresh optimizer，固定25端点。

B0 SourceFedAvg从Sstar续训25轮；B1只目标2500CE步；B2普通第三client，num_examples32/2552，100步/轮；B3/4/5均服务器100个优化步骤/轮，每步10批SourceCE+1批TargetCE（每轮1000源批/100目标批），目标2500步40000呈现。B3/4仅adapted反馈开关，B4/5仅原型选择性参数加权开关；stats/EMA.8/warmup3/minscale.3相同、辅助损失关闭。B3主评价也用adapted25，不用其返回aggregated25替代。

新增entry只绑定pilot授权、输出和必要数据角色，仍调用原PhaseClient/strategy/DA，B1复制原循环。普通server无数据；客户端各自数据；DAserver源2520+目标32，无旧500截断。SOURCE旧授权不复用。冻结文件不改。

所有新执行代码与观察器SHA在EXECUTION_EXTENSION_FINAL.json锁定，真实启动外层命令在launch.run前逐项assert。内层校验entry/common/runtime/contract/spec/Sstar和canonicalpack。普通用户localhost8080，不改端口/防火墙/管理员权限；原Flower1.23原生回调不变。观察器仅计数/有限性/快照，不改变损失、参数或随机种子。

训练进程audit hook按角色白名单允许canonical .npy，拒绝其他.npy/.npz及所有dataset/data1原始文件，数据包禁止额外文件与symlink；Dev/Test无loader/回调。该约束是Python进程内访问合同及审计，不宣称OS级网络隔离或隐私保护。B0审计恢复fixture可读取合法Calibration，但不训练/不算准确率/不选模型。

预注册评价：C5Day22、源C1R4Day21/C2R4Day16，全部六端点锁定之后才打开/哈希Test。早期C2Day20字面假设在登记前被canonical身份表纠正，未更改任何原身份。C1/C2保持是历史留出暴露，C2Day16早于Day18接入；不能称未来在线采集。

本轮预登记导入同名与日期断言失败发生在真实训练之前，PREREGISTRATION_ISSUE及EXECUTION_EXTENSION记录保留；没有正式pilot重跑。
