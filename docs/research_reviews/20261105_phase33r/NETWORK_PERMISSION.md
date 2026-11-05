# 网络权限与执行环境合同

默认沙箱：同一127.0.0.1:8080可bind，但connect返回Socket10013，本轮一字节无模型诊断再次确认。不能从bind成功推断可握手，默认沙箱不可用于SOURCE网络执行。

本轮按用户授权在允许localhost的宿主执行上下文运行合成test；仍为同一用户，IsUserAnAdmin=False，无UAC管理员令牌，地址固定127.0.0.1:8080，不监听0.0.0.0。没有修改防火墙、扫描或更换端口、安装新服务或改变模型/数据权限。宿主访问许可来自本次明确合成网络授权，不等于真实SOURCE授权。

ENVIRONMENT_LOCK保存Python3.12.4可执行文件SHA、Flower1.23.0、gRPC1.76.0、Torch2.5.1及numpy/scipy/sklearn/psutil版本、Windows版本、匿名化用户哈希、线程/环境变量和实际代码SHA。网络进程身份、argv、退出及真实bind/connect/资源释放记录在network_smoke_01；一字节默认诊断只界定权限，不替代原生Flower验收。

未来SOURCE必须在同等允许localhost的受控环境、同版本与固定端口核验通过后执行。不能用关闭防火墙、额外提升权限或任意换端口来绕过预检。端口占用就BLOCKED，不自动改地址。

执行前必须调用execution_environment.clean_environment清除PYTHONPATH及旧FEDONGAS观察器变量，在新进程先核对runtime与contract都来自本版本。MODULE_IMPORT_PRECHECK已用内存模拟污染环境验证；不会实际启动旧观察器。环境清理和模块路径是attempt2授权前必要条件。
