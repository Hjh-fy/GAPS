# Feishu → GitHub Issue → Feishu Reply Bridge v1

这个 v1 中间服务用于把飞书机器人收到的任务消息转成 GitHub Issue，并把 Issue 链接回复到飞书消息下。

## 1. 工作流

```text
飞书群 @GAPS-Codex
  → FastAPI /feishu/events
  → 解析 /task-code、/task-review、/task-exp、/task-doc、/task-issue
  → GitHub REST API 创建 Issue
  → 飞书回复 Issue 链接
```

v1 不直接调用 Codex。建议先让任务稳定进入 GitHub Issue，后续再接入 Codex Cloud、GitHub Action 或 PR 评论触发。

## 2. 飞书开放平台配置

在飞书开放平台创建企业自建应用：

1. 开启机器人能力。
2. 事件订阅方式选择“将事件发送至开发者服务器”。
3. 请求地址填入：

```text
https://你的域名/feishu/events
```

4. 订阅事件：

```text
im.message.receive_v1
```

5. 权限建议至少包含机器人接收消息和回复消息相关权限。
6. v1 推荐先关闭事件内容加密；生产环境建议填写 `FEISHU_ENCRYPT_KEY` 并开启签名校验。

## 3. GitHub Token 权限

建议使用 fine-grained token：

```text
Repository access: 只选择 GAPS 仓库
Permissions:
- Metadata: Read
- Issues: Read and write
```

如果后面要自动评论 PR 或触发 Codex，再补 Pull requests: Read and write。

## 4. 本地运行

```bash
cd tools/feishu_github_codex_bridge_v1
cp .env.example .env
# 编辑 .env，填入飞书和 GitHub 配置

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

uvicorn app.main:app --host 0.0.0.0 --port 8088
```

健康检查：

```bash
curl http://127.0.0.1:8088/healthz
```

## 5. 飞书任务格式

```text
@GAPS-Codex /task-code

模块：QC v3
标题：修复 rank/margin risk 占位问题

背景：
当前 class_response_rank_risk 和 class_response_margin_risk 仍然是 0，导致 QC v3 风险解释性不足。

涉及文件：
- evaluate_single_window_reliability.py
- route_aware_response_anchoring.py
- build_deployment_output_package.py

要求：
1. 保持 CSV 字段兼容
2. 不改变训练主流程
3. 增加最小 smoke test
4. 输出验证命令

验收：
生成新的 guardrail_summary.csv 和 deployment_output.csv，字段兼容旧版。
```

## 6. Docker 运行

```bash
docker build -t feishu-github-bridge:v1 .
docker run --env-file .env -p 8088:8088 feishu-github-bridge:v1
```

## 7. systemd 示例

把 `deploy/feishu-github-bridge.service` 复制到 `/etc/systemd/system/` 后：

```bash
sudo systemctl daemon-reload
sudo systemctl enable feishu-github-bridge
sudo systemctl start feishu-github-bridge
sudo journalctl -u feishu-github-bridge -f
```

## 8. 给 Codex 的下一步任务

PR 合并后，可以在 GitHub Issue 或 PR 中评论：

```text
@codex Please review tools/feishu_github_codex_bridge_v1 and help configure it for the GAPS repository. Check security, Feishu callback validation, GitHub issue creation, and deployment README completeness.
```

建议 Codex 重点检查：

- 飞书事件订阅 URL 验证是否完整。
- `im.message.receive_v1` 消息结构解析是否兼容群聊和私聊。
- GitHub fine-grained token 权限是否最小化。
- `.env` 是否被 `.gitignore` 排除。
- 是否需要补充 Redis 去重、飞书事件解密、GitHub Action 回传飞书。

## 9. 注意事项

- 不要把 `.env` 提交到 GitHub。
- GitHub 仓库要开启 Issues。
- 飞书回调地址正式使用时建议配 HTTPS。
- v1 使用内存去重，服务重启后去重状态会丢失；后续可以接 Redis。
- v1 支持签名校验，但没有实现飞书加密事件体解密；生产环境如开启事件加密，需要补充解密逻辑或接入飞书官方 SDK。
