from __future__ import annotations

import json
import logging
from collections import OrderedDict
from typing import Dict

from fastapi import FastAPI, HTTPException, Request

from app.config import get_settings
from app.feishu_client import FeishuClient
from app.feishu_event import get_payload_token, is_url_verification, parse_message_event
from app.github_client import GitHubClient
from app.security import verify_feishu_signature
from app.task_parser import parse_task


settings = get_settings()

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("feishu-github-bridge")

app = FastAPI(title="Feishu GitHub Bridge v1", version="1.0.0")

feishu = FeishuClient(
    app_id=settings.feishu_app_id,
    app_secret=settings.feishu_app_secret,
)
github = GitHubClient(
    token=settings.github_token,
    owner=settings.github_owner,
    repo=settings.github_repo,
    api_version=settings.github_api_version,
)

# Simple in-memory deduplication for Feishu event retries.
_SEEN_EVENT_IDS: "OrderedDict[str, None]" = OrderedDict()
_MAX_SEEN = 2048


def remember_event(event_id: str) -> bool:
    """Return True if this event is new, False if it was seen before."""
    if not event_id:
        return True
    if event_id in _SEEN_EVENT_IDS:
        return False
    _SEEN_EVENT_IDS[event_id] = None
    if len(_SEEN_EVENT_IDS) > _MAX_SEEN:
        _SEEN_EVENT_IDS.popitem(last=False)
    return True


@app.get("/healthz")
async def healthz() -> Dict[str, str]:
    return {"status": "ok", "service": "feishu-github-bridge-v1"}


@app.post("/feishu/events")
async def feishu_events(request: Request):
    raw_body = await request.body()

    if not verify_feishu_signature(
        raw_body=raw_body,
        headers=request.headers,
        encrypt_key=settings.feishu_encrypt_key,
    ):
        raise HTTPException(status_code=401, detail="invalid feishu signature")

    try:
        payload = json.loads(raw_body.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="invalid json") from exc

    # URL verification when configuring the callback URL in Feishu.
    if is_url_verification(payload):
        token = get_payload_token(payload)
        if token and token != settings.feishu_verification_token:
            raise HTTPException(status_code=401, detail="invalid verification token")
        return {"challenge": payload.get("challenge")}

    # Verification token check when token is present in payload/header.
    token = get_payload_token(payload)
    if token and token != settings.feishu_verification_token:
        raise HTTPException(status_code=401, detail="invalid verification token")

    event = parse_message_event(payload)
    if event is None:
        return {"ok": True, "ignored": True}

    if not remember_event(event.event_id):
        logger.info("duplicated event ignored: %s", event.event_id)
        return {"ok": True, "duplicated": True}

    logger.info("received message event=%s message=%s sender=%s", event.event_id, event.message_id, event.sender_id)

    try:
        task = parse_task(event.text, default_labels=settings.default_labels)
        issue_title = f"[{task.command}] {task.title}"
        issue_body = task.to_issue_body(
            sender=event.sender_id,
            codex_trigger_text=settings.codex_trigger_text,
            auto_mention_codex=settings.auto_mention_codex,
        )
        issue = await github.create_issue(
            title=issue_title,
            body=issue_body,
            labels=task.labels,
        )
        reply = (
            "✅ 已创建 GitHub Issue\n"
            f"标题：{issue.title}\n"
            f"链接：{issue.html_url}\n"
            f"标签：{', '.join(task.labels) if task.labels else '无'}"
        )
        await feishu.reply_text(message_id=event.message_id, text=reply)
        return {"ok": True, "issue": issue.html_url}

    except Exception as exc:
        logger.exception("failed to process Feishu task")
        try:
            await feishu.reply_text(
                message_id=event.message_id,
                text=(
                    "❌ 创建 GitHub Issue 失败\n"
                    f"错误：{type(exc).__name__}: {exc}\n"
                    "请检查 GitHub Token、仓库名、Issues 是否开启，以及飞书机器人权限。"
                ),
            )
        except Exception:
            logger.exception("failed to reply error to Feishu")
        return {"ok": False, "error": str(exc)}
