from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class FeishuMessageEvent:
    event_id: str
    event_type: str
    message_id: str
    chat_id: str
    text: str
    sender_id: str


def is_url_verification(payload: Dict[str, Any]) -> bool:
    if payload.get("type") == "url_verification" and payload.get("challenge"):
        return True
    header = payload.get("header") or {}
    return header.get("event_type") == "url_verification" and bool(payload.get("challenge"))


def get_payload_token(payload: Dict[str, Any]) -> Optional[str]:
    if payload.get("token"):
        return str(payload["token"])
    header = payload.get("header") or {}
    if header.get("token"):
        return str(header["token"])
    return None


def parse_text_content(content: Any) -> str:
    if isinstance(content, dict):
        return str(content.get("text", ""))
    if not content:
        return ""
    if isinstance(content, str):
        try:
            obj = json.loads(content)
            if isinstance(obj, dict):
                return str(obj.get("text", ""))
        except json.JSONDecodeError:
            return content
    return str(content)


def parse_message_event(payload: Dict[str, Any]) -> Optional[FeishuMessageEvent]:
    header = payload.get("header") or {}
    event = payload.get("event") or {}

    event_type = str(header.get("event_type") or payload.get("type") or "")
    if event_type != "im.message.receive_v1":
        return None

    message = event.get("message") or {}
    sender = event.get("sender") or {}

    message_id = str(message.get("message_id") or "")
    if not message_id:
        return None

    text = parse_text_content(message.get("content"))

    sender_id_obj = sender.get("sender_id") or {}
    sender_id = (
        sender_id_obj.get("open_id")
        or sender_id_obj.get("user_id")
        or sender_id_obj.get("union_id")
        or ""
    )

    return FeishuMessageEvent(
        event_id=str(header.get("event_id") or ""),
        event_type=event_type,
        message_id=message_id,
        chat_id=str(message.get("chat_id") or ""),
        text=text,
        sender_id=str(sender_id),
    )
