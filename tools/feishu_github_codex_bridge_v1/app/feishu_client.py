from __future__ import annotations

import json
import time
from typing import Optional

import httpx


class FeishuClient:
    def __init__(self, *, app_id: str, app_secret: str):
        self.app_id = app_id
        self.app_secret = app_secret
        self.base_url = "https://open.feishu.cn"
        self._tenant_token: Optional[str] = None
        self._tenant_token_expire_at: float = 0.0

    async def get_tenant_access_token(self) -> str:
        now = time.time()
        if self._tenant_token and now < self._tenant_token_expire_at - 120:
            return self._tenant_token

        url = f"{self.base_url}/open-apis/auth/v3/tenant_access_token/internal"
        payload = {"app_id": self.app_id, "app_secret": self.app_secret}
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()

        if data.get("code") != 0:
            raise RuntimeError(f"Feishu token error: {data}")

        self._tenant_token = data["tenant_access_token"]
        self._tenant_token_expire_at = now + int(data.get("expire", 7200))
        return self._tenant_token

    async def reply_text(self, *, message_id: str, text: str) -> None:
        token = await self.get_tenant_access_token()
        url = f"{self.base_url}/open-apis/im/v1/messages/{message_id}/reply"
        payload = {
            "msg_type": "text",
            "content": json.dumps({"text": text}, ensure_ascii=False),
        }
        headers = {"Authorization": f"Bearer {token}"}
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()

        if data.get("code") != 0:
            raise RuntimeError(f"Feishu reply error: {data}")
