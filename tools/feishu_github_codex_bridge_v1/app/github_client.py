from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import httpx


@dataclass
class GitHubIssue:
    number: int
    html_url: str
    title: str


class GitHubClient:
    def __init__(self, *, token: str, owner: str, repo: str, api_version: str = "2026-03-10"):
        self.token = token
        self.owner = owner
        self.repo = repo
        self.api_version = api_version
        self.base_url = "https://api.github.com"

    async def create_issue(self, *, title: str, body: str, labels: Optional[List[str]] = None) -> GitHubIssue:
        url = f"{self.base_url}/repos/{self.owner}/{self.repo}/issues"
        payload = {
            "title": title,
            "body": body,
            "labels": labels or [],
        }
        headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {self.token}",
            "X-GitHub-Api-Version": self.api_version,
        }
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
        return GitHubIssue(
            number=int(data["number"]),
            html_url=str(data["html_url"]),
            title=str(data["title"]),
        )
