from functools import lru_cache
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Feishu / Lark
    feishu_app_id: str
    feishu_app_secret: str
    feishu_verification_token: str
    feishu_encrypt_key: str = ""

    # GitHub
    github_token: str
    github_owner: str
    github_repo: str
    github_api_version: str = "2026-03-10"

    # Service
    app_env: str = "dev"
    log_level: str = "INFO"
    issue_default_labels: str = "codex-task,needs-review"
    auto_mention_codex: bool = False
    codex_trigger_text: str = (
        "@codex Please implement this task. Follow AGENTS.md and keep the PR small and reviewable."
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @property
    def default_labels(self) -> List[str]:
        return [x.strip() for x in self.issue_default_labels.split(",") if x.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
