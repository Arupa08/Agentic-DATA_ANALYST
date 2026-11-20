"""API-only settings (no secrets here: model and tracing keys stay in `agent.config`)."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class APISettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="API_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # browsers may call the API only from the UI's origin; JSON list in the env var
    cors_origins: list[str] = ["http://localhost:8501"]
    # per client IP, slowapi syntax; behind a proxy, set FORWARDED_ALLOW_IPS for uvicorn
    rate_limit: str = "10/minute"
    # a local model answers one question in minutes; queueing more only adds latency
    max_concurrent: int = Field(default=2, ge=1, le=32)
    # a question is at most 1,000 characters; anything much bigger is not a question
    max_body_bytes: int = Field(default=4_096, ge=256, le=65_536)
    docs: bool = True  # /docs and /openapi.json
