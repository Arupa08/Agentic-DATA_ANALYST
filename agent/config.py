"""Typed settings loaded from the environment (and .env in dev).

The app must refuse to start when a key it needs is missing, so validation
runs at construction time rather than at first use.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# provider prefix in a "provider:model" spec -> settings field holding its key
# None = no key needed (local models)
PROVIDER_KEY_FIELDS: dict[str, str | None] = {
    "openai": "openai_api_key",
    "google_genai": "google_api_key",  # Gemini API (Google AI Studio key)
    "ollama": None,  # local models via the Ollama app
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", populate_by_name=True
    )

    cheap_model: str = "openai:gpt-4o-mini"
    strong_model: str = "openai:gpt-4o"
    openai_api_key: SecretStr | None = None
    # Google's SDKs accept either name for an AI Studio key, so we do too
    google_api_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("GOOGLE_API_KEY", "GEMINI_API_KEY")
    )
    # how models return structured output; None = the provider's best supported method
    structured_output_method: Literal["function_calling", "json_schema"] | None = None
    llm_max_retries: int = Field(default=3, ge=0, le=10)
    # client-side throttle shared by every model call; set it to stay under free-tier quotas
    llm_requests_per_minute: float | None = Field(default=None, gt=0)
    # local models (provider "ollama"); Ollama's own default context silently truncates our prompts
    ollama_base_url: str = "http://localhost:11434"
    ollama_num_ctx: int = Field(default=8192, ge=2048, le=131072)

    langsmith_tracing: bool = False
    langsmith_api_key: SecretStr | None = None
    langsmith_project: str = "nl-analytics-agent"

    database_url: str = "sqlite:///data/shop.db"
    # business glossary in the planner prompt; switch off to measure its effect in evals
    use_glossary: bool = True
    # optional: web search is disabled when unset
    tavily_api_key: SecretStr | None = None
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # hard per-request caps (enforced by the graph in later phases)
    max_replans: int = Field(default=3, ge=0, le=5)
    max_cost_usd: float = Field(default=0.25, gt=0)
    max_tool_calls: int = Field(default=15, ge=1, le=50)
    # the critic's LLM check (rule checks always run); off = cheaper, for routing experiments
    critic_llm: bool = True
    # when the planner wants to ask, nudge it once to state an assumption instead.
    # Off by default: on the local 4B model it took underspecified cases from 60% to 9%
    # (evals/reports/2026-11-18_1927_critic-rules-local-regraded.md), because asking passes.
    clarify_nudge: bool = False

    @model_validator(mode="after")
    def _require_keys(self) -> Settings:
        missing: list[str] = []
        for spec in (self.cheap_model, self.strong_model):
            provider, sep, model = spec.partition(":")
            if not sep or not model:
                raise ValueError(
                    f"model spec must look like 'provider:model' (e.g. "
                    f"'google_genai:gemini-3.5-flash-lite' or 'openai:gpt-4o-mini'), got {spec!r}"
                )
            if provider not in PROVIDER_KEY_FIELDS:
                raise ValueError(
                    f"unsupported provider {provider!r}; add it to PROVIDER_KEY_FIELDS"
                )
            key_field = PROVIDER_KEY_FIELDS[provider]
            if key_field is None:
                continue
            key: SecretStr | None = getattr(self, key_field)
            if key is None or not key.get_secret_value().strip():
                missing.append(key_field.upper())
        if self.langsmith_tracing and self.langsmith_api_key is None:
            missing.append("LANGSMITH_API_KEY")
        if missing:
            # names only — never echo values
            raise ValueError(f"missing required settings: {', '.join(sorted(set(missing)))}")
        return self

    @property
    def db_path(self) -> Path:
        prefix = "sqlite:///"
        if not self.database_url.startswith(prefix):
            raise ValueError("only sqlite:/// database URLs are supported until the Postgres phase")
        return Path(self.database_url.removeprefix(prefix))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
