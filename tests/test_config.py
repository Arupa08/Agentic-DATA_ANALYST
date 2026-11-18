from __future__ import annotations

import pytest
from pydantic import ValidationError

from agent.config import Settings

# _env_file=None keeps a developer's real .env out of these tests
BASE: dict[str, object] = {"_env_file": None, "langsmith_tracing": False}


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (
        "OPENAI_API_KEY",
        "LANGSMITH_API_KEY",
        "LANGSMITH_TRACING",
        "CHEAP_MODEL",
        "STRONG_MODEL",
    ):
        monkeypatch.delenv(var, raising=False)


def test_refuses_to_start_without_llm_key() -> None:
    with pytest.raises(ValidationError, match="OPENAI_API_KEY"):
        Settings(**BASE)  # type: ignore[arg-type]


def test_loads_with_key() -> None:
    s = Settings(**BASE, openai_api_key="sk-test")  # type: ignore[arg-type]
    assert s.cheap_model.startswith("openai:")


def test_tracing_requires_langsmith_key() -> None:
    with pytest.raises(ValidationError, match="LANGSMITH_API_KEY"):
        Settings(**{**BASE, "langsmith_tracing": True}, openai_api_key="sk-test")  # type: ignore[arg-type]


def test_rejects_malformed_model_spec() -> None:
    with pytest.raises(ValidationError, match="provider:model"):
        Settings(**BASE, openai_api_key="sk-test", cheap_model="gpt-4o-mini")  # type: ignore[arg-type]


def test_rejects_unknown_provider() -> None:
    with pytest.raises(ValidationError, match="unsupported provider"):
        Settings(**BASE, openai_api_key="sk-test", strong_model="acme:big")  # type: ignore[arg-type]


def test_secret_never_in_repr_or_errors() -> None:
    secret = "sk-very-secret-value"
    s = Settings(**BASE, openai_api_key=secret)  # type: ignore[arg-type]
    assert secret not in repr(s)
    assert secret not in str(s.model_dump())
    with pytest.raises(ValidationError) as exc:
        Settings(**{**BASE, "langsmith_tracing": True}, openai_api_key=secret)  # type: ignore[arg-type]
    assert secret not in str(exc.value)
