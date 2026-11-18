"""LangSmith tracing with masking: emails and anything key-shaped are replaced before a
run leaves the process. Only this tracer is used, so no unmasked copy is sent."""

from __future__ import annotations

import os
import re

from langchain_core.tracers.langchain import LangChainTracer
from langsmith import Client
from langsmith.anonymizer import create_anonymizer

from agent.config import Settings

MASK_RULES = [
    {"pattern": re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "replace": "<email>"},
    {"pattern": re.compile(r"\b(sk|lsv2|tvly)[-_][A-Za-z0-9_-]{6,}"), "replace": "<secret>"},
]
_ENV_TRACING_FLAGS = ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2", "LANGCHAIN_TRACING")


def build_tracer(settings: Settings) -> LangChainTracer | None:
    if not settings.langsmith_tracing or settings.langsmith_api_key is None:
        return None
    # env-driven auto-tracing would send a second, unmasked copy of every run
    for flag in _ENV_TRACING_FLAGS:
        os.environ.pop(flag, None)
    client = Client(
        api_key=settings.langsmith_api_key.get_secret_value(),
        anonymizer=create_anonymizer(MASK_RULES),  # type: ignore[arg-type]
    )
    return LangChainTracer(project_name=settings.langsmith_project, client=client)
