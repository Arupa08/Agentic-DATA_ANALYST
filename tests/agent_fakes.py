"""Test doubles for the agent: a scripted model and a deps builder."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from pydantic import BaseModel

from agent.config import Settings
from agent.deps import AgentDeps
from agent.llm import Tier, UsageCallback, ensure_budget
from agent.tools.sql_tool import SqlTool
from agent.tools.web_search import SearchBackend, WebSearchTool

T = TypeVar("T", bound=BaseModel)
Scripted = BaseModel | Exception | Callable[[list[BaseMessage]], BaseModel]


class ScriptedLLM:
    """Returns queued outputs in order and records every call. Each call costs
    `tokens_per_call` gpt-4o tokens, recorded like a real call, after the same budget check."""

    def __init__(
        self, outputs: list[Scripted], *, max_cost_usd: float = 1.0, tokens_per_call: int = 1_000
    ) -> None:
        self.outputs = list(outputs)
        self.max_cost_usd = max_cost_usd
        self.tokens_per_call = tokens_per_call
        self.calls: list[tuple[Tier, str, list[BaseMessage]]] = []

    def structured(
        self,
        tier: Tier,
        schema: type[T],
        messages: list[BaseMessage],
        *,
        run_name: str,
        text_field: str | None = None,
    ) -> T:
        ensure_budget(self.max_cost_usd)
        self.calls.append((tier, run_name, messages))
        msg = AIMessage(
            content="",
            usage_metadata={
                "input_tokens": self.tokens_per_call,
                "output_tokens": 0,
                "total_tokens": self.tokens_per_call,
            },
        )
        UsageCallback(tier, "gpt-4o").on_llm_end(
            LLMResult(generations=[[ChatGeneration(message=msg)]])
        )
        if not self.outputs:
            raise AssertionError(f"unexpected extra model call: {run_name}")
        item = self.outputs.pop(0)
        if isinstance(item, Exception):
            raise item
        if not isinstance(item, BaseModel):
            item = item(messages)
        if not isinstance(item, schema):
            raise AssertionError(
                f"{run_name} expected {schema.__name__}, got {type(item).__name__}"
            )
        return item

    def prompt_text(self, call: int) -> str:
        return "\n".join(str(m.content) for m in self.calls[call][2])


def make_deps(
    db_path: Path,
    llm: ScriptedLLM,
    *,
    web: SearchBackend | None = None,
    **settings: Any,
) -> AgentDeps:
    base: dict[str, Any] = {
        "_env_file": None,
        "openai_api_key": "sk-test-not-real",
        "langsmith_tracing": False,
        "critic_llm": False,  # tests opt in to the critic's model check explicitly
        **settings,
    }
    return AgentDeps(
        settings=Settings(**base),
        llm=llm,
        sql=SqlTool(db_path),
        web=WebSearchTool(web),
    )
