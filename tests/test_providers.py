"""Provider wiring: Gemini alongside OpenAI, structured-output method choice, throttling,
and rate-limit handling end to end (the harness stops instead of scoring quota errors)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from agent.config import Settings
from agent.graph import Agent
from agent.llm import (
    LangChainLLM,
    RateLimited,
    _rate_limiter,
    estimate_cost,
    get_model,
    is_rate_limit,
)
from agent.state import SynthesisOutput
from evals.report import summarize
from evals.run_evals import QUOTA_STOP, run_suite
from tests.agent_fakes import ScriptedLLM, make_deps
from tests.eval_fixtures import case

GEMINI: dict[str, Any] = {
    "_env_file": None,
    "langsmith_tracing": False,
    "cheap_model": "google_genai:gemini-3.5-flash-lite",
    "strong_model": "google_genai:gemini-3.8-flash",
}


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    names = ("OPENAI_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY", "CHEAP_MODEL", "STRONG_MODEL")
    for var in names:
        monkeypatch.delenv(var, raising=False)


def test_gemini_only_needs_the_google_key() -> None:
    s = Settings(**GEMINI, google_api_key="AIza-test")
    assert s.openai_api_key is None
    with pytest.raises(ValidationError, match="GOOGLE_API_KEY"):
        Settings(**GEMINI)


def test_empty_key_counts_as_missing() -> None:
    with pytest.raises(ValidationError, match="GOOGLE_API_KEY"):
        Settings(**GEMINI, google_api_key="   ")


def test_mixed_providers_need_both_keys() -> None:
    mixed = {**GEMINI, "strong_model": "openai:gpt-4o"}
    with pytest.raises(ValidationError, match="OPENAI_API_KEY"):
        Settings(**mixed, google_api_key="AIza-test")
    Settings(**mixed, google_api_key="AIza-test", openai_api_key="sk-test")


def test_gemini_model_built_with_retries_and_throttle() -> None:
    s = Settings(
        **GEMINI, google_api_key="AIza-test", llm_requests_per_minute=12, llm_max_retries=5
    )
    model = get_model("cheap", s, max_tokens=300)
    assert type(model).__name__ == "ChatGoogleGenerativeAI"
    assert getattr(model, "model", "").endswith("gemini-3.5-flash-lite")
    assert getattr(model, "max_retries", None) == 5
    assert getattr(model, "rate_limiter", None) is _rate_limiter(12)  # one shared limiter


def test_structured_method_per_provider_and_override() -> None:
    assert LangChainLLM(Settings(**GEMINI, google_api_key="k"))._methods == {
        "cheap": "json_schema",
        "strong": "json_schema",
    }
    openai = Settings(_env_file=None, langsmith_tracing=False, openai_api_key="sk-t")  # type: ignore[call-arg]
    assert LangChainLLM(openai)._methods["cheap"] == "function_calling"
    forced = Settings(**GEMINI, google_api_key="k", structured_output_method="function_calling")
    assert set(LangChainLLM(forced)._methods.values()) == {"function_calling"}


def test_gemini_pricing_is_known() -> None:
    assert estimate_cost("gemini-3.5-flash-lite", 1_000_000, 1_000_000) == pytest.approx(2.80)
    assert estimate_cost("gemini-3.8-flash", 1_000_000, 0) == pytest.approx(0.75)


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("429 RESOURCE_EXHAUSTED: Quota exceeded for metric", True),
        ("You exceeded your current quota", True),
        ("Rate limit reached for requests", True),
        ("400 Invalid argument: schema", False),
        ("connection reset", False),
    ],
)
def test_rate_limit_detection(message: str, expected: bool) -> None:
    assert is_rate_limit(RuntimeError(message)) is expected


class _Raising:
    """Stands in for a chat model whose provider refuses with a quota error."""

    def with_structured_output(self, *_: Any, **__: Any) -> _Raising:
        return self

    def invoke(self, *_: Any, **__: Any) -> Any:
        raise RuntimeError("429 RESOURCE_EXHAUSTED: quota exceeded (key AIza-secret)")


def test_quota_errors_become_rate_limited_without_echoing_provider_text() -> None:
    llm = LangChainLLM(Settings(**GEMINI, google_api_key="k"))
    llm._models["cheap"] = _Raising()  # type: ignore[assignment]
    with pytest.raises(RateLimited) as exc:
        llm.structured("cheap", SynthesisOutput, [], run_name="t")
    assert "AIza" not in str(exc.value)


def test_agent_propagates_rate_limits(shop_db_path: Path) -> None:
    agent = Agent(make_deps(shop_db_path, ScriptedLLM([RateLimited("quota")])))
    with pytest.raises(RateLimited):
        agent.ask("What was total revenue in Q3 2025?")


def test_harness_stops_on_quota_and_marks_runs_not_run(shop_db_path: Path) -> None:
    agent = Agent(make_deps(shop_db_path, ScriptedLLM([RateLimited("quota")])))
    c = case("simple_01")
    grades = run_suite(agent, ScriptedLLM([]), [c], runs=3, workers=1, label="t", progress=False)
    assert [g.grading_error for g in grades] == [QUOTA_STOP] * 3  # 1 hit the limit, 2 skipped
    assert not any(g.graded for g in grades)
    summary = summarize(grades, [c])
    assert summary["not_run_quota"] == 3 and summary["pass_rate"] is None


def test_gemini_key_accepted_under_either_env_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-from-gemini-name")
    s = Settings(**GEMINI)
    assert s.google_api_key is not None
    assert s.google_api_key.get_secret_value() == "AIza-from-gemini-name"


def test_bare_provider_name_gets_a_helpful_error() -> None:
    with pytest.raises(ValidationError, match=re.escape("google_genai:gemini-3.5-flash-lite")):
        Settings(**{**GEMINI, "cheap_model": "Gemini"}, google_api_key="k")


OLLAMA: dict[str, Any] = {
    "_env_file": None,
    "langsmith_tracing": False,
    "cheap_model": "ollama:qwen3:4b-instruct",
    "strong_model": "ollama:qwen3:4b-instruct",
}


def test_local_models_need_no_key() -> None:
    s = Settings(**OLLAMA)
    assert s.openai_api_key is None and s.google_api_key is None


def test_ollama_model_gets_context_and_output_caps() -> None:
    s = Settings(**OLLAMA, ollama_num_ctx=8192)
    model = get_model("cheap", s, max_tokens=700)
    assert type(model).__name__ == "ChatOllama"
    assert getattr(model, "model", None) == "qwen3:4b-instruct"  # tag with a colon survives
    assert getattr(model, "num_ctx", None) == 8192  # never Ollama's silent small default
    assert getattr(model, "num_predict", None) == 700
    assert getattr(model, "base_url", None) == "http://localhost:11434"


def test_ollama_uses_tool_calling_not_a_grammar() -> None:
    assert set(LangChainLLM(Settings(**OLLAMA))._methods.values()) == {"function_calling"}


def test_local_calls_are_free_but_still_counted() -> None:
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, LLMResult

    from agent.llm import UsageCallback, track_costs

    msg = AIMessage(
        content="",
        usage_metadata={"input_tokens": 2_000, "output_tokens": 300, "total_tokens": 2_300},
    )
    with track_costs() as tracker:
        UsageCallback("cheap", "qwen3:4b-instruct", local=True).on_llm_end(
            LLMResult(generations=[[ChatGeneration(message=msg)]])
        )
    assert tracker.total_cost_usd == 0.0 and tracker.total_tokens == 2_300
    assert not tracker.has_unpriced_calls


class _Replying:
    """A chat model that answers the plan schema with JSON in its message body."""

    def __init__(self, content: str) -> None:
        self.content = content

    def with_structured_output(self, *_: Any, **__: Any) -> _Replying:
        return self

    def invoke(self, *_: Any, **__: Any) -> Any:
        from langchain_core.messages import AIMessage

        return {"raw": AIMessage(content=self.content), "parsed": None, "parsing_error": None}


PLAN_JSON = (
    '{"outcome": "plan", "steps": [{"id": 1, "goal": "Q3 revenue", "expected_output": "one row",'
    ' "tool": "sql", "sql": "SELECT 1 AS x"}]}'
)


@pytest.mark.parametrize("content", [PLAN_JSON, f"```json\n{PLAN_JSON}\n```"])
def test_local_model_json_in_reply_is_accepted_and_validated(content: str) -> None:
    from agent.state import Plan

    llm = LangChainLLM(Settings(**OLLAMA))
    llm._models["cheap"] = _Replying(content)  # type: ignore[assignment]
    plan = llm.structured("cheap", Plan, [], run_name="planner")
    assert plan.steps[0].sql == "SELECT 1 AS x"


@pytest.mark.parametrize(
    "content",
    [
        "Here is the plan: " + PLAN_JSON,  # prose around it
        PLAN_JSON.replace('"outcome": "plan"', '"outcome": "plan", "permissions": "all"'),
        PLAN_JSON.replace('"tool": "sql"', '"tool": "shell"'),
        "not json at all",
    ],
)
def test_json_in_reply_is_still_strictly_validated(content: str) -> None:
    from agent.llm import InvalidModelOutput
    from agent.state import Plan

    llm = LangChainLLM(Settings(**OLLAMA))
    llm._models["cheap"] = _Replying(content)  # type: ignore[assignment]
    with pytest.raises(InvalidModelOutput):
        llm.structured("cheap", Plan, [], run_name="planner")


def test_cloud_models_never_read_json_from_the_reply_body() -> None:
    from agent.llm import InvalidModelOutput
    from agent.state import Plan

    llm = LangChainLLM(Settings(**GEMINI, google_api_key="k"))
    llm._models["cheap"] = _Replying(PLAN_JSON)  # type: ignore[assignment]
    with pytest.raises(InvalidModelOutput):
        llm.structured("cheap", Plan, [], run_name="planner")


def test_prose_reply_becomes_the_answer_only_where_opted_in() -> None:
    from agent.llm import InvalidModelOutput
    from agent.state import Plan

    prose = "4,172 orders were placed in December 2025, including cancelled ones."
    llm = LangChainLLM(Settings(**OLLAMA))
    llm._models["strong"] = _Replying(prose)  # type: ignore[assignment]
    out = llm.structured("strong", SynthesisOutput, [], run_name="synthesizer", text_field="answer")
    assert out.answer == prose and out.caveats == []
    # without the opt-in (plans, verdicts) prose is still rejected
    with pytest.raises(InvalidModelOutput):
        llm.structured("strong", Plan, [], run_name="planner")
    with pytest.raises(InvalidModelOutput):
        llm.structured("strong", SynthesisOutput, [], run_name="synthesizer")
