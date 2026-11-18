from __future__ import annotations

import json
import logging

import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from agent.config import Settings
from agent.llm import UsageCallback, estimate_cost, get_model, track_costs
from agent.logging_setup import JsonFormatter, request_id_var


def _result(input_tokens: int, output_tokens: int) -> LLMResult:
    msg = AIMessage(
        content="hi",
        usage_metadata={
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    )
    return LLMResult(generations=[[ChatGeneration(message=msg)]])


def test_estimate_cost_known_and_unknown_model() -> None:
    assert estimate_cost("gpt-4o-mini", 1_000_000, 1_000_000) == pytest.approx(0.75)
    assert estimate_cost("not-a-model", 10, 10) is None


def test_callback_accumulates_into_active_tracker() -> None:
    cb = UsageCallback("cheap", "gpt-4o-mini")
    with track_costs() as tracker:
        cb.on_llm_end(_result(1_000, 500))
        cb.on_llm_end(_result(2_000, 0))
    assert tracker.total_tokens == 3_500
    assert tracker.total_cost_usd == pytest.approx((3_000 * 0.15 + 500 * 0.60) / 1e6)
    assert not tracker.has_unpriced_calls


def test_trackers_do_not_leak_between_scopes() -> None:
    cb = UsageCallback("strong", "gpt-4o")
    with track_costs() as first:
        cb.on_llm_end(_result(10, 10))
    with track_costs() as second:
        pass
    cb.on_llm_end(_result(10, 10))  # no active tracker: logged only
    assert len(first.records) == 1
    assert second.records == []


def test_unpriced_model_is_flagged() -> None:
    with track_costs() as tracker:
        UsageCallback("cheap", "mystery").on_llm_end(_result(5, 5))
    assert tracker.has_unpriced_calls


def test_get_model_routes_by_tier() -> None:
    settings = Settings(  # type: ignore[call-arg]
        _env_file=None,
        langsmith_tracing=False,
        openai_api_key="sk-test",
        cheap_model="openai:gpt-4o-mini",
        strong_model="openai:gpt-4o",
    )
    cheap, strong = get_model("cheap", settings), get_model("strong", settings)
    assert getattr(cheap, "model_name", None) == "gpt-4o-mini"
    assert getattr(strong, "model_name", None) == "gpt-4o"
    callbacks = cheap.callbacks
    assert isinstance(callbacks, list)
    assert any(isinstance(cb, UsageCallback) and cb.tier == "cheap" for cb in callbacks)


def test_json_log_carries_request_id_and_extras() -> None:
    record = logging.LogRecord("t", logging.INFO, __file__, 1, "llm_call", None, None)
    record.cost_usd = 0.01
    token = request_id_var.set("req-123")
    try:
        payload = json.loads(JsonFormatter().format(record))
    finally:
        request_id_var.reset(token)
    assert payload["request_id"] == "req-123"
    assert payload["cost_usd"] == 0.01
    assert payload["msg"] == "llm_call"
