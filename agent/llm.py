"""Model router: `get_model(tier)` returns a chat model for the cheap or strong tier,
with a callback that logs tokens and $ cost for every call.

Costs accumulate into the active `CostTracker` (see `track_costs`) so a request,
an eval case, or a routing experiment can read its own total.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Literal, Protocol, TypeVar

from langchain.chat_models import init_chat_model
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from langchain_core.rate_limiters import InMemoryRateLimiter
from pydantic import BaseModel, ValidationError

from agent.config import PROVIDER_KEY_FIELDS, Settings, get_settings

log = logging.getLogger(__name__)

Tier = Literal["cheap", "strong"]

# USD per 1M tokens (input, output). Verify against the provider's pricing page
# before quoting numbers; unknown models are logged with cost_usd=None.
PRICING_PER_MTOK: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    # Gemini paid-tier list prices (ai.google.dev/gemini-api/docs/pricing, checked 2026-10-30).
    # On the free tier the actual spend is $0; these keep $/query comparable across routings.
    # 3.8/3.6 Flash rise to (1.50, 7.50) on 2027-01-01.
    "gemini-3.8-flash": (0.75, 3.75),
    "gemini-3.6-flash": (0.75, 3.75),
    "gemini-3.5-flash-lite": (0.30, 2.50),
    "gemini-3.1-flash-lite": (0.25, 1.50),
    "gemini-3-flash-preview": (0.50, 3.00),
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-flash-lite": (0.10, 0.40),
}
# structured-output method per provider, unless Settings.structured_output_method overrides it
DEFAULT_STRUCTURED_METHOD: dict[str, Literal["function_calling", "json_schema"]] = {
    "openai": "function_calling",
    # Gemini's native response schema; Google discourages tool calling for structured output
    "google_genai": "json_schema",
    # Measured on qwen3:4b-instruct: under Ollama's schema grammar it never closed the SQL
    # string (2,000-token runaways); with tool calling it wrote a correct plan in ~150 tokens.
    "ollama": "function_calling",
}
LOCAL_PROVIDERS = frozenset({"ollama"})


@dataclass(frozen=True)
class UsageRecord:
    tier: Tier
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float | None


@dataclass
class CostTracker:
    records: list[UsageRecord] = field(default_factory=list)

    @property
    def total_cost_usd(self) -> float:
        return sum(r.cost_usd or 0.0 for r in self.records)

    @property
    def total_tokens(self) -> int:
        return sum(r.input_tokens + r.output_tokens for r in self.records)

    @property
    def has_unpriced_calls(self) -> bool:
        return any(r.cost_usd is None for r in self.records)


_current_tracker: ContextVar[CostTracker | None] = ContextVar("cost_tracker", default=None)


@contextmanager
def track_costs() -> Iterator[CostTracker]:
    tracker = CostTracker()
    token = _current_tracker.set(tracker)
    try:
        yield tracker
    finally:
        _current_tracker.reset(token)


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float | None:
    price = PRICING_PER_MTOK.get(model)
    if price is None:
        return None
    return (input_tokens * price[0] + output_tokens * price[1]) / 1_000_000


def _extract_usage(response: LLMResult) -> tuple[int, int]:
    for generations in response.generations:
        for gen in generations:
            if isinstance(gen, ChatGeneration) and isinstance(gen.message, AIMessage):
                usage = gen.message.usage_metadata
                if usage:
                    return usage["input_tokens"], usage["output_tokens"]
    token_usage = (response.llm_output or {}).get("token_usage") or {}
    return int(token_usage.get("prompt_tokens", 0)), int(token_usage.get("completion_tokens", 0))


class UsageCallback(BaseCallbackHandler):
    def __init__(self, tier: Tier, model: str, *, local: bool = False) -> None:
        self.tier = tier
        self.model = model
        self.local = local  # runs on this machine: tokens are counted, cost is $0

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        input_tokens, output_tokens = _extract_usage(response)
        record = UsageRecord(
            tier=self.tier,
            model=self.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=0.0 if self.local else estimate_cost(self.model, input_tokens, output_tokens),
        )
        log.info(
            "llm_call",
            extra={
                "tier": record.tier,
                "model": record.model,
                "input_tokens": record.input_tokens,
                "output_tokens": record.output_tokens,
                "cost_usd": record.cost_usd,
            },
        )
        if (tracker := _current_tracker.get()) is not None:
            tracker.records.append(record)


def _rate_limiter(requests_per_minute: float) -> InMemoryRateLimiter:
    """One limiter per rate, shared by every model and thread in the process.
    (float() matters: lru_cache keys the int 12 and the float 12.0 separately.)"""
    return _limiter_for(float(requests_per_minute))


@lru_cache(maxsize=4)
def _limiter_for(requests_per_minute: float) -> InMemoryRateLimiter:
    return InMemoryRateLimiter(
        requests_per_second=requests_per_minute / 60, check_every_n_seconds=0.1, max_bucket_size=1
    )


def provider_of(spec: str) -> str:
    return spec.partition(":")[0]


def get_model(
    tier: Tier, settings: Settings | None = None, *, max_tokens: int | None = None
) -> BaseChatModel:
    settings = settings or get_settings()
    spec = settings.cheap_model if tier == "cheap" else settings.strong_model
    provider, _, model = spec.partition(":")
    extra: dict[str, Any] = {}
    if settings.llm_requests_per_minute:
        extra["rate_limiter"] = _rate_limiter(settings.llm_requests_per_minute)
    if provider in LOCAL_PROVIDERS:
        local: BaseChatModel = init_chat_model(
            model,
            model_provider=provider,
            base_url=settings.ollama_base_url,
            temperature=0,
            num_predict=max_tokens,  # Ollama's name for the output cap
            num_ctx=settings.ollama_num_ctx,
            callbacks=[UsageCallback(tier, model, local=True)],
            **extra,
        )
        return local
    key_field = PROVIDER_KEY_FIELDS[provider]
    if key_field is None:
        raise ValueError(f"provider {provider!r} needs no key but isn't a local provider")
    api_key = getattr(settings, key_field)
    chat: BaseChatModel = init_chat_model(
        model,
        model_provider=provider,
        api_key=api_key.get_secret_value(),
        temperature=0,
        max_tokens=max_tokens,
        max_retries=settings.llm_max_retries,
        callbacks=[UsageCallback(tier, model)],
        **extra,
    )
    return chat


# ---------------------------------------------------------------- structured calls

T = TypeVar("T", bound=BaseModel)
MAX_OUTPUT_TOKENS: dict[Tier, int] = {"cheap": 2_000, "strong": 1_500}


class InvalidModelOutput(ValueError):
    """The model's output did not validate against the schema. Never parsed loosely."""


class BudgetExceeded(RuntimeError):
    """The request already spent its cost cap; no further model calls are made."""


class RateLimited(RuntimeError):
    """The provider refused for quota/rate reasons even after retries. Not the agent's fault:
    callers (the eval harness) should stop or back off rather than score it as a failure."""


_RATE_LIMIT_MARKERS = (
    "429",
    "resource_exhausted",
    "resourceexhausted",
    "rate limit",
    "ratelimit",
    "quota",
)


def is_rate_limit(exc: BaseException) -> bool:
    text = f"{type(exc).__name__} {exc}".lower()
    return any(marker in text for marker in _RATE_LIMIT_MARKERS)


def ensure_budget(max_cost_usd: float) -> None:
    tracker = _current_tracker.get()
    if tracker is not None and tracker.total_cost_usd >= max_cost_usd:
        raise BudgetExceeded(f"request cost cap of ${max_cost_usd:.2f} reached")


def json_object_in(message: Any) -> dict[str, Any] | None:
    """The single JSON object a reply's text consists of (optionally fenced), else None.
    Only the transport is lenient: the object is still validated strictly by the caller."""
    text = str(getattr(message, "content", "") or "").strip()
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start or text[:start].strip() or text[end + 1 :].strip():
        return None  # prose around the object: not a clean answer, reject
    try:
        value = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


class StructuredLLM(Protocol):
    def structured(
        self,
        tier: Tier,
        schema: type[T],
        messages: list[BaseMessage],
        *,
        run_name: str,
        text_field: str | None = None,
    ) -> T:
        """`text_field`: if the model replies in plain prose (no tool call, no JSON), use the
        prose as that field. Only for display text; never for plans or verdicts."""
        ...


class LangChainLLM:
    """Real implementation: provider-native structured output, validated with Pydantic."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._methods = {
            tier: settings.structured_output_method or DEFAULT_STRUCTURED_METHOD[provider_of(spec)]
            for tier, spec in (("cheap", settings.cheap_model), ("strong", settings.strong_model))
        }
        self._json_in_content = {
            tier: provider_of(spec) in LOCAL_PROVIDERS
            for tier, spec in (("cheap", settings.cheap_model), ("strong", settings.strong_model))
        }
        self._models = {
            tier: get_model(tier, settings, max_tokens=MAX_OUTPUT_TOKENS[tier])
            for tier in ("cheap", "strong")
        }

    def structured(
        self,
        tier: Tier,
        schema: type[T],
        messages: list[BaseMessage],
        *,
        run_name: str,
        text_field: str | None = None,
    ) -> T:
        ensure_budget(self._settings.max_cost_usd)
        runnable = self._models[tier].with_structured_output(
            schema, method=self._methods[tier], include_raw=True
        )
        try:
            out = runnable.invoke(messages, config={"run_name": run_name})
        except Exception as exc:
            if is_rate_limit(exc):
                # generic message: provider errors are never echoed verbatim
                raise RateLimited(f"{tier} model rate-limited or out of quota") from None
            raise
        parsed = out.get("parsed") if isinstance(out, dict) else None
        error = out.get("parsing_error") if isinstance(out, dict) else "no output"
        if (
            isinstance(out, dict)
            and parsed is None
            and error is None
            and self._json_in_content[tier]
        ):
            # small local models often answer the tool schema with JSON in the message body
            # instead of a formal tool call; it still has to pass the same strict validation
            parsed = json_object_in(out.get("raw"))
        if parsed is None and error is None and text_field and isinstance(out, dict):
            text = str(getattr(out.get("raw"), "content", "") or "").strip()
            if text and json_object_in(out.get("raw")) is None:
                parsed = {text_field: text}  # still validated against the schema below
        if error is not None or parsed is None:
            raise InvalidModelOutput(f"{schema.__name__} output rejected: {str(error)[:300]}")
        try:
            return schema.model_validate(parsed)
        except ValidationError as exc:
            raise InvalidModelOutput(f"{schema.__name__} output rejected: {exc}") from None
