from __future__ import annotations

import pytest
from pydantic import SecretStr

from agent.config import Settings
from agent.tools.web_search import SearchHit, WebSearchTool, build_web_search


class FakeBackend:
    def __init__(self, hits: list[SearchHit] | None = None, fail: Exception | None = None) -> None:
        self.hits = hits or []
        self.fail = fail
        self.queries: list[str] = []

    def search(self, query: str, max_results: int) -> list[SearchHit]:
        self.queries.append(query)
        if self.fail:
            raise self.fail
        return self.hits[:max_results]


HOLIDAY = SearchHit(
    title="US holidays 2025", url="https://example.com/h", snippet="Black Friday is Nov 28, 2025."
)


def test_results_are_labeled_external_and_untrusted() -> None:
    tool = WebSearchTool(FakeBackend([HOLIDAY]))
    r = tool.run("US public holidays November 2025")
    assert r.ok and len(r.hits) == 1
    text = r.to_llm_text()
    assert text.startswith("EXTERNAL web results")
    assert '<untrusted source="web">' in text


def test_disabled_without_backend() -> None:
    r = WebSearchTool(None).run("anything")
    assert not r.ok and r.error == "web search is not configured"


def test_build_from_settings_disabled_when_key_empty() -> None:
    s = Settings(
        _env_file=None, openai_api_key="sk-test", langsmith_tracing=False, tavily_api_key=""
    )  # type: ignore[call-arg]
    assert not build_web_search(s).enabled
    s2 = Settings(
        _env_file=None,
        openai_api_key="sk-test",
        langsmith_tracing=False,
        tavily_api_key=SecretStr("tvly-x"),
    )  # type: ignore[call-arg]
    assert build_web_search(s2).enabled


def test_non_http_urls_dropped() -> None:
    bad = SearchHit(title="x", url="javascript:alert(1)", snippet="x")
    r = WebSearchTool(FakeBackend([bad, HOLIDAY])).run("holidays")
    assert [h.url for h in r.hits] == [HOLIDAY.url]


def test_backend_error_is_generic_and_key_free() -> None:
    backend = FakeBackend(fail=RuntimeError("401 for key tvly-SECRET-123"))
    r = WebSearchTool(backend).run("holidays")
    assert not r.ok and r.error == "web search failed"
    assert "SECRET" not in r.model_dump_json()


def test_snippets_truncated() -> None:
    long = SearchHit(title="t", url="https://e.com", snippet="x" * 5_000)
    r = WebSearchTool(FakeBackend([long])).run("q")
    assert len(r.hits[0].snippet) == 500


@pytest.mark.parametrize("query", ["", "   ", "x" * 201])
def test_malformed_queries_rejected(query: str) -> None:
    backend = FakeBackend([HOLIDAY])
    assert not WebSearchTool(backend).run(query).ok
    assert backend.queries == []
