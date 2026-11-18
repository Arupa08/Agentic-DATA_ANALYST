"""Optional web search for external context (e.g. holiday dates), always labeled external.

Two rules from the safety design:
  - outbound: a query must never carry company data. Emails, long digit runs and any
    caller-supplied tainted values (e.g. strings from SQL results) are rejected.
  - inbound: results are untrusted text. They are wrapped as untrusted and labeled
    EXTERNAL, and can inform an answer but never trigger a tool call.

Disabled (returns a clear error) when no search API key is configured.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from typing import Protocol

import httpx
from pydantic import BaseModel, SecretStr

from agent.config import Settings
from agent.safety.untrusted import wrap_untrusted

log = logging.getLogger(__name__)

MAX_QUERY_CHARS = 200
MAX_HITS = 5
MAX_SNIPPET_CHARS = 500
TAINT_MIN_CHARS = 4  # shorter tainted values (e.g. "USD", ids) would block normal queries

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_LONG_DIGITS = re.compile(r"\d[\d\s-]{5,}\d")
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")


class SearchHit(BaseModel):
    title: str
    url: str
    snippet: str


class WebSearchResult(BaseModel):
    ok: bool
    query: str
    hits: list[SearchHit] = []
    error: str | None = None

    def to_llm_text(self) -> str:
        if not self.ok:
            return f"Web search error: {self.error}"
        body = "\n\n".join(
            f"[{i}] {h.title}\n{h.url}\n{h.snippet}" for i, h in enumerate(self.hits, 1)
        )
        return "EXTERNAL web results (not company data; cite as external):\n" + wrap_untrusted(
            "web", body or "no results"
        )


class SearchBackend(Protocol):
    def search(self, query: str, max_results: int) -> list[SearchHit]: ...


class TavilyBackend:
    URL = "https://api.tavily.com/search"

    def __init__(self, api_key: SecretStr, timeout_s: float = 10.0) -> None:
        self._api_key = api_key
        self._timeout_s = timeout_s

    def search(self, query: str, max_results: int) -> list[SearchHit]:
        resp = httpx.post(
            self.URL,
            json={"query": query, "max_results": max_results, "search_depth": "basic"},
            headers={"Authorization": f"Bearer {self._api_key.get_secret_value()}"},
            timeout=self._timeout_s,
        )
        resp.raise_for_status()
        return [
            SearchHit(
                title=str(r.get("title", "")),
                url=str(r.get("url", "")),
                snippet=str(r.get("content", "")),
            )
            for r in resp.json().get("results", [])
        ]


class QueryRejected(ValueError):
    pass


def check_query(query: str, tainted: Iterable[str] = ()) -> str:
    q = " ".join(query.split())
    if not q:
        raise QueryRejected("empty query")
    if len(q) > MAX_QUERY_CHARS:
        raise QueryRejected(f"query longer than {MAX_QUERY_CHARS} characters")
    if _CONTROL.search(query.replace("\n", " ").replace("\t", " ")):
        raise QueryRejected("query contains control characters")
    if _EMAIL.search(q):
        raise QueryRejected("queries may not contain email addresses")
    if _LONG_DIGITS.search(q):
        raise QueryRejected("queries may not contain long numbers (ids, phones, card numbers)")
    lowered = q.casefold()
    for value in tainted:
        v = str(value).strip().casefold()
        if len(v) >= TAINT_MIN_CHARS and v in lowered:
            raise QueryRejected("queries may not contain values taken from company data")
    return q


class WebSearchTool:
    def __init__(self, backend: SearchBackend | None) -> None:
        self.backend = backend

    @property
    def enabled(self) -> bool:
        return self.backend is not None

    def run(self, query: str, *, tainted: Iterable[str] = ()) -> WebSearchResult:
        if self.backend is None:
            return WebSearchResult(ok=False, query=query, error="web search is not configured")
        try:
            q = check_query(query, tainted)
        except QueryRejected as exc:
            log.info("web_search_rejected", extra={"reason": str(exc)})
            return WebSearchResult(ok=False, query=query, error=f"rejected: {exc}")
        try:
            raw_hits = self.backend.search(q, MAX_HITS)
        except Exception as exc:  # any backend failure becomes a generic, key-free error
            log.warning("web_search_failed", extra={"error_type": type(exc).__name__})
            return WebSearchResult(ok=False, query=q, error="web search failed")
        hits = [
            SearchHit(title=h.title[:200], url=h.url[:500], snippet=h.snippet[:MAX_SNIPPET_CHARS])
            for h in raw_hits[:MAX_HITS]
            if h.url.startswith(("https://", "http://"))
        ]
        log.info("web_search_ok", extra={"hits": len(hits)})
        return WebSearchResult(ok=True, query=q, hits=hits)


def build_web_search(settings: Settings) -> WebSearchTool:
    key = settings.tavily_api_key
    return WebSearchTool(TavilyBackend(key) if key and key.get_secret_value() else None)
