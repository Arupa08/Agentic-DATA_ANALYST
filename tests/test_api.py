"""POST /ask over HTTP: the response contract and the controls a public URL needs
(validation, size limit, rate limit, concurrency cap, CORS, headers, generic errors)."""

from __future__ import annotations

import base64
import json
import threading
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from agent.graph import MAX_QUESTION_CHARS, Agent, AgentResponse, Chart
from agent.llm import RateLimited
from api.config import APISettings
from api.main import SECURITY_HEADERS, create_app
from tests.agent_fakes import ScriptedLLM, make_deps
from tests.test_agent_graph import ANSWER, sql_plan
from ui.ask_support import RemoteAgent, RemoteBusy

PNG = b"\x89PNG\r\n\x1a\nnot-a-real-image"


class FakeAgent:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.questions: list[str] = []

    def ask(self, question: str, *, tags: tuple[str, ...] = ()) -> AgentResponse:
        self.questions.append(question)
        if self.error is not None:
            raise self.error
        return AgentResponse(
            request_id="abc123",
            question=question,
            status="answered",
            answer="Q3 2025 revenue was $1,783,489.61.",
            sql=["SELECT 1 LIMIT 5001"],
            charts=[Chart(png=PNG, alt_text="a bar chart")],
            cost_usd=0.0012,
            trace_id="run-1",
        )


def client(agent: Any = None, **settings: Any) -> TestClient:
    api = APISettings(_env_file=None, **settings)  # type: ignore[call-arg]
    app = create_app(agent if agent is not None else FakeAgent(), api)
    return TestClient(app, raise_server_exceptions=False)


def test_ask_returns_answer_sql_chart_cost_and_trace() -> None:
    with client() as c:
        r = c.post("/ask", json={"question": "What was total revenue in Q3 2025?"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"].startswith("Q3 2025 revenue")
    assert body["sql"] == ["SELECT 1 LIMIT 5001"]
    assert body["cost_usd"] == 0.0012 and body["trace_id"] == "run-1"
    assert base64.b64decode(body["charts"][0]["png"]) == PNG  # PNG travels as base64
    assert AgentResponse.model_validate_json(r.content).charts[0].png == PNG  # and back


def test_real_agent_end_to_end_over_http(shop_db_path: Path) -> None:
    agent = Agent(make_deps(shop_db_path, ScriptedLLM([sql_plan(), ANSWER])))
    with client(agent) as c:
        r = c.post("/ask", json={"question": "What was total revenue in Q3 2025?"})
    assert r.status_code == 200 and r.json()["status"] == "answered"
    assert r.json()["sql"][0].endswith("LIMIT 5001")


def test_health() -> None:
    with client() as c:
        assert c.get("/health").json() == {"status": "ok"}


@pytest.mark.parametrize(
    "payload",
    [
        {"question": ""},
        {"question": "x" * (MAX_QUESTION_CHARS + 1)},
        {"question": 42},  # strict: no coercion
        {"question": "ok", "tools": ["python"]},  # extra fields are refused
        {},
    ],
)
def test_invalid_questions_are_rejected_without_echo(payload: dict[str, Any]) -> None:
    agent = FakeAgent()
    with client(agent) as c:
        r = c.post("/ask", json=payload)
    assert r.status_code == 422 and r.json()["error"] == "invalid request"
    assert '"input"' not in r.text and "x" * 50 not in r.text  # the value is never echoed
    assert agent.questions == []


def test_oversized_body_is_refused_before_it_is_read() -> None:
    agent = FakeAgent()
    with client(agent, max_body_bytes=1024) as c:
        big = json.dumps({"question": "x" * 5_000})
        r = c.post("/ask", content=big, headers={"Content-Type": "application/json"})
        chunked = c.post(  # no Content-Length: counted as it streams in
            "/ask",
            content=iter([big[:600].encode(), big[600:].encode()]),
            headers={"Content-Type": "application/json"},
        )
    assert r.status_code == 413 and r.json() == {"error": "request too large"}
    assert chunked.status_code == 413
    assert agent.questions == []


def test_rate_limit_per_client() -> None:
    with client(rate_limit="2/minute") as c:
        codes = [c.post("/ask", json={"question": "hi"}).status_code for _ in range(3)]
        r = c.post("/ask", json={"question": "hi"})
    assert codes == [200, 200, 429]
    assert r.json()["error"].startswith("too many requests") and r.headers["Retry-After"]


def test_concurrency_cap_turns_extra_requests_away() -> None:
    release = threading.Event()
    started = threading.Event()

    class Slow(FakeAgent):
        def ask(self, question: str, *, tags: tuple[str, ...] = ()) -> AgentResponse:
            started.set()
            release.wait(5)
            return super().ask(question, tags=tags)

    with client(Slow(), max_concurrent=1) as c:
        first: dict[str, int] = {}
        t = threading.Thread(
            target=lambda: first.update(code=c.post("/ask", json={"question": "a"}).status_code)
        )
        t.start()
        assert started.wait(5)
        busy = c.post("/ask", json={"question": "b"})
        release.set()
        t.join(5)
    assert busy.status_code == 503 and busy.headers["Retry-After"] == "30"
    assert first["code"] == 200


def test_provider_quota_is_503_and_internal_errors_are_generic() -> None:
    with client(FakeAgent(RateLimited("quota"))) as c:
        quota = c.post("/ask", json={"question": "hi"})
    with client(FakeAgent(RuntimeError("db at /secret/path exploded"))) as c:
        boom = c.post("/ask", json={"question": "hi"})
    assert quota.status_code == 503 and "rate-limited" in quota.json()["error"]
    assert boom.status_code == 500 and boom.json()["error"] == "internal error"
    assert "secret" not in boom.text and "Traceback" not in boom.text


def test_security_headers_on_every_response() -> None:
    with client(max_body_bytes=256) as c:
        responses = [
            c.get("/health"),
            c.post("/ask", json={"question": "hi"}),
            c.post("/ask", json={"question": ""}),
            c.post("/ask", content=b"x" * 1_000, headers={"Content-Type": "application/json"}),
            c.get("/nope"),
        ]
    for r in responses:
        for name, value in SECURITY_HEADERS.items():
            assert r.headers.get(name) == value, (r.request.url, name)
        assert "Strict-Transport-Security" not in r.headers  # only over https


def test_cors_allows_only_the_ui_origin() -> None:
    with client(cors_origins=["https://ui.example"]) as c:
        ok = c.options(
            "/ask",
            headers={"Origin": "https://ui.example", "Access-Control-Request-Method": "POST"},
        )
        bad = c.options(
            "/ask",
            headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"},
        )
    assert ok.headers.get("access-control-allow-origin") == "https://ui.example"
    assert "access-control-allow-origin" not in bad.headers


def test_docs_can_be_switched_off() -> None:
    with client(docs=False) as c:
        assert c.get("/docs").status_code == 404 and c.get("/openapi.json").status_code == 404


def test_missing_model_key_refuses_to_start() -> None:
    def factory() -> Any:
        raise ValueError("missing required settings: OPENAI_API_KEY")

    app = create_app(None, APISettings(_env_file=None), agent_factory=factory)  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="missing required settings"), TestClient(app):
        pass


# ---------------------------------------------------------------- the UI's HTTP client


def _remote(handler: Any) -> RemoteAgent:
    return RemoteAgent("http://api", client=httpx.Client(transport=httpx.MockTransport(handler)))


def test_remote_agent_round_trips_the_response_and_forwards_the_client_ip() -> None:
    seen: dict[str, Any] = {}
    payload = FakeAgent().ask("q").model_dump_json()

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        seen["xff"] = request.headers.get("x-forwarded-for")
        return httpx.Response(200, content=payload)

    r = _remote(handler).ask("q", forwarded_for="203.0.113.7", tags=("ui",))
    assert r.answer.startswith("Q3 2025") and r.charts[0].png == PNG
    assert seen == {"body": {"question": "q"}, "xff": "203.0.113.7"}


@pytest.mark.parametrize(
    ("status", "error"),
    [(422, ValueError), (429, RemoteBusy), (503, RemoteBusy), (500, RuntimeError)],
)
def test_remote_agent_maps_errors(status: int, error: type[Exception]) -> None:
    with pytest.raises(error):
        _remote(lambda _: httpx.Response(status, json={"error": "x"})).ask("q")


def test_remote_agent_health() -> None:
    assert _remote(lambda _: httpx.Response(200, json={"status": "ok"})).healthy() is True

    def down(_: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    assert _remote(down).healthy() is False
