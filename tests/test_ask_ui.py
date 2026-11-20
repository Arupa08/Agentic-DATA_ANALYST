"""The "Ask the agent" page: its helpers, the agent's progress callbacks, and the page
itself driven through Streamlit's AppTest with a scripted (offline) agent."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest

from agent.config import Settings
from agent.graph import Agent
from tests.agent_fakes import ScriptedLLM, make_deps
from tests.test_agent_graph import ANSWER, sql_plan
from ui import ask_support as support

ROOT = Path(__file__).resolve().parents[1]


def _settings(**kw: Any) -> Settings:
    base: dict[str, Any] = {
        "_env_file": None,
        "openai_api_key": "sk-test-not-real",
        "langsmith_tracing": False,
        **kw,
    }
    return Settings(**base)


# ---------------------------------------------------------------- helpers


def test_recorder_keeps_only_graph_nodes() -> None:
    rec = support.NodeRecorder()
    for name in ("nl_agent", "planner", "RunnableSequence", "executor", "critic", "synthesizer"):
        rec.on_chain_start({}, {}, run_id=None, name=name)  # type: ignore[arg-type]
    assert [e.node for e in rec.events] == ["planner", "executor", "critic", "synthesizer"]


def test_timeline_numbers_replans() -> None:
    t0 = 100.0
    events = [
        support.NodeEvent("planner", 100.5),
        support.NodeEvent("executor", 101.0),
        support.NodeEvent("critic", 102.0),
        support.NodeEvent("planner", 103.25),
    ]
    lines = support.timeline(events, t0)
    assert lines[0] == "  0.5s  Planning"
    assert lines[3].startswith("  3.2s  Replanning (attempt 2)")


def test_model_summary_shows_names_never_keys() -> None:
    s = _settings(cheap_model="openai:gpt-4o-mini", strong_model="openai:gpt-4o")
    text = "\n".join(support.model_summary(s))
    assert "gpt-4o-mini" in text and "gpt-4o" in text
    assert "sk-test" not in text
    same = _settings(cheap_model="openai:gpt-4o", strong_model="openai:gpt-4o")
    assert support.model_summary(same) == ["Model: openai:gpt-4o"]


def test_local_model_reachable(monkeypatch: pytest.MonkeyPatch) -> None:
    assert support.local_model_reachable(_settings(cheap_model="openai:gpt-4o")) is None

    local = _settings(
        cheap_model="ollama:qwen3:4b-instruct", strong_model="ollama:qwen3:4b-instruct"
    )

    def down(*_: Any, **__: Any) -> httpx.Response:
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(support.httpx, "get", down)
    assert support.local_model_reachable(local) is False
    monkeypatch.setattr(support.httpx, "get", lambda *a, **k: httpx.Response(200))
    assert support.local_model_reachable(local) is True


def test_extra_callbacks_see_every_node(shop_db_path: Path) -> None:
    rec = support.NodeRecorder()
    agent = Agent(make_deps(shop_db_path, ScriptedLLM([sql_plan(), ANSWER])))
    r = agent.ask("What was total revenue in Q3 2025?", extra_callbacks=[rec])
    assert r.status == "answered"
    assert [e.node for e in rec.events] == ["planner", "executor", "critic", "synthesizer"]


# ---------------------------------------------------------------- the page

testing = pytest.importorskip("streamlit.testing.v1")


class Boom:
    def ask(self, *_: Any, **__: Any) -> Any:
        raise RuntimeError("secret internal detail")


@pytest.fixture
def page(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Patch the page's agent factory, settings and model check; return a runner."""
    import streamlit as st

    import agent.config

    st.cache_resource.clear()
    settings = _settings()
    monkeypatch.setattr(agent.config, "get_settings", lambda: settings)
    monkeypatch.setattr(support, "local_model_reachable", lambda _s: None)

    def run(agent_obj: Any) -> Any:
        monkeypatch.setattr(support, "make_agent", lambda: agent_obj)
        at = testing.AppTest.from_file(str(ROOT / "ui" / "ask_page.py"), default_timeout=60)
        return at.run()

    yield run
    st.cache_resource.clear()


def _all_text(at: Any) -> str:
    parts = [e.value for e in at.text] + [e.value for e in at.markdown]
    parts += [e.value for e in at.error] + [e.value for e in at.warning]
    return "\n".join(str(p) for p in parts)


def test_page_empty_state(page: Any) -> None:
    at = page(Boom())
    assert not at.exception
    assert at.title[0].value.startswith("💬 Ask the agent")
    assert len([b for b in at.button if b.key and b.key.startswith("example_")]) == len(
        support.EXAMPLES
    )


def test_page_answers_an_example(page: Any, shop_db_path: Path) -> None:
    agent = Agent(make_deps(shop_db_path, ScriptedLLM([sql_plan(), ANSWER])))
    at = page(agent)
    at.button(key="example_0").click().run()

    assert not at.exception
    text = _all_text(at)
    assert ANSWER.answer in text
    assert "✅ Answered" in text
    assert "Planning" in "\n".join(c.value for c in at.code)
    assert any("SELECT" in c.value for c in at.code)  # the SQL that ran
    assert at.text_area(key="question_box").value == support.EXAMPLES[0][1]
    assert [m.label for m in at.metric][:2] == ["Time", "Tokens"]


def test_page_hides_internal_errors(page: Any) -> None:
    at = page(Boom())
    at.text_area(key="question_box").input("What was revenue?")
    next(b for b in at.button if b.label == "Ask").click().run()  # the form's submit

    assert not at.exception
    assert [e.value for e in at.error] == ["Something went wrong. Please try again."]
    assert "secret internal detail" not in _all_text(at)


def test_page_uses_the_api_when_configured(page: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    from agent.graph import AgentResponse

    monkeypatch.setenv(support.API_URL_ENV, "http://api:8000")
    answer = AgentResponse(
        request_id="r1", question="q", status="answered", answer="Remote answer: 42."
    ).model_dump_json()
    asked: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        asked.append(request.content.decode())
        return httpx.Response(200, content=answer)

    remote = support.RemoteAgent(
        "http://api:8000", client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    at = page(remote)
    assert "🟢 The agent API is up" in [c.value for c in at.caption]
    at.button(key="example_0").click().run()

    assert not at.exception
    assert "Remote answer: 42." in _all_text(at)
    assert len(asked) == 1 and support.EXAMPLES[0][1] in asked[0]
