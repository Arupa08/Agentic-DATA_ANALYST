"""Malicious requests against POST /ask: none reach the agent, none echo input back."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from api.config import APISettings
from api.main import create_app

ESCALATE = b'{"question": "hi", "tools": ["python"], "max_replans": 99}'
DUPLICATE_KEY = b'{"question": "hi", "question": "' + b"A" * 2_000 + b'"}'


class RecordingAgent:
    def __init__(self) -> None:
        self.questions: list[str] = []

    def ask(self, question: str, *, tags: tuple[str, ...] = ()) -> Any:
        self.questions.append(question)
        raise AssertionError("a malicious request reached the agent")


@pytest.fixture
def setup() -> tuple[TestClient, RecordingAgent]:
    agent = RecordingAgent()
    app = create_app(agent, APISettings(_env_file=None))  # type: ignore[call-arg]
    return TestClient(app, raise_server_exceptions=False), agent


@pytest.mark.parametrize(
    ("body", "headers"),
    [
        (b'{"question": "hi"', {"Content-Type": "application/json"}),  # truncated JSON
        (b"question=hi", {"Content-Type": "application/x-www-form-urlencoded"}),
        (b'{"question": ["DROP TABLE orders"]}', {"Content-Type": "application/json"}),
        (b'{"question": {"$gt": ""}}', {"Content-Type": "application/json"}),
        # trying to widen what the agent may do: tool permissions are fixed in code
        (ESCALATE, {"Content-Type": "application/json"}),
        (DUPLICATE_KEY, {"Content-Type": "application/json"}),  # the last duplicate wins
    ],
)
def test_malformed_and_escalating_bodies_never_reach_the_agent(
    setup: tuple[TestClient, RecordingAgent], body: bytes, headers: dict[str, str]
) -> None:
    c, agent = setup
    r = c.post("/ask", content=body, headers=headers)
    assert r.status_code in (413, 422)
    assert "DROP" not in r.text and "$gt" not in r.text and "A" * 100 not in r.text
    assert agent.questions == []


@pytest.mark.parametrize("length", ["-1", "abc", "99999999999999999999"])
def test_lying_content_length_is_refused(
    setup: tuple[TestClient, RecordingAgent], length: str
) -> None:
    c, agent = setup
    r = c.post(
        "/ask",
        content=b'{"question": "hi"}',
        headers={"Content-Type": "application/json", "Content-Length": length},
    )
    assert r.status_code in (400, 413)
    assert agent.questions == []


def test_only_get_and_post_routes_exist(setup: tuple[TestClient, RecordingAgent]) -> None:
    c, _ = setup
    assert c.delete("/ask").status_code == 405
    assert c.put("/ask", json={"question": "hi"}).status_code == 405
    assert c.get("/../../etc/passwd").status_code == 404
