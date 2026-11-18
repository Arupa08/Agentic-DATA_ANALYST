"""The Phase 4 graph end to end, offline: a scripted model stands in for the LLM, and
the real tools run against the real (test-built) database."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from agent.graph import ERROR_ANSWER, MAX_QUESTION_CHARS, Agent
from agent.llm import InvalidModelOutput
from agent.nodes.synthesizer import PLAN_FAILED_ANSWER
from agent.state import Plan, SynthesisOutput
from tests.agent_fakes import ScriptedLLM, make_deps

Q3_SQL = """
SELECT ROUND(SUM(oi.line_total), 2) AS revenue
FROM order_items oi JOIN orders o ON o.order_id = oi.order_id
WHERE o.status = 'completed' AND o.order_ts >= '2025-07-01' AND o.order_ts < '2025-10-01'
"""


def sql_plan(sql: str = Q3_SQL, **kw: Any) -> Plan:
    step = {
        "id": 1,
        "tool": "sql",
        "goal": "Q3 2025 net revenue",
        "expected_output": "1 row",
        "sql": sql,
    }
    return Plan.model_validate({"outcome": "plan", "steps": [step], **kw})


ANSWER = SynthesisOutput(answer="Q3 2025 revenue was $1,783,489.61.", caveats=["net of refunds"])


def test_happy_path_single_sql(shop_db_path: Path) -> None:
    llm = ScriptedLLM([sql_plan(), ANSWER])
    r = Agent(make_deps(shop_db_path, llm)).ask(
        "What was total revenue in Q3 2025?", case_id="simple_01"
    )

    assert r.status == "answered"
    assert r.answer == ANSWER.answer
    assert r.caveats == ["net of refunds"]
    assert len(r.sql) == 1 and r.sql[0].endswith("LIMIT 5001")  # the guarded SQL that ran
    assert r.steps[0].ok and r.steps[0].row_count == 1
    assert [c[0] for c in llm.calls] == ["cheap", "strong"]  # planner cheap, synthesizer strong
    assert r.cost_usd > 0 and r.total_tokens == 2_000 and r.latency_ms > 0

    planner_prompt, synth_prompt = llm.prompt_text(0), llm.prompt_text(1)
    assert "CREATE TABLE order_items" in planner_prompt
    assert "Business glossary" in planner_prompt
    assert "web_search" not in planner_prompt  # disabled tools aren't offered
    assert '<untrusted source="database">' in synth_prompt
    assert "1783489.61" in synth_prompt


@pytest.mark.usefixtures("warm_sandbox")
def test_sql_python_chart_pipeline(shop_db_path: Path) -> None:
    plan = Plan.model_validate(
        {
            "outcome": "plan",
            "assumptions": ["2025 only"],
            "steps": [
                {
                    "id": 1,
                    "tool": "sql",
                    "goal": "net revenue by quarter",
                    "expected_output": "4 rows: quarter, revenue",
                    "sql": "SELECT (CAST(strftime('%m', o.order_ts) AS INTEGER) + 2) / 3 "
                    "AS quarter, SUM(oi.line_total) AS revenue FROM order_items oi JOIN orders o "
                    "ON o.order_id = oi.order_id WHERE o.status = 'completed' "
                    "AND o.order_ts >= '2025-01-01' AND o.order_ts < '2026-01-01' "
                    "GROUP BY quarter",
                },
                {
                    "id": 2,
                    "tool": "python",
                    "goal": "Q4 share of the year",
                    "expected_output": "a percentage",
                    "code": "r = step_1.revenue\nresult = round(100 * r.iloc[-1] / r.sum(), 1)",
                    "inputs": [1],
                },
                {
                    "id": 3,
                    "tool": "chart",
                    "goal": "plot quarters",
                    "expected_output": "bar chart",
                    "chart": {
                        "kind": "bar",
                        "x": "quarter",
                        "y": ["revenue"],
                        "title": "2025 revenue",
                    },
                    "inputs": [1],
                },
            ],
        }
    )
    llm = ScriptedLLM([plan, ANSWER])
    r = Agent(make_deps(shop_db_path, llm)).ask("How much of 2025 revenue came in Q4?")
    assert r.status == "answered", [s.error for s in r.steps]
    assert r.assumptions == ["2025 only"]
    assert len(r.charts) == 1 and r.charts[0].png.startswith(b"\x89PNG")
    synth_prompt = llm.prompt_text(1)
    assert "31.9" in synth_prompt  # python result reached the synthesizer
    assert "Chart rendered" in synth_prompt


def test_declined_plan_skips_tools_and_synthesis_model(shop_db_path: Path) -> None:
    declined = Plan(outcome="decline", message="The data has no NPS survey results.")
    llm = ScriptedLLM([declined])
    r = Agent(make_deps(shop_db_path, llm)).ask("What was our NPS last year?")
    assert r.status == "declined"
    assert r.answer == "The data has no NPS survey results."
    assert r.steps == [] and r.sql == []
    assert len(llm.calls) == 1


def test_invalid_plan_is_retried_once_with_the_reason(shop_db_path: Path) -> None:
    llm = ScriptedLLM([InvalidModelOutput("Plan output rejected: bad step"), sql_plan(), ANSWER])
    r = Agent(make_deps(shop_db_path, llm)).ask("Revenue in Q3 2025?")
    assert r.status == "answered"
    assert "Your plan was rejected: Plan output rejected: bad step" in llm.prompt_text(1)


def test_two_invalid_plans_fail_gracefully(shop_db_path: Path) -> None:
    llm = ScriptedLLM([InvalidModelOutput("x"), InvalidModelOutput("y")])
    r = Agent(make_deps(shop_db_path, llm)).ask("Revenue in Q3 2025?")
    assert r.status == "failed"
    assert r.answer == PLAN_FAILED_ANSWER
    assert len(llm.calls) == 2


def test_web_step_rejected_when_web_disabled(shop_db_path: Path) -> None:
    web_plan = Plan.model_validate(
        {
            "outcome": "plan",
            "steps": [
                {
                    "id": 1,
                    "tool": "web_search",
                    "goal": "get data",
                    "expected_output": "x",
                    "query": "holidays",
                }
            ],
        }
    )
    llm = ScriptedLLM([web_plan, sql_plan(), ANSWER])
    r = Agent(make_deps(shop_db_path, llm)).ask("Revenue in Q3 2025?")
    assert r.status == "answered"
    assert "web_search, which is not available" in llm.prompt_text(1)


@pytest.mark.usefixtures("warm_sandbox")
def test_failed_step_stops_execution_and_replans(shop_db_path: Path) -> None:
    plan = Plan.model_validate(
        {
            "outcome": "plan",
            "steps": [
                {
                    "id": 1,
                    "tool": "sql",
                    "goal": "get data",
                    "expected_output": "x",
                    "sql": "SELECT revenue FROM orders",
                },
                {
                    "id": 2,
                    "tool": "python",
                    "goal": "get data",
                    "expected_output": "x",
                    "code": "result = 1",
                    "inputs": [1],
                },
            ],
        }
    )
    llm = ScriptedLLM([plan, sql_plan(), ANSWER])
    r = Agent(make_deps(shop_db_path, llm)).ask("Revenue?")
    # step 1 failed, so step 2 never ran; the critic sent it back for a replan that worked
    assert r.status == "answered" and r.replans == 1
    assert r.attempts[0].failed_step_id == 1 and "revenue" in r.attempts[0].reason
    assert [v.passed for v in r.verdicts] == [False, True]
    assert "Rejected at step 1 by a rule check" in llm.prompt_text(1)


def test_glossary_can_be_switched_off(shop_db_path: Path) -> None:
    llm = ScriptedLLM([sql_plan(), ANSWER])
    Agent(make_deps(shop_db_path, llm, use_glossary=False)).ask("Revenue in Q3 2025?")
    assert "Business glossary" not in llm.prompt_text(0)


def test_cost_cap_stops_further_model_calls(shop_db_path: Path) -> None:
    # one planner call costs 100k gpt-4o input tokens = $0.25, which hits a $0.20 cap
    llm = ScriptedLLM([sql_plan(), ANSWER], max_cost_usd=0.20, tokens_per_call=100_000)
    r = Agent(make_deps(shop_db_path, llm)).ask("Revenue in Q3 2025?")
    assert r.status == "budget_exceeded"
    assert len(llm.calls) == 1  # synthesizer never called
    assert r.cost_usd == pytest.approx(0.25)


def test_synthesis_rejection_fails_gracefully(shop_db_path: Path) -> None:
    llm = ScriptedLLM([sql_plan(), InvalidModelOutput("bad synthesis")])
    r = Agent(make_deps(shop_db_path, llm)).ask("Revenue in Q3 2025?")
    assert r.status == "failed" and r.answer == ERROR_ANSWER
    # what did run is still reported, so the failure is traceable
    assert len(r.sql) == 1 and r.steps[0].ok


@pytest.mark.parametrize("question", ["", "   ", "x" * (MAX_QUESTION_CHARS + 1)])
def test_question_validation(shop_db_path: Path, question: str) -> None:
    with pytest.raises(ValueError):
        Agent(make_deps(shop_db_path, ScriptedLLM([]))).ask(question)
