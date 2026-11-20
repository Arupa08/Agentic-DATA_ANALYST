"""Phase 5 exit checks, offline: injected failures recover with a *different* plan, the
loop is capped, and the trace shows the cycle. The model is scripted; the database,
tools, critic rules and graph are real."""

from __future__ import annotations

import shutil
import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage

from agent.config import Settings
from agent.graph import Agent
from agent.llm import InvalidModelOutput
from agent.state import CriticOutput, Plan, SynthesisOutput
from tests.agent_fakes import ScriptedLLM, make_deps

ANSWER = SynthesisOutput(answer="answer")


def sql_plan(sql: str, *, expected: str = "one row", assumptions: list[str] | None = None) -> Plan:
    return Plan.model_validate(
        {
            "outcome": "plan",
            "assumptions": assumptions or [],
            "steps": [
                {
                    "id": 1,
                    "tool": "sql",
                    "goal": "compute it",
                    "expected_output": expected,
                    "sql": sql,
                }
            ],
        }
    )


def q3_revenue(ts_column: str) -> str:
    return (
        "SELECT ROUND(SUM(oi.line_total), 2) AS revenue FROM order_items oi "
        "JOIN orders o ON o.order_id = oi.order_id WHERE o.status = 'completed' "
        f"AND o.{ts_column} >= '2025-07-01' AND o.{ts_column} < '2025-10-01'"
    )


SPEND_2025 = (
    "SELECT ROUND(SUM(ms.amount * fx.usd_rate), 2) AS spend_usd FROM marketing_spend ms "
    "JOIN fx_rates fx ON fx.currency = ms.currency WHERE ms.month >= '2025-01-01'"
)


@pytest.fixture
def mutated_db(shop_db_path: Path, tmp_path: Path) -> Callable[[str], Path]:
    """A private copy of the shop DB with one statement applied: injects a failure."""

    def make(statement: str) -> Path:
        path = tmp_path / "mutated.db"
        shutil.copy(shop_db_path, path)
        conn = sqlite3.connect(path)
        conn.execute(statement)
        conn.commit()
        conn.close()
        return path

    return make


def test_renamed_column_recovers_with_a_different_plan(mutated_db: Callable[[str], Path]) -> None:
    db = mutated_db("ALTER TABLE orders RENAME COLUMN order_ts TO ordered_at")
    first, second = sql_plan(q3_revenue("order_ts")), sql_plan(q3_revenue("ordered_at"))

    def replan(messages: list[BaseMessage]) -> Plan:
        prompt = "\n".join(str(m.content) for m in messages)
        # the planner is told which step failed, why, and which columns really exist
        assert "Rejected at step 1" in prompt and "no such column" in prompt
        assert "ordered_at" in prompt
        return second

    llm = ScriptedLLM([first, replan, ANSWER])
    r = Agent(make_deps(db, llm)).ask("What was total revenue in Q3 2025?")

    assert r.status == "answered"
    assert r.replans == 1 and len(r.attempts) == 1
    assert r.attempts[0].plan.fingerprint() != r.plan.fingerprint()  # type: ignore[union-attr]
    assert "ordered_at" in r.sql[0]
    assert [(v.passed, v.source) for v in r.verdicts] == [(False, "rule"), (True, "rule")]


def test_empty_table_ends_in_a_decline(mutated_db: Callable[[str], Path]) -> None:
    db = mutated_db("DELETE FROM marketing_spend")
    decline = Plan(outcome="decline", message="No marketing spend is recorded for 2025.")
    llm = ScriptedLLM([sql_plan(SPEND_2025), decline])
    r = Agent(make_deps(db, llm)).ask("How much did we spend on marketing in 2025?")
    assert r.status == "declined"
    assert r.answer == "No marketing spend is recorded for 2025."
    assert "entirely NULL" in r.attempts[0].reason


def test_loop_is_capped_and_explains_what_was_tried(mutated_db: Callable[[str], Path]) -> None:
    db = mutated_db("DELETE FROM marketing_spend")
    plans = [sql_plan(f"{SPEND_2025} AND ms.region_id <> {i}") for i in range(4)]
    llm = ScriptedLLM(list(plans))  # 1 plan + 3 replans, and no synthesizer call
    r = Agent(make_deps(db, llm)).ask("How much did we spend on marketing in 2025?")
    assert r.status == "retries_exhausted"
    assert r.replans == 3 and len(r.attempts) == 4
    assert [c[1] for c in llm.calls] == ["planner"] * 4
    assert "after 4 attempt(s)" in r.answer and r.answer.count("rejected because") == 4
    assert "$" not in r.answer  # no number slips into a give-up answer


def test_repeating_a_failed_plan_is_rejected_in_code(shop_db_path: Path) -> None:
    bad = sql_plan("SELECT nope FROM orders")
    good = sql_plan(q3_revenue("order_ts"))
    llm = ScriptedLLM([bad, bad, good, ANSWER])
    r = Agent(make_deps(shop_db_path, llm)).ask("What was total revenue in Q3 2025?")
    assert r.status == "answered" and r.replans == 1
    assert "plan repeats attempt 1" in llm.prompt_text(2)


def test_ambiguous_question_can_still_ask_after_one_nudge(shop_db_path: Path) -> None:
    ask = Plan(outcome="clarify", message="Do you mean revenue, churn, or support volume?")
    llm = ScriptedLLM([ask, ask])  # nudged once, still asks: the question stands
    r = Agent(make_deps(shop_db_path, llm, clarify_nudge=True)).ask("How are things going?")
    assert r.status == "needs_clarification"
    assert r.answer == "Do you mean revenue, churn, or support volume?"
    assert [c[1] for c in llm.calls] == ["planner", "planner_assume"]
    assert "Don't ask. Pick the most reasonable reading" in llm.prompt_text(1)


def test_needless_question_becomes_an_assumption(shop_db_path: Path) -> None:
    # live baseline: "How many support tickets were opened in Q4 2025?" was met with a question
    ask = Plan(outcome="clarify", message="Created or resolved in Q4 2025?")
    plan = sql_plan(q3_revenue("order_ts"), assumptions=["'opened' means created"])
    llm = ScriptedLLM([ask, plan, ANSWER])
    r = Agent(make_deps(shop_db_path, llm, clarify_nudge=True)).ask(
        "How many tickets were opened in Q4 2025?"
    )
    assert r.status == "answered" and r.assumptions == ["'opened' means created"]


def test_unusable_nudged_plan_keeps_the_original_question(shop_db_path: Path) -> None:
    ask = Plan(outcome="clarify", message="Which metric?")
    llm = ScriptedLLM([ask, InvalidModelOutput("junk")])
    r = Agent(make_deps(shop_db_path, llm, clarify_nudge=True)).ask("How are things going?")
    assert r.status == "needs_clarification" and r.answer == "Which metric?"


def test_nudge_is_off_by_default(shop_db_path: Path) -> None:
    # it cost underspecified cases 51 pp on the local model: asking is a valid answer there
    assert Settings(_env_file=None, openai_api_key="sk-test-not-real").clarify_nudge is False
    ask = Plan(outcome="clarify", message="Which metric?")
    llm = ScriptedLLM([ask])
    r = Agent(make_deps(shop_db_path, llm)).ask("How are things going?")
    assert r.status == "needs_clarification" and len(llm.calls) == 1


def test_ambiguous_question_can_state_an_assumption(shop_db_path: Path) -> None:
    plan = sql_plan(q3_revenue("order_ts"), assumptions=["'Doing' means net revenue, Q3 2025"])
    r = Agent(make_deps(shop_db_path, ScriptedLLM([plan, ANSWER]))).ask("How did Q3 do?")
    assert r.status == "answered"
    assert r.assumptions == ["'Doing' means net revenue, Q3 2025"]


NAIVE_REGIONS = (
    "SELECT r.region_name AS region, COUNT(*) AS customers FROM customers c "
    "JOIN regions r ON r.region_id = c.region_id GROUP BY r.region_name"
)
CANONICAL_REGIONS = (
    "SELECT r.region_name AS region, COUNT(DISTINCT LOWER(c.email)) AS customers "
    "FROM customers c JOIN regions r ON r.region_id = "
    "CASE c.region_id WHEN 6 THEN 1 WHEN 7 THEN 3 ELSE c.region_id END GROUP BY r.region_name"
)


def test_region_split_is_caught_by_a_rule_without_a_model_call(shop_db_path: Path) -> None:
    # live baseline: this grouping passed the critic and produced confident wrong answers
    llm = ScriptedLLM(
        [
            sql_plan(NAIVE_REGIONS, expected="rows: region, customers"),
            sql_plan(CANONICAL_REGIONS, expected="rows: region, customers"),
            CriticOutput(passed=True, reason="canonical regions"),
            ANSWER,
        ]
    )
    r = Agent(make_deps(shop_db_path, llm, critic_llm=True)).ask("Customers per region?")
    assert r.status == "answered" and r.replans == 1
    assert [(v.passed, v.source) for v in r.verdicts] == [(False, "rule"), (True, "llm")]
    assert "separate regions" in r.attempts[0].reason
    assert "counts customer rows, not people" in r.attempts[0].reason  # both issues at once
    assert "CASE c.region_id WHEN 6 THEN 1" in llm.prompt_text(1)  # the fix reached the replan
    assert [c[1] for c in llm.calls] == ["planner", "planner", "critic", "synthesizer"]


ALL_STATUSES_REVENUE = (
    "SELECT cat.category_name AS category, ROUND(SUM(oi.line_total), 2) AS revenue "
    "FROM order_items oi JOIN orders o ON o.order_id = oi.order_id "
    "JOIN products p ON p.product_id = oi.product_id "
    "JOIN categories cat ON cat.category_id = p.category_id GROUP BY cat.category_name"
)
COMPLETED_REVENUE = ALL_STATUSES_REVENUE.replace(
    "GROUP BY", "WHERE o.status = 'completed' GROUP BY"
)


def test_llm_critic_catches_what_rules_cannot(shop_db_path: Path) -> None:
    # no rule encodes "revenue excludes cancelled orders"; only the model check sees it
    reject = CriticOutput(
        passed=False,
        reason="revenue includes cancelled orders",
        fix="filter o.status = 'completed'",
    )
    llm = ScriptedLLM(
        [
            sql_plan(ALL_STATUSES_REVENUE, expected="rows: category, revenue"),
            reject,
            sql_plan(COMPLETED_REVENUE, expected="rows: category, revenue"),
            CriticOutput(passed=True, reason="completed orders only"),
            ANSWER,
        ]
    )
    r = Agent(make_deps(shop_db_path, llm, critic_llm=True)).ask("Revenue by category?")
    assert r.status == "answered" and r.replans == 1
    assert [(v.passed, v.source) for v in r.verdicts] == [(False, "llm"), (True, "llm")]
    critic_prompt = llm.prompt_text(1)
    assert "Electronics" in critic_prompt and '<untrusted source="database">' in critic_prompt
    assert "filter o.status = 'completed'" in llm.prompt_text(2)  # the fix reached the replan
    assert [c[0] for c in llm.calls] == ["cheap", "cheap", "cheap", "cheap", "strong"]


def test_unusable_critic_output_falls_back_to_rules(shop_db_path: Path) -> None:
    llm = ScriptedLLM([sql_plan(q3_revenue("order_ts")), InvalidModelOutput("junk"), ANSWER])
    r = Agent(make_deps(shop_db_path, llm, critic_llm=True)).ask("Q3 2025 revenue?")
    assert r.status == "answered"
    assert r.verdicts[0].passed and "invalid" in r.verdicts[0].reason


def test_tool_call_cap_stops_replanning(shop_db_path: Path) -> None:
    llm = ScriptedLLM([sql_plan("SELECT nope FROM orders")])
    r = Agent(make_deps(shop_db_path, llm, max_tool_calls=1)).ask("Q3 2025 revenue?")
    assert r.status == "retries_exhausted"
    assert len(llm.calls) == 1  # no replan once the tool budget is spent


class NodeRecorder(BaseCallbackHandler):
    """Stands in for the LangSmith tracer: records which graph nodes ran, in order."""

    NODES = frozenset({"planner", "executor", "critic", "synthesizer"})

    def __init__(self) -> None:
        self.nodes: list[str] = []

    def on_chain_start(
        self, serialized: dict[str, Any] | None, inputs: Any, *, run_id: UUID, **kwargs: Any
    ) -> None:
        name = kwargs.get("name")
        if name in self.NODES:
            self.nodes.append(name)


def test_trace_shows_the_replan_cycle(mutated_db: Callable[[str], Path]) -> None:
    db = mutated_db("ALTER TABLE orders RENAME COLUMN order_ts TO ordered_at")
    llm = ScriptedLLM(
        [sql_plan(q3_revenue("order_ts")), sql_plan(q3_revenue("ordered_at")), ANSWER]
    )
    recorder = NodeRecorder()
    Agent(make_deps(db, llm), tracer=recorder).ask("Q3 2025 revenue?", case_id="inject_rename")
    assert recorder.nodes == [
        "planner", "executor", "critic",  # first plan fails at the critic
        "planner", "executor", "critic",  # the replan passes
        "synthesizer",
    ]  # fmt: skip
