"""Critic rule checks, one per failure mode, plus the cases that must NOT fail."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from agent.deps import AgentDeps
from agent.nodes.critic import rule_check
from agent.state import PlanStep, StepResult
from tests.agent_fakes import ScriptedLLM, make_deps


@pytest.fixture(scope="module")
def deps(shop_db_path: Path) -> AgentDeps:
    return make_deps(shop_db_path, ScriptedLLM([]))


def sql_step(expected: str = "rows", sql: str = "SELECT 1") -> PlanStep:
    return PlanStep(id=1, tool="sql", goal="get data", expected_output=expected, sql=sql)


def sql_result(columns: list[str], rows: list[list[Any]], **kw: Any) -> StepResult:
    return StepResult(
        step_id=1, tool="sql", ok=True, columns=columns, rows=rows, row_count=len(rows), **kw
    )


def test_missing_column_fix_lists_real_columns(deps: AgentDeps) -> None:
    step = sql_step(sql="SELECT ordered_at FROM orders")
    result = StepResult(step_id=1, tool="sql", ok=False, error="no such column: ordered_at")
    reason, fix = rule_check(step, result, deps) or ("", None)
    assert "no such column" in reason
    assert fix is not None and "orders(order_id, customer_id, order_ts" in fix
    assert "customers(" not in fix  # only the tables the query used


def test_other_errors_fail_without_fix(deps: AgentDeps) -> None:
    result = StepResult(step_id=1, tool="sql", ok=False, error="rejected by SQL guard: DROP")
    assert rule_check(sql_step(), result, deps) == (
        "sql step failed: rejected by SQL guard: DROP",
        None,
    )


@pytest.mark.parametrize(
    ("result", "needle"),
    [
        (sql_result(["x"], []), "no rows"),
        (sql_result(["x"], [[1]], truncated=True), "row cap"),
        (sql_result(["revenue"], [[None]]), "entirely NULL"),
        (sql_result(["v"], [[None], [None], [None], [1]]), "mostly NULL"),
        (sql_result(["customers"], [[-3]]), "negative"),
        (sql_result(["refund_rate"], [[140.0]]), "outside 0-100"),
        (sql_result(["spend"], [[2.7e13]]), "implausible magnitude"),
    ],
)
def test_rule_failures(deps: AgentDeps, result: StepResult, needle: str) -> None:
    failure = rule_check(sql_step(), result, deps)
    assert failure is not None and needle in failure[0]


def test_one_row_expectation(deps: AgentDeps) -> None:
    step = sql_step(expected="One row: region, delta_pp")
    assert rule_check(step, sql_result(["region"], [["a"], ["b"]]), deps) is not None
    assert rule_check(step, sql_result(["region"], [["a"]]), deps) is None


@pytest.mark.parametrize(
    "result",
    [
        sql_result(["revenue"], [[1783489.61]]),
        sql_result(["growth_pct"], [[136.2]]),  # changes may exceed 100
        sql_result(["delta_customers"], [[-40]]),  # and may be negative
        sql_result(["refund_rate"], [[0.179]]),  # fractions are fine
        sql_result(["net_revenue"], [[-120.5], [300.0]]),  # revenue is not a count
        sql_result(["region", "v"], [["a", None], ["b", 1], ["c", 2], ["d", 3]]),  # 1/4 NULL
    ],
)
def test_plausible_results_pass(deps: AgentDeps, result: StepResult) -> None:
    assert rule_check(sql_step(), result, deps) is None


def test_python_without_result(deps: AgentDeps) -> None:
    step = PlanStep(
        id=2, tool="python", goal="compute", expected_output="x", code="x = 1", inputs=[1]
    )
    result = StepResult(step_id=2, tool="python", ok=True, value=None)
    failure = rule_check(step, result, deps)
    assert failure is not None and "result" in failure[0]


def test_python_table_result_is_checked_like_sql(deps: AgentDeps) -> None:
    step = PlanStep(id=2, tool="python", goal="compute", expected_output="x", code="x", inputs=[1])
    value = {"columns": ["share_pct"], "rows": [[250.0]], "row_count": 1}
    failure = rule_check(step, StepResult(step_id=2, tool="python", ok=True, value=value), deps)
    assert failure is not None and "outside 0-100" in failure[0]


@pytest.mark.parametrize(
    "result",
    [
        sql_result(["refund_rate"], [[100.0]]),  # live baseline multi_02: refunds over refunds
        sql_result(["share_pct"], [[100]]),
        sql_result(["refund_rate"], [[1.0], [0.4]]),  # fractions: 1.0 is the 100% case
    ],
)
def test_a_rate_of_exactly_100_percent_is_suspicious(deps: AgentDeps, result: StepResult) -> None:
    failure = rule_check(sql_step(), result, deps)
    assert failure is not None and "exactly 100%" in failure[0]


@pytest.mark.parametrize(
    "result",
    [
        sql_result(["refund_rate"], [[17.9]]),
        sql_result(["refund_rate"], [[99.9]]),
        sql_result(["growth_pct"], [[100.0]]),  # a change can be +100%
        sql_result(["refund_rate"], [[1.0], [17.9]]),  # 1% among percentages is fine
    ],
)
def test_ordinary_rates_pass(deps: AgentDeps, result: StepResult) -> None:
    assert rule_check(sql_step(), result, deps) is None


def test_semantic_rules_run_on_successful_sql_steps(deps: AgentDeps) -> None:
    sql = "SELECT COUNT(DISTINCT o.customer_id) AS buyers FROM orders o"
    result = sql_result(["buyers"], [[7925]], sql=sql)
    failure = rule_check(sql_step(sql=sql), result, deps)
    assert failure is not None and "accounts" in failure[0]
