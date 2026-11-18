"""Critic: checks each step's result before anything is built on it.

Rule checks run first: deterministic, free, and they catch the common failures (errors,
empty or truncated results, NULL-dominated columns, implausible values). Only a step that
passes them gets one cheap-tier LLM check: does this result answer the step's goal?

A failed verdict records an Attempt; the graph then routes back to the planner with
every attempt so far, so the next plan has to be different.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from agent.deps import AgentDeps
from agent.llm import InvalidModelOutput
from agent.nodes.sql_rules import check_sql_semantics
from agent.prompts import raw, render
from agent.safety.sql_guard import referenced_tables
from agent.safety.untrusted import UNTRUSTED_POLICY, neutralize
from agent.state import (
    AgentState,
    Attempt,
    CriticOutput,
    NodeFn,
    PlanStep,
    StepResult,
    Verdict,
)

log = logging.getLogger(__name__)

NULL_DOMINANCE = 0.5
ATTEMPT_ROWS = 50
MAX_MAGNITUDE = 1e12
_RATE_NAME = re.compile(r"(pct|percent|rate|share|ratio)", re.IGNORECASE)
_CHANGE_NAME = re.compile(r"(growth|change|delta|diff|vs|yoy|qoq)", re.IGNORECASE)
_COUNT_NAME = re.compile(
    r"(^n$|count|^num_|customers|orders|units|tickets|subscriptions|people)", re.IGNORECASE
)
_ONE_ROW = re.compile(r"^\s*(one|1|a single|single) row", re.IGNORECASE)
_MISSING_NAME = re.compile(r"no such (column|table)", re.IGNORECASE)

RuleFailure = tuple[str, str | None]  # (reason, fix)


def _table_of(result: StepResult) -> tuple[list[str], list[list[Any]]] | None:
    if result.tool == "sql":
        return result.columns, result.rows
    value = result.value
    if isinstance(value, dict) and "columns" in value and "rows" in value:
        return list(value["columns"]), list(value["rows"])
    return None


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _check_values(columns: list[str], rows: list[list[Any]]) -> RuleFailure | None:
    for i, col in enumerate(columns):
        values = [row[i] for row in rows if i < len(row)]
        nulls = sum(v is None for v in values)
        if values and nulls == len(values):
            return (
                f"column {col!r} is entirely NULL",
                "the filter probably matched nothing: check date formats ('YYYY-MM-DD' text), "
                "status values and spellings against the schema's listed values",
            )
        if len(values) >= 4 and nulls / len(values) > NULL_DOMINANCE:
            return (f"column {col!r} is mostly NULL ({nulls}/{len(values)})", None)
        numbers = [n for v in values if (n := _number(v)) is not None]
        if not numbers:
            continue
        if any(abs(n) > MAX_MAGNITUDE for n in numbers):
            return (
                f"column {col!r} has an implausible magnitude ({max(numbers, key=abs):,.0f})",
                "check for unconverted currencies or a join that multiplies rows",
            )
        if _COUNT_NAME.search(col) and not _CHANGE_NAME.search(col) and min(numbers) < 0:
            return (f"count column {col!r} has a negative value", None)
        is_rate = _RATE_NAME.search(col) and not _CHANGE_NAME.search(col)
        if is_rate and (min(numbers) < 0 or max(numbers) > 100):
            return (f"rate column {col!r} is outside 0-100", None)
        top = max(numbers)
        if is_rate and (top == 100 or (top == 1 and max(abs(n) for n in numbers) <= 1)):
            # live baseline: refunds / sales computed over refund rows only gave "100%"
            return (
                f"rate column {col!r} is exactly 100%, which usually means numerator and "
                "denominator were computed over the same rows",
                "compute the denominator over the full population (e.g. all sold units), "
                "not only the rows that match the numerator",
            )
    return None


def rule_check(step: PlanStep, result: StepResult, deps: AgentDeps) -> RuleFailure | None:
    """Deterministic checks. Returns (reason, fix) for a failure, None to pass."""
    if not result.ok:
        error = result.error or "unknown error"
        reason = f"{step.tool} step failed: {error}"
        if step.tool == "sql" and _MISSING_NAME.search(error):
            real_columns = deps.sql.table_columns()
            used = sorted(referenced_tables(step.sql or ""))
            shown = [t for t in used if t in real_columns] or sorted(real_columns)
            real = "; ".join(f"{t}({', '.join(real_columns[t])})" for t in shown)
            return reason, f"use only columns that exist: {real}"
        if "timed out" in error:
            return reason, "aggregate in SQL and narrow the filters so the query is cheaper"
        return reason, None
    if step.tool == "python" and result.value is None:
        return "python step did not assign `result`", "assign the final value to `result`"
    table = _table_of(result)
    if table is None:
        return None
    columns, rows = table
    if result.truncated:
        return "result hit the row cap", "aggregate in SQL instead of returning raw rows"
    if not rows:
        return (
            "step returned no rows",
            "check the filters (date ranges as text, status values, spellings in the schema), "
            "or decline if the data is genuinely missing",
        )
    if _ONE_ROW.match(step.expected_output) and len(rows) != 1:
        return (
            f"expected one row but got {len(rows)}",
            "aggregate to a single row, or fix the grouping",
        )
    values = _check_values(columns, rows)
    if values is not None:
        return values
    if step.tool == "sql" and result.sql:
        return check_sql_semantics(result.sql, f"{step.goal} {step.expected_output}")
    return None


def critic_messages(
    deps: AgentDeps, question: str, step: PlanStep, result: StepResult
) -> list[BaseMessage]:
    system = render(
        "critic",
        glossary=raw("glossary") if deps.settings.use_glossary else "",
        untrusted_policy=UNTRUSTED_POLICY,
    )
    work = step.sql or step.code or ""
    body = (
        f"Question: {neutralize(question)}\n"
        f"Step {step.id} ({step.tool}) goal: {neutralize(step.goal)}\n"
        f"Expected output: {neutralize(step.expected_output)}\n"
        f"{'SQL' if step.sql else 'Code'}:\n{neutralize(work)}\n\n"
        f"Result:\n{result.llm_text}"
    )
    return [SystemMessage(system), HumanMessage(body)]


def make_critic(deps: AgentDeps) -> NodeFn:
    def critic(state: AgentState) -> AgentState:
        plan = state["plan"]
        if plan is None:
            raise RuntimeError("critic reached without a plan")
        step = plan.steps[state.get("step_index", 0) - 1]
        result = state.get("results", {})[step.id]
        attempt = state.get("retry_count", 0)

        verdict: Verdict
        failure = rule_check(step, result, deps)
        if failure is not None:
            verdict = Verdict(
                attempt=attempt, step_id=step.id, passed=False, source="rule",
                reason=failure[0], fix=failure[1],
            )  # fmt: skip
        elif deps.settings.critic_llm and step.tool in ("sql", "python"):
            try:
                out = deps.llm.structured(
                    "cheap",
                    CriticOutput,
                    critic_messages(deps, state["question"], step, result),
                    run_name="critic",
                )
                verdict = Verdict(
                    attempt=attempt, step_id=step.id, passed=out.passed, source="llm",
                    reason=out.reason, fix=out.fix,
                )  # fmt: skip
            except InvalidModelOutput:
                # an unusable critic output must not sink a result that passed every rule
                log.warning("critic_output_rejected", extra={"step": step.id})
                verdict = Verdict(
                    attempt=attempt, step_id=step.id, passed=True, source="rule",
                    reason="rule checks passed (critic model output was invalid)",
                )  # fmt: skip
        else:
            verdict = Verdict(
                attempt=attempt, step_id=step.id, passed=True, source="rule",
                reason="rule checks passed",
            )  # fmt: skip

        log.info(
            "critic_verdict",
            extra={"step": step.id, "passed": verdict.passed, "source": verdict.source,
                   "attempt": attempt},
        )  # fmt: skip
        update: AgentState = {"verdicts": [*state.get("verdicts", []), verdict]}
        if not verdict.passed:
            update["attempts"] = [
                *state.get("attempts", []),
                Attempt(
                    plan=plan,
                    failed_step_id=step.id,
                    reason=verdict.reason,
                    fix=verdict.fix,
                    source=verdict.source,
                    columns=result.columns,
                    rows=result.rows[:ATTEMPT_ROWS],
                    value=result.value,
                ),
            ]
        return update

    return critic
