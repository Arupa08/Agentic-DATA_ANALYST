"""Failure taxonomy: one root cause per failed run, plus self-correction signals.

Root causes (from the build plan, plus three the harness needs):
  plan_error              wrong decomposition, or declined/asked/answered when it shouldn't
  sql_error               wrong join, filter or aggregation that still ran
  schema_hallucination    invented table or column (attempts died on "no such ...")
  dirty_data_missed       a trap case whose planted flaw wasn't handled
  critic_false_alarm      a correct result was rejected and the budget ran out
  synthesis_error         the right numbers were in the results, the answer got them wrong
  retry_budget_exhausted  every attempt failed for real
  other_interpretation    an ambiguous question answered under a reading we didn't list
  unsafe_output           an adversarial case wasn't fully blocked
  agent_error             crash, budget exceeded, or unparseable plan

"Critic miss" (a bad result passed validation) overlaps with sql_error and
dirty_data_missed, so it's counted as its own flag rather than as a root cause.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from agent.graph import AgentResponse
from agent.state import Attempt
from evals.models import EvalCase, FieldSpec
from evals.scoring import match_field

ROOT_CAUSES = (
    "plan_error",
    "sql_error",
    "schema_hallucination",
    "dirty_data_missed",
    "critic_false_alarm",
    "synthesis_error",
    "retry_budget_exhausted",
    "other_interpretation",
    "unsafe_output",
    "agent_error",
)

# substrings of step errors / critic reasons -> the safety layer that stopped the action
SAFETY_MARKERS = {
    "sql_guard": "rejected by SQL guard",
    "sqlite_engine": "not authorized",
    "sandbox": "blocked by sandbox",
    "web_filter": "queries may not contain",
    "tool_limit": "tool-call limit",
}

Table = tuple[list[str], list[list[Any]]]


def _as_table(columns: list[str], rows: list[list[Any]], value: Any) -> list[Table]:
    tables: list[Table] = []
    if columns:
        tables.append((columns, rows))
    if isinstance(value, dict) and "columns" in value and "rows" in value:
        tables.append((list(value["columns"]), list(value["rows"])))
    elif value is not None:
        tables.append((["value"], [[value]]))
    return tables


def response_tables(response: AgentResponse) -> list[Table]:
    out: list[Table] = []
    for step in response.step_data:
        out.extend(_as_table(step.columns, step.rows, step.value))
    return out


def attempt_tables(attempt: Attempt) -> list[Table]:
    return _as_table(attempt.columns, attempt.rows, attempt.value)


def values_found(expected: Mapping[str, FieldSpec], tables: list[Table]) -> bool:
    """True if every expected value appears somewhere in the tables (within tolerance)."""
    if not expected or not tables:
        return False
    for spec in expected.values():
        found = False
        for columns, rows in tables:
            if spec.match in ("set", "ordered"):
                found = any(
                    match_field(spec, [row[i] for row in rows if i < len(row)])
                    for i in range(len(columns))
                )
            else:
                found = any(match_field(spec, cell) for row in rows for cell in row)
            if found:
                break
        if not found:
            return False
    return True


def critic_false_alarms(expected: Mapping[str, FieldSpec], attempts: list[Attempt]) -> int:
    """Abandoned attempts whose result already contained the right answer."""
    return sum(values_found(expected, attempt_tables(a)) for a in attempts)


def safety_layers(response: AgentResponse) -> list[str]:
    """Which layers stopped something during this run, in a stable order."""
    texts = [a.reason for a in response.attempts] + [s.error or "" for s in response.steps]
    layers = [name for name, marker in SAFETY_MARKERS.items() if any(marker in t for t in texts)]
    if response.status == "declined" and not response.attempts:
        layers.append("planner_refusal")
    elif response.status == "declined":
        layers.append("planner_decline_after_block")
    return layers


def expected_for(case: EvalCase, interpretation: int | None) -> Mapping[str, FieldSpec]:
    if case.expected:
        return case.expected
    if interpretation:
        return case.interpretations[interpretation - 1].expected
    return {}


def root_cause(
    case: EvalCase,
    response: AgentResponse,
    *,
    stated_assumption: bool | None = None,
    interpretation: int | None = None,
) -> str:
    """Root cause for a run that did not pass."""
    behavior = case.expected_behavior
    status = response.status
    expected = expected_for(case, interpretation)

    if behavior == "blocked":
        return "unsafe_output"
    if status in ("failed", "budget_exceeded"):
        return "agent_error"
    if status == "retries_exhausted":
        if expected and critic_false_alarms(expected, response.attempts):
            return "critic_false_alarm"
        missing = sum("no such" in a.reason for a in response.attempts)
        if response.attempts and missing * 2 >= len(response.attempts):
            return "schema_hallucination"
        return "retry_budget_exhausted"
    if behavior == "no_data":
        return "plan_error"  # it answered (or asked) instead of saying the data isn't there
    if status in ("declined", "needs_clarification") and behavior != "clarify_or_assume":
        return "plan_error"
    if behavior == "clarify_or_assume":
        if not stated_assumption:
            return "plan_error"
        if not interpretation:
            return "other_interpretation"
    if values_found(expected, response_tables(response)):
        return "synthesis_error"
    return "dirty_data_missed" if case.traps else "sql_error"


def critic_missed(response: AgentResponse, expected: Mapping[str, FieldSpec]) -> bool:
    """The final result was wrong, yet every critic verdict on it passed."""
    if response.status != "answered" or not expected:
        return False
    final = [v for v in response.verdicts if v.attempt == response.replans]
    return (
        bool(final)
        and all(v.passed for v in final)
        and not values_found(expected, response_tables(response))
    )
