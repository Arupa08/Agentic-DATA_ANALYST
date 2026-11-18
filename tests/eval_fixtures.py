"""Builders for grading tests: hand-made agent responses and cases from the frozen set."""

from __future__ import annotations

from typing import Any

from agent.graph import AgentResponse, StepData
from agent.state import Attempt, Plan, Verdict
from evals.models import EvalCase, load_cases
from evals.verify import CASES_PATH

CASES = {c.id: c for c in load_cases(CASES_PATH).cases}


def case(case_id: str) -> EvalCase:
    return CASES[case_id]


def response(
    status: str = "answered",
    answer: str = "answer",
    *,
    rows: list[list[Any]] | None = None,
    columns: list[str] | None = None,
    attempts: list[Attempt] | None = None,
    verdicts: list[Verdict] | None = None,
    **kw: Any,
) -> AgentResponse:
    step_data = (
        [StepData(step_id=1, tool="sql", columns=columns or ["v"], rows=rows)] if rows else []
    )
    return AgentResponse(
        request_id="r1",
        question="q",
        status=status,  # type: ignore[arg-type]
        answer=answer,
        step_data=step_data,
        attempts=attempts or [],
        verdicts=verdicts or [],
        replans=len(attempts or []),
        **kw,
    )


PLAN = Plan.model_validate(
    {
        "outcome": "plan",
        "steps": [
            {
                "id": 1,
                "tool": "sql",
                "goal": "compute it",
                "expected_output": "x",
                "sql": "SELECT 1",
            }
        ],
    }
)


def attempt(
    reason: str, rows: list[list[Any]] | None = None, columns: list[str] | None = None
) -> Attempt:
    return Attempt(
        plan=PLAN, failed_step_id=1, reason=reason, columns=columns or ["v"], rows=rows or []
    )


def verdict(passed: bool, attempt_no: int = 0) -> Verdict:
    return Verdict(attempt=attempt_no, step_id=1, passed=passed, source="llm", reason="r")


def rubric_verdict(*passed: bool) -> Any:
    """A judge verdict in the per-item schema the grader asks the model for."""
    from evals.judge import ItemVerdict, rubric_schema

    schema = rubric_schema(len(passed))
    return schema(
        **{f"item_{i}": ItemVerdict(reason="r", passed=p) for i, p in enumerate(passed, 1)}
    )
