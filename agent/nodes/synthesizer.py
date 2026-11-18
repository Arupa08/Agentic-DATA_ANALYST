"""Synthesizer (strong tier): writes the answer from step results. SQL citations and
charts are attached by code from what actually ran, never re-typed by the model."""

from __future__ import annotations

import logging

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from agent.deps import AgentDeps
from agent.prompts import render
from agent.safety.untrusted import UNTRUSTED_POLICY, neutralize
from agent.state import AgentState, Attempt, NodeFn, Plan, StepResult, SynthesisOutput

log = logging.getLogger(__name__)

PLAN_FAILED_ANSWER = (
    "I couldn't work out how to answer this from the available data. "
    "Try rephrasing the question or naming the metric and period you're interested in."
)


def _evidence(question: str, plan: Plan, results: dict[int, StepResult]) -> str:
    # Text outside the wrappers (question, model-written SQL and goals, errors) is
    # neutralized too, so nothing but the wrapper code can emit a real <untrusted> tag.
    parts = [f"Question: {neutralize(question)}"]
    if plan.assumptions:
        listed = "\n".join(f"- {neutralize(a)}" for a in plan.assumptions)
        parts.append(f"Assumptions made:\n{listed}")
    for step in plan.steps:
        res = results.get(step.id)
        header = f"Step {step.id} ({step.tool}): {neutralize(step.goal)}"
        if res is None:
            parts.append(f"{header}\nNOT RUN (an earlier step failed)")
        elif not res.ok:
            parts.append(f"{header}\nFAILED: {neutralize(res.error or '')}")
        else:
            sql = f"SQL: {neutralize(res.sql)}\n" if res.sql else ""
            parts.append(f"{header}\n{sql}{res.llm_text}")
    return "\n\n".join(parts)


def synthesis_messages(
    question: str, plan: Plan, results: dict[int, StepResult]
) -> list[BaseMessage]:
    system = render("synthesizer", untrusted_policy=UNTRUSTED_POLICY)
    return [SystemMessage(system), HumanMessage(_evidence(question, plan, results))]


def exhausted_answer(attempts: list[Attempt]) -> str:
    """Deterministic: no model call, so it can't slip in a number nobody verified."""
    lines = [
        f"I couldn't get a result I trust after {len(attempts)} attempt(s), so I'm not "
        "giving a number. What I tried:"
    ]
    for n, attempt in enumerate(attempts, 1):
        step = next((s for s in attempt.plan.steps if s.id == attempt.failed_step_id), None)
        what = step.goal if step else "the plan"
        lines.append(f"{n}. {what}: rejected because {attempt.reason}")
    lines.append("Try narrowing the question (a specific metric, period or segment).")
    return "\n".join(lines)


def make_synthesizer(deps: AgentDeps) -> NodeFn:
    def synthesizer(state: AgentState) -> AgentState:
        plan = state.get("plan")
        if plan is None:
            return {"synthesis": SynthesisOutput(answer=PLAN_FAILED_ANSWER), "status": "failed"}
        if plan.outcome == "decline":
            # the planner's own explanation; no second model call, no data involved
            return {"synthesis": SynthesisOutput(answer=plan.message or ""), "status": "declined"}
        if plan.outcome == "clarify":
            return {
                "synthesis": SynthesisOutput(answer=plan.message or ""),
                "status": "needs_clarification",
            }
        verdicts = state.get("verdicts", [])
        if verdicts and not verdicts[-1].passed:
            # the critic rejected a step and the replan budget is spent
            answer = exhausted_answer(state.get("attempts", []))
            return {"synthesis": SynthesisOutput(answer=answer), "status": "retries_exhausted"}
        results = state.get("results", {})
        out = deps.llm.structured(
            "strong",
            SynthesisOutput,
            synthesis_messages(state["question"], plan, results),
            run_name="synthesizer",
            text_field="answer",  # the answer is display text; citations come from code
        )
        log.info("synthesis_done", extra={"replans": state.get("retry_count", 0)})
        return {"synthesis": out, "status": "answered"}

    return synthesizer
