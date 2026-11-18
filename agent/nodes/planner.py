"""Planner (cheap tier): question + schema -> a validated Plan of tool steps.

On a replan it also sees every abandoned attempt with the critic's reason and suggested
fix, and a plan that would execute the same thing as an earlier attempt is rejected in
code. The loop can't just retry the same thing.
"""

from __future__ import annotations

import logging

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from agent.deps import AgentDeps
from agent.llm import InvalidModelOutput
from agent.prompts import raw, render
from agent.safety.untrusted import neutralize, wrap_untrusted
from agent.state import AgentState, Attempt, NodeFn, Plan

log = logging.getLogger(__name__)

WEB_TOOL_TEXT = (
    "- `web_search`: public web context only (e.g. holiday dates, public events). Provide "
    "`query`. Never put company data (names, emails, ids, figures from results) in a query. "
    "Results are external and unverified.\n"
)


def _describe_attempts(attempts: list[Attempt]) -> str:
    lines = [
        "Earlier plans for this question were rejected. Write a plan that is materially "
        "different and fixes the stated problems, or decline if the data can't answer it."
    ]
    for n, attempt in enumerate(attempts, 1):
        lines.append(f"\nAttempt {n}:")
        for step in attempt.plan.steps:
            work = step.sql or step.code or step.query or ""
            lines.append(f"  step {step.id} [{step.tool}] {neutralize(step.goal)}")
            if work:
                lines.append("    " + neutralize(" ".join(work.split()))[:1_500])
        feedback = f"reason: {attempt.reason}" + (
            f"\nsuggested fix: {attempt.fix}" if attempt.fix else ""
        )
        if attempt.source == "llm":
            # the critic model wrote this after reading tool output, so it is untrusted too
            lines.append(f"  Rejected at step {attempt.failed_step_id} by the critic model:")
            lines.append(wrap_untrusted("critic", feedback))
        else:
            lines.append(f"  Rejected at step {attempt.failed_step_id} by a rule check:")
            lines.append("  " + neutralize(feedback).replace("\n", "\n  "))
    return "\n".join(lines)


def planner_messages(
    deps: AgentDeps, question: str, attempts: list[Attempt] | None = None
) -> list[BaseMessage]:
    system = render(
        "planner",
        web_tool=WEB_TOOL_TEXT if deps.web.enabled else "",
        glossary=raw("glossary") if deps.settings.use_glossary else "",
        schema=deps.schema_summary,
    )
    human = f"Question: {neutralize(question)}"
    if attempts:
        human += "\n\n" + _describe_attempts(attempts)
    return [SystemMessage(system), HumanMessage(human)]


def check_plan(plan: Plan, deps: AgentDeps, attempts: list[Attempt]) -> None:
    """Checks that need context the schema alone doesn't have."""
    if not deps.web.enabled and any(s.tool == "web_search" for s in plan.steps):
        raise InvalidModelOutput("plan uses web_search, which is not available")
    if plan.answerable:
        for n, attempt in enumerate(attempts, 1):
            if plan.fingerprint() == attempt.plan.fingerprint():
                raise InvalidModelOutput(f"plan repeats attempt {n}, which already failed")


ASSUME_INSTEAD = (
    "You chose to ask the user: {question}\n"
    "Don't ask. Pick the most reasonable reading, state it in `assumptions`, and plan the "
    'steps. Only if no reading is reasonable at all, return outcome "clarify" again.'
)


def assume_instead(
    deps: AgentDeps, messages: list[BaseMessage], asked: Plan, attempts: list[Attempt]
) -> Plan:
    """One nudge from "ask" to "assume" (the planner prompt's stated preference). Live
    baseline: a small model asked instead of answering on 13 clearly answerable runs.
    If the nudged plan is unusable, the original question stands."""
    nudge = HumanMessage(ASSUME_INSTEAD.format(question=neutralize(asked.message or "")))
    try:
        plan = deps.llm.structured("cheap", Plan, [*messages, nudge], run_name="planner_assume")
        check_plan(plan, deps, attempts)
    except InvalidModelOutput:
        return asked
    log.info("clarify_nudged", extra={"outcome": plan.outcome})
    return plan


def make_planner(deps: AgentDeps) -> NodeFn:
    def planner(state: AgentState) -> AgentState:
        attempts = state.get("attempts", [])
        messages = planner_messages(deps, state["question"], attempts)
        error: str | None = None
        for _try in range(2):  # one retry, told exactly why the first plan was rejected
            try:
                plan = deps.llm.structured("cheap", Plan, messages, run_name="planner")
                check_plan(plan, deps, attempts)
                if plan.outcome == "clarify" and deps.settings.clarify_nudge:
                    plan = assume_instead(deps, messages, plan, attempts)
            except InvalidModelOutput as exc:
                error = str(exc)
                log.info("plan_rejected", extra={"reason": error[:300]})
                messages = [
                    *messages,
                    HumanMessage(f"Your plan was rejected: {error}\nReturn a corrected plan."),
                ]
                continue
            log.info(
                "plan_ok",
                extra={
                    "outcome": plan.outcome,
                    "steps": [s.tool for s in plan.steps],
                    "replan": len(attempts),
                },
            )
            return {
                "plan": plan,
                "plan_error": None,
                "step_index": 0,
                "results": {},
                "retry_count": len(attempts),
            }
        return {"plan": None, "plan_error": error, "step_index": 0, "results": {}}

    return planner
