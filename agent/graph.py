"""LangGraph wiring and the public entry point.

    START -> planner --plan--> executor -> critic --pass, more steps--> executor
                 |                            |---pass, done-----------> synthesizer -> END
                 |                            '---fail, budget left----> planner (replan)
                 '--decline / clarify / unparseable-------------------> synthesizer

The critic checks every step. On a failure, the planner is re-entered with all abandoned
attempts and the critic's reasons, and a plan that repeats one is rejected in code. At
most `max_replans` replans and `max_tool_calls` tool calls per request; past that the
synthesizer explains what was tried instead of answering.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Callable
from typing import Any, Literal, cast

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel

from agent.deps import AgentDeps
from agent.llm import BudgetExceeded, InvalidModelOutput, RateLimited, track_costs
from agent.logging_setup import request_id_var
from agent.nodes.critic import make_critic
from agent.nodes.executor import make_executor
from agent.nodes.planner import make_planner
from agent.nodes.synthesizer import make_synthesizer
from agent.prompts import PROMPT_VERSION
from agent.state import AgentState, Attempt, Plan, Status, StepResult, Verdict
from agent.tracing import build_tracer

log = logging.getLogger(__name__)

MAX_QUESTION_CHARS = 1_000
STEP_DATA_ROWS = 200
# each replan costs at most 1 + 2*MAX_STEPS supersteps; 4 plans fit comfortably
RECURSION_LIMIT = 80
BUDGET_ANSWER = "This question needed more work than the per-request budget allows."
ERROR_ANSWER = "Something went wrong while answering. Please try again."


def route_after_planner(state: AgentState) -> Literal["executor", "synthesizer"]:
    plan = state.get("plan")
    return "executor" if plan is not None and plan.answerable else "synthesizer"


def make_route_after_critic(
    max_replans: int, max_tool_calls: int
) -> Callable[[AgentState], Literal["executor", "planner", "synthesizer"]]:
    def route_after_critic(state: AgentState) -> Literal["executor", "planner", "synthesizer"]:
        plan = state["plan"]
        verdict = state["verdicts"][-1]
        if verdict.passed:
            more = plan is not None and state.get("step_index", 0) < len(plan.steps)
            return "executor" if more else "synthesizer"
        budget_left = (
            state.get("retry_count", 0) < max_replans
            and state.get("tool_calls", 0) < max_tool_calls
        )
        return "planner" if budget_left else "synthesizer"

    return route_after_critic


def build_graph(deps: AgentDeps) -> CompiledStateGraph[AgentState, Any, AgentState, AgentState]:
    settings = deps.settings
    graph = StateGraph(AgentState)
    graph.add_node("planner", make_planner(deps))
    graph.add_node("executor", make_executor(deps))
    graph.add_node("critic", make_critic(deps))
    graph.add_node("synthesizer", make_synthesizer(deps))
    graph.add_edge(START, "planner")
    graph.add_conditional_edges("planner", route_after_planner)
    graph.add_edge("executor", "critic")
    graph.add_conditional_edges(
        "critic", make_route_after_critic(settings.max_replans, settings.max_tool_calls)
    )
    graph.add_edge("synthesizer", END)
    return graph.compile()


class Chart(BaseModel):
    png: bytes
    alt_text: str


class StepData(BaseModel):
    """A step's output, capped, for the UI's "how I got here" panel and for grading."""

    step_id: int
    tool: str
    columns: list[str] = []
    rows: list[list[Any]] = []
    value: Any = None


class StepSummary(BaseModel):
    id: int
    tool: str
    goal: str
    ok: bool | None  # None = not run
    error: str | None = None
    row_count: int = 0


class AgentResponse(BaseModel):
    request_id: str
    question: str
    status: Status
    answer: str
    assumptions: list[str] = []
    caveats: list[str] = []
    sql: list[str] = []  # exactly what executed, from the tool, not from the model
    charts: list[Chart] = []
    plan: Plan | None = None  # the final plan
    steps: list[StepSummary] = []
    attempts: list[Attempt] = []  # abandoned plans and why
    verdicts: list[Verdict] = []
    replans: int = 0
    step_data: list[StepData] = []
    cost_usd: float = 0.0
    total_tokens: int = 0
    unpriced_calls: bool = False
    latency_ms: int = 0
    prompt_version: str = PROMPT_VERSION


class Agent:
    def __init__(self, deps: AgentDeps, *, tracer: BaseCallbackHandler | None = None) -> None:
        self.deps = deps
        self.graph = build_graph(deps)
        self.tracer = tracer if tracer is not None else build_tracer(deps.settings)

    def ask(
        self,
        question: str,
        *,
        case_id: str | None = None,
        tags: tuple[str, ...] = (),
        extra_callbacks: list[BaseCallbackHandler] | None = None,
    ) -> AgentResponse:
        """`extra_callbacks` lets a caller watch the run (e.g. the UI's live progress)."""
        question = " ".join(question.split())
        if not question or len(question) > MAX_QUESTION_CHARS:
            raise ValueError(f"question must be 1-{MAX_QUESTION_CHARS} characters")
        request_id = uuid.uuid4().hex[:12]
        token = request_id_var.set(request_id)
        start = time.monotonic()
        config: RunnableConfig = {
            "run_name": "nl_agent",
            "recursion_limit": RECURSION_LIMIT,
            "tags": [*tags, f"prompt:{PROMPT_VERSION}", *([f"case:{case_id}"] if case_id else [])],
            "metadata": {
                "request_id": request_id,
                "case_id": case_id,
                "prompt_version": PROMPT_VERSION,
            },
            "callbacks": ([self.tracer] if self.tracer is not None else [])
            + list(extra_callbacks or []),
        }
        state: AgentState = {}
        status: Status
        try:
            with track_costs() as tracker:
                try:
                    initial: AgentState = {"question": question, "request_id": request_id}
                    # stream full states so that if a late node fails, the plan, steps and
                    # SQL that did run are still reported (and traceable) in the response
                    for snapshot in self.graph.stream(initial, config, stream_mode="values"):
                        state = cast(AgentState, snapshot)
                    status = state.get("status", "failed")
                except BudgetExceeded:
                    status = "budget_exceeded"
                except InvalidModelOutput:
                    log.warning("synthesis_rejected")
                    status = "failed"
                except RateLimited:
                    raise  # provider quota, not an agent failure: the caller decides
                except Exception:
                    log.exception("agent_error")  # stack trace to server logs only
                    status = "failed"
            response = self._response(question, request_id, status, state)
            response.cost_usd = round(tracker.total_cost_usd, 6)
            response.total_tokens = tracker.total_tokens
            response.unpriced_calls = tracker.has_unpriced_calls
            response.latency_ms = int((time.monotonic() - start) * 1000)
            log.info(
                "request_done",
                extra={
                    "status": status,
                    "cost_usd": response.cost_usd,
                    "latency_ms": response.latency_ms,
                    "case_id": case_id,
                },
            )
            return response
        finally:
            request_id_var.reset(token)

    @staticmethod
    def _response(
        question: str, request_id: str, status: Status, state: AgentState
    ) -> AgentResponse:
        plan = state.get("plan")
        results: dict[int, StepResult] = state.get("results", {})
        synthesis = state.get("synthesis")
        if status == "budget_exceeded":
            answer = BUDGET_ANSWER
        elif synthesis is not None:
            answer = synthesis.answer
        else:
            answer = ERROR_ANSWER
        steps = []
        for s in plan.steps if plan else []:
            r = results.get(s.id)
            steps.append(
                StepSummary(
                    id=s.id,
                    tool=s.tool,
                    goal=s.goal,
                    ok=None if r is None else r.ok,
                    error=None if r is None else r.error,
                    row_count=0 if r is None else r.row_count,
                )
            )
        ordered = [results[k] for k in sorted(results)]
        return AgentResponse(
            request_id=request_id,
            question=question,
            status=status,
            answer=answer,
            assumptions=plan.assumptions if plan else [],
            caveats=synthesis.caveats if synthesis else [],
            sql=[r.sql for r in ordered if r.sql and r.ok],
            charts=[Chart(png=r.png, alt_text=r.alt_text) for r in ordered if r.png and r.ok],
            plan=plan,
            steps=steps,
            attempts=state.get("attempts", []),
            verdicts=state.get("verdicts", []),
            replans=state.get("retry_count", 0),
            step_data=[
                StepData(
                    step_id=r.step_id,
                    tool=r.tool,
                    columns=r.columns,
                    rows=r.rows[:STEP_DATA_ROWS],
                    value=r.value,
                )
                for r in ordered
                if r.ok
            ],
        )
