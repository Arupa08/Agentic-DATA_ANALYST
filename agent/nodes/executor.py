"""Executor: runs the current plan step through its fixed, guarded tool and records the
result or error. Tool dispatch is a table in code; the plan can only pick a key."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable

from agent.deps import AgentDeps
from agent.safety.sandbox import TableInput, run_python
from agent.safety.untrusted import wrap_untrusted
from agent.state import AgentState, NodeFn, PlanStep, StepResult, ToolName
from agent.tools.chart_tool import render_chart

log = logging.getLogger(__name__)

MAX_TAINTED_VALUES = 2_000


def _table(result: StepResult) -> TableInput | None:
    """A step's tabular output: SQL rows, or a python `result` shaped as a table."""
    if result.tool == "sql":
        return TableInput(columns=result.columns, rows=result.rows)
    value = result.value
    if isinstance(value, dict) and "columns" in value and "rows" in value:
        return TableInput(columns=value["columns"], rows=value["rows"])
    return None


def _inputs(step: PlanStep, results: dict[int, StepResult]) -> dict[str, TableInput] | str:
    tables: dict[str, TableInput] = {}
    for i in step.inputs:
        src = results.get(i)
        if src is None or not src.ok:
            return f"input step {i} has no successful result"
        table = _table(src)
        if table is None:
            return f"input step {i} did not produce a table"
        tables[f"step_{i}"] = table
    return tables


def _run_sql(step: PlanStep, _results: dict[int, StepResult], deps: AgentDeps) -> StepResult:
    r = deps.sql.run(step.sql or "")
    return StepResult(
        step_id=step.id,
        tool="sql",
        ok=r.ok,
        error=r.error,
        sql=r.sql,
        columns=r.columns,
        rows=r.rows,
        row_count=r.row_count,
        truncated=r.truncated,
        llm_text=r.to_llm_text(),
        elapsed_ms=r.elapsed_ms,
    )


def _run_python(step: PlanStep, results: dict[int, StepResult], deps: AgentDeps) -> StepResult:
    tables = _inputs(step, results)
    if isinstance(tables, str):
        return StepResult(step_id=step.id, tool="python", ok=False, error=tables)
    r = run_python(step.code or "", tables)
    shown = json.dumps(r.result, default=str)[:4_000]
    body = f"stdout:\n{r.stdout[:4_000]}\nresult:\n{shown}"
    return StepResult(
        step_id=step.id,
        tool="python",
        ok=r.ok,
        error=r.error,
        value=r.result,
        stdout=r.stdout,
        png=r.figures[0] if r.figures else None,
        llm_text=wrap_untrusted("python", body) if r.ok else "",
        elapsed_ms=r.elapsed_ms,
    )


def _run_chart(step: PlanStep, results: dict[int, StepResult], deps: AgentDeps) -> StepResult:
    tables = _inputs(step, results)
    if isinstance(tables, str):
        return StepResult(step_id=step.id, tool="chart", ok=False, error=tables)
    if step.chart is None:  # PlanStep validation guarantees it; fail closed regardless
        return StepResult(step_id=step.id, tool="chart", ok=False, error="missing chart spec")
    (table,) = tables.values()
    r = render_chart(step.chart, table.columns, table.rows)
    return StepResult(
        step_id=step.id,
        tool="chart",
        ok=r.ok,
        error=r.error,
        png=r.png,
        alt_text=r.alt_text,
        llm_text=wrap_untrusted("chart", f"Chart rendered. {r.alt_text}") if r.ok else "",
    )


def _tainted(results: dict[int, StepResult]) -> list[str]:
    """String values already pulled from company data; web queries may not contain them."""
    values: list[str] = []
    for res in results.values():
        for row in res.rows:
            values.extend(str(v) for v in row if isinstance(v, str))
        if len(values) >= MAX_TAINTED_VALUES:
            break
    return values[:MAX_TAINTED_VALUES]


def _run_web(step: PlanStep, results: dict[int, StepResult], deps: AgentDeps) -> StepResult:
    r = deps.web.run(step.query or "", tainted=_tainted(results))
    return StepResult(
        step_id=step.id,
        tool="web_search",
        ok=r.ok,
        error=r.error,
        llm_text=r.to_llm_text() if r.ok else "",
    )


TOOLS: dict[ToolName, Callable[[PlanStep, dict[int, StepResult], AgentDeps], StepResult]] = {
    "sql": _run_sql,
    "python": _run_python,
    "chart": _run_chart,
    "web_search": _run_web,
}


def make_executor(deps: AgentDeps) -> NodeFn:
    def executor(state: AgentState) -> AgentState:
        plan = state["plan"]
        if plan is None:
            raise RuntimeError("executor reached without a plan")
        index = state.get("step_index", 0)
        step = plan.steps[index]
        results = dict(state.get("results", {}))
        if state.get("tool_calls", 0) >= deps.settings.max_tool_calls:
            # cap spans every replan of this request; the critic turns this into a stop
            result = StepResult(
                step_id=step.id, tool=step.tool, ok=False, error="tool-call limit reached"
            )
        else:
            result = TOOLS[step.tool](step, results, deps)
        results[step.id] = result
        log.info(
            "step_done",
            extra={"step": step.id, "tool": step.tool, "ok": result.ok, "rows": result.row_count},
        )
        return {
            "results": results,
            "step_index": index + 1,
            "tool_calls": state.get("tool_calls", 0) + 1,
        }

    return executor
