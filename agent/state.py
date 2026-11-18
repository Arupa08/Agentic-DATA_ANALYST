"""Graph state and the strict schemas the models must produce.

The planner's output is a `Plan`. Tool choice is a closed Literal, so a model cannot
name a tool that isn't wired in code, and every tool-specific field is required for its
tool and forbidden for the others. Invalid output fails validation and is rejected.
"""

from __future__ import annotations

from typing import Any, Literal, Protocol, TypedDict

from pydantic import BaseModel, ConfigDict, Field, model_validator

from agent.tools.chart_tool import ChartSpec

ToolName = Literal["sql", "python", "chart", "web_search"]
MAX_STEPS = 6


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


_PAYLOAD_FIELD: dict[str, str] = {
    "sql": "sql",
    "python": "python_code",  # model-facing alias of `code`: plain "code" invites SQL
    "chart": "chart",
    "web_search": "query",
}
# order matters for constrained decoders: the payload key is written before `tool`, so
# choosing to write `sql` commits the step to tool "sql" rather than the other way round
_LEAD_FIELDS = ("id", "goal", "expected_output")


def _one_shape_per_tool(schema: dict[str, Any]) -> None:
    """Rewrite the step's JSON schema (what models see) as one strict shape per tool.

    With a flat schema every payload field is optional on every step, so constrained
    decoders happily put SQL under `code`. Here a `sql` step can only carry `sql`, a
    chart step exactly one input, and so on, so a wrong shape can't even be generated.
    The Python validators below stay the real gate; this just stops the model wasting a try.
    """
    props: dict[str, Any] = schema["properties"]

    def non_null(prop: dict[str, Any]) -> dict[str, Any]:
        branches = [b for b in prop.get("anyOf", []) if b.get("type") != "null"]
        base = branches[0] if branches else prop
        return {**base, **{k: v for k, v in prop.items() if k == "description"}}

    variants = []
    for tool, field in _PAYLOAD_FIELD.items():
        shape: dict[str, Any] = {name: props[name] for name in _LEAD_FIELDS}
        shape[field] = non_null(props[field])
        required = [*_LEAD_FIELDS, field]
        if tool in ("python", "chart"):
            items = {"type": "array", "items": {"type": "integer", "minimum": 1}, "minItems": 1}
            if tool == "chart":
                items["maxItems"] = 1
            shape["inputs"] = {**items, "description": props["inputs"].get("description", "")}
            required.append("inputs")
        shape["tool"] = {"const": tool, "type": "string"}
        required.append("tool")
        variants.append(
            {
                "type": "object",
                "properties": shape,
                "required": required,
                "additionalProperties": False,
            }
        )
    title = schema.get("title")
    schema.clear()
    schema.update({"title": title, "anyOf": variants})


class PlanStep(_Strict):
    model_config = ConfigDict(
        extra="forbid", populate_by_name=True, json_schema_extra=_one_shape_per_tool
    )

    id: int = Field(ge=1, le=MAX_STEPS)
    tool: ToolName
    goal: str = Field(min_length=3, max_length=300, description="What this step establishes.")
    expected_output: str = Field(
        max_length=300, description="Shape of the result, e.g. 'one row: region, delta_pp'."
    )
    sql: str | None = Field(
        default=None,
        max_length=2_500,
        description="sql steps only: one SQLite SELECT, no comments.",
    )
    code: str | None = Field(
        default=None,
        alias="python_code",
        max_length=3_000,
        description="python steps only: pandas code; inputs are DataFrames named step_<id>; "
        "assign the answer to `result`.",
    )
    chart: ChartSpec | None = Field(default=None, description="chart steps only.")
    query: str | None = Field(default=None, max_length=200, description="web_search steps only.")
    inputs: list[int] = Field(
        default=[], description="Ids of earlier steps whose results this step reads."
    )

    @model_validator(mode="after")
    def _fields_match_tool(self) -> PlanStep:
        payload = {
            "sql": self.sql,
            "python": self.code,
            "chart": self.chart,
            "web_search": self.query,
        }
        if payload[self.tool] is None:
            raise ValueError(f"{self.tool} step needs its {self.tool} payload")
        extra = [t for t, v in payload.items() if t != self.tool and v is not None]
        if extra:
            raise ValueError(f"{self.tool} step must not carry payloads for {extra}")
        if self.tool in ("python", "chart") and not self.inputs:
            raise ValueError(f"{self.tool} step must read at least one earlier step")
        if self.tool == "chart" and len(self.inputs) != 1:
            raise ValueError("chart step reads exactly one earlier step")
        if self.tool in ("sql", "web_search") and self.inputs:
            raise ValueError(f"{self.tool} step takes no inputs")
        return self


class Plan(_Strict):
    outcome: Literal["plan", "decline", "clarify"] = Field(
        description="plan: answer with steps. decline: the data can't answer it, or the request "
        "is unsafe. clarify: too ambiguous to pick any reasonable reading; ask one question."
    )
    message: str | None = Field(
        default=None,
        max_length=500,
        description="decline: why. clarify: the single question to ask the user.",
    )
    assumptions: list[str] = Field(
        default=[], max_length=5, description="Interpretations chosen for anything ambiguous."
    )
    steps: list[PlanStep] = Field(default=[], max_length=MAX_STEPS)

    @property
    def answerable(self) -> bool:
        return self.outcome == "plan"

    @model_validator(mode="after")
    def _consistent(self) -> Plan:
        if self.outcome != "plan":
            if not self.message:
                raise ValueError(f"a {self.outcome} outcome needs a message")
            if self.steps:
                raise ValueError(f"a {self.outcome} outcome has no steps")
            return self
        if not self.steps:
            raise ValueError("a plan needs at least one step")
        ids = [s.id for s in self.steps]
        if ids != list(range(1, len(ids) + 1)):
            raise ValueError("step ids must be 1..n in order")
        for step in self.steps:
            if any(i < 1 or i >= step.id for i in step.inputs):
                raise ValueError(
                    f"step {step.id} may only read earlier steps (ids 1 to {step.id - 1}); "
                    "python and chart steps work on results of earlier sql steps, so the "
                    "first step that queries the database must be a sql step with `sql`"
                )
        return self

    def fingerprint(self) -> tuple[tuple[str, str], ...]:
        """What the plan would actually execute, whitespace- and case-normalized."""

        def norm(text: str | None) -> str:
            return " ".join((text or "").split()).lower()

        def payload(step: PlanStep) -> str:
            chart = step.chart.model_dump_json() if step.chart else ""
            return norm(step.sql or step.code or step.query or chart)

        return tuple((s.tool, payload(s)) for s in self.steps)


class CriticOutput(_Strict):
    """What the critic model writes."""

    passed: bool = Field(description="true unless you can name a concrete problem.")
    reason: str = Field(min_length=1, max_length=500)
    fix: str | None = Field(
        default=None, max_length=500, description="How a new plan should differ."
    )


class Verdict(BaseModel):
    attempt: int  # 0 = first plan, 1 = first replan, ...
    step_id: int
    passed: bool
    source: Literal["rule", "llm"]
    reason: str
    fix: str | None = None


class Attempt(BaseModel):
    """A plan that was abandoned, and why. Fed back to the planner on replan."""

    plan: Plan
    failed_step_id: int
    reason: str
    fix: str | None = None
    source: Literal["rule", "llm"] = "rule"
    # what the rejected step returned (capped), so evals can spot critic false alarms
    columns: list[str] = []
    rows: list[list[Any]] = []
    value: Any = None


class SynthesisOutput(_Strict):
    """What the synthesizer model writes. SQL citations and charts are attached by code."""

    answer: str = Field(
        min_length=1,
        max_length=4_000,
        description="Direct answer first, with the key numbers, then brief supporting detail.",
    )
    caveats: list[str] = Field(
        default=[], max_length=5, description="Data-quality handling and limits worth knowing."
    )


class StepResult(BaseModel):
    step_id: int
    tool: ToolName
    ok: bool
    error: str | None = None
    sql: str | None = None  # executed (guarded) SQL
    columns: list[str] = []
    rows: list[list[Any]] = []
    row_count: int = 0
    truncated: bool = False
    value: Any = None  # python `result`
    stdout: str = ""
    png: bytes | None = None
    alt_text: str = ""
    llm_text: str = ""  # untrusted-wrapped rendering for prompts
    elapsed_ms: int = 0


Status = Literal[
    "answered",
    "declined",
    "needs_clarification",
    "failed",
    "retries_exhausted",
    "budget_exceeded",
]


class AgentState(TypedDict, total=False):
    question: str
    request_id: str
    plan: Plan | None
    plan_error: str | None
    step_index: int
    results: dict[int, StepResult]
    verdicts: list[Verdict]
    attempts: list[Attempt]  # abandoned plans, oldest first
    retry_count: int  # replans so far
    tool_calls: int
    synthesis: SynthesisOutput | None
    status: Status


class NodeFn(Protocol):
    """A graph node: reads the state, returns the keys it updates."""

    def __call__(self, state: AgentState) -> AgentState: ...
