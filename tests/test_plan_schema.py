from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from agent.state import MAX_STEPS, Plan, PlanStep

SQL_STEP: dict[str, Any] = {
    "id": 1,
    "tool": "sql",
    "goal": "get revenue",
    "expected_output": "1 row",
    "sql": "SELECT 1",
}
CHART = {"kind": "bar", "x": "a", "y": ["b"], "title": "t"}


def plan(*steps: dict[str, Any], **kw: Any) -> Plan:
    return Plan.model_validate({"outcome": "plan", "steps": list(steps), **kw})


def test_valid_multi_step_plan() -> None:
    p = plan(
        SQL_STEP,
        {
            "id": 2,
            "tool": "python",
            "goal": "compute",
            "expected_output": "x",
            "code": "result = 1",
            "inputs": [1],
        },
        {
            "id": 3,
            "tool": "chart",
            "goal": "plot",
            "expected_output": "png",
            "chart": CHART,
            "inputs": [1],
        },
    )
    assert [s.tool for s in p.steps] == ["sql", "python", "chart"]


@pytest.mark.parametrize(
    "step",
    [
        {**SQL_STEP, "tool": "shell"},  # tools are a closed set
        {**SQL_STEP, "permissions": ["write"]},  # no extra fields
        {**SQL_STEP, "sql": None},  # sql step needs sql
        {**SQL_STEP, "code": "print(1)"},  # payload for another tool
        {**SQL_STEP, "inputs": [1]},
        {"id": 1, "tool": "python", "goal": "x", "expected_output": "x", "code": "1"},  # no inputs
        {
            "id": 1,
            "tool": "web_search",
            "goal": "x",
            "expected_output": "x",
            "query": "q",
            "inputs": [1],
        },
    ],
)
def test_bad_steps_rejected(step: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        PlanStep.model_validate(step)


def test_steps_may_only_read_earlier_steps() -> None:
    with pytest.raises(ValidationError):
        plan(
            {**SQL_STEP},
            {
                "id": 2,
                "tool": "chart",
                "goal": "plot",
                "expected_output": "png",
                "chart": CHART,
                "inputs": [2],
            },
        )


def test_ids_must_be_sequential() -> None:
    with pytest.raises(ValidationError):
        plan({**SQL_STEP, "id": 2})


def test_step_cap() -> None:
    with pytest.raises(ValidationError):
        plan(*({**SQL_STEP, "id": i} for i in range(1, MAX_STEPS + 2)))


def test_chart_reads_exactly_one_step() -> None:
    with pytest.raises(ValidationError):
        PlanStep.model_validate(
            {
                "id": 3,
                "tool": "chart",
                "goal": "g",
                "expected_output": "p",
                "chart": CHART,
                "inputs": [1, 2],
            }
        )


@pytest.mark.parametrize("outcome", ["decline", "clarify"])
def test_decline_and_clarify_need_a_message_and_no_steps(outcome: str) -> None:
    p = Plan.model_validate({"outcome": outcome, "message": "no NPS data / which metric?"})
    assert not p.answerable
    with pytest.raises(ValidationError):
        Plan.model_validate({"outcome": outcome})
    with pytest.raises(ValidationError):
        Plan.model_validate({"outcome": outcome, "message": "x", "steps": [SQL_STEP]})


def test_plan_needs_steps() -> None:
    with pytest.raises(ValidationError):
        Plan.model_validate({"outcome": "plan", "steps": []})


def test_fingerprint_ignores_whitespace_and_case_only() -> None:
    a = plan({**SQL_STEP, "sql": "SELECT  1\n FROM regions"})
    b = plan({**SQL_STEP, "sql": "select 1 from REGIONS", "goal": "reworded goal"})
    c = plan({**SQL_STEP, "sql": "SELECT 2 FROM regions"})
    assert a.fingerprint() == b.fingerprint()
    assert a.fingerprint() != c.fingerprint()


# ---------------------------------------------------------------- what models see (live-run fixes)


def test_model_facing_schema_has_one_shape_per_tool() -> None:
    step_schema = Plan.model_json_schema()["$defs"]["PlanStep"]
    shapes = {v["properties"]["tool"]["const"]: v for v in step_schema["anyOf"]}
    payload = {"sql": "sql", "python": "python_code", "chart": "chart", "web_search": "query"}
    assert set(shapes) == set(payload)
    for tool, shape in shapes.items():
        others = set(payload.values()) - {payload[tool]}
        assert payload[tool] in shape["required"]
        assert not others & set(shape["properties"])  # e.g. no `python_code` on a sql step
        assert shape["additionalProperties"] is False
    assert shapes["chart"]["properties"]["inputs"]["maxItems"] == 1
    assert shapes["python"]["properties"]["inputs"]["items"]["minimum"] == 1


def test_python_code_alias_and_internal_name_are_the_same_field() -> None:
    base = {"id": 2, "tool": "python", "goal": "compute", "expected_output": "x", "inputs": [1]}
    by_alias = PlanStep.model_validate({**base, "python_code": "result = 1"})
    by_name = PlanStep.model_validate({**base, "code": "result = 1"})
    assert by_alias.code == by_name.code == "result = 1"


@pytest.mark.parametrize("bad_input", [0, -1])
def test_inputs_must_name_a_real_earlier_step(bad_input: int) -> None:
    # live run: a 4B model labelled SQL as a python step reading "step 0", and it validated
    with pytest.raises(ValidationError, match="earlier steps"):
        plan(
            {
                "id": 1,
                "tool": "python",
                "goal": "compute",
                "expected_output": "x",
                "code": "SELECT 1",
                "inputs": [bad_input],
            }
        )


@pytest.mark.parametrize(("field", "limit"), [("sql", 2_500), ("code", 3_000), ("query", 200)])
def test_payload_length_caps(field: str, limit: int) -> None:
    # live run: a 4B model kept writing comments inside the SQL until it hit its token cap
    tool = {"sql": "sql", "code": "python", "query": "web_search"}[field]
    step = {
        "id": 2,
        "tool": tool,
        "goal": "do it",
        "expected_output": "x",
        field: "x" * (limit + 1),
    }
    if tool == "python":
        step["inputs"] = [1]
    with pytest.raises(ValidationError):
        PlanStep.model_validate(step)
