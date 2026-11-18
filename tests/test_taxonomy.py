"""One root cause per failure mode, plus the self-correction signals."""

from __future__ import annotations

from evals.taxonomy import (
    critic_false_alarms,
    critic_missed,
    root_cause,
    safety_layers,
    values_found,
)
from tests.eval_fixtures import attempt, case, response, verdict

Q3_REVENUE = 1783489.61


def test_values_found_respects_tolerance_and_kinds() -> None:
    simple = case("simple_01").expected
    assert values_found(simple, [(["revenue"], [[Q3_REVENUE * 1.005]])])
    assert not values_found(simple, [(["revenue"], [[Q3_REVENUE * 1.05]])])
    multi = case("multi_01").expected  # region exact + delta numeric
    assert values_found(multi, [(["region", "delta_pp"], [["Asia Pacific", 0.0], ["Europe", 8.9]])])
    assert not values_found(multi, [(["region"], [["Europe"]])])  # delta missing


def test_synthesis_error_when_the_data_was_right() -> None:
    r = response(rows=[[Q3_REVENUE]], answer="Revenue was $2.1M")
    assert root_cause(case("simple_01"), r) == "synthesis_error"


def test_sql_error_vs_dirty_data_missed() -> None:
    wrong = response(rows=[[999.0]])
    assert root_cause(case("simple_01"), wrong) == "sql_error"
    assert (
        root_cause(case("dirty_01"), response(rows=[[7925]])) == "dirty_data_missed"
    )  # naive count


def test_exhaustion_split_three_ways() -> None:
    right_rows = [[Q3_REVENUE]]
    false_alarm = response("retries_exhausted", attempts=[attempt("looked odd", right_rows)])
    assert root_cause(case("simple_01"), false_alarm) == "critic_false_alarm"
    schema = response(
        "retries_exhausted",
        attempts=[attempt("sql step failed: no such column: x"), attempt("no such table: y")],
    )
    assert root_cause(case("simple_01"), schema) == "schema_hallucination"
    real = response("retries_exhausted", attempts=[attempt("step returned no rows")])
    assert root_cause(case("simple_01"), real) == "retry_budget_exhausted"


def test_behavior_mismatches_are_plan_errors() -> None:
    assert root_cause(case("simple_01"), response("declined")) == "plan_error"
    assert root_cause(case("noans_01"), response("answered", "NPS was 42")) == "plan_error"
    under = case("under_01")
    assert root_cause(under, response(), stated_assumption=False) == "plan_error"
    assert (
        root_cause(under, response(), stated_assumption=True, interpretation=None)
        == "other_interpretation"
    )


def test_other_causes() -> None:
    assert root_cause(case("adv_01"), response("declined")) == "unsafe_output"
    assert root_cause(case("simple_01"), response("failed")) == "agent_error"
    assert root_cause(case("simple_01"), response("budget_exceeded")) == "agent_error"


def test_critic_signals() -> None:
    expected = case("simple_01").expected
    assert critic_false_alarms(expected, [attempt("x", [[Q3_REVENUE]]), attempt("y", [[1.0]])]) == 1
    assert critic_missed(response(rows=[[1.0]], verdicts=[verdict(True)]), expected)
    assert not critic_missed(response(rows=[[Q3_REVENUE]], verdicts=[verdict(True)]), expected)
    assert not critic_missed(response(rows=[[1.0]]), expected)  # no critic ran on it


def test_safety_layers_named_from_reasons() -> None:
    r = response(
        "declined",
        attempts=[
            attempt("sql step failed: rejected by SQL guard: DROP is not allowed"),
            attempt("python step failed: SandboxViolation: blocked by sandbox: os.system"),
            attempt("web_search step failed: rejected: queries may not contain email addresses"),
        ],
    )
    assert safety_layers(r) == ["sql_guard", "sandbox", "web_filter", "planner_decline_after_block"]
    assert safety_layers(response("declined")) == ["planner_refusal"]
