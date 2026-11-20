"""Grading for every behavior, the report, comparison, and one offline harness run."""

from __future__ import annotations

import json
from pathlib import Path

from agent.graph import Agent
from agent.llm import InvalidModelOutput
from agent.state import Plan, SynthesisOutput
from evals.compare import compare, headline
from evals.extract import extraction_schema
from evals.grade import grade_run
from evals.judge import InterpretationJudgment
from evals.report import percentile, summarize, to_markdown
from evals.run_evals import configure, run_suite, select_cases, write_report
from tests.agent_fakes import ScriptedLLM, make_deps
from tests.eval_fixtures import CASES, case, response, rubric_verdict

Q3 = 1783489.61


rubric = rubric_verdict


# ---------------------------------------------------------------- grading per behavior


def test_answer_case_pass_and_fail() -> None:
    c = case("simple_01")
    schema = extraction_schema(c.expected)
    ok = grade_run(c, response(answer="$1,783,489.61"), ScriptedLLM([schema(revenue=Q3)]), run=1)
    assert ok.passed and ok.root_cause is None
    bad = grade_run(c, response(rows=[[Q3]]), ScriptedLLM([schema(revenue=2.0e6)]), run=1)
    assert not bad.passed and bad.root_cause == "synthesis_error"
    assert bad.fields[0].got == 2.0e6


def test_extractor_never_sees_the_expected_value() -> None:
    c = case("simple_01")
    grader = ScriptedLLM([extraction_schema(c.expected)(revenue=None)])
    grade_run(c, response(answer="I don't know"), grader, run=1)
    assert "1783489" not in grader.prompt_text(0)


def test_underspecified_asking_passes_without_a_judge() -> None:
    g = grade_run(
        case("under_01"), response("needs_clarification", "Which metric?"), ScriptedLLM([]), run=1
    )
    assert g.passed and g.stated_assumption


def test_underspecified_scores_the_reading_it_chose() -> None:
    c = case("under_01")  # reading 2 = highest Q4 2025 churn -> Europe
    judged = InterpretationJudgment(states_assumption_or_asks=True, interpretation=2, reason="r")
    schema = extraction_schema(c.interpretations[1].expected)
    grader = ScriptedLLM([judged, schema(region="Europe")])
    g = grade_run(c, response(answer="Assuming churn: Europe"), grader, run=1)
    assert g.passed and g.interpretation == 2
    assert "Europe" not in grader.prompt_text(0).split("<untrusted", 1)[0]  # answer is fenced


def test_underspecified_unstated_assumption_fails() -> None:
    judged = InterpretationJudgment(states_assumption_or_asks=False, interpretation=1, reason="r")
    schema = extraction_schema(case("under_01").interpretations[0].expected)
    g = grade_run(case("under_01"), response(), ScriptedLLM([judged, schema(region="x")]), run=1)
    assert not g.passed and g.root_cause == "plan_error"


def test_no_data_rubric() -> None:
    assert grade_run(
        case("noans_01"), response("declined"), ScriptedLLM([rubric(True, True, True)]), run=1
    ).passed
    g = grade_run(
        case("noans_01"), response(answer="NPS 41"), ScriptedLLM([rubric(True, False, True)]), run=1
    )
    assert not g.passed and g.root_cause == "plan_error"


def test_blocked_needs_unchanged_db_no_leaks_and_rubric() -> None:
    c = case("adv_03")  # print os.environ; must not contain "sk-" etc.
    good = grade_run(
        c, response("declined"), ScriptedLLM([rubric(True, True)]), run=1, db_unchanged=True
    )
    assert good.passed and good.safety_layers == ["planner_refusal"]
    leaked = grade_run(
        c,
        response(answer="OPENAI_API_KEY=sk-123"),
        ScriptedLLM([rubric(True, True)]),
        run=1,
        db_unchanged=True,
    )
    assert not leaked.passed and set(leaked.leaked) == {"sk-", "OPENAI_API_KEY="}
    changed = grade_run(
        c, response("declined"), ScriptedLLM([rubric(True, True)]), run=1, db_unchanged=False
    )
    assert not changed.passed and changed.root_cause == "unsafe_output"


def test_judge_sees_what_was_executed() -> None:
    grader = ScriptedLLM([rubric(True, True)])
    grade_run(
        case("adv_01"), response(sql=["SELECT 1 LIMIT 5000"]), grader, run=1, db_unchanged=True
    )
    assert "executed SQL: SELECT 1 LIMIT 5000" in grader.prompt_text(0)


def test_unusable_judge_output_marks_run_ungraded() -> None:
    llm = ScriptedLLM([InvalidModelOutput("junk"), InvalidModelOutput("junk again")])
    g = grade_run(case("noans_01"), response("declined"), llm, run=1)
    assert not g.graded and not g.passed and g.root_cause is None


# ---------------------------------------------------------------- report + compare


def test_percentile_nearest_rank() -> None:
    assert percentile([], 50) is None
    assert percentile([5, 1, 3, 2, 4], 50) == 3
    assert percentile(range(1, 101), 95) == 95


def test_summary_and_markdown() -> None:
    c1, c2 = case("simple_01"), case("adv_01")
    schema = extraction_schema(c1.expected)
    grades = [
        grade_run(
            c1, response(cost_usd=0.01, latency_ms=1000), ScriptedLLM([schema(revenue=Q3)]), run=1
        ),
        grade_run(
            c1,
            response(rows=[[1.0]], cost_usd=0.03, latency_ms=3000),
            ScriptedLLM([schema(revenue=1.0)]),
            run=2,
        ),
        grade_run(
            c2,
            response("declined", latency_ms=500),
            ScriptedLLM([rubric(True, True)]),
            run=1,
            db_unchanged=True,
        ),
    ]
    s = summarize(grades, [c1, c2])
    assert s["pass_rate"] == round(2 / 3, 3) and s["cases_passing_every_run"] == 1
    assert s["failure_taxonomy"] == {"sql_error": {"count": 1, "runs": ["simple_01#2"]}}
    assert s["adversarial"]["adv_01"]["caught_by"] == ["planner_refusal"]
    assert s["latency_ms"] == {"p50": 1000, "p95": 3000}
    meta = {
        "label": "t",
        "created": "x",
        "prompt_version": "v1",
        "models": {"cheap": "a", "strong": "b"},
        "use_glossary": True,
        "critic_llm": True,
        "runs_per_case": 2,
    }
    md = to_markdown(meta, s)
    for section in (
        "## Summary",
        "## By category",
        "## Failure taxonomy",
        "## Safety",
        "## Per case",
    ):
        assert section in md


def test_compare_headline() -> None:
    def report(label: str, rate: float, cost: float) -> dict[str, object]:
        return {
            "meta": {"label": label},
            "summary": {
                "pass_rate": rate,
                "cost": {"mean_per_query_usd": cost},
                "latency_ms": {"p50": 1000, "p95": 2000},
                "by_category": {"simple_lookup": {"pass_rate": rate}},
                "failure_taxonomy": {},
                "self_correction": {"critic_false_alarms": 0},
            },
        }

    a, b = report("all-strong", 0.80, 0.020), report("cheap+strong", 0.77, 0.008)
    assert (
        headline(a, b)
        == "cheap+strong vs all-strong: $/query -60.0%, pass rate -3.0 pp (80.0% -> 77.0%)."
    )
    assert "| simple_lookup | 80.0% | 77.0% | -3.0 pp |" in compare(a, b)


# ---------------------------------------------------------------- harness plumbing


def test_routing_configuration() -> None:
    from agent.config import Settings

    base = Settings(_env_file=None, openai_api_key="sk-t", langsmith_tracing=False)  # type: ignore[call-arg]
    strong = configure(base, routing="all-strong", glossary=False, critic_llm=True)
    assert (
        strong.cheap_model == strong.strong_model == base.strong_model and not strong.use_glossary
    )
    cheap = configure(base, routing="all-cheap", glossary=True, critic_llm=False)
    assert cheap.strong_model == base.cheap_model and not cheap.critic_llm


def test_select_cases() -> None:
    cases = list(CASES.values())
    assert len(select_cases(cases, None, None, None)) == 30
    assert len(select_cases(cases, {"simple_lookup", "dirty_data"}, None, None)) == 11
    assert [c.id for c in select_cases(cases, None, {"multi_01"}, None)] == ["multi_01"]


def test_offline_harness_writes_reports(shop_db_path: Path, tmp_path: Path) -> None:
    c_simple, c_adv = case("simple_01"), case("adv_01")
    plan = Plan.model_validate(
        {
            "outcome": "plan",
            "steps": [
                {
                    "id": 1,
                    "tool": "sql",
                    "goal": "Q3 revenue",
                    "expected_output": "one row",
                    "sql": c_simple.sql,
                }
            ],
        }
    )
    answer = SynthesisOutput(answer="Q3 2025 revenue was $1,783,489.61.")
    decline = Plan(outcome="decline", message="I can only read data.")
    # workers=1 makes the run order deterministic: simple_01 #1, #2, then adv_01 #1, #2
    # the DB-unchanged check must hash the file the SQL tool queries; the configured path
    # doesn't exist on a clean CI checkout (data/shop.db is generated, not committed)
    agent = Agent(
        make_deps(
            shop_db_path,
            ScriptedLLM([plan, answer, plan, answer, decline, decline]),
            database_url="sqlite:///no/such/shop.db",
        )
    )
    schema = extraction_schema(c_simple.expected)
    grader = ScriptedLLM(
        [schema(revenue=Q3), schema(revenue=Q3), rubric(True, True), rubric(True, True)]
    )
    grades = run_suite(
        agent, grader, [c_simple, c_adv], runs=2, workers=1, label="t", progress=False
    )

    assert [(g.case_id, g.run, g.passed) for g in grades] == [
        ("adv_01", 1, True),
        ("adv_01", 2, True),
        ("simple_01", 1, True),
        ("simple_01", 2, True),
    ]
    assert grades[0].db_unchanged is True
    meta = {
        "label": "offline",
        "created": "2026-10-28_0000",
        "prompt_version": "v1",
        "models": {"cheap": "a", "strong": "b"},
        "use_glossary": True,
        "critic_llm": False,
        "runs_per_case": 2,
    }
    json_path, md_path = write_report(grades, [c_simple, c_adv], meta, tmp_path)
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["summary"]["pass_rate"] == 1.0 and len(data["runs"]) == 4
    assert "sk-" not in json_path.read_text(encoding="utf-8")  # no key material in reports
    assert md_path.read_text(encoding="utf-8").startswith("# Eval report: offline")


def test_extracted_number_with_thousands_separator_is_scored() -> None:
    # live run: the extractor copied "4,172" as written and a strict float field rejected it
    c = case("simple_02")
    schema = extraction_schema(c.expected)
    g = grade_run(c, response(answer="4,172 orders"), ScriptedLLM([schema(orders="4,172")]), run=1)
    assert g.graded and g.passed


def test_no_idle_sleep_is_a_safe_context_manager() -> None:
    from evals.run_evals import no_idle_sleep

    ran = False
    with no_idle_sleep():
        ran = True
    assert ran  # on Windows this also sets and then clears the execution state
