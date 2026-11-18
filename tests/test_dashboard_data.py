"""Dashboard data layer and the live progress log (no Streamlit needed)."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pytest

from evals.extract import extraction_schema
from evals.grade import RunGrade, grade_run
from evals.progress import ProgressLog
from evals.run_evals import write_report
from tests.agent_fakes import ScriptedLLM
from tests.eval_fixtures import case, response, rubric_verdict
from ui.dashboard_data import (
    STALE_AFTER,
    case_rows,
    case_runs,
    category_rows,
    compare_rows,
    kpis,
    latest_live,
    layer_counts,
    list_reports,
    load_report,
    read_live,
    safety_rows,
    taxonomy_rows,
    trend_rows,
)

Q3 = 1783489.61


def _grades() -> list[RunGrade]:
    simple, adv = case("simple_01"), case("adv_01")
    schema = extraction_schema(simple.expected)
    ok_rubric = rubric_verdict(True, True)
    return [
        grade_run(
            simple,
            response(cost_usd=0.01, latency_ms=1000),
            ScriptedLLM([schema(revenue=Q3)]),
            run=1,
        ),
        grade_run(
            simple,
            response(rows=[[1.0]], latency_ms=3000),
            ScriptedLLM([schema(revenue=1.0)]),
            run=2,
        ),
        grade_run(
            adv,
            response("declined", latency_ms=500),
            ScriptedLLM([ok_rubric]),
            run=1,
            db_unchanged=True,
        ),
    ]


def _meta(label: str, created: str) -> dict[str, object]:
    return {
        "label": label,
        "created": created,
        "prompt_version": "v1",
        "models": {"cheap": "ollama:m", "strong": "ollama:m"},
        "use_glossary": True,
        "critic_llm": True,
        "runs_per_case": 2,
    }


@pytest.fixture
def reports_dir(tmp_path: Path) -> Path:
    cases = [case("simple_01"), case("adv_01")]
    write_report(_grades(), cases, _meta("baseline", "2026-11-17_1200"), tmp_path)
    write_report(_grades()[:1], [case("simple_01")], _meta("after", "2026-11-18_0900"), tmp_path)
    (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")
    return tmp_path


def test_list_reports_newest_first_and_skips_broken(reports_dir: Path) -> None:
    refs = list_reports(reports_dir)
    assert [r.label for r in refs] == ["after", "baseline"]
    assert refs[1].title == "2026-11-17 1200 · baseline · 3 runs · 67%"


def test_report_views(reports_dir: Path) -> None:
    report = load_report(next(r.path for r in list_reports(reports_dir) if r.label == "baseline"))
    k = kpis(report)
    assert k["pass_rate"] == pytest.approx(2 / 3, abs=1e-3) and k["p95_latency_s"] == 3.0
    assert {r["category"]: r["pass_rate_pct"] for r in category_rows(report)} == {
        "adversarial": 100.0,
        "simple_lookup": 50.0,
    }
    assert taxonomy_rows(report) == [
        {"root_cause": "sql_error", "runs": 1, "examples": "simple_01#2"}
    ]
    simple = next(r for r in case_rows(report) if r["case_id"] == "simple_01")
    assert simple["passed"] == "1/2" and simple["root_causes"] == "sql_error x1"
    assert [r["run"] for r in case_runs(report, "simple_01")] == [1, 2]
    assert safety_rows(report)[0]["caught_by"] == "planner_refusal"
    assert layer_counts(report) == [{"layer": "planner_refusal", "runs": 1}]


def test_compare_and_trend(reports_dir: Path) -> None:
    refs = list_reports(reports_dir)
    after, base = load_report(refs[0].path), load_report(refs[1].path)
    rows = {r["metric"]: r for r in compare_rows(base, after)}
    assert rows["Pass rate"]["change"] == pytest.approx(33.3, abs=0.1)
    assert rows["Pass rate: adversarial"]["other"] is None  # category absent in the newer report
    assert [t["label"] for t in trend_rows(refs)] == ["baseline", "after"]


# ---------------------------------------------------------------- live progress


def test_progress_log_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "live" / "x.progress.jsonl"
    log = ProgressLog(path, label="baseline", total=4, meta={"label": "baseline"})
    for g in _grades()[:2]:
        log.run(g)
    status = read_live(path)
    assert status is not None and status.done == 2 and status.passed == 1 and not status.ended
    assert status.running_pass_rate == 0.5
    now = dt.datetime.now()
    eta = status.eta_seconds(now)
    assert eta is not None and eta >= 0
    assert status.state(now) == "running"
    log.end(tmp_path / "report.json", stopped_early=False)
    finished = read_live(path)
    assert finished is not None and finished.ended and finished.state(now) == "finished"
    assert finished.eta_seconds(now) is None


def test_progress_log_never_contains_answers(tmp_path: Path) -> None:
    path = tmp_path / "p.jsonl"
    log = ProgressLog(path, label="l", total=1, meta={})
    grade = _grades()[0]
    grade.answer = "Revenue was $1,783,489.61 for kara.davis5550@outlook.com"
    log.run(grade)
    assert "kara.davis" not in path.read_text(encoding="utf-8")


def test_half_written_line_and_staleness(tmp_path: Path) -> None:
    path = tmp_path / "p.progress.jsonl"
    started = dt.datetime.now() - STALE_AFTER - dt.timedelta(minutes=5)
    lines = [
        json.dumps({"type": "start", "label": "l", "total": 3, "started": started.isoformat()}),
        json.dumps(
            {
                "type": "run",
                "case_id": "a",
                "passed": True,
                "graded": True,
                "at": started.isoformat(),
            }
        ),
        '{"type": "run", "case_id": "b", "pass',  # crashed mid-write
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    status = read_live(path)
    assert status is not None and status.done == 1
    assert status.state(dt.datetime.now()) == "no progress lately (stopped?)"


def test_latest_live_picks_newest(tmp_path: Path) -> None:
    assert latest_live(tmp_path) is None
    ProgressLog(tmp_path / "old.progress.jsonl", label="old", total=1, meta={})
    import os
    import time

    os.utime(tmp_path / "old.progress.jsonl", (time.time() - 100, time.time() - 100))
    ProgressLog(tmp_path / "new.progress.jsonl", label="new", total=1, meta={})
    status = latest_live(tmp_path)
    assert status is not None and status.label == "new"


def test_dashboard_never_renders_model_text_as_html() -> None:
    source = (Path(__file__).resolve().parents[1] / "ui" / "dashboard.py").read_text(
        encoding="utf-8"
    )
    assert "unsafe_allow_html" not in source
    assert 'st.text(run.get("answer")' in source  # answers as plain text, not markdown


def test_ungraded_runs_are_not_shown_as_failures() -> None:
    graded_pass = {"case_id": "c", "category": "x", "run": 1, "passed": True, "graded": True}
    ungraded = {"case_id": "c", "category": "x", "run": 2, "passed": False, "graded": False}
    only_ungraded = {"case_id": "d", "category": "x", "run": 1, "passed": False, "graded": False}
    rows = {r["case_id"]: r for r in case_rows({"runs": [graded_pass, ungraded, only_ungraded]})}
    assert rows["c"]["passed"] == "1/1 (1 ungraded)" and rows["c"]["pass_fraction"] == 1.0
    assert rows["d"]["passed"] == "- (1 ungraded)" and rows["d"]["pass_fraction"] is None


@pytest.mark.parametrize(
    ("seconds", "shown"),
    [
        (1854, "0h 30m"),
        (1236, "0h 20m"),
        (5400, "1h 30m"),
        (59, "0h 00m"),
        (-5, "0h 00m"),
        (7322, "2h 02m"),
    ],
)
def test_hours_minutes_truncates_instead_of_rounding(seconds: float, shown: str) -> None:
    # live check: 1,854 s was displayed as "1h 31m" because hours were rounded, not truncated
    from ui.dashboard_data import hours_minutes

    assert hours_minutes(seconds) == shown
