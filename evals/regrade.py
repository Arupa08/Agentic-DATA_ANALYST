"""Re-grade a finished report with the current grader, without re-running the agent.

    uv run python -m evals.regrade evals/reports/2026-11-17_1502_baseline-local.json

Only the judge-graded behaviors are re-scored (underspecified, no-answer, adversarial):
the agent's answers, SQL and statuses are taken from the report exactly as recorded.
Number-scored runs are copied over unchanged. The result is a NEW report labelled
"<label>-regraded"; the original stays as it was.

Limits: a report stores the agent's answer but not its step data, so for a re-graded
underspecified run that still fails, "synthesis_error" can't be told apart from
"sql_error". Reports written before attempt reasons and step errors were recorded give
the safety judge only the executed SQL as context.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any

from agent.config import get_settings
from agent.graph import AgentResponse
from agent.llm import LangChainLLM, StructuredLLM
from agent.logging_setup import configure_logging
from evals.grade import RunGrade, grade_run
from evals.models import EvalCase, load_cases
from evals.run_evals import REPORTS_DIR, no_idle_sleep, write_report
from evals.verify import CASES_PATH, HOLDOUT_PATH

JUDGED = frozenset({"clarify_or_assume", "no_data", "blocked"})
# what a re-grade cannot recompute from a stored run, so it keeps the recorded values
KEPT_FROM_RUN = ("safety_layers", "critic_false_alarms", "critic_miss", "replans")


def _response(record: dict[str, Any], case: EvalCase) -> AgentResponse:
    return AgentResponse(
        request_id=str(record.get("request_id", "")),
        question=case.question,
        status=record["status"],
        answer=str(record.get("answer", "")),
        assumptions=list(record.get("assumptions", [])),
        sql=list(record.get("sql", [])),
        replans=int(record.get("replans", 0)),
        cost_usd=float(record.get("cost_usd", 0.0)),
        total_tokens=int(record.get("tokens", 0)),
        latency_ms=int(record.get("latency_ms", 0)),
    )


def regrade_run(record: dict[str, Any], case: EvalCase, grader: StructuredLLM) -> RunGrade:
    if record.get("behavior") not in JUDGED or str(record.get("grading_error", "")).startswith(
        "not run:"
    ):
        return RunGrade.model_validate(record)  # number-scored or never run: unchanged
    grade = grade_run(
        case,
        _response(record, case),
        grader,
        run=int(record["run"]),
        db_unchanged=record.get("db_unchanged"),
        attempt_reasons=list(record.get("attempt_reasons", [])),
        step_errors=list(record.get("step_errors", [])),
    )
    for name in KEPT_FROM_RUN:
        if name in record:
            setattr(grade, name, record[name])
    return grade


def regrade_report(
    report: dict[str, Any], grader: StructuredLLM, cases: dict[str, EvalCase]
) -> list[RunGrade]:
    return [regrade_run(r, cases[r["case_id"]], grader) for r in report["runs"]]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("report", type=Path, help="a finished report JSON in evals/reports/")
    args = parser.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    meta = report["meta"]
    case_file = HOLDOUT_PATH if meta.get("holdout") else CASES_PATH
    cases = {c.id: c for c in load_cases(case_file).cases}
    missing = {r["case_id"] for r in report["runs"]} - set(cases)
    if missing:
        sys.exit(f"report has cases not in {case_file.name}: {sorted(missing)}")

    configure_logging("WARNING")
    grader = LangChainLLM(get_settings())
    judged = sum(r.get("behavior") in JUDGED for r in report["runs"])
    print(f"re-grading {judged} judge-graded runs of {len(report['runs'])} ...", flush=True)
    with no_idle_sleep():
        grades = regrade_report(report, grader, cases)

    new_meta = {
        **meta,
        "label": f"{meta['label']}-regraded",
        "created": dt.datetime.now().strftime("%Y-%m-%d_%H%M"),
        "regraded_from": args.report.name,
        "agent_run_created": meta.get("agent_run_created", meta["created"]),
    }
    used = [cases[cid] for cid in dict.fromkeys(r["case_id"] for r in report["runs"])]
    json_path, md_path = write_report(grades, used, new_meta, REPORTS_DIR)
    print(f"\n{md_path.read_text(encoding='utf-8')}\nreport: {json_path}\n        {md_path}")


if __name__ == "__main__":
    main()
