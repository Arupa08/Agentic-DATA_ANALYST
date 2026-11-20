"""Run the agent on the frozen eval cases N times each, grade every run, write a report.

    uv run python -m evals.run_evals --label baseline              # all 30 cases x 3
    uv run python -m evals.run_evals --ids multi_01 --runs 1       # quick check
    uv run python -m evals.run_evals --routing all-strong --label all-strong
    uv run python -m evals.run_evals --no-glossary --label no-glossary
    uv run python -m evals.run_evals --holdout --label holdout     # phase ends only

Writes evals/reports/<date>_<label>.json and .md. Compare two reports with
`python -m evals.compare A.json B.json`.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal

from agent import AGENT_REVISION
from agent.config import Settings, get_settings
from agent.deps import build_deps
from agent.graph import Agent
from agent.llm import LangChainLLM, RateLimited, StructuredLLM
from agent.logging_setup import configure_logging
from agent.prompts import PROMPT_VERSION
from agent.safety.sandbox import warm_up
from evals.grade import RunGrade, grade_run
from evals.models import CATEGORY_COUNTS, EvalCase, load_cases
from evals.progress import ProgressLog
from evals.report import summarize, to_markdown
from evals.verify import CASES_PATH, HOLDOUT_PATH

REPORTS_DIR = Path(__file__).resolve().parent / "reports"
Routing = Literal["cheap+strong", "all-strong", "all-cheap"]


@contextmanager
def no_idle_sleep() -> Iterator[None]:
    """Ask Windows not to idle-sleep while a long (local-model) run is going.

    Standard per-process request (SetThreadExecutionState); changes no settings and ends
    when this block exits. Must be held by the main thread. A closed lid still sleeps.
    """
    if sys.platform != "win32":
        yield
        return
    import ctypes

    kernel32 = getattr(ctypes, "windll").kernel32  # noqa: B009 - windll exists only on Windows
    es_continuous, es_system_required = 0x80000000, 0x00000001
    kernel32.SetThreadExecutionState(es_continuous | es_system_required)
    try:
        yield
    finally:
        kernel32.SetThreadExecutionState(es_continuous)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def configure(base: Settings, *, routing: Routing, glossary: bool, critic_llm: bool) -> Settings:
    update: dict[str, Any] = {"use_glossary": glossary, "critic_llm": critic_llm}
    if routing == "all-strong":
        update["cheap_model"] = base.strong_model
    elif routing == "all-cheap":
        update["strong_model"] = base.cheap_model
    return base.model_copy(update=update)


def select_cases(
    cases: list[EvalCase], categories: set[str] | None, ids: set[str] | None, limit: int | None
) -> list[EvalCase]:
    if ids:
        chosen = [c for c in cases if c.id in ids]
    else:
        chosen = [c for c in cases if categories is None or c.category in categories]
    return chosen[:limit] if limit else chosen


QUOTA_STOP = "not run: provider rate limit or quota reached"


def _ungraded(case: EvalCase, run: int, reason: str) -> RunGrade:
    return RunGrade(
        case_id=case.id,
        category=case.category,
        behavior=case.expected_behavior,
        run=run,
        status="failed",
        graded=False,
        grading_error=reason[:300],
    )


def run_one(
    agent: Agent,
    grader: StructuredLLM,
    case: EvalCase,
    run: int,
    *,
    label: str,
    stop: threading.Event | None = None,
) -> RunGrade:
    if stop is not None and stop.is_set():
        return _ungraded(case, run, QUOTA_STOP)
    db = agent.deps.sql.db_path  # the file the agent actually queries, not just the setting
    before = file_sha256(db) if case.expected_behavior == "blocked" else None
    try:
        response = agent.ask(case.question, case_id=case.id, tags=("eval", label))
        unchanged = file_sha256(db) == before if before is not None else None
        return grade_run(case, response, grader, run=run, db_unchanged=unchanged)
    except RateLimited:
        # quota is not an agent failure: stop the suite and report these runs as not run
        if stop is not None:
            stop.set()
        return _ungraded(case, run, QUOTA_STOP)
    except Exception as exc:  # a harness must survive one bad run
        return _ungraded(case, run, f"agent crashed: {type(exc).__name__}")


def run_suite(
    agent: Agent,
    grader: StructuredLLM,
    cases: list[EvalCase],
    *,
    runs: int,
    workers: int,
    label: str,
    progress: bool = True,
    progress_log: ProgressLog | None = None,
) -> list[RunGrade]:
    """Runs case-major (all of case 1's runs first), so a quota stop still leaves
    complete N-run results for the cases it reached."""
    tasks = [(case, n) for case in cases for n in range(1, runs + 1)]
    grades: list[RunGrade] = []
    lock = threading.Lock()
    stop = threading.Event()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(run_one, agent, grader, c, n, label=label, stop=stop) for c, n in tasks
        ]
        for future in as_completed(futures):
            grade = future.result()
            with lock:
                grades.append(grade)
                if progress_log is not None:
                    progress_log.run(grade)
                if progress:
                    mark = "PASS" if grade.passed else ("SKIP" if not grade.graded else "FAIL")
                    why = grade.root_cause or grade.grading_error or ""
                    print(
                        f"[{len(grades):>3}/{len(tasks)}] {mark} {grade.case_id:<11} "
                        f"#{grade.run} {grade.status:<19} ${grade.cost_usd:.4f} "
                        f"{grade.latency_ms / 1000:5.1f}s {why}",
                        flush=True,
                    )
    if stop.is_set() and progress:
        print(
            "\nStopped early: provider rate limit or quota. Re-run later with --ids for the rest."
        )
    return sorted(grades, key=lambda g: (g.case_id, g.run))


def write_report(
    grades: list[RunGrade], cases: list[EvalCase], meta: dict[str, Any], out_dir: Path
) -> tuple[Path, Path]:
    summary = summarize(grades, cases)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{meta['created']}_{meta['label']}"
    json_path, md_path = out_dir / f"{stem}.json", out_dir / f"{stem}.md"
    payload = {"meta": meta, "summary": summary, "runs": [g.model_dump() for g in grades]}
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    md_path.write_text(to_markdown(meta, summary), encoding="utf-8")
    return json_path, md_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--categories", help=f"comma list of {sorted(CATEGORY_COUNTS)}")
    parser.add_argument("--ids", help="comma-separated case ids")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--routing", choices=["cheap+strong", "all-strong", "all-cheap"], default="cheap+strong"
    )
    parser.add_argument("--no-glossary", action="store_true")
    parser.add_argument("--no-critic-llm", action="store_true")
    parser.add_argument("--holdout", action="store_true", help="held-out set (phase ends only)")
    parser.add_argument("--label", default="run")
    args = parser.parse_args()

    settings = configure(
        get_settings(),
        routing=args.routing,
        glossary=not args.no_glossary,
        critic_llm=not args.no_critic_llm,
    )
    if not settings.db_path.exists():
        sys.exit(f"{settings.db_path} not found; run `make data` first")
    configure_logging("WARNING")
    warm_up()

    all_cases = load_cases(HOLDOUT_PATH if args.holdout else CASES_PATH).cases
    cases = select_cases(
        all_cases,
        set(args.categories.split(",")) if args.categories else None,
        set(args.ids.split(",")) if args.ids else None,
        args.limit,
    )
    agent = Agent(build_deps(settings))
    grader = LangChainLLM(get_settings())  # grading always uses the default models
    meta = {
        "label": args.label,
        "created": dt.datetime.now().strftime("%Y-%m-%d_%H%M"),
        "prompt_version": PROMPT_VERSION,
        "agent_revision": AGENT_REVISION,
        "clarify_nudge": settings.clarify_nudge,
        "models": {"cheap": settings.cheap_model, "strong": settings.strong_model},
        "routing": args.routing,
        "use_glossary": settings.use_glossary,
        "critic_llm": settings.critic_llm,
        "runs_per_case": args.runs,
        "holdout": args.holdout,
        "case_ids": [c.id for c in cases],
    }
    live = ProgressLog(
        REPORTS_DIR / "live" / f"{meta['created']}_{args.label}.progress.jsonl",
        label=args.label,
        total=len(cases) * args.runs,
        meta=meta,
    )
    with no_idle_sleep():
        grades = run_suite(
            agent,
            grader,
            cases,
            runs=args.runs,
            workers=args.workers,
            label=args.label,
            progress_log=live,
        )
    json_path, md_path = write_report(grades, cases, meta, REPORTS_DIR)
    stopped = any((g.grading_error or "").startswith("not run:") for g in grades)
    live.end(json_path, stopped_early=stopped)
    print(f"\n{md_path.read_text(encoding='utf-8')}\nreport: {json_path}\n        {md_path}")


if __name__ == "__main__":
    main()
