"""Data layer for the eval dashboard: plain Python, no Streamlit, fully unit-tested.

Reads only files inside evals/reports/ (finished reports) and evals/reports/live/
(progress logs). Callers pick reports from `list_reports()`; nothing here takes a
user-typed path, so there is no path traversal to worry about.
"""

from __future__ import annotations

import datetime as dt
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPORTS_DIR = Path(__file__).resolve().parents[1] / "evals" / "reports"
LIVE_DIR = REPORTS_DIR / "live"
STALE_AFTER = dt.timedelta(minutes=15)  # no progress line for this long: likely stopped


@dataclass(frozen=True)
class ReportRef:
    path: Path
    label: str
    created: str  # "YYYY-MM-DD_HHMM"
    runs: int
    pass_rate: float | None

    @property
    def title(self) -> str:
        rate = "n/a" if self.pass_rate is None else f"{self.pass_rate * 100:.0f}%"
        return f"{self.created.replace('_', ' ')} · {self.label} · {self.runs} runs · {rate}"


def load_report(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


def list_reports(reports_dir: Path = REPORTS_DIR) -> list[ReportRef]:
    """Finished reports, newest first. Unreadable files are skipped, not fatal."""
    refs: list[ReportRef] = []
    for path in reports_dir.glob("*.json"):
        try:
            data = load_report(path)
            meta, summary = data["meta"], data["summary"]
            refs.append(
                ReportRef(
                    path=path,
                    label=str(meta.get("label", path.stem)),
                    created=str(meta.get("created", "")),
                    runs=int(summary.get("runs", 0)),
                    pass_rate=summary.get("pass_rate"),
                )
            )
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return sorted(refs, key=lambda r: (r.created, r.path.name), reverse=True)


# ---------------------------------------------------------------- one report


def kpis(report: dict[str, Any]) -> dict[str, Any]:
    s = report["summary"]
    sc = s.get("self_correction", {})
    return {
        "pass_rate": s.get("pass_rate"),
        "passed": s.get("passed", 0),
        "graded": s.get("graded", 0),
        "runs": s.get("runs", 0),
        "ungraded": s.get("ungraded", 0),
        "not_run_quota": s.get("not_run_quota", 0),
        "cases_passing_every_run": s.get("cases_passing_every_run", 0),
        "cases": s.get("cases", 0),
        "p50_latency_s": _secs(s.get("latency_ms", {}).get("p50")),
        "p95_latency_s": _secs(s.get("latency_ms", {}).get("p95")),
        "mean_tokens": s.get("mean_tokens_per_query"),
        "mean_cost_usd": s.get("cost", {}).get("mean_per_query_usd"),
        "runs_with_replan": sc.get("runs_with_replan", 0),
        "passed_after_replan": sc.get("passed_after_replan", 0),
        "critic_false_alarms": sc.get("critic_false_alarms", 0),
        "critic_misses": sc.get("critic_misses", 0),
    }


def _secs(ms: float | None) -> float | None:
    return None if ms is None else round(ms / 1000, 1)


def category_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for name, c in report["summary"].get("by_category", {}).items():
        rate = c.get("pass_rate")
        rows.append(
            {
                "category": name,
                "pass_rate_pct": None if rate is None else round(rate * 100, 1),
                "passed": c.get("passed", 0),
                "runs": c.get("runs", 0),
                "mean_cost_usd": c.get("mean_cost_usd"),
                "p50_latency_s": _secs(c.get("p50_latency_ms")),
            }
        )
    return sorted(rows, key=lambda r: r["category"])


def taxonomy_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [
        {
            "root_cause": cause,
            "runs": info.get("count", 0),
            "examples": ", ".join(info.get("runs", [])[:8]),
        }
        for cause, info in report["summary"].get("failure_taxonomy", {}).items()
    ]
    return sorted(rows, key=lambda r: (-r["runs"], r["root_cause"]))


def case_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    """One row per case: how many runs passed, what happened, cost and latency."""
    by_case: dict[str, list[dict[str, Any]]] = {}
    for run in report.get("runs", []):
        by_case.setdefault(run["case_id"], []).append(run)
    rows = []
    for case_id, runs in sorted(by_case.items()):
        passed = sum(bool(r.get("passed")) for r in runs)
        graded = sum(bool(r.get("graded", True)) for r in runs)
        ungraded = len(runs) - graded
        statuses = Counter(r.get("status", "?") for r in runs)
        causes = Counter(r["root_cause"] for r in runs if r.get("root_cause"))
        latencies = [r.get("latency_ms", 0) for r in runs]
        rows.append(
            {
                "case_id": case_id,
                "category": runs[0].get("category", ""),
                # over graded runs only: an ungraded run is not a failure
                "passed": (f"{passed}/{graded}" if graded else "-")
                + (f" ({ungraded} ungraded)" if ungraded else ""),
                "pass_fraction": passed / graded if graded else None,
                "statuses": ", ".join(f"{k} x{v}" for k, v in statuses.most_common()),
                "root_causes": ", ".join(f"{k} x{v}" for k, v in causes.most_common()),
                "mean_latency_s": round(sum(latencies) / len(latencies) / 1000, 1)
                if latencies
                else None,
                "ungraded": ungraded,
            }
        )
    return rows


def case_runs(report: dict[str, Any], case_id: str) -> list[dict[str, Any]]:
    return sorted(
        (r for r in report.get("runs", []) if r.get("case_id") == case_id),
        key=lambda r: r.get("run", 0),
    )


def safety_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "case_id": cid,
            "blocked": a.get("passed", ""),
            "caught_by": ", ".join(a.get("caught_by", [])) or "none recorded",
            "db_unchanged": bool(a.get("db_unchanged", True)),
            "leaks": ", ".join(a.get("leaked", [])) or "none",
        }
        for cid, a in sorted(report["summary"].get("adversarial", {}).items())
    ]


def layer_counts(report: dict[str, Any]) -> list[dict[str, Any]]:
    """How many adversarial runs each safety layer stopped something in."""
    counts: Counter[str] = Counter()
    for run in report.get("runs", []):
        if run.get("behavior") == "blocked":
            counts.update(run.get("safety_layers", []))
    return [{"layer": k, "runs": v} for k, v in counts.most_common()]


# ---------------------------------------------------------------- across reports


def compare_rows(base: dict[str, Any], other: dict[str, Any]) -> list[dict[str, Any]]:
    """Metric-by-metric comparison; `change` is other minus base."""
    kb, ko = kpis(base), kpis(other)
    rows: list[dict[str, Any]] = []

    def add(metric: str, a: float | None, b: float | None, unit: str) -> None:
        change = None if a is None or b is None else round(b - a, 3)
        rows.append({"metric": metric, "base": a, "other": b, "change": change, "unit": unit})

    def pct(v: float | None) -> float | None:
        return None if v is None else round(v * 100, 1)

    add("Pass rate", pct(kb["pass_rate"]), pct(ko["pass_rate"]), "pp")
    add("Mean cost per query", kb["mean_cost_usd"], ko["mean_cost_usd"], "USD")
    add("p50 latency", kb["p50_latency_s"], ko["p50_latency_s"], "s")
    add("p95 latency", kb["p95_latency_s"], ko["p95_latency_s"], "s")
    add("Mean tokens per query", kb["mean_tokens"], ko["mean_tokens"], "tokens")
    add("Critic false alarms", kb["critic_false_alarms"], ko["critic_false_alarms"], "runs")
    cats_b = {r["category"]: r["pass_rate_pct"] for r in category_rows(base)}
    cats_o = {r["category"]: r["pass_rate_pct"] for r in category_rows(other)}
    for cat in sorted(set(cats_b) | set(cats_o)):
        add(f"Pass rate: {cat}", cats_b.get(cat), cats_o.get(cat), "pp")
    return rows


def trend_rows(refs: list[ReportRef]) -> list[dict[str, Any]]:
    """Pass rate over time, oldest first, skipping reports with nothing graded."""
    return [
        {
            "created": r.created,
            "label": r.label,
            "pass_rate_pct": round(r.pass_rate * 100, 1),
            "runs": r.runs,
        }
        for r in sorted(refs, key=lambda r: r.created)
        if r.pass_rate is not None
    ]


# ---------------------------------------------------------------- live runs


def hours_minutes(seconds: float) -> str:
    """Whole hours and minutes, truncated (0.515 h is "0h 30m", not a rounded-up "1h")."""
    total_minutes = max(0, int(seconds // 60))
    hours, minutes = divmod(total_minutes, 60)
    return f"{hours}h {minutes:02d}m"


@dataclass
class LiveStatus:
    path: Path
    label: str
    total: int
    started: dt.datetime | None
    runs: list[dict[str, Any]] = field(default_factory=list)
    ended: bool = False
    stopped_early: bool = False
    report: str | None = None
    last_event: dt.datetime | None = None

    @property
    def done(self) -> int:
        return len(self.runs)

    @property
    def passed(self) -> int:
        return sum(bool(r.get("passed")) for r in self.runs)

    @property
    def graded(self) -> int:
        return sum(bool(r.get("graded")) for r in self.runs)

    @property
    def running_pass_rate(self) -> float | None:
        return self.passed / self.graded if self.graded else None

    def eta_seconds(self, now: dt.datetime) -> float | None:
        if self.ended or not self.started or not self.done:
            return None
        per_run = (now - self.started).total_seconds() / self.done
        return per_run * (self.total - self.done)

    def is_stale(self, now: dt.datetime) -> bool:
        last = self.last_event or self.started
        return not self.ended and last is not None and now - last > STALE_AFTER

    def state(self, now: dt.datetime) -> str:
        if self.ended:
            return "stopped early" if self.stopped_early else "finished"
        return "no progress lately (stopped?)" if self.is_stale(now) else "running"


def _when(value: Any) -> dt.datetime | None:
    try:
        return dt.datetime.fromisoformat(str(value))
    except ValueError:
        return None


def read_live(path: Path) -> LiveStatus | None:
    """Parse a progress log; tolerant of a half-written last line."""
    status: LiveStatus | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        kind = event.get("type")
        if kind == "start":
            status = LiveStatus(
                path=path,
                label=str(event.get("label", "")),
                total=int(event.get("total", 0)),
                started=_when(event.get("started")),
            )
            status.last_event = status.started
        elif status is None:
            continue
        elif kind == "run":
            status.runs.append(event)
            status.last_event = _when(event.get("at")) or status.last_event
        elif kind == "end":
            status.ended = True
            status.stopped_early = bool(event.get("stopped_early"))
            status.report = event.get("report")
            status.last_event = _when(event.get("at")) or status.last_event
    return status


def latest_live(live_dir: Path = LIVE_DIR) -> LiveStatus | None:
    files = sorted(live_dir.glob("*.progress.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    for path in files:
        status = read_live(path)
        if status is not None:
            return status
    return None
