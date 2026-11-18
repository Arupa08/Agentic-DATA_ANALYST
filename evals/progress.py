"""Live progress for long eval runs: one JSON line per event, for the dashboard.

    {"type": "start", "label": ..., "total": 90, "started": ..., "meta": {...}}
    {"type": "run", "case_id": ..., "passed": ..., "latency_ms": ..., "at": ...}   (per run)
    {"type": "end", "at": ..., "report": "evals/reports/<stem>.json", "stopped_early": false}

Files live in evals/reports/live/ (gitignored); the finished JSON/markdown report is the
record. Only grading summaries are written here, never answers or rows.
"""

from __future__ import annotations

import datetime as dt
import json
import threading
from pathlib import Path
from typing import Any

from evals.grade import RunGrade


def _now() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


class ProgressLog:
    def __init__(self, path: Path, *, label: str, total: int, meta: dict[str, Any]) -> None:
        self.path = path
        self._lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
        self._write(
            {"type": "start", "label": label, "total": total, "started": _now(), "meta": meta}
        )

    def _write(self, event: dict[str, Any]) -> None:
        with self._lock, self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, default=str) + "\n")

    def run(self, grade: RunGrade) -> None:
        self._write(
            {
                "type": "run",
                "at": _now(),
                "case_id": grade.case_id,
                "category": grade.category,
                "run": grade.run,
                "passed": grade.passed,
                "graded": grade.graded,
                "status": grade.status,
                "root_cause": grade.root_cause,
                "grading_error": (grade.grading_error or "")[:200] or None,
                "latency_ms": grade.latency_ms,
                "cost_usd": grade.cost_usd,
                "tokens": grade.tokens,
                "replans": grade.replans,
            }
        )

    def end(self, report: Path | None, *, stopped_early: bool = False) -> None:
        self._write(
            {
                "type": "end",
                "at": _now(),
                "report": report.as_posix() if report else None,
                "stopped_early": stopped_early,
            }
        )
