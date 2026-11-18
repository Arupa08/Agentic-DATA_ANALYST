"""Phase 2 exit check: every eval case has a verified ground-truth query and result.

    uv run python -m evals.verify            # verify against data/shop.db
    uv run python -m evals.verify --show     # print what each query returns
    uv run python -m evals.verify --freeze   # re-lock the case files after a deliberate edit

For every case (and every interpretation of an underspecified case) this re-runs the
ground-truth SQL on a read-only connection and requires the stored answer to match it
exactly. For dirty-data cases it also runs the naive SQL and requires that it FAILS the
case's own tolerance, which proves the trap is real.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
import sys
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from data.generate import DEFAULT_OUT
from evals.models import CATEGORY_COUNTS, EvalCase, EvalSet, FieldSpec, load_cases
from evals.scoring import all_pass, score_fields

EVALS_DIR = Path(__file__).resolve().parent
CASES_PATH = EVALS_DIR / "cases.yaml"
HOLDOUT_PATH = EVALS_DIR / "holdout" / "holdout.yaml"
LOCK_PATH = EVALS_DIR / "cases.lock"
HOLDOUT_SIZE = 5


def open_readonly(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    conn.execute("PRAGMA query_only = ON")
    return conn


def run_truth(
    conn: sqlite3.Connection, sql: str, expected: Mapping[str, FieldSpec]
) -> dict[str, Any]:
    """Run SQL and map its result columns onto the expected field names."""
    cur = conn.execute(sql)
    cols = [d[0] for d in cur.description]
    rows = cur.fetchall()
    if missing := set(expected) - set(cols):
        raise ValueError(f"query does not return columns {sorted(missing)}; got {cols}")
    if any(spec.match in ("set", "ordered") for spec in expected.values()):
        return {name: [row[cols.index(name)] for row in rows] for name in expected}
    if len(rows) != 1:
        raise ValueError(f"scalar fields need exactly one row, query returned {len(rows)}")
    return {name: rows[0][cols.index(name)] for name in expected}


def _same(stored: Any, computed: Any) -> bool:
    if isinstance(stored, int | float) and isinstance(computed, int | float):
        return math.isclose(stored, computed, rel_tol=1e-9, abs_tol=1e-9)
    return bool(stored == computed)


def _check_truth(
    conn: sqlite3.Connection, label: str, sql: str, expected: Mapping[str, FieldSpec]
) -> list[str]:
    try:
        got = run_truth(conn, sql, expected)
    except (sqlite3.Error, ValueError) as exc:
        return [f"{label}: ground-truth query failed: {exc}"]
    return [
        f"{label}.{name}: stored {spec.value!r} but query returns {got[name]!r}"
        for name, spec in expected.items()
        if not _same(spec.value, got[name])
    ]


def verify_case(conn: sqlite3.Connection, case: EvalCase) -> list[str]:
    problems: list[str] = []
    if case.sql is not None:
        problems += _check_truth(conn, case.id, case.sql, case.expected)
    for i, interp in enumerate(case.interpretations):
        problems += _check_truth(conn, f"{case.id}[{i}]", interp.sql, interp.expected)
    if case.naive_sql is not None:
        try:
            naive = run_truth(conn, case.naive_sql, case.expected)
        except (sqlite3.Error, ValueError) as exc:
            problems.append(f"{case.id}: naive query failed to run: {exc}")
        else:
            if all_pass(score_fields(case.expected, naive)):
                problems.append(f"{case.id}: naive query PASSES ({naive}); the trap is not a trap")
    return problems


def verify(conn: sqlite3.Connection, main: EvalSet, holdout: EvalSet) -> list[str]:
    problems: list[str] = []
    counts = Counter(c.category for c in main.cases)
    if dict(counts) != CATEGORY_COUNTS:
        problems.append(f"category counts {dict(counts)} != plan {CATEGORY_COUNTS}")
    if len(holdout.cases) != HOLDOUT_SIZE:
        problems.append(f"holdout has {len(holdout.cases)} cases, expected {HOLDOUT_SIZE}")
    if overlap := {c.id for c in main.cases} & {c.id for c in holdout.cases}:
        problems.append(f"ids shared between main and holdout: {sorted(overlap)}")
    if overlap_q := {c.question for c in main.cases} & {c.question for c in holdout.cases}:
        problems.append(f"questions shared between main and holdout: {sorted(overlap_q)}")
    for case in [*main.cases, *holdout.cases]:
        problems += verify_case(conn, case)
    return problems


def file_digest(path: Path) -> str:
    # normalize line endings so the lock holds on Windows and Linux checkouts alike
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def current_lock() -> dict[str, str]:
    return {p.relative_to(EVALS_DIR).as_posix(): file_digest(p) for p in (CASES_PATH, HOLDOUT_PATH)}


def check_lock() -> list[str]:
    if not LOCK_PATH.exists():
        return ["evals/cases.lock missing; run `python -m evals.verify --freeze`"]
    locked = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    return [
        f"{name} changed since it was frozen; if deliberate, re-run with --freeze"
        for name, digest in current_lock().items()
        if locked.get(name) != digest
    ]


def _show(conn: sqlite3.Connection, cases: list[EvalCase]) -> None:
    for case in cases:
        targets = [(case.id, case.sql, case.expected)]
        targets += [
            (f"{case.id}[{i}]", it.sql, it.expected) for i, it in enumerate(case.interpretations)
        ]
        if case.naive_sql is not None:
            targets.append((f"{case.id} naive", case.naive_sql, case.expected))
        for label, sql, expected in targets:
            if sql is None:
                continue
            try:
                print(f"{label:<18} {run_truth(conn, sql, expected)}")
            except (sqlite3.Error, ValueError) as exc:
                print(f"{label:<18} ERROR {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--db", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()

    main_set, holdout_set = load_cases(CASES_PATH), load_cases(HOLDOUT_PATH)
    conn = open_readonly(args.db)
    try:
        if args.show:
            _show(conn, [*main_set.cases, *holdout_set.cases])
            return
        problems = verify(conn, main_set, holdout_set)
    finally:
        conn.close()
    if args.freeze and not problems:
        LOCK_PATH.write_text(json.dumps(current_lock(), indent=2) + "\n", encoding="utf-8")
        print(f"froze {LOCK_PATH.name}")
    problems += check_lock()
    for p in problems:
        print(f"FAIL {p}")
    total = len(main_set.cases) + len(holdout_set.cases)
    if problems:
        sys.exit(f"{len(problems)} problem(s) across {total} cases")
    print(f"OK: {len(main_set.cases)} cases + {len(holdout_set.cases)} held-out verified")


if __name__ == "__main__":
    main()
