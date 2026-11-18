"""Read-only SQL tool: schema introspection plus guarded query execution.

Layers, each sufficient on its own to stop a write:
  1. sql_guard: AST allowlist (single SELECT, allowlisted tables, LIMIT cap)
  2. connection opened with SQLite mode=ro and PRAGMA query_only
  3. SQLite authorizer: only SELECT, reads of allowlisted tables, safe functions
  4. statement timeout (progress handler) and a 1 MB cap on any single value
"""

from __future__ import annotations

import logging
import sqlite3
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from agent.safety.sql_guard import (
    ALLOWED_TABLES,
    DEFAULT_ROW_CAP,
    FORBIDDEN_FUNCTION_PREFIXES,
    FORBIDDEN_FUNCTIONS,
    SqlGuardError,
    guard_sql,
)
from agent.safety.untrusted import wrap_untrusted

log = logging.getLogger(__name__)

DEFAULT_TIMEOUT_S = 5.0
MAX_VALUE_BYTES = 1_000_000
PREVIEW_ROWS = 50
MAX_CELL_CHARS = 200
LOW_CARDINALITY = 12

_ALLOWED_ACTIONS = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_RECURSIVE}


def _authorizer(
    action: int, arg1: str | None, arg2: str | None, db: str | None, src: str | None
) -> int:
    if action in _ALLOWED_ACTIONS:
        return sqlite3.SQLITE_OK
    if action == sqlite3.SQLITE_READ:
        if (arg1 or "").lower() in ALLOWED_TABLES and db in (None, "main"):
            return sqlite3.SQLITE_OK
        return sqlite3.SQLITE_DENY
    if action == sqlite3.SQLITE_FUNCTION:
        name = (arg2 or "").lower()
        if name in FORBIDDEN_FUNCTIONS or name.startswith(FORBIDDEN_FUNCTION_PREFIXES):
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK
    return sqlite3.SQLITE_DENY


def open_agent_connection(
    db_path: Path, timeout_s: float = DEFAULT_TIMEOUT_S
) -> sqlite3.Connection:
    """The only connection the agent's queries ever run on."""
    if not db_path.exists():
        # mode=ro would fail anyway, but never let sqlite create an empty file
        raise FileNotFoundError(db_path)
    conn = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True, check_same_thread=False)
    conn.execute("PRAGMA query_only = ON")
    conn.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, MAX_VALUE_BYTES)
    conn.setlimit(sqlite3.SQLITE_LIMIT_ATTACHED, 0)
    conn.set_authorizer(_authorizer)
    deadline = time.monotonic() + timeout_s

    def _past_deadline() -> int:
        return 1 if time.monotonic() > deadline else 0

    conn.set_progress_handler(_past_deadline, 10_000)
    return conn


class SqlResult(BaseModel):
    ok: bool
    sql: str  # what actually ran (guard output), or the rejected input
    columns: list[str] = []
    rows: list[list[Any]] = []
    row_count: int = 0
    truncated: bool = False  # hit the row cap; more rows exist
    error: str | None = None
    elapsed_ms: int = 0

    def to_llm_text(self, max_rows: int = PREVIEW_ROWS) -> str:
        """Compact rendering for prompts; row values are wrapped as untrusted."""
        if not self.ok:
            return f"SQL error: {self.error}"
        header = f"{self.row_count} row(s){' (truncated at cap)' if self.truncated else ''}"
        lines = [" | ".join(self.columns)]
        for row in self.rows[:max_rows]:
            lines.append(" | ".join(_cell(v) for v in row))
        if self.row_count > max_rows:
            lines.append(f"... {self.row_count - max_rows} more row(s) not shown")
        return f"{header}\n{wrap_untrusted('database', chr(10).join(lines))}"


def _cell(value: Any) -> str:
    text = "NULL" if value is None else str(value)
    return text if len(text) <= MAX_CELL_CHARS else text[:MAX_CELL_CHARS] + "…"


class SqlTool:
    def __init__(
        self,
        db_path: Path,
        *,
        row_cap: int = DEFAULT_ROW_CAP,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> None:
        self.db_path = db_path
        self.row_cap = row_cap
        self.timeout_s = timeout_s

    def run(self, sql: str) -> SqlResult:
        start = time.monotonic()
        try:
            guarded = guard_sql(sql, row_cap=self.row_cap + 1)  # +1 row detects truncation
        except SqlGuardError as exc:
            log.info("sql_rejected", extra={"reason": str(exc)})
            return SqlResult(ok=False, sql=sql, error=f"rejected by SQL guard: {exc}")
        conn = open_agent_connection(self.db_path, self.timeout_s)
        try:
            cur = conn.execute(guarded.sql)
            columns = [d[0] for d in cur.description]
            rows = [list(r) for r in cur.fetchmany(self.row_cap + 1)]
        except sqlite3.Error as exc:
            message = "query timed out" if "interrupted" in str(exc) else str(exc)
            log.info("sql_failed", extra={"error": message})
            return SqlResult(ok=False, sql=guarded.sql, error=message, elapsed_ms=_ms(start))
        finally:
            conn.close()
        truncated = len(rows) > self.row_cap
        rows = rows[: self.row_cap]
        log.info(
            "sql_ok",
            extra={"rows": len(rows), "truncated": truncated, "tables": sorted(guarded.tables)},
        )
        return SqlResult(
            ok=True,
            sql=guarded.sql,
            columns=columns,
            rows=rows,
            row_count=len(rows),
            truncated=truncated,
            elapsed_ms=_ms(start),
        )

    def schema_summary(self) -> str:
        return _schema_summary(self.db_path, self.db_path.stat().st_mtime_ns)

    def table_columns(self) -> dict[str, list[str]]:
        """Real column names of every allowlisted table (trusted introspection)."""
        return _table_columns(self.db_path, self.db_path.stat().st_mtime_ns)


def _ms(start: float) -> int:
    return int((time.monotonic() - start) * 1000)


def _quote(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


@lru_cache(maxsize=4)
def _table_columns(db_path: Path, _mtime_ns: int) -> dict[str, list[str]]:
    conn = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    try:
        return {
            table: [row[1] for row in conn.execute(f"PRAGMA table_info({_quote(table)})")]
            for table in sorted(ALLOWED_TABLES)
        }
    finally:
        conn.close()


@lru_cache(maxsize=4)
def _schema_summary(db_path: Path, _mtime_ns: int) -> str:
    """CREATE statements for the allowlisted tables plus the values of low-cardinality
    text columns. Runs on a trusted connection: no agent input reaches it."""
    conn = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    try:
        ddl = conn.execute(
            "SELECT name, sql FROM sqlite_master WHERE type = 'table' ORDER BY name"
        ).fetchall()
        parts: list[str] = []
        for table, create_sql in ddl:
            if table not in ALLOWED_TABLES:
                continue
            parts.append(create_sql.strip() + ";")
            for _cid, col, col_type, *_ in conn.execute(f"PRAGMA table_info({_quote(table)})"):
                if col_type.upper() != "TEXT":
                    continue
                # identifiers come from introspection and are quoted; no agent input
                c, t, n = _quote(col), _quote(table), LOW_CARDINALITY + 1
                values = conn.execute(
                    f"SELECT DISTINCT {c} FROM {t} WHERE {c} IS NOT NULL LIMIT {n}"  # noqa: S608  # nosec B608
                ).fetchall()
                if 0 < len(values) <= LOW_CARDINALITY:
                    shown = ", ".join(repr(v[0]) for v in values)
                    parts.append(f"-- {table}.{col} values: {shown}")
            parts.append("")
        return "\n".join(parts).strip()
    finally:
        conn.close()
