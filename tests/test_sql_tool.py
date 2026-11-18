from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from agent.tools.sql_tool import SqlTool
from evals.models import load_cases
from evals.verify import CASES_PATH, HOLDOUT_PATH


@pytest.fixture(scope="module")
def tool(shop_db_path: Path) -> SqlTool:
    return SqlTool(shop_db_path, row_cap=100, timeout_s=2.0)


def test_runs_a_query(tool: SqlTool) -> None:
    r = tool.run("SELECT region_id, region_name FROM regions ORDER BY region_id")
    assert r.ok
    assert r.columns == ["region_id", "region_name"]
    assert r.row_count == 7
    assert r.rows[5] == [6, "N. America"]
    assert not r.truncated


def test_truncates_at_row_cap(tool: SqlTool) -> None:
    r = tool.run("SELECT order_id FROM orders")
    assert r.ok
    assert r.row_count == 100
    assert r.truncated
    assert "LIMIT" in r.sql.upper()


def test_respects_smaller_limit(tool: SqlTool) -> None:
    r = tool.run("SELECT order_id FROM orders ORDER BY order_id LIMIT 3")
    assert r.rows == [[1], [2], [3]]
    assert not r.truncated


def test_bad_column_returns_informative_error(tool: SqlTool) -> None:
    r = tool.run("SELECT revenue FROM orders")
    assert not r.ok
    assert r.error is not None
    assert "revenue" in r.error  # the planner needs this to replan


def test_slow_query_times_out(tool: SqlTool) -> None:
    r = tool.run(
        "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) SELECT MAX(x) FROM c"
    )
    assert not r.ok
    assert r.error == "query timed out"


def test_llm_text_wraps_rows_as_untrusted(tool: SqlTool) -> None:
    text = tool.run("SELECT region_name FROM regions").to_llm_text(max_rows=3)
    assert '<untrusted source="database">' in text
    assert "4 more row(s) not shown" in text


def test_schema_summary_shows_tables_and_values(tool: SqlTool) -> None:
    summary = tool.schema_summary()
    assert "CREATE TABLE order_items" in summary
    assert "'N. America'" in summary  # dirty values are visible to the planner, as to an analyst
    assert "negative for refunds" in summary  # schema comments are kept
    assert "sqlite_" not in summary


def test_every_eval_query_survives_the_guard_unchanged(
    shop_db_path: Path, shop_db: sqlite3.Connection
) -> None:
    """The tool executes SQL regenerated from the AST; prove that never changes an answer."""
    tool = SqlTool(shop_db_path)
    cases = [*load_cases(CASES_PATH).cases, *load_cases(HOLDOUT_PATH).cases]
    checked = 0
    for case in cases:
        sqls = [case.sql, case.naive_sql, *(i.sql for i in case.interpretations)]
        for sql in filter(None, sqls):
            result = tool.run(sql)
            assert result.ok, (case.id, result.error)
            assert result.rows == [list(r) for r in shop_db.execute(sql)], case.id
            checked += 1
    assert checked == 38  # every ground-truth, interpretation and naive query
