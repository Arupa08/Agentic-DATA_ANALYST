"""Malicious SQL. Every layer is tested on its own: the AST guard, then the engine-level
connection with the guard bypassed. A write must fail at each layer independently."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from agent.safety.sql_guard import MAX_SQL_CHARS, SqlGuardError, guard_sql
from agent.tools.sql_tool import SqlTool, open_agent_connection

ATTACKS = [
    # DDL / DML
    "DROP TABLE orders",
    "DELETE FROM orders WHERE status = 'cancelled'",
    "UPDATE products SET list_price = 0",
    "INSERT INTO regions VALUES (99, 'x')",
    "REPLACE INTO regions VALUES (1, 'x')",
    "CREATE TABLE t (x)",
    "CREATE TEMP TABLE t AS SELECT * FROM customers",
    "ALTER TABLE orders ADD COLUMN x",
    "SELECT * INTO backup FROM orders",
    # stacked statements, including ones hidden after comments
    "SELECT 1; DROP TABLE orders",
    "SELECT COUNT(*) FROM orders; UPDATE products SET list_price = 0",
    "SELECT 1 /* harmless */; DELETE FROM orders",
    "SELECT 1 -- comment\n; DROP TABLE orders",
    # engine / file access
    "PRAGMA writable_schema = 1",
    "PRAGMA table_info(orders)",
    "ATTACH DATABASE '/tmp/evil.db' AS evil",
    "DETACH DATABASE main",
    "VACUUM INTO '/tmp/copy.db'",
    "REINDEX",
    "ANALYZE",
    "BEGIN TRANSACTION",
    "SELECT load_extension('/tmp/evil.so')",
    "SELECT readfile('/etc/passwd')",
    # system tables and table-valued pragmas
    "SELECT * FROM sqlite_master",
    "SELECT sql FROM sqlite_schema",
    "SELECT * FROM sqlite_temp_master",
    "SELECT * FROM pragma_table_info('orders')",
    "SELECT name FROM pragma_database_list",
    "SELECT sqlite_version()",
    "SELECT * FROM main.orders",
    "SELECT * FROM temp.anything",
    # resource abuse
    "SELECT randomblob(1000000000)",
    "SELECT zeroblob(1000000000)",
    # non-literal LIMIT tricks
    "SELECT * FROM orders LIMIT (SELECT 1000000000)",
    "SELECT * FROM orders LIMIT 10 OFFSET 5",
    # unknown tables
    "SELECT * FROM users",
    "",
    "   ",
    "x" * (MAX_SQL_CHARS + 1),
    "SELEC * FROM orders",
]


@pytest.mark.parametrize("sql", ATTACKS, ids=lambda s: s[:40] or "empty")
def test_guard_rejects(sql: str) -> None:
    with pytest.raises(SqlGuardError):
        guard_sql(sql)


@pytest.mark.parametrize("sql", ATTACKS[:12], ids=lambda s: s[:40])
def test_tool_rejects_and_db_unchanged(shop_db_path: Path, sql: str) -> None:
    before = shop_db_path.read_bytes()
    result = SqlTool(shop_db_path).run(sql)
    assert not result.ok
    assert result.error is not None and result.error.startswith("rejected by SQL guard")
    assert shop_db_path.read_bytes() == before


def test_guard_strips_comments_and_injects_limit() -> None:
    g = guard_sql("SELECT order_id /* note */ FROM orders -- trailing")
    assert "/*" not in g.sql and "--" not in g.sql
    assert g.sql.endswith("LIMIT 5000")


def test_guard_caps_large_limit() -> None:
    assert guard_sql("SELECT order_id FROM orders LIMIT 999999", row_cap=100).limit == 100


@pytest.mark.parametrize(
    "sql",
    [
        "WITH x AS (SELECT region_id FROM regions) SELECT * FROM x",
        "SELECT region_id FROM regions UNION SELECT category_id FROM categories",
        "SELECT o.order_id, SUM(oi.line_total) OVER (PARTITION BY o.customer_id) FROM orders o "
        "JOIN order_items oi ON oi.order_id = o.order_id",
        "SELECT * FROM (SELECT customer_id, COUNT(*) AS n FROM orders GROUP BY 1) WHERE n > 5",
        "WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM n WHERE i < 5) "
        "SELECT i FROM n",
        "SELECT julianday(resolved_ts) - julianday(created_ts) FROM support_tickets",
    ],
)
def test_guard_allows_legitimate_analytics(sql: str) -> None:
    guard_sql(sql)


# ------------------------------------------------------------ engine layer, guard bypassed

ENGINE_ATTACKS = [
    "DELETE FROM orders",
    "UPDATE products SET list_price = 0",
    "DROP TABLE orders",
    "CREATE TEMP TABLE t (x)",
    "ATTACH DATABASE 'evil.db' AS evil",
    "PRAGMA writable_schema = 1",
    "SELECT * FROM sqlite_master",
    "SELECT * FROM pragma_table_info('orders')",
    "SELECT load_extension('evil')",
    "SELECT randomblob(10)",
]


@pytest.mark.parametrize("sql", ENGINE_ATTACKS, ids=lambda s: s[:40])
def test_engine_layer_blocks_without_guard(shop_db_path: Path, sql: str) -> None:
    conn = open_agent_connection(shop_db_path)
    try:
        with pytest.raises(sqlite3.Error):
            conn.execute(sql)
    finally:
        conn.close()


def test_engine_rejects_stacked_statements(shop_db_path: Path) -> None:
    conn = open_agent_connection(shop_db_path)
    try:
        with pytest.raises((sqlite3.Error, sqlite3.ProgrammingError)):
            conn.execute("SELECT 1; DELETE FROM orders")
    finally:
        conn.close()


def test_connection_is_read_only_even_without_authorizer(shop_db_path: Path) -> None:
    conn = open_agent_connection(shop_db_path)
    conn.set_authorizer(None)
    try:
        with pytest.raises(sqlite3.OperationalError, match=r"readonly|read-only|query_only"):
            conn.execute("DELETE FROM orders")
    finally:
        conn.close()


def test_missing_db_is_never_created(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        open_agent_connection(tmp_path / "nope.db")
    assert not (tmp_path / "nope.db").exists()


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT COUNT(*) FROM orders WHERE order_ts < '2026-01-01' AND o.quantity > 0;'}, '",
        "SELECT 'unterminated FROM orders",
    ],
)
def test_unreadable_sql_is_rejected_not_raised(sql: str) -> None:
    # live baseline: a TokenError escaped the guard and crashed dirty_04 three times
    with pytest.raises(SqlGuardError):  # rejected (whichever rule fires), never raised raw
        guard_sql(sql)
