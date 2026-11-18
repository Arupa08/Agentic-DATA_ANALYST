"""SQL guard: the AST layer of the read-only defense.

A query is accepted only if it parses to exactly one SELECT (or a UNION / INTERSECT /
EXCEPT of SELECTs), touches only allowlisted tables, calls no dangerous functions,
and carries a LIMIT no larger than the cap. What gets executed is the SQL regenerated
from the checked AST, so comments and anything the parser didn't see never run.

This is one of four layers. The others are the read-only connection, the SQLite
authorizer (agent/tools/sql_tool.py) and the statement timeout. Each must hold on its
own.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import sqlglot
from sqlglot import TokenType, exp
from sqlglot.errors import ParseError, SqlglotError

ALLOWED_TABLES = frozenset(
    {
        "regions",
        "categories",
        "products",
        "customers",
        "orders",
        "order_items",
        "subscriptions",
        "churn_events",
        "support_tickets",
        "marketing_spend",
        "fx_rates",
    }
)

MAX_SQL_CHARS = 10_000
DEFAULT_ROW_CAP = 5_000

# Statement / clause node types that must never appear anywhere in the tree.
# Looked up by name so a sqlglot upgrade that renames one fails closed in tests, not silently.
_FORBIDDEN_NODE_NAMES = (
    "Insert",
    "Update",
    "Delete",
    "Merge",
    "Drop",
    "Create",
    "Alter",
    "AlterTable",
    "TruncateTable",
    "Command",
    "Pragma",
    "Attach",
    "Detach",
    "Transaction",
    "Commit",
    "Rollback",
    "Copy",
    "LoadData",
    "Into",
    "Set",
    "Use",
    "Analyze",
    "Lock",
)
FORBIDDEN_NODES: tuple[type[exp.Expression], ...] = tuple(
    t for name in _FORBIDDEN_NODE_NAMES if isinstance(t := getattr(exp, name, None), type)
)

# SQLite functions that read files, load code, expose internals, or allocate unbounded memory.
FORBIDDEN_FUNCTIONS = frozenset(
    {
        "load_extension",
        "readfile",
        "writefile",
        "edit",
        "fts3_tokenizer",
        "zipfile",
        "randomblob",
        "zeroblob",
        "unlikely",  # harmless, but no reason to let plans steer the query planner
        "likelihood",
    }
)
FORBIDDEN_FUNCTION_PREFIXES = ("sqlite_", "pragma_")


class SqlGuardError(ValueError):
    """Raised when a query is rejected. The message is safe to show to the planner."""


@dataclass(frozen=True)
class GuardedQuery:
    sql: str  # regenerated, comment-free SQL to execute
    tables: frozenset[str]
    limit: int


def _function_name(node: exp.Expr) -> str | None:
    if isinstance(node, exp.Anonymous):
        return str(node.name).lower()
    if isinstance(node, exp.Func):
        return node.sql_name().lower()
    return None


def _cte_names(root: exp.Expr) -> set[str]:
    return {cte.alias_or_name.lower() for cte in root.find_all(exp.CTE)}


def _check_tables(root: exp.Expression) -> frozenset[str]:
    ctes = _cte_names(root)
    used: set[str] = set()
    for table in root.find_all(exp.Table):
        if not isinstance(table.this, exp.Identifier):
            # table-valued function in FROM, e.g. pragma_table_info('orders')
            raise SqlGuardError("table-valued functions are not allowed in FROM")
        if table.args.get("db") is not None or table.args.get("catalog") is not None:
            raise SqlGuardError("schema-qualified table names are not allowed")
        name = table.name.lower()
        if name in ctes:
            continue
        if name not in ALLOWED_TABLES:
            raise SqlGuardError(f"table {table.name!r} is not available")
        used.add(name)
    return frozenset(used)


def _check_nodes(root: exp.Expression) -> None:
    for node in root.walk():
        if isinstance(node, FORBIDDEN_NODES):
            raise SqlGuardError(f"{type(node).__name__.upper()} is not allowed; only SELECT")
        name = _function_name(node)
        if name and (name in FORBIDDEN_FUNCTIONS or name.startswith(FORBIDDEN_FUNCTION_PREFIXES)):
            raise SqlGuardError(f"function {name}() is not allowed")


def _check_tokens(sql: str) -> None:
    """Catch forbidden function names in the raw token stream too. sqlglot maps some
    SQLite functions onto typed nodes (sqlite_version() becomes CurrentVersion), which
    hides the original name from the AST check."""
    try:
        tokens = sqlglot.tokenize(sql, read="sqlite")
    except Exception:  # sqlglot raises TokenError; treat any tokenizer failure as a reject
        raise SqlGuardError("could not tokenize SQL") from None
    for tok, nxt in pairwise(tokens):
        if tok.token_type == TokenType.STRING or nxt.token_type != TokenType.L_PAREN:
            continue
        name = tok.text.lower()
        if name in FORBIDDEN_FUNCTIONS or name.startswith(FORBIDDEN_FUNCTION_PREFIXES):
            raise SqlGuardError(f"function {name}() is not allowed")


def _apply_limit(root: exp.Query, cap: int) -> tuple[exp.Query, int]:
    existing = root.args.get("limit")
    if existing is None:
        return root.limit(cap, copy=True), cap
    value = existing.expression
    if not (isinstance(value, exp.Literal) and value.is_int):
        raise SqlGuardError("LIMIT must be a plain integer")
    n = int(value.this)
    if root.args.get("offset") is not None or existing.args.get("offset") is not None:
        raise SqlGuardError("OFFSET is not supported")
    if n <= cap:
        return root, n
    return root.limit(cap, copy=True), cap


def guard_sql(sql: str, *, row_cap: int = DEFAULT_ROW_CAP) -> GuardedQuery:
    if not sql or not sql.strip():
        raise SqlGuardError("empty query")
    if len(sql) > MAX_SQL_CHARS:
        raise SqlGuardError(f"query longer than {MAX_SQL_CHARS} characters")
    try:
        statements = [s for s in sqlglot.parse(sql, read="sqlite") if s is not None]
    except ParseError as exc:
        raise SqlGuardError(
            f"could not parse SQL: {exc.errors[0]['description'] if exc.errors else exc}"
        ) from None
    except SqlglotError:
        # e.g. TokenError on an unterminated string. Live baseline: this escaped as an
        # exception and crashed the whole run instead of letting the critic trigger a replan.
        raise SqlGuardError("could not read the SQL (check quotes and brackets)") from None
    if len(statements) != 1:
        raise SqlGuardError("exactly one statement is allowed")
    root = statements[0]
    if not isinstance(root, exp.Select | exp.Union | exp.Intersect | exp.Except):
        raise SqlGuardError(f"{type(root).__name__.upper()} is not allowed; only SELECT")
    _check_tokens(sql)
    _check_nodes(root)
    tables = _check_tables(root)
    limited, limit = _apply_limit(root, row_cap)
    return GuardedQuery(
        sql=limited.sql(dialect="sqlite", comments=False), tables=tables, limit=limit
    )


def referenced_tables(sql: str) -> set[str]:
    """Best-effort table names a query reads (for error hints only, never for security)."""
    try:
        statements = sqlglot.parse(sql, read="sqlite")
    except SqlglotError:
        return set()
    root = next((s for s in statements if s is not None), None)
    if root is None:
        return set()
    ctes = _cte_names(root)
    return {t.name.lower() for t in root.find_all(exp.Table) if t.name.lower() not in ctes}
