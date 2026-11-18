"""Deterministic checks of WHAT a query computes, against the business definitions.

The critic's result rules catch broken output (errors, empty or NULL results); these
catch plausible-looking answers to the wrong question. Each rule mirrors a definition in
the business glossary (agent/prompts/glossary_v1.md), so the rules hold for any question
about this data, not just the ones in the eval set. They read the SQL that actually ran.

Found in the first live baseline, where all of these got confident wrong answers through:
  - grouping or filtering customer regions without merging alternate spellings
  - counting customer accounts (customer_id) instead of people (email)
  - averaging line items for a per-order value
"""

from __future__ import annotations

import re

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError

RuleFailure = tuple[str, str | None]  # (reason, fix)

# glossary: region ids 6 ('N. America') and 7 ('APAC') are spellings of 1 and 3
REGION_ALIASES: dict[str, tuple[str, ...]] = {
    "1": ("6",),
    "3": ("7",),
    "north america": ("n. america",),
    "asia pacific": ("apac",),
}
CANONICAL_REGION = "CASE c.region_id WHEN 6 THEN 1 WHEN 7 THEN 3 ELSE c.region_id END"
_REGION_COLUMNS = {"region_id", "region_name"}
_LINE_COLUMNS = {"line_total", "unit_price", "quantity"}
_PER_ORDER = re.compile(r"(order[_ ]value|per[_ ]order|\baov\b|basket)", re.IGNORECASE)


def _literals(node: exp.Expr) -> set[str]:
    return {str(lit.this).strip().lower() for lit in node.find_all(exp.Literal)}


def _mentions_region(node: exp.Expr) -> bool:
    return any(col.name.lower() in _REGION_COLUMNS for col in node.find_all(exp.Column))


def _maps_region_spellings(tree: exp.Expr) -> bool:
    """A CASE over a region column that folds both alternate spellings."""
    for case in tree.find_all(exp.Case):
        if _mentions_region(case):
            lits = _literals(case)
            if {"6", "7"} <= lits or {"n. america", "apac"} <= lits:
                return True
    return False


def _region_rule(tree: exp.Expr, tables: set[str]) -> RuleFailure | None:
    if "customers" not in tables or _maps_region_spellings(tree):
        return None  # only customers carry the alternate spellings
    fix = f"merge alternate spellings first: group or filter by {CANONICAL_REGION}"
    for select in tree.find_all(exp.Select):
        group = select.args.get("group")
        if group is not None and _mentions_region(group):
            return (
                "groups customers by region without merging alternate spellings, so "
                "'N. America' and 'APAC' are counted as separate regions",
                fix,
            )
    for cond in tree.find_all(exp.EQ, exp.In):
        if not _mentions_region(cond):
            continue
        lits = _literals(cond)
        for canonical, variants in REGION_ALIASES.items():
            if canonical in lits and not set(variants) & lits:
                return (
                    f"filters to region {canonical!r} but leaves out its alternate "
                    f"spelling {variants[0]!r}",
                    fix,
                )
    return None


def _people_rule(tree: exp.Expr, tables: set[str], sql: str) -> RuleFailure | None:
    if "lower(" in sql.lower() and "email" in sql.lower():
        return None  # people are already counted by email
    fix = "count people with COUNT(DISTINCT LOWER(c.email)): one person can have several accounts"
    only_customer_tables = tables <= {"customers", "regions"}
    for count in tree.find_all(exp.Count):
        cols = {c.name.lower() for c in count.find_all(exp.Column)}
        distinct = count.find(exp.Distinct) is not None
        star = count.find(exp.Star) is not None or not cols
        if "customer_id" in cols and (distinct or only_customer_tables):
            return ("counts customer accounts (customer_id), not people", fix)
        if star and only_customer_tables and "customers" in tables:
            return ("counts customer rows, not people", fix)
    return None


def _per_order_rule(tree: exp.Expr, goal: str) -> RuleFailure | None:
    for avg in tree.find_all(exp.Avg):
        cols = {c.name.lower() for c in avg.find_all(exp.Column)}
        if not cols & _LINE_COLUMNS:
            continue
        alias = avg.parent.alias if isinstance(avg.parent, exp.Alias) else ""
        if _PER_ORDER.search(alias) or _PER_ORDER.search(goal):
            return (
                "averages line items, but an order's value is the sum of its lines",
                "average per order: SUM(oi.line_total) / COUNT(DISTINCT o.order_id)",
            )
    return None


def check_sql_semantics(sql: str, goal: str) -> RuleFailure | None:
    """Return (reason, fix) if the query computes something other than what the business
    definitions say, else None. Unparseable SQL is left to the other rules."""
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except SqlglotError:
        return None
    tables = {t.name.lower() for t in tree.find_all(exp.Table)}
    found = [
        f
        for f in (
            _region_rule(tree, tables),
            _people_rule(tree, tables, sql),
            _per_order_rule(tree, goal),
        )
        if f is not None
    ]
    if not found:
        return None
    # report every violation at once, so one replan can fix them all
    reasons = "; ".join(r for r, _ in found)
    fixes = "; ".join(f for _, f in found if f)
    return reasons, fixes or None
