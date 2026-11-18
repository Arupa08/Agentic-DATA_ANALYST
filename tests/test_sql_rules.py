"""Critic SQL rules: each catches its live-baseline failure, and none fires on correct SQL."""

from __future__ import annotations

import pytest

from agent.nodes.sql_rules import check_sql_semantics
from evals.models import load_cases
from evals.verify import CASES_PATH

MAPPED = "CASE c.region_id WHEN 6 THEN 1 WHEN 7 THEN 3 ELSE c.region_id END"


@pytest.mark.parametrize(
    ("sql", "needle"),
    [
        # live baseline dirty_02 / dirty_06: grouped by the raw region name
        (
            "SELECT r.region_name, SUM(oi.line_total) FROM order_items oi JOIN orders o "
            "ON o.order_id = oi.order_id JOIN customers c ON c.customer_id = o.customer_id "
            "JOIN regions r ON r.region_id = c.region_id GROUP BY r.region_name",
            "separate regions",
        ),
        (
            "SELECT COUNT(DISTINCT LOWER(c.email)) FROM customers c JOIN regions r "
            "ON r.region_id = c.region_id WHERE r.region_name = 'North America'",
            "alternate spelling 'n. america'",
        ),
        ("SELECT COUNT(DISTINCT LOWER(email)) FROM customers WHERE region_id = 3", "'7'"),
        # people vs accounts
        ("SELECT COUNT(DISTINCT o.customer_id) FROM orders o", "accounts"),
        ("SELECT COUNT(*) FROM customers", "customer rows"),
        # live baseline simple_04: average line instead of average order
        (
            "SELECT AVG(line_total) AS average_order_value FROM order_items oi "
            "JOIN orders o ON o.order_id = oi.order_id",
            "averages line items",
        ),
    ],
)
def test_rules_catch_the_baseline_mistakes(sql: str, needle: str) -> None:
    failure = check_sql_semantics(sql, "")
    assert failure is not None and needle in (failure[0] + (failure[1] or ""))


def test_per_order_rule_also_reads_the_step_goal() -> None:
    sql = "SELECT AVG(oi.line_total) AS x FROM order_items oi"
    assert check_sql_semantics(sql, "Average order value of completed orders") is not None
    assert check_sql_semantics(sql, "Average line amount") is None


@pytest.mark.parametrize(
    "sql",
    [
        f"SELECT r.region_name, COUNT(DISTINCT LOWER(c.email)) FROM customers c JOIN regions r "
        f"ON r.region_id = {MAPPED} GROUP BY r.region_name",
        "SELECT COUNT(DISTINCT LOWER(email)) FROM customers WHERE region_id IN (1, 6)",
        "SELECT region_id, SUM(amount) FROM marketing_spend GROUP BY region_id",  # no spellings
        "SELECT COUNT(*) AS orders FROM orders WHERE order_ts >= '2025-12-01'",
        "SELECT SUM(oi.line_total) / COUNT(DISTINCT o.order_id) AS aov FROM order_items oi "
        "JOIN orders o ON o.order_id = oi.order_id",
        "SELECT AVG(csat) AS avg_csat FROM support_tickets",
        "not sql at all",  # left to the guard, never a crash here
    ],
)
def test_correct_or_unrelated_sql_passes(sql: str) -> None:
    assert check_sql_semantics(sql, "") is None


def test_every_violation_is_reported_at_once() -> None:
    sql = (
        "SELECT r.region_name, COUNT(*) FROM customers c JOIN regions r "
        "ON r.region_id = c.region_id GROUP BY r.region_name"
    )
    failure = check_sql_semantics(sql, "")
    assert failure is not None
    assert "separate regions" in failure[0] and "not people" in failure[0]


def test_no_rule_fires_on_any_ground_truth_query() -> None:
    """A rule that flags the answer key would cost a correct answer a replan."""
    flagged = []
    for case in load_cases(CASES_PATH).cases:
        for sql in filter(None, [case.sql, *(i.sql for i in case.interpretations)]):
            if check_sql_semantics(sql, case.question) is not None:
                flagged.append(case.id)
    assert flagged == []
