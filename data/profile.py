"""Measure every planted signal and flaw in the generated DB.

    uv run python -m data.profile [path/to/shop.db]

Each check pairs the naive query a careless analyst (or agent) would write with the
correct one. The numbers printed here are the ones quoted in DATA_CARD.md.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path
from typing import Any

from data.generate import DEFAULT_OUT

# Plain SQL literals on purpose: this file is the readable answer key.
# Canonical region: CASE c.region_id WHEN 6 THEN 1 WHEN 7 THEN 3 ELSE c.region_id END
# folds the duplicate spellings ("N. America", "APAC") back onto their real region.

# Canonical revenue: net of refund lines, completed orders only, order must exist and be dated.
REVENUE_BY_QUARTER = """
SELECT strftime('%Y', o.order_ts) || '-Q' || ((CAST(strftime('%m', o.order_ts) AS INTEGER) + 2) / 3)
         AS quarter,
       ROUND(SUM(oi.line_total), 2) AS net_revenue
FROM order_items oi JOIN orders o ON o.order_id = oi.order_id
WHERE o.status = 'completed' AND o.order_ts IS NOT NULL
GROUP BY quarter ORDER BY quarter
"""

CHURN_BY_REGION_QUARTER = """
WITH q(quarter, q_start, q_end) AS (
    VALUES ('2025-Q3', '2025-07-01', '2025-09-30'), ('2025-Q4', '2025-10-01', '2025-12-31')
),
base AS (
    SELECT q.quarter, r.region_name, s.subscription_id,
           CASE WHEN s.end_date BETWEEN q.q_start AND q.q_end THEN 1 ELSE 0 END AS churned
    FROM q
    JOIN subscriptions s ON s.start_date < q.q_start AND (s.end_date IS NULL OR s.end_date >= q.q_start)
    JOIN customers c ON c.customer_id = s.customer_id
    JOIN regions r ON r.region_id = CASE c.region_id WHEN 6 THEN 1 WHEN 7 THEN 3 ELSE c.region_id END
)
SELECT region_name, quarter, COUNT(*) AS active_at_start, SUM(churned) AS churned,
       ROUND(100.0 * SUM(churned) / COUNT(*), 1) AS churn_pct
FROM base GROUP BY region_name, quarter ORDER BY region_name, quarter
"""

CHECKS: list[tuple[str, str]] = [
    ("Revenue by quarter (canonical)", REVENUE_BY_QUARTER),
    (
        "Naive vs canonical total revenue",
        """
        SELECT
          (SELECT ROUND(SUM(line_total), 2) FROM order_items) AS naive_all_items,
          (SELECT ROUND(SUM(line_total), 2) FROM order_items WHERE line_total > 0) AS naive_gross_only,
          (SELECT ROUND(SUM(oi.line_total), 2) FROM order_items oi JOIN orders o USING (order_id)
             WHERE o.status = 'completed' AND o.order_ts IS NOT NULL) AS canonical_net
        """,
    ),
    ("Churn by region, 2025 Q3 vs Q4 (canonical regions)", CHURN_BY_REGION_QUARTER),
    (
        "Churn reasons, Europe Q4 2025 vs everyone else in 2025",
        """
        SELECT CASE WHEN c.region_id = 2 AND ce.churn_date >= '2025-10-01'
                    THEN 'Europe Q4' ELSE 'other 2025' END AS cohort,
               ce.reason, COUNT(*) AS n
        FROM churn_events ce JOIN customers c USING (customer_id)
        WHERE ce.churn_date >= '2025-01-01'
        GROUP BY cohort, ce.reason ORDER BY cohort, n DESC
        """,
    ),
    (
        "Refund rate by category (refunded units / sold units, completed orders)",
        """
        SELECT cat.category_name,
               SUM(CASE WHEN oi.quantity > 0 THEN oi.quantity ELSE 0 END) AS units_sold,
               -SUM(CASE WHEN oi.quantity < 0 THEN oi.quantity ELSE 0 END) AS units_refunded,
               ROUND(100.0 * -SUM(CASE WHEN oi.quantity < 0 THEN oi.quantity ELSE 0 END)
                     / SUM(CASE WHEN oi.quantity > 0 THEN oi.quantity ELSE 0 END), 1) AS refund_pct
        FROM order_items oi
        JOIN orders o USING (order_id)
        JOIN products p USING (product_id)
        JOIN categories cat USING (category_id)
        WHERE o.status = 'completed'
        GROUP BY cat.category_name ORDER BY refund_pct DESC
        """,
    ),
    (
        "Unique customers who bought in 2025: naive vs deduplicated",
        """
        SELECT COUNT(DISTINCT o.customer_id) AS naive_by_id,
               COUNT(DISTINCT LOWER(c.email)) AS dedup_by_email
        FROM orders o JOIN customers c USING (customer_id)
        WHERE o.status = 'completed' AND o.order_ts >= '2025-01-01' AND o.order_ts < '2026-01-01'
        """,
    ),
    (
        "Duplicate customer accounts (same email ignoring case)",
        """
        SELECT COUNT(*) AS customer_rows, COUNT(DISTINCT LOWER(email)) AS distinct_people,
               COUNT(*) - COUNT(DISTINCT LOWER(email)) AS duplicate_rows
        FROM customers
        """,
    ),
    (
        "Customers per region: raw region rows",
        """
        SELECT r.region_id, r.region_name, COUNT(*) AS customers
        FROM customers c JOIN regions r USING (region_id) GROUP BY r.region_id ORDER BY r.region_id
        """,
    ),
    (
        "Null timestamps",
        """
        SELECT (SELECT COUNT(*) FROM orders WHERE order_ts IS NULL) AS orders_null_ts,
               (SELECT COUNT(*) FROM orders) AS orders_total,
               (SELECT COUNT(*) FROM support_tickets WHERE created_ts IS NULL) AS tickets_null_created,
               (SELECT COUNT(*) FROM support_tickets WHERE resolved_ts IS NULL) AS tickets_open
        """,
    ),
    (
        "Orphaned rows (FK points at a purged order)",
        """
        SELECT (SELECT COUNT(*) FROM order_items oi
                 WHERE NOT EXISTS (SELECT 1 FROM orders o WHERE o.order_id = oi.order_id)) AS orphan_items,
               (SELECT ROUND(SUM(line_total), 2) FROM order_items oi
                 WHERE NOT EXISTS (SELECT 1 FROM orders o WHERE o.order_id = oi.order_id)) AS orphan_value,
               (SELECT COUNT(*) FROM support_tickets t WHERE t.order_id IS NOT NULL
                 AND NOT EXISTS (SELECT 1 FROM orders o WHERE o.order_id = t.order_id)) AS orphan_tickets
        """,
    ),
    (
        "Cancelled orders that still carry line items",
        """
        SELECT COUNT(DISTINCT o.order_id) AS cancelled_orders, ROUND(SUM(oi.line_total), 2) AS their_value
        FROM orders o JOIN order_items oi USING (order_id) WHERE o.status = 'cancelled'
        """,
    ),
    (
        "Marketing spend 2025: naive SUM(amount) vs USD-converted",
        """
        SELECT ROUND(SUM(ms.amount), 2) AS naive_sum,
               ROUND(SUM(ms.amount * fx.usd_rate), 2) AS usd_total
        FROM marketing_spend ms JOIN fx_rates fx USING (currency)
        WHERE ms.month >= '2025-01-01'
        """,
    ),
    (
        "Latin America: spend (USD) and new signups by quarter, 2025",
        """
        SELECT s.quarter, s.spend_usd, COALESCE(g.signups, 0) AS signups FROM
          (SELECT '2025-Q' || ((CAST(strftime('%m', ms.month) AS INTEGER) + 2) / 3) AS quarter, ROUND(SUM(ms.amount * fx.usd_rate), 0) AS spend_usd
             FROM marketing_spend ms JOIN fx_rates fx USING (currency)
            WHERE ms.region_id = 4 AND ms.month >= '2025-01-01' GROUP BY quarter) s
        LEFT JOIN
          (SELECT '2025-Q' || ((CAST(strftime('%m', c.signup_date) AS INTEGER) + 2) / 3) AS quarter, COUNT(*) AS signups
             FROM customers c WHERE c.region_id = 4 AND c.signup_date >= '2025-01-01'
            GROUP BY quarter) g USING (quarter)
        ORDER BY s.quarter
        """,
    ),
    (
        "Shipping tickets: volume and avg resolution hours by month, H2 2025",
        """
        SELECT substr(created_ts, 1, 7) AS month, COUNT(*) AS tickets,
               ROUND(AVG((julianday(resolved_ts) - julianday(created_ts)) * 24), 1) AS avg_hours
        FROM support_tickets
        WHERE category = 'shipping' AND created_ts >= '2025-07-01'
        GROUP BY month ORDER BY month
        """,
    ),
    (
        "Average order value by channel (completed, net)",
        """
        SELECT o.channel, COUNT(DISTINCT o.order_id) AS orders,
               ROUND(SUM(oi.line_total) / COUNT(DISTINCT o.order_id), 2) AS aov
        FROM orders o JOIN order_items oi USING (order_id)
        WHERE o.status = 'completed' GROUP BY o.channel ORDER BY aov DESC
        """,
    ),
]


def run_check(conn: sqlite3.Connection, sql: str) -> tuple[list[str], list[tuple[Any, ...]]]:
    cur = conn.execute(sql)
    return [d[0] for d in cur.description], cur.fetchall()


def _print_table(cols: list[str], rows: list[tuple[Any, ...]]) -> None:
    cells = [cols] + [[f"{v:,}" if isinstance(v, int | float) else str(v) for v in r] for r in rows]
    widths = [max(len(row[i]) for row in cells) for i in range(len(cols))]
    for k, row in enumerate(cells):
        print("  " + "  ".join(c.rjust(w) for c, w in zip(row, widths, strict=True)))
        if k == 0:
            print("  " + "  ".join("-" * w for w in widths))


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT
    conn = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    try:
        for title, sql in CHECKS:
            print(f"\n## {title}")
            _print_table(*run_check(conn, sql))
    finally:
        conn.close()


if __name__ == "__main__":
    main()
