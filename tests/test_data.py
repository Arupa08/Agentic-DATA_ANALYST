"""Phase 1 exit checks: the dataset is reproducible, big enough, carries every planted
signal, and every planted flaw breaks the naive query it is meant to break."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from data.generate import Config, build_database
from data.profile import CHURN_BY_REGION_QUARTER, REVENUE_BY_QUARTER


def scalar(conn: sqlite3.Connection, sql: str) -> float:
    return float(conn.execute(sql).fetchone()[0])


# ---------------------------------------------------------------- reproducibility


def test_same_seed_same_content(tmp_path: Path) -> None:
    cfg = Config(scale=0.05)
    assert build_database(tmp_path / "a.db", cfg) == build_database(tmp_path / "b.db", cfg)


def test_different_seed_different_content(tmp_path: Path) -> None:
    a = build_database(tmp_path / "a.db", Config(scale=0.05, seed=1))
    b = build_database(tmp_path / "b.db", Config(scale=0.05, seed=2))
    assert a != b


# ---------------------------------------------------------------- shape


def test_scale_meets_plan(shop_db: sqlite3.Connection) -> None:
    assert scalar(shop_db, "SELECT COUNT(*) FROM customers") >= 10_000
    assert scalar(shop_db, "SELECT COUNT(*) FROM orders") >= 50_000


def test_eight_quarters_of_orders(shop_db: sqlite3.Connection) -> None:
    quarters = [r[0] for r in shop_db.execute(REVENUE_BY_QUARTER)]
    assert quarters == [f"{y}-Q{q}" for y in (2024, 2025) for q in (1, 2, 3, 4)]


# ---------------------------------------------------------------- planted signals


def test_europe_churn_rose_most_q3_to_q4(shop_db: sqlite3.Connection) -> None:
    rates: dict[str, dict[str, float]] = {}
    for region, quarter, _active, _churned, pct in shop_db.execute(CHURN_BY_REGION_QUARTER):
        rates.setdefault(region, {})[quarter] = pct
    deltas = {r: q["2025-Q4"] - q["2025-Q3"] for r, q in rates.items()}
    assert max(deltas, key=lambda r: deltas[r]) == "Europe"
    assert deltas["Europe"] > 5.0


def test_home_appliances_refund_rate_stands_out(shop_db: sqlite3.Connection) -> None:
    rows = shop_db.execute(
        """
        SELECT cat.category_name,
               -1.0 * SUM(CASE WHEN oi.quantity < 0 THEN oi.quantity ELSE 0 END)
                    / SUM(CASE WHEN oi.quantity > 0 THEN oi.quantity ELSE 0 END)
        FROM order_items oi JOIN orders o USING (order_id)
        JOIN products p USING (product_id) JOIN categories cat USING (category_id)
        WHERE o.status = 'completed' GROUP BY cat.category_name
        """
    ).fetchall()
    rates = dict(rows)
    assert max(rates, key=lambda c: rates[c]) == "Home Appliances"
    assert rates["Home Appliances"] > 0.15
    assert all(r < 0.05 for c, r in rates.items() if c != "Home Appliances")


def test_q4_is_peak_quarter_each_year(shop_db: sqlite3.Connection) -> None:
    revenue = dict(shop_db.execute(REVENUE_BY_QUARTER).fetchall())
    for year in (2024, 2025):
        q4 = revenue[f"{year}-Q4"]
        assert all(q4 > revenue[f"{year}-Q{q}"] for q in (1, 2, 3))


def test_latam_campaign_quarter_has_most_signups(shop_db: sqlite3.Connection) -> None:
    rows = shop_db.execute(
        """
        SELECT (CAST(strftime('%m', signup_date) AS INTEGER) + 2) / 3 AS q, COUNT(*)
        FROM customers WHERE region_id = 4 AND signup_date >= '2025-01-01' GROUP BY q
        """
    ).fetchall()
    signups = dict(rows)
    assert max(signups, key=lambda q: signups[q]) == 2


def test_holiday_shipping_tickets_slow_down(shop_db: sqlite3.Connection) -> None:
    def avg_hours(lo: str, hi: str) -> float:
        return scalar(
            shop_db,
            f"""
            SELECT AVG((julianday(resolved_ts) - julianday(created_ts)) * 24) FROM support_tickets
            WHERE category = 'shipping' AND created_ts >= '{lo}' AND created_ts < '{hi}'
            """,
        )

    assert avg_hours("2025-11-01", "2026-01-01") > 2 * avg_hours("2025-07-01", "2025-11-01")


# ---------------------------------------------------------------- planted dirt breaks naive queries


def test_duplicate_customers_inflate_naive_unique_count(shop_db: sqlite3.Connection) -> None:
    naive, dedup = shop_db.execute(
        """
        SELECT COUNT(DISTINCT o.customer_id), COUNT(DISTINCT LOWER(c.email))
        FROM orders o JOIN customers c USING (customer_id)
        WHERE o.status = 'completed' AND o.order_ts >= '2025-01-01' AND o.order_ts < '2026-01-01'
        """
    ).fetchone()
    assert naive > dedup * 1.01  # outside a ±1% tolerance


def test_region_spelling_variants_split_groups(shop_db: sqlite3.Connection) -> None:
    names = {r[0] for r in shop_db.execute("SELECT region_name FROM regions")}
    assert {"North America", "N. America", "Asia Pacific", "APAC"} <= names
    assert scalar(shop_db, "SELECT COUNT(*) FROM customers WHERE region_id IN (6, 7)") > 500


def test_null_order_timestamps_present(shop_db: sqlite3.Connection) -> None:
    assert scalar(shop_db, "SELECT COUNT(*) FROM orders WHERE order_ts IS NULL") > 100


def test_refunds_are_negative_line_items(shop_db: sqlite3.Connection) -> None:
    bad = scalar(
        shop_db,
        "SELECT COUNT(*) FROM order_items WHERE (quantity < 0) != (line_total < 0)",
    )
    assert bad == 0
    assert scalar(shop_db, "SELECT COUNT(*) FROM order_items WHERE quantity < 0") > 1_000


def test_orphaned_order_items_present(shop_db: sqlite3.Connection) -> None:
    orphans = scalar(
        shop_db,
        """
        SELECT COUNT(*) FROM order_items oi
        WHERE NOT EXISTS (SELECT 1 FROM orders o WHERE o.order_id = oi.order_id)
        """,
    )
    assert orphans > 100


def test_naive_revenue_overstates_canonical(shop_db: sqlite3.Connection) -> None:
    naive = scalar(shop_db, "SELECT SUM(line_total) FROM order_items")
    canonical = scalar(
        shop_db,
        """
        SELECT SUM(oi.line_total) FROM order_items oi JOIN orders o USING (order_id)
        WHERE o.status = 'completed' AND o.order_ts IS NOT NULL
        """,
    )
    assert naive > canonical * 1.01


def test_marketing_spend_mixes_currencies(shop_db: sqlite3.Connection) -> None:
    currencies = {r[0] for r in shop_db.execute("SELECT DISTINCT currency FROM marketing_spend")}
    assert currencies == {"USD", "EUR", "JPY"}
    naive = scalar(shop_db, "SELECT SUM(amount) FROM marketing_spend")
    usd = scalar(
        shop_db,
        "SELECT SUM(amount * usd_rate) FROM marketing_spend JOIN fx_rates USING (currency)",
    )
    assert naive > usd * 1.01


@pytest.mark.parametrize(
    "table",
    ["orders", "order_items", "customers", "subscriptions", "churn_events", "support_tickets"],
)
def test_primary_keys_unique(shop_db: sqlite3.Connection, table: str) -> None:
    total, distinct = shop_db.execute(
        f"SELECT COUNT(*), COUNT(DISTINCT rowid) FROM {table}"
    ).fetchone()
    assert total == distinct
