"""Generate the synthetic e-commerce SQLite database with planted signals and planted dirt.

    uv run python -m data.generate                      # writes data/shop.db
    uv run python -m data.generate --scale 0.1 --out x.db

Same seed + scale => same content hash (printed at the end). Every signal and flaw
planted here is documented in DATA_CARD.md, which is the answer key for the evals.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import sqlite3
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from faker import Faker
from numpy.typing import NDArray

DATA_DIR = Path(__file__).resolve().parent
SCHEMA_PATH = DATA_DIR / "schema.sql"
DEFAULT_OUT = DATA_DIR / "shop.db"
DEFAULT_SEED = 20240101

Arr = NDArray[Any]
Row = tuple[Any, ...]

# observation window: 8 quarters
PERIOD_MONTHS = np.arange("2024-01", "2026-01", dtype="datetime64[M]")
SIGNUP_MONTHS = np.arange("2023-01", "2026-01", dtype="datetime64[M]")
SNAPSHOT_TS = np.datetime64("2026-01-01T00:00:00", "s")  # "now" for open tickets / subs
ONE_MONTH = np.timedelta64(1, "M")
ONE_DAY = np.timedelta64(1, "D")
ONE_SECOND = np.timedelta64(1, "s")

# --- regions -------------------------------------------------------------------------
NORTH_AMERICA, EUROPE, ASIA_PACIFIC, LATIN_AMERICA, MEA = 1, 2, 3, 4, 5
REGIONS: dict[int, str] = {
    NORTH_AMERICA: "North America",
    EUROPE: "Europe",
    ASIA_PACIFIC: "Asia Pacific",
    LATIN_AMERICA: "Latin America",
    MEA: "Middle East & Africa",
    # DIRT: duplicate spellings stored as separate region rows
    6: "N. America",
    7: "APAC",
}
CANONICAL_REGION_IDS = np.array([1, 2, 3, 4, 5])
CANONICAL_REGION_WEIGHTS = np.array([0.35, 0.25, 0.20, 0.12, 0.08])
SPELLING_VARIANTS: dict[int, tuple[int, float]] = {
    NORTH_AMERICA: (6, 0.30),
    ASIA_PACIFIC: (7, 0.35),
}

# --- catalog -------------------------------------------------------------------------
HOME_APPLIANCES = 2
CATEGORIES: dict[int, str] = {
    1: "Electronics",
    HOME_APPLIANCES: "Home Appliances",
    3: "Apparel",
    4: "Beauty",
    5: "Sports & Outdoors",
    6: "Books",
    7: "Toys",
    8: "Grocery",
}
# (median list price, lognormal sigma, popularity weight)
CATEGORY_PROFILE: dict[int, tuple[float, float, float]] = {
    1: (240.0, 0.7, 1.2),
    2: (170.0, 0.6, 0.8),
    3: (45.0, 0.5, 1.5),
    4: (28.0, 0.5, 1.0),
    5: (65.0, 0.6, 0.9),
    6: (18.0, 0.4, 0.8),
    7: (32.0, 0.5, 0.7),
    8: (12.0, 0.4, 1.1),
}
CATEGORY_NOUNS: dict[int, list[str]] = {
    1: ["Headphones", "Smartwatch", "Tablet", "Speaker", "Monitor", "Webcam", "Keyboard"],
    2: ["Blender", "Air Fryer", "Vacuum", "Microwave", "Espresso Machine", "Dehumidifier"],
    3: ["Jacket", "Sneakers", "Hoodie", "Jeans", "Dress", "Scarf", "Polo"],
    4: ["Serum", "Moisturizer", "Lip Balm", "Shampoo", "Sunscreen", "Face Mask"],
    5: ["Yoga Mat", "Tent", "Water Bottle", "Dumbbells", "Backpack", "Bike Light"],
    6: ["Cookbook", "Novel", "Journal", "Atlas", "Biography", "Puzzle Book"],
    7: ["Building Set", "Plush Bear", "RC Car", "Board Game", "Puzzle", "Doll House"],
    8: ["Coffee Beans", "Olive Oil", "Granola", "Green Tea", "Honey", "Pasta"],
}
ADJECTIVES = ["Nova", "Apex", "Lumen", "Terra", "Pulse", "Vista", "Orbit", "Zen", "Atlas", "Echo"]
PRODUCTS_PER_CATEGORY = 25
BASE_REFUND_RATE = 0.03
REFUND_RATE: dict[int, float] = {HOME_APPLIANCES: 0.18}  # SIGNAL

# --- customers / subscriptions --------------------------------------------------------
SEGMENTS = np.array(["consumer", "smb", "enterprise"])
SEGMENT_WEIGHTS = np.array([0.80, 0.15, 0.05])
SEGMENT_ORDER_MULT = {"consumer": 1.0, "smb": 1.4, "enterprise": 2.0}
EMAIL_DOMAINS = ["gmail.com", "outlook.com", "yahoo.com", "proton.me", "icloud.com"]
PLANS = np.array(["basic", "plus", "pro"])
PLAN_WEIGHTS = np.array([0.50, 0.35, 0.15])
PLAN_PRICE = {"basic": 9.99, "plus": 19.99, "pro": 49.99}
SUBSCRIBER_SHARE = 0.40
BASE_MONTHLY_CHURN = 0.022
EUROPE_Q4_2025_CHURN = 0.055  # SIGNAL
CHURN_REASONS = np.array(["price", "competitor", "product_issue", "support", "other"])
CHURN_REASON_WEIGHTS = np.array([0.25, 0.25, 0.20, 0.15, 0.15])
EUROPE_Q4_REASON_WEIGHTS = np.array([0.55, 0.15, 0.10, 0.10, 0.10])  # SIGNAL: price-driven

# --- marketing ------------------------------------------------------------------------
SPEND_CHANNELS = {"search": 1.4, "social": 1.0, "display": 0.7, "email": 0.25}
REGION_BASE_SPEND_USD = {
    NORTH_AMERICA: 60_000.0,
    EUROPE: 45_000.0,
    ASIA_PACIFIC: 35_000.0,
    LATIN_AMERICA: 15_000.0,
    MEA: 10_000.0,
}
LATAM_CAMPAIGN_MONTHS = np.arange("2025-04", "2025-07", dtype="datetime64[M]")  # SIGNAL
FX_USD_RATE = {"USD": 1.0, "EUR": 1.08, "JPY": 0.0067}

# --- dirt rates -----------------------------------------------------------------------
DUPLICATE_CUSTOMER_SHARE = 0.03
NULL_ORDER_TS_SHARE = 0.015
PURGED_ORDER_SHARE = 0.004
NULL_TICKET_TS_SHARE = 0.005


@dataclass(frozen=True)
class Config:
    seed: int = DEFAULT_SEED
    scale: float = 1.0

    @property
    def n_customers(self) -> int:
        return max(300, int(12_000 * self.scale))

    @property
    def base_monthly_orders(self) -> float:
        return 2_150.0 * self.scale


def _month_season(month: np.datetime64) -> float:
    m = int(str(month)[5:7])
    return {1: 0.85, 11: 1.35, 12: 1.5}.get(m, 1.0)  # SIGNAL: holiday peak


def _days_in_month(months: Arr) -> Arr:
    days: Arr = (months + ONE_MONTH).astype("datetime64[D]") - months.astype("datetime64[D]")
    return days.astype(int)


def _ts_str(ts: Arr) -> list[str]:
    return [s.replace("T", " ") for s in np.datetime_as_string(ts, unit="s").tolist()]


def _date_str(d: Arr) -> list[str]:
    return [str(s) for s in np.datetime_as_string(d, unit="D").tolist()]


def _slug(name: str) -> str:
    return re.sub(r"[^a-z]", "", name.lower())


def _email_case_variant(email: str, kind: int) -> str:
    local, _, domain = email.partition("@")
    if kind == 0:
        return f"{local.title()}@{domain}"
    if kind == 1:
        return f"{local}@{domain.upper()}"
    return email.upper()


def build_tables(cfg: Config) -> dict[str, list[Row]]:
    rng = np.random.default_rng(cfg.seed)
    fake = Faker("en_US")
    fake.seed_instance(cfg.seed)
    tables: dict[str, list[Row]] = {}

    tables["regions"] = list(REGIONS.items())
    tables["categories"] = list(CATEGORIES.items())
    tables["fx_rates"] = list(FX_USD_RATE.items())

    # ---------------------------------------------------------------- products
    prod_ids: list[int] = []
    prod_cat: list[int] = []
    prod_price: list[float] = []
    prod_weight: list[float] = []
    product_rows: list[Row] = []
    launch_days = (np.datetime64("2023-12-31") - np.datetime64("2021-01-01")).astype(int)
    for cat_id, (median, sigma, popularity) in CATEGORY_PROFILE.items():
        prices = np.round(median * rng.lognormal(0.0, sigma, PRODUCTS_PER_CATEGORY), 2)
        weights = popularity * rng.lognormal(0.0, 0.5, PRODUCTS_PER_CATEGORY)
        launched = np.datetime64("2021-01-01") + rng.integers(0, launch_days, PRODUCTS_PER_CATEGORY)
        for i in range(PRODUCTS_PER_CATEGORY):
            pid = len(prod_ids) + 1
            noun = CATEGORY_NOUNS[cat_id][int(rng.integers(len(CATEGORY_NOUNS[cat_id])))]
            adj = ADJECTIVES[int(rng.integers(len(ADJECTIVES)))]
            name = f"{adj} {noun} {chr(65 + i % 26)}{int(rng.integers(100, 1000))}"
            price = max(float(prices[i]), 2.99)
            product_rows.append((pid, name, cat_id, price, str(launched[i])))
            prod_ids.append(pid)
            prod_cat.append(cat_id)
            prod_price.append(price)
            prod_weight.append(float(weights[i]))
    tables["products"] = product_rows
    product_ids = np.array(prod_ids)
    product_cat = np.array(prod_cat)
    product_price = np.array(prod_price)
    product_p = np.array(prod_weight) / sum(prod_weight)

    # ---------------------------------------------------------------- customers
    n = cfg.n_customers
    cust_ids = np.arange(1, n + 1)
    canon = rng.choice(CANONICAL_REGION_IDS, size=n, p=CANONICAL_REGION_WEIGHTS)
    region = canon.copy()
    for canonical_id, (variant_id, share) in SPELLING_VARIANTS.items():
        region[(canon == canonical_id) & (rng.random(n) < share)] = variant_id
    segment = rng.choice(SEGMENTS, size=n, p=SEGMENT_WEIGHTS)

    signup_month = np.empty(n, dtype="datetime64[M]")
    growth_w = 1.03 ** np.arange(len(SIGNUP_MONTHS))
    for r in CANONICAL_REGION_IDS:
        idx = np.flatnonzero(canon == r)
        w = growth_w.copy()
        if r == LATIN_AMERICA:
            w[np.isin(SIGNUP_MONTHS, LATAM_CAMPAIGN_MONTHS)] *= 3.0  # SIGNAL: campaign signups
        signup_month[idx] = SIGNUP_MONTHS[
            rng.choice(len(SIGNUP_MONTHS), size=len(idx), p=w / w.sum())
        ]
    day_offset = (rng.random(n) * _days_in_month(signup_month)).astype(int)
    signup = signup_month.astype("datetime64[D]") + day_offset

    first_names = [fake.first_name() for _ in range(n)]
    last_names = [fake.last_name() for _ in range(n)]
    domains = rng.choice(len(EMAIL_DOMAINS), size=n)
    emails = [
        f"{_slug(f)}.{_slug(last)}{cid}@{EMAIL_DOMAINS[d]}"
        for f, last, cid, d in zip(first_names, last_names, cust_ids.tolist(), domains, strict=True)
    ]

    # ---------------------------------------------------------------- subscriptions + churn
    has_sub = rng.random(n) < SUBSCRIBER_SHARE
    sub_cust_idx = np.flatnonzero(has_sub)
    sub_start = signup[sub_cust_idx] + rng.integers(0, 60, len(sub_cust_idx))
    in_window = sub_start < SNAPSHOT_TS.astype("datetime64[D]")
    sub_cust_idx, sub_start = sub_cust_idx[in_window], sub_start[in_window]
    m = len(sub_cust_idx)
    sub_plan = rng.choice(PLANS, size=m, p=PLAN_WEIGHTS)
    sub_canon = canon[sub_cust_idx]
    churn_date = np.full(m, np.datetime64("NaT", "D"))
    europe_q4 = np.datetime64("2025-10", "M")
    for month in SIGNUP_MONTHS:
        m_start = month.astype("datetime64[D]")
        m_end = (month + ONE_MONTH).astype("datetime64[D]") - ONE_DAY
        draws = rng.random(m)
        offsets = rng.random(m)
        active = (sub_start <= m_end) & np.isnat(churn_date)
        hazard = np.where(
            (sub_canon == EUROPE) & (month >= europe_q4), EUROPE_Q4_2025_CHURN, BASE_MONTHLY_CHURN
        )
        churned = active & (draws < hazard)
        lo = np.maximum(sub_start[churned], m_start)
        span = (m_end - lo).astype(int) + 1
        churn_date[churned] = lo + (offsets[churned] * span).astype(int)

    churned_mask = ~np.isnat(churn_date)
    is_europe_q4 = (sub_canon == EUROPE) & (churn_date >= np.datetime64("2025-10-01"))
    reason_default = rng.choice(CHURN_REASONS, size=m, p=CHURN_REASON_WEIGHTS)
    reason_europe = rng.choice(CHURN_REASONS, size=m, p=EUROPE_Q4_REASON_WEIGHTS)
    reasons = np.where(is_europe_q4, reason_europe, reason_default)

    sub_ids = np.arange(1, m + 1)
    start_s, churn_s = _date_str(sub_start), _date_str(churn_date)
    tables["subscriptions"] = [
        (
            int(sub_ids[i]),
            int(cust_ids[sub_cust_idx[i]]),
            str(sub_plan[i]),
            PLAN_PRICE[str(sub_plan[i])],
            start_s[i],
            churn_s[i] if churned_mask[i] else None,
            "cancelled" if churned_mask[i] else "active",
        )
        for i in range(m)
    ]
    churn_order = np.flatnonzero(churned_mask)
    churn_order = churn_order[np.argsort(churn_date[churn_order], kind="stable")]
    tables["churn_events"] = [
        (k + 1, int(sub_ids[i]), int(cust_ids[sub_cust_idx[i]]), churn_s[i], str(reasons[i]))
        for k, i in enumerate(churn_order.tolist())
    ]
    cust_churn = np.full(n, np.datetime64("NaT", "s"))
    cust_churn[sub_cust_idx[churned_mask]] = churn_date[churned_mask].astype("datetime64[s]")

    # ---------------------------------------------------------------- orders
    seg_mult = np.array([SEGMENT_ORDER_MULT[s] for s in segment.tolist()])
    propensity = rng.gamma(0.8, 1.0, n) * seg_mult
    signup_ts = signup.astype("datetime64[s]")
    picks_all: list[Arr] = []
    ts_all: list[Arr] = []
    for k, month in enumerate(PERIOD_MONTHS):
        n_orders = int(rng.poisson(cfg.base_monthly_orders * 1.012**k * _month_season(month)))
        m_start = month.astype("datetime64[s]")
        m_end = (month + ONE_MONTH).astype("datetime64[s]")
        churned_before = cust_churn < m_start  # NaT compares False
        w = propensity * (signup_ts < m_end) * np.where(churned_before, 0.4, 1.0)
        picks = rng.choice(n, size=n_orders, p=w / w.sum())
        lo = np.maximum(signup_ts[picks], m_start)
        span = (m_end - lo).astype(np.int64)
        ts = lo + (rng.random(n_orders) * span).astype(np.int64)
        picks_all.append(picks)
        ts_all.append(ts)
    order_cust_idx = np.concatenate(picks_all)
    order_ts = np.concatenate(ts_all)
    chrono = np.argsort(order_ts, kind="stable")
    order_cust_idx, order_ts = order_cust_idx[chrono], order_ts[chrono]
    n_orders_total = len(order_ts)
    order_ids = np.arange(1, n_orders_total + 1)
    channel = rng.choice(
        np.array(["web", "mobile", "marketplace"]), size=n_orders_total, p=[0.55, 0.35, 0.10]
    )
    status = np.where(rng.random(n_orders_total) < 0.05, "cancelled", "completed")

    # ---------------------------------------------------------------- DIRT: duplicate customers
    eligible_dups = np.flatnonzero(signup < np.datetime64("2025-06-01"))
    dup_src = np.sort(
        rng.choice(eligible_dups, size=int(DUPLICATE_CUSTOMER_SHARE * n), replace=False)
    )
    dup_ids = np.arange(n + 1, n + 1 + len(dup_src))
    dup_signup = np.minimum(
        signup[dup_src] + rng.integers(30, 300, len(dup_src)), np.datetime64("2025-12-31")
    )
    dup_kind = rng.integers(0, 3, len(dup_src))
    dup_region = canon[dup_src].copy()
    flip = rng.random(len(dup_src)) < 0.5
    for canonical_id, (variant_id, _) in SPELLING_VARIANTS.items():
        dup_region[(canon[dup_src] == canonical_id) & flip] = variant_id
    # reassign roughly half of each original's later orders to the duplicate account
    order_cust_id = cust_ids[order_cust_idx].copy()
    reassign_draw = rng.random(n_orders_total)
    dup_lookup = dict(zip(dup_src.tolist(), range(len(dup_src)), strict=True))
    for j, ci in enumerate(order_cust_idx.tolist()):
        d = dup_lookup.get(ci)
        if (
            d is not None
            and order_ts[j] >= dup_signup[d].astype("datetime64[s]")
            and reassign_draw[j] < 0.5
        ):
            order_cust_id[j] = dup_ids[d]

    signup_s = _date_str(signup)
    customer_rows: list[Row] = [
        (
            int(cust_ids[i]),
            emails[i],
            first_names[i],
            last_names[i],
            int(region[i]),
            str(segment[i]),
            signup_s[i],
        )
        for i in range(n)
    ]
    dup_signup_s = _date_str(dup_signup)
    for d, src in enumerate(dup_src.tolist()):
        customer_rows.append(
            (
                int(dup_ids[d]),
                _email_case_variant(emails[src], int(dup_kind[d])),
                first_names[src],
                last_names[src],
                int(dup_region[d]),
                str(segment[src]),
                dup_signup_s[d],
            )
        )
    tables["customers"] = customer_rows

    # ---------------------------------------------------------------- order items (+ refunds)
    n_items = 1 + rng.poisson(np.where(channel == "mobile", 0.6, 1.3))
    item_order_idx = np.repeat(np.arange(n_orders_total), n_items)
    k_items = len(item_order_idx)
    item_product = rng.choice(product_ids, size=k_items, p=product_p)
    qty = 1 + rng.poisson(0.35, k_items)
    discount = rng.choice(np.array([0.0, 0.10, 0.20]), size=k_items, p=[0.70, 0.20, 0.10])
    unit_price = np.round(product_price[item_product - 1] * (1 - discount), 2)
    line_total = np.round(qty * unit_price, 2)

    item_cat = product_cat[item_product - 1]
    refund_p = np.array([REFUND_RATE.get(int(c), BASE_REFUND_RATE) for c in range(1, 9)])[
        item_cat - 1
    ]
    refunded = (status[item_order_idx] == "completed") & (rng.random(k_items) < refund_p)
    refund_src = np.flatnonzero(refunded)

    item_rows: list[Row] = list(
        zip(
            range(1, k_items + 1),
            order_ids[item_order_idx].tolist(),
            item_product.tolist(),
            qty.tolist(),
            unit_price.tolist(),
            line_total.tolist(),
            strict=True,
        )
    )
    for r, src in enumerate(refund_src.tolist()):
        item_rows.append(
            (
                k_items + r + 1,
                int(order_ids[item_order_idx[src]]),
                int(item_product[src]),
                -int(qty[src]),
                float(unit_price[src]),
                -float(line_total[src]),
            )
        )
    tables["order_items"] = item_rows

    # ---------------------------------------------------------------- DIRT: purged orders, null ts
    completed_idx = np.flatnonzero(status == "completed")
    purged = np.zeros(n_orders_total, dtype=bool)
    purged[
        rng.choice(completed_idx, size=int(PURGED_ORDER_SHARE * n_orders_total), replace=False)
    ] = True
    null_ts = rng.random(n_orders_total) < NULL_ORDER_TS_SHARE
    order_ts_s = _ts_str(order_ts)
    tables["orders"] = [
        (
            int(order_ids[j]),
            int(order_cust_id[j]),
            None if null_ts[j] else order_ts_s[j],
            str(status[j]),
            str(channel[j]),
            "USD",
        )
        for j in range(n_orders_total)
        if not purged[j]
    ]

    # ---------------------------------------------------------------- support tickets
    order_has_refund = np.zeros(n_orders_total, dtype=bool)
    order_has_refund[item_order_idx[refund_src]] = True
    holiday_2025 = order_ts >= np.datetime64("2025-11-01T00:00:00", "s")
    t_draw = rng.random(n_orders_total)
    spike_draw = rng.random(n_orders_total)
    base_ticket = t_draw < np.where(order_has_refund, 0.55, 0.10)
    spike_ticket = holiday_2025 & (spike_draw < 0.06)  # SIGNAL: holiday shipping backlog

    base_cats = np.array(["shipping", "billing", "product_defect", "returns", "account"])
    cat_base = rng.choice(base_cats, size=n_orders_total, p=[0.35, 0.20, 0.20, 0.15, 0.10])
    cat_refund = rng.choice(np.array(["returns", "product_defect"]), size=n_orders_total)
    t_order_idx = np.concatenate([np.flatnonzero(base_ticket), np.flatnonzero(spike_ticket)])
    t_category = np.concatenate(
        [
            np.where(order_has_refund, cat_refund, cat_base)[base_ticket],
            np.full(int(spike_ticket.sum()), "shipping"),
        ]
    )
    lag = (rng.exponential(3 * 86_400, len(t_order_idx))).astype(np.int64)
    t_created = np.minimum(order_ts[t_order_idx] + lag, SNAPSHOT_TS - ONE_SECOND)
    t_cust = order_cust_id[t_order_idx]
    t_order = order_ids[t_order_idx].astype(object)

    n_standalone = int(0.15 * n)
    s_cust_idx = rng.integers(0, n, n_standalone)
    s_lo = np.maximum(signup_ts[s_cust_idx], np.datetime64("2024-01-01T00:00:00", "s"))
    s_created = s_lo + (rng.random(n_standalone) * (SNAPSHOT_TS - s_lo).astype(np.int64)).astype(
        np.int64
    )
    s_category = rng.choice(np.array(["account", "billing"]), size=n_standalone)

    all_created = np.concatenate([t_created, s_created])
    all_cust = np.concatenate([t_cust, cust_ids[s_cust_idx]])
    all_order = np.concatenate([t_order, np.full(n_standalone, None, dtype=object)])
    all_cat = np.concatenate([t_category, s_category])
    n_tickets = len(all_created)

    slow = (all_cat == "shipping") & (all_created >= np.datetime64("2025-11-01T00:00:00", "s"))
    median_hours = np.where(slow, 52.0, 18.0)  # SIGNAL: holiday resolution slowdown
    hours = median_hours * rng.lognormal(0.0, 0.8, n_tickets)
    resolved = all_created + (hours * 3600).astype(np.int64)
    is_open = (resolved >= SNAPSHOT_TS) | (rng.random(n_tickets) < 0.02)
    priority = np.where(
        all_cat == "product_defect",
        rng.choice(np.array(["low", "medium", "high"]), size=n_tickets, p=[0.15, 0.45, 0.40]),
        rng.choice(np.array(["low", "medium", "high"]), size=n_tickets, p=[0.35, 0.50, 0.15]),
    )
    csat_raw = 5 - 1.1 * np.log10(np.maximum(hours, 1.0) / 8.0) + rng.normal(0.0, 0.6, n_tickets)
    csat = np.clip(np.round(csat_raw), 1, 5).astype(int)
    csat_missing = is_open | (rng.random(n_tickets) < 0.35)
    created_missing = rng.random(n_tickets) < NULL_TICKET_TS_SHARE

    t_chrono = np.argsort(all_created, kind="stable")
    created_s, resolved_s = _ts_str(all_created), _ts_str(resolved)
    tables["support_tickets"] = [
        (
            tid,
            int(all_cust[i]),
            None if all_order[i] is None else int(all_order[i]),
            str(all_cat[i]),
            str(priority[i]),
            None if created_missing[i] else created_s[i],
            None if is_open[i] else resolved_s[i],
            None if csat_missing[i] else int(csat[i]),
        )
        for tid, i in enumerate(t_chrono.tolist(), start=1)
    ]

    # ---------------------------------------------------------------- marketing spend
    spend_rows: list[Row] = []
    for k, month in enumerate(PERIOD_MONTHS):
        for region_id, base in REGION_BASE_SPEND_USD.items():
            for ch, ch_mult in SPEND_CHANNELS.items():
                usd = base * ch_mult * 1.01**k * (1.3 if _month_season(month) > 1.2 else 1.0)
                usd *= rng.lognormal(0.0, 0.1)
                if region_id == LATIN_AMERICA and month in LATAM_CAMPAIGN_MONTHS:
                    usd *= 2.2  # SIGNAL
                # DIRT: mixed currency — Europe books in EUR, APAC switched to JPY in 2025
                if region_id == EUROPE:
                    cur = "EUR"
                elif region_id == ASIA_PACIFIC and month >= np.datetime64("2025-01", "M"):
                    cur = "JPY"
                else:
                    cur = "USD"
                amount = round(usd / FX_USD_RATE[cur], 2)
                spend_rows.append((len(spend_rows) + 1, region_id, ch, f"{month}-01", amount, cur))
    tables["marketing_spend"] = spend_rows
    return tables


TABLE_ORDER = [
    "regions",
    "categories",
    "fx_rates",
    "products",
    "customers",
    "orders",
    "order_items",
    "subscriptions",
    "churn_events",
    "support_tickets",
    "marketing_spend",
]


def content_hash(conn: sqlite3.Connection) -> str:
    """Hash of every row in every table, independent of SQLite file layout."""
    h = hashlib.sha256()
    for table in TABLE_ORDER:
        h.update(table.encode())
        for row in conn.execute(f"SELECT * FROM {table} ORDER BY 1"):  # noqa: S608  # nosec B608
            h.update(repr(row).encode())
    return h.hexdigest()


def build_database(out: Path, cfg: Config) -> str:
    tables = build_tables(cfg)
    out.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(suffix=".db", dir=out.parent)
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        conn = sqlite3.connect(tmp)
        try:
            conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
            for table in TABLE_ORDER:
                rows = tables[table]
                placeholders = ",".join("?" * len(rows[0]))
                conn.executemany(f"INSERT INTO {table} VALUES ({placeholders})", rows)  # noqa: S608  # nosec B608
            conn.commit()
            digest = content_hash(conn)
            conn.execute("VACUUM")
        finally:
            conn.close()
        os.replace(tmp, out)
    finally:
        tmp.unlink(missing_ok=True)
    return digest


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--scale", type=float, default=1.0)
    args = parser.parse_args()
    digest = build_database(args.out, Config(seed=args.seed, scale=args.scale))
    conn = sqlite3.connect(args.out)
    try:
        for table in TABLE_ORDER:
            (count,) = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()  # noqa: S608  # nosec B608
            print(f"{table:<16} {count:>8,}")
    finally:
        conn.close()
    print(f"content_hash     {digest}")


if __name__ == "__main__":
    main()
