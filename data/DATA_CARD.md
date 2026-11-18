# Data Card — synthetic e-commerce DB (`data/shop.db`)

This is the **answer key**. Every signal and every flaw below was planted on purpose by
[`generate.py`](generate.py). The numbers come from [`profile.py`](profile.py) at the
default seed, so they are reproducible, not hand-typed estimates.

All data is synthetic (Faker + seeded NumPy). No real personal data exists anywhere in
this project.

## Reproduce

```bash
make data       # or: uv run python -m data.generate
make profile    # or: uv run python -m data.profile
```

| | |
|---|---|
| Seed | `20240101` (`--seed` to change) |
| Window | 2024-01-01 → 2025-12-31 (8 quarters); snapshot "now" = 2026-01-01 |
| Content hash | `5a3e1d4ccb2d5dfc06d4b70655b39b2f10fe97ed9b1a9a75cf304361d2a8d162` |

The hash covers row content, not file bytes, and is checked by
`tests/test_data.py::test_same_seed_same_content`. It is stable for a given lockfile
on a given platform. If a NumPy or Faker upgrade changes it, regenerate, re-run the
profile and update the numbers in this card **and** the eval ground truth together.

## Tables

| Table | Rows | Notes |
|---|---:|---|
| `regions` | 7 | 5 real regions + 2 duplicate spellings (see D2) |
| `categories` | 8 | |
| `products` | 200 | 25 per category; `list_price` is the catalog price |
| `customers` | 12,360 | 12,000 people + 360 duplicate accounts (D1) |
| `orders` | 63,393 | `status` ∈ completed / cancelled; `currency` is always USD |
| `order_items` | 135,948 | includes negative refund lines (D4) and orphans (D5) |
| `subscriptions` | 4,624 | one per subscriber; `end_date` NULL while active |
| `churn_events` | 1,253 | one per cancelled subscription |
| `support_tickets` | 11,009 | `order_id` NULL for account/billing tickets not tied to an order |
| `marketing_spend` | 480 | 5 regions × 4 channels × 24 months, **mixed currency** (D6) |
| `fx_rates` | 3 | USD per unit: USD 1.0, EUR 1.08, JPY 0.0067 |

Foreign keys are declared in `schema.sql` but were not enforced on load, as in a legacy
import. That is how the orphans in D5 got in.

## Canonical metric definitions

The evals score against these definitions. When a question is ambiguous, these are the
defaults the agent should state as its assumption.

| Metric | Definition |
|---|---|
| **Revenue** | `SUM(order_items.line_total)` joined to `orders`, `status = 'completed'`, `order_ts IS NOT NULL`. **Net of refunds** (refund lines are included, so they subtract). Refunds count in the original order's period. |
| **Gross revenue** | Same, but only `line_total > 0`. |
| **Region** | Canonical region: `CASE region_id WHEN 6 THEN 1 WHEN 7 THEN 3 ELSE region_id END`. |
| **Unique customers** | `COUNT(DISTINCT LOWER(email))`, not `customer_id`. |
| **Churn rate (quarter)** | Subscriptions active at quarter start that ended during the quarter ÷ subscriptions active at quarter start. |
| **Marketing spend** | `SUM(amount * fx_rates.usd_rate)`, always in USD. |
| **Resolution time** | `resolved_ts − created_ts` in hours; open tickets (`resolved_ts IS NULL`) and NULL `created_ts` excluded. |

## Planted signals (questions with interesting answers)

### S1 — Europe churn spikes in Q4 2025
Europe's monthly churn hazard rises from 2.2% to 5.5% starting October 2025.

| Region | Q3 2025 | Q4 2025 | Δ (pp) |
|---|---:|---:|---:|
| **Europe** | 6.8% | **15.7%** | **+8.9** |
| Latin America | 6.2% | 8.0% | +1.8 |
| Asia Pacific | 5.8% | 5.8% | 0.0 |
| North America | 5.5% | 5.3% | −0.2 |
| Middle East & Africa | 10.8% | 6.8% | −4.0 |

MEA's Q3 value is ordinary noise from a small base (195 subs), not a planted signal.
**Answer: Europe, +8.9 pp.**

### S2 — Europe Q4 churn is price-driven
Of Europe's Q4 2025 churn events, 69 of 127 (54%) give `reason = 'price'`, against
179 of 656 (27%) for all other 2025 churn.

### S3 — Home Appliances has a high refund rate
17.9% of units refunded (2,606 / 14,583). Every other category sits at 2.9–3.1%.

### S4 — Q4 holiday peak, with year-over-year growth
Canonical net revenue by quarter:

| | Q1 | Q2 | Q3 | Q4 |
|---|---:|---:|---:|---:|
| 2024 | 1,409,244.66 | 1,493,389.03 | 1,585,843.03 | 2,097,260.52 |
| 2025 | 1,646,826.45 | 1,758,160.59 | **1,783,489.61** | 2,429,004.98 |

November and December carry 1.35× and 1.5× order volume; January dips to 0.85×.

### S5 — Latin America campaign, Q2 2025
LatAm marketing spend runs at 2.2× from April to June 2025, and LatAm signups triple
over the same months.

| 2025 | Spend (USD) | New LatAm customers |
|---|---:|---:|
| Q1 | 176,276 | 135 |
| **Q2** | **393,233** | **421** |
| Q3 | 186,050 | 172 |
| Q4 | 241,157 | 164 |

### S6 — Holiday shipping backlog, Nov–Dec 2025
Shipping tickets jump and resolution slows sharply. This happens in 2025 only; the
2024 holidays have more tickets but normal resolution times.

| 2025 | Jul | Aug | Sep | Oct | **Nov** | **Dec** |
|---|---:|---:|---:|---:|---:|---:|
| Shipping tickets | 69 | 98 | 103 | 77 | **306** | **429** |
| Avg hours to resolve | 25.2 | 25.5 | 23.8 | 19.9 | **71.7** | **58.2** |

### S7 — Mobile orders are smaller
Average order value (completed, net): marketplace $276.43, web $267.07, mobile $185.95.

## Planted flaws (every one breaks a naive query)

| # | Flaw | Naive query | Wrong answer | Right answer |
|---|---|---|---|---|
| D1 | **Duplicate customers**: 360 people (3%) have a second account whose email differs only in case (`Jane.Doe12@…`, `jane.doe12@GMAIL.COM`, or all caps). About half of the person's later orders sit on the duplicate. | `COUNT(DISTINCT customer_id)` of 2025 buyers | 7,925 | **7,785** (`LOWER(email)`), 1.8% off |
| D2 | **Region spellings**: `N. America` (id 6) duplicates North America, and `APAC` (id 7) duplicates Asia Pacific, as separate rows. 30% of NA and 35% of APAC customers sit on the variant. Duplicate accounts often use the other spelling from their original. | `GROUP BY region_name` | NA 3,015 customers | **NA 4,187 people** (4,304 accounts on ids 1 + 6, deduplicated by email) |
| D3 | **Null timestamps**: 902 orders (1.4%) have `order_ts IS NULL`; 57 tickets have `created_ts IS NULL`. | time-bucketed queries that `COALESCE` or assume non-null | silently mis-bucketed | exclude, and say so |
| D4 | **Refunds as negative line items**: a refund is an extra `order_items` row with negative `quantity` and `line_total` on the same order (5,346 rows). | `SUM(line_total) WHERE line_total > 0`, or `SUM(quantity)` treated as units sold | gross 16.19M | **net 14.20M** |
| D5 | **Orphaned rows**: 254 completed orders were purged but their 530 `order_items` ($65,824.11) and 43 tickets remain. | `SUM(line_total) FROM order_items` with no join | — | must join `orders` |
| D6 | **Mixed currency**: `marketing_spend.amount` is EUR for Europe, JPY for Asia Pacific from 2025-01, USD otherwise. | `SUM(amount)` for 2025 | 273,592,589.91 | **8,624,031.82 USD**, 31.7× off |
| D7 | **Cancelled orders keep their line items**: 3,254 cancelled orders still carry $821,452.46 of positive lines. | revenue without `status = 'completed'` | — | filter on status |

D4 + D5 + D7 combined: the fully naive `SELECT SUM(line_total) FROM order_items` returns
**15,301,274.38**, while canonical net revenue is **14,203,218.87** (7.7% high).

## Known non-flaws (don't "fix" these)

- `order_items.unit_price` can be below `products.list_price`: discounts of 0, 10 or 20%.
- Open tickets (`resolved_ts IS NULL`, 427 rows) are real open tickets at snapshot time,
  not missing data.
- `csat` is NULL for about 35% of resolved tickets: the survey wasn't answered.
- Customers signed up from 2023 onward, so some have no orders in the window.
- Subscriptions active at snapshot have `end_date IS NULL` and `status = 'active'`.
- **No NPS, no web-traffic, and no inventory data exist.** Questions about them must be
  answered with "the data doesn't contain this" (the eval category "No answer in data").
