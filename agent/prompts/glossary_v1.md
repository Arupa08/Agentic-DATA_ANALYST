## Business glossary (company metric definitions)

- Dataset window: 2024-01-01 to 2025-12-31. Treat "today" as 2026-01-01, so "last year" is 2025 and "current" or "active" means as of 2025-12-31.
- Revenue: SUM(order_items.line_total) over line items joined to an existing order in `orders` with status = 'completed', dated by orders.order_ts. Refunds are separate line items with negative quantity and line_total; revenue is net of them unless "gross" is asked for. Cancelled orders are not revenue.
- Units sold: SUM(order_items.quantity) on completed orders, net of refunds unless gross is asked for.
- Customer: a person, identified by LOWER(email). One person can have several customer_id rows, so count people with COUNT(DISTINCT LOWER(email)).
- Region: report canonical regions. Region ids 6 ('N. America') and 7 ('APAC') are alternate spellings of 1 (North America) and 3 (Asia Pacific): map with CASE region_id WHEN 6 THEN 1 WHEN 7 THEN 3 ELSE region_id END.
- Churn rate for a period: subscriptions active at the period start (start_date < start AND (end_date IS NULL OR end_date >= start)) that ended during the period, divided by subscriptions active at the period start.
- Marketing spend: always in USD, as SUM(amount * fx_rates.usd_rate) joined on currency.
- Ticket resolution time: hours between created_ts and resolved_ts, for resolved tickets only.
- Rows with a NULL timestamp cannot be placed in time: exclude them from time-bucketed figures.
