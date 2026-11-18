-- Synthetic e-commerce schema. Foreign keys are declared but the loader does not
-- enforce them (legacy-import realism): see DATA_CARD.md for the orphans that result.

CREATE TABLE regions (
    region_id    INTEGER PRIMARY KEY,
    region_name  TEXT NOT NULL
);

CREATE TABLE categories (
    category_id    INTEGER PRIMARY KEY,
    category_name  TEXT NOT NULL
);

CREATE TABLE products (
    product_id    INTEGER PRIMARY KEY,
    product_name  TEXT NOT NULL,
    category_id   INTEGER NOT NULL REFERENCES categories(category_id),
    list_price    REAL NOT NULL,
    launched_on   TEXT NOT NULL            -- YYYY-MM-DD
);

CREATE TABLE customers (
    customer_id  INTEGER PRIMARY KEY,
    email        TEXT NOT NULL,
    first_name   TEXT NOT NULL,
    last_name    TEXT NOT NULL,
    region_id    INTEGER NOT NULL REFERENCES regions(region_id),
    segment      TEXT NOT NULL CHECK (segment IN ('consumer', 'smb', 'enterprise')),
    signup_date  TEXT NOT NULL             -- YYYY-MM-DD
);

CREATE TABLE orders (
    order_id     INTEGER PRIMARY KEY,
    customer_id  INTEGER NOT NULL REFERENCES customers(customer_id),
    order_ts     TEXT,                     -- YYYY-MM-DD HH:MM:SS, UTC; nullable
    status       TEXT NOT NULL CHECK (status IN ('completed', 'cancelled')),
    channel      TEXT NOT NULL CHECK (channel IN ('web', 'mobile', 'marketplace')),
    currency     TEXT NOT NULL DEFAULT 'USD'
);

CREATE TABLE order_items (
    order_item_id  INTEGER PRIMARY KEY,
    order_id       INTEGER NOT NULL REFERENCES orders(order_id),
    product_id     INTEGER NOT NULL REFERENCES products(product_id),
    quantity       INTEGER NOT NULL,       -- negative for refund lines
    unit_price     REAL NOT NULL,
    line_total     REAL NOT NULL           -- quantity * unit_price; negative for refunds
);

CREATE TABLE subscriptions (
    subscription_id  INTEGER PRIMARY KEY,
    customer_id      INTEGER NOT NULL REFERENCES customers(customer_id),
    plan             TEXT NOT NULL CHECK (plan IN ('basic', 'plus', 'pro')),
    monthly_price    REAL NOT NULL,
    start_date       TEXT NOT NULL,        -- YYYY-MM-DD
    end_date         TEXT,                 -- NULL while active
    status           TEXT NOT NULL CHECK (status IN ('active', 'cancelled'))
);

CREATE TABLE churn_events (
    churn_event_id   INTEGER PRIMARY KEY,
    subscription_id  INTEGER NOT NULL REFERENCES subscriptions(subscription_id),
    customer_id      INTEGER NOT NULL REFERENCES customers(customer_id),
    churn_date       TEXT NOT NULL,        -- YYYY-MM-DD
    reason           TEXT NOT NULL
);

CREATE TABLE support_tickets (
    ticket_id    INTEGER PRIMARY KEY,
    customer_id  INTEGER NOT NULL REFERENCES customers(customer_id),
    order_id     INTEGER REFERENCES orders(order_id),
    category     TEXT NOT NULL,
    priority     TEXT NOT NULL CHECK (priority IN ('low', 'medium', 'high')),
    created_ts   TEXT,                     -- nullable
    resolved_ts  TEXT,                     -- NULL while open
    csat         INTEGER CHECK (csat BETWEEN 1 AND 5)  -- NULL when survey not answered
);

CREATE TABLE marketing_spend (
    spend_id     INTEGER PRIMARY KEY,
    region_id    INTEGER NOT NULL REFERENCES regions(region_id),
    channel      TEXT NOT NULL,
    month        TEXT NOT NULL,            -- YYYY-MM-01
    amount       REAL NOT NULL,            -- in `currency`, NOT always USD
    currency     TEXT NOT NULL
);

CREATE TABLE fx_rates (
    currency  TEXT PRIMARY KEY,
    usd_rate  REAL NOT NULL                -- USD per 1 unit of currency
);

CREATE INDEX idx_orders_customer ON orders(customer_id);
CREATE INDEX idx_orders_ts ON orders(order_ts);
CREATE INDEX idx_items_order ON order_items(order_id);
CREATE INDEX idx_items_product ON order_items(product_id);
CREATE INDEX idx_subs_customer ON subscriptions(customer_id);
CREATE INDEX idx_churn_date ON churn_events(churn_date);
CREATE INDEX idx_tickets_created ON support_tickets(created_ts);
