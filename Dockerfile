# syntax=docker/dockerfile:1
# One image, two processes: the API (default) and the Streamlit UI (see compose.yaml).
# No secrets are baked in: model keys arrive as environment variables at run time.

# ---- build: locked dependencies + the synthetic database -----------------------------
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS build
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --group ui --no-install-project

COPY agent ./agent
COPY api ./api
COPY data ./data
COPY evals ./evals
COPY ui ./ui
COPY .streamlit ./.streamlit

# generate the database and refuse to ship one whose content differs from the data card,
# since every eval answer is computed from that exact content
RUN /app/.venv/bin/python -m data.generate > /tmp/generate.log \
    && expected="$(grep -oE '[0-9a-f]{64}' data/DATA_CARD.md | head -n 1)" \
    && grep -q "content_hash *${expected}" /tmp/generate.log \
    || { echo "generated data does not match data/DATA_CARD.md"; cat /tmp/generate.log; exit 1; }

# ---- runtime: slim, non-root, code and data read-only ---------------------------------
FROM python:3.13-slim-bookworm
RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin app
WORKDIR /app
# owned by root and not writable by `app`: the agent can't change its own code or data
COPY --from=build /app /app
RUN chmod 0444 /app/data/shop.db

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_BROWSER_SERVER_ADDRESS=localhost \
    STREAMLIT_SERVER_FILE_WATCHER_TYPE=none
USER app
EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"

# --proxy-headers: the client IP for rate limiting comes from X-Forwarded-For, trusted
# only from the addresses in FORWARDED_ALLOW_IPS (default 127.0.0.1)
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--no-server-header"]
