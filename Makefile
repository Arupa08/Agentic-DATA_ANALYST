.PHONY: install data profile evals-verify evals evals-routing dashboard test lint format typecheck security check hello

install:            ## install deps from the lockfile + git hooks
	uv sync --locked
	uv run pre-commit install

data:               ## rebuild data/shop.db deterministically from the seed
	uv run python -m data.generate

profile:            ## print every planted signal / flaw (numbers in DATA_CARD.md)
	uv run python -m data.profile

evals-verify:       ## Phase 2 exit: recompute every case's ground truth, check the freeze lock
	uv run python -m evals.verify

evals:              ## live: all 30 cases x3, graded, report in evals/reports/ (costs money; needs .env)
	uv run python -m evals.run_evals --label $${LABEL:-baseline}

evals-routing:      ## live: the all-strong arm of the routing experiment
	uv run python -m evals.run_evals --routing all-strong --label all-strong

dashboard:          ## demo UI (ask the agent + eval results) at http://localhost:8501
	uv run --group ui streamlit run ui/app.py

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

typecheck:
	uv run mypy

security:           ## code + dependency security scans (secret scan runs in pre-commit / CI)
	uv run bandit -q -c pyproject.toml -r agent data evals scripts
	mkdir -p .cache
	uv export --locked --no-hashes --no-emit-project --format requirements-txt -o .cache/requirements.txt
	uv run pip-audit --strict -r .cache/requirements.txt

check: lint typecheck security test   ## everything CI runs

hello:              ## Phase 0 exit check: one traced LLM call (needs .env)
	uv run python scripts/hello_trace.py
