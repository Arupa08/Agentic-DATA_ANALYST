# NL Analytics Agent

A LangGraph agent (Planner → Executor → Critic → Synthesizer, with a Critic → Planner
replan edge) that answers natural-language business questions over a deliberately messy
SQLite e-commerce DB, and proves its reliability with a frozen eval set.

Build plan: 8 phases. **Done: Phases 0–6.** Live runs use a free local model (Ollama `qwen3:4b-instruct`). Local baseline (2026-11-17): 50.0% pass rate, multi-step 0%; the 4B judge mis-graded some rubric cases, now fixed (reason-first but optional and short, one field per rubric item, "may" items are permissions, one nudged retry for unreadable verdicts). Demo UI: `ui/app.py` (Ask page `ui/ask_page.py` + eval dashboard `ui/dashboard.py`). Re-graded baseline 53.5%; with critic SQL rules and the clarify nudge off (now the default): 60.5% (2026-11-20). Next: Phase 7 (ship).

## Commands

`make` isn't installed on the dev machine (Windows); use the `uv run` equivalents.

| make | uv equivalent |
|---|---|
| `make data` | `uv run python -m data.generate` |
| `make profile` | `uv run python -m data.profile` |
| `make evals-verify` | `uv run python -m evals.verify` |
| `make test` | `uv run pytest` |
| `make lint` | `uv run ruff check . && uv run ruff format --check .` |
| `make typecheck` | `uv run mypy` |
| `make hello` | `uv run python scripts/hello_trace.py` |
| ask one question | `uv run python scripts/ask.py "..."` |
| run evals (live, costs money) | `uv run python -m evals.run_evals --label baseline [--runs 3] [--ids ...] [--routing all-strong] [--no-glossary] [--no-critic-llm]` |
| compare two reports | `uv run python -m evals.compare A.json B.json` |
| demo UI (ask + eval dashboard) | `uv run --group ui streamlit run ui/app.py` (localhost:8501) |
| re-grade a report (grader changes only) | `uv run python -m evals.regrade evals/reports/<report>.json` |

## Working rules

- Build one phase at a time. Before writing code, list which standards below apply to the phase.
- Measuring stick first: `data/DATA_CARD.md` is the answer key, and eval cases are frozen
  in Phase 2 before any agent code exists. Never tune the evals to the agent.
- Every number in `DATA_CARD.md` must come from `data/profile.py`, not from estimates.
  If the generator changes, regenerate, re-profile and update the card and eval ground truth together.
- `evals/cases.yaml` and `evals/holdout/holdout.yaml` are frozen by `evals/cases.lock`. Never edit a
  case to make the agent pass. A deliberate, reviewed fix is re-locked with `python -m evals.verify --freeze`.
- Never read `evals/holdout/` while building or tuning the agent or its prompts.
- Tools live in `agent/tools/`, safety layers in `agent/safety/`. Never run model-written code in-process:
  Python goes through `agent/safety/sandbox.run_python`; SQL only through `SqlTool.run`.
- The sandbox's memory/CPU rlimits are POSIX-only; on Windows only the wall-clock limit is OS-enforced.
  Their tests are skipped on Windows and first run in Linux CI.
- Graph code: `agent/graph.py` (wiring + `Agent.ask`), nodes in `agent/nodes/`, prompts versioned in `agent/prompts/*_v1.md`.
  Bump `PROMPT_VERSION` for prompt changes that should be compared in evals.
- Local models: provider `ollama` (no key, $0; `OLLAMA_NUM_CTX` must be set, Ollama's default silently truncates).
  Use tool calling, not Ollama's schema grammar (it made the 4B model loop until its token cap). Small models often
  reply with JSON in the message body: accepted for local providers only, still strictly validated. Only the
  synthesizer may fall back to plain prose (`text_field="answer"`); plans and verdicts never do.
- uv on this machine: `C:\Users\ASUS\AppData\Local\Programs\Python\Python313\python.exe -m uv` (bare `python` is 3.15, no uv).
- Models: the default `.env.example` uses the free Gemini tier (`google_genai:gemini-3.5-flash-lite` cheap,
  `google_genai:gemini-3.8-flash` strong); OpenAI is supported too. Gemini uses `json_schema` structured output,
  OpenAI `function_calling` (override: `STRUCTURED_OUTPUT_METHOD`). Quota errors raise `RateLimited`; the eval
  harness stops and marks remaining runs "not run" rather than scoring them. Pricing in `agent/llm.py` is the
  paid list price, so free-tier $/query is notional.
- Agent tests use `tests/agent_fakes.ScriptedLLM`; they never call a real model. CI never runs live evals.
- Eval reports in `evals/reports/` are committed. Keep the baseline and every "after" report; never delete a bad one.
- After each phase, review the code against this checklist and the security tests, then fix gaps.

## Engineering and security standards

No task counts as done until it meets these.

**Secrets and configuration**
- All keys come from env vars via the Pydantic `Settings` class (`agent/config.py`); the app refuses to start if a required key is missing.
- `.env` is gitignored; only `.env.example` with placeholders is committed.
- gitleaks runs in pre-commit and CI.
- Keys never appear in logs, error messages, traces, prompts or the UI (use `SecretStr`).

**Database and SQL**
- The agent connects read-only (SQLite `mode=ro`; Postgres role with SELECT only on allowed tables).
- Every query passes `sql_guard` (sqlglot AST): a single SELECT only; no DDL, DML, PRAGMA, ATTACH, COPY or stacked statements.
- Table allowlist; system tables blocked.
- LIMIT and statement timeout enforced on every query.
- Any user-supplied value used by app code (not the agent) goes through parameterized queries, never string formatting.

**Code execution sandbox**
- LLM-written code never runs in the API process; no bare `exec()` or `eval()` anywhere.
- Separate process or container: CPU, memory and 10s wall-clock limits; no network; import allowlist; temp dir deleted after each run.
- The sandbox has no access to env vars or the database file.

**LLM and prompt-injection safety**
- Plans and Critic verdicts use strict Pydantic schemas; invalid output is rejected, not parsed loosely.
- Tool permissions are fixed in code, never decided by model output.
- Data values and web results are wrapped as untrusted content and cannot trigger tool calls.
- Hard caps per request: max replans (3), max tool calls, max tokens, max cost.

**API and deployment**
- Input validation on `/ask`: max question length, request size limit, type checks.
- Rate limiting per IP (slowapi) plus a provider-side monthly spend cap.
- CORS restricted to the UI's origin; security headers set.
- Users see generic errors; stack traces go only to server logs.
- Docker: slim base, non-root user, no secrets in the image, deps pinned via lockfile.
- HTTPS only on the hosted URL.

**Logging and privacy**
- Structured JSON logs with request IDs (`agent/logging_setup.py`); no secrets or full raw rows in logs.
- LangSmith tracing masks sensitive fields.
- Synthetic data only; no real personal data ever enters the project.

**Testing and CI gates**
- Unit tests for every tool, guard and node, plus a dedicated `tests/security/` suite of malicious inputs (SQL injection, DROP/DELETE, stacked queries, sandbox escapes, prompt injection).
- CI blocks merge unless all pass: tests, ruff, mypy, bandit, pip-audit, secret scan.
- Dependabot is enabled for dependency updates.
