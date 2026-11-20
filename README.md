# NL Analytics Agent

A LangGraph agent that answers natural-language business questions over a messy
e-commerce database, catches its own mistakes, and proves its reliability with an
eval suite.

> **Status:** Phase 6 of 8 done (agent, evals, demo UI). Phase 7 (ship) is next.

## Quick start

```bash
uv sync --locked
uv run python -m data.generate   # make data
uv run pytest                    # make test
```

To check one traced LLM call (the Phase 0 exit), copy `.env.example` to `.env`, fill in
your keys, then run `uv run python scripts/hello_trace.py`.

## The dataset

12k customers, 63k orders and 136k line items across 8 quarters, generated from a fixed
seed. The data hides real signals: a Q4 churn spike in Europe, a category with high
refunds, and a holiday shipping backlog. It also hides seven flaws that break naive SQL,
including duplicate customers, split region spellings, negative refund lines, orphaned
rows and mixed currencies. Every one is documented, with measured numbers, in
[`data/DATA_CARD.md`](data/DATA_CARD.md).

## The eval set

[`evals/cases.yaml`](evals/cases.yaml) holds 30 cases written before any agent code, plus
5 held-out cases kept apart for checking generalization:

| Category | Cases | Passes when |
|---|---:|---|
| Simple lookup | 5 | number within ±1% |
| Multi-step / joins | 7 | every field matches (entity + value) |
| Dirty-data traps | 6 | duplicates, spellings, refunds and currency handled |
| Underspecified | 4 | states an assumption (or asks) and matches that reading's answer |
| No answer in data | 3 | says the data doesn't contain it |
| Adversarial | 5 | unsafe part blocked, DB unchanged, no leaks |

**Results:** `qwen3:4b-instruct` via Ollama on a laptop, $0, 30 cases x 3 runs, graded by the
same 4B model (judge fixes applied by re-grading; the original reports are kept as recorded).

| Run | Pass rate | Simple | Dirty data | Multi-step | Underspecified | No answer | Adversarial | p50 latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline ([report](evals/reports/2026-11-18_1915_baseline-local-regraded.md)) | **53.5%** (46/86) | 60% | 50% | 0% | 60% | 100% | 100% | 70s |
| + critic SQL rules, guard fix, clarify nudge ([report](evals/reports/2026-11-18_1927_critic-rules-local-regraded.md)) | **51.2%** (44/86) | 67% | 67% | 0% | 9% | 100% | 100% | 92s |
| Same, clarify nudge off ([report](evals/reports/2026-11-20_1145_no-nudge-local.md)) | **60.5%** (52/86) | 80% | 67% | 0% | 60% | 100% | 100% | 63s |
| Held-out set, 5 unseen cases ([report](evals/reports/2026-11-18_1532_holdout-local.md)) | **73.3%** (11/15) | 100% | 100% | 0% | 67% | n/a | 100% | |

The critic's new SQL rules helped where they aimed (dirty data +17 pp, simple lookups up to
+20 pp, dirty-data-missed failures 7 to 3). A nudge to state an assumption instead of asking
cost the underspecified cases 51 pp (asking is a valid answer there), so it is now off by
default; with it off, the pass rate is **60.5%, +7.0 pp over the baseline**, with 16/30 cases
passing every run. Multi-step questions remain out of reach for a 4B model. Every adversarial
run was refused with the database unchanged and no leaks. The held-out run used the nudge.

After a grader change, re-score a report's judge-graded runs without re-running the agent
(writes a new `<label>-regraded` report and leaves the original as recorded):

```bash
uv run python -m evals.regrade evals/reports/2026-11-17_1502_baseline-local.json
```

## Demo UI

One Streamlit app (localhost only) with two pages:

- **Ask the agent:** type a question or pick one of eight examples (simple lookup, dirty data,
  multi-step, chart, ambiguous, not in the data, and two safety attacks). A live timeline shows
  each planning, execution, checking and replanning step as it happens. The answer comes with
  time, tokens, cost and replans, plus a "How I got here" panel: the final plan, the exact SQL
  that ran, result tables, the critic's verdicts and any abandoned attempts.
- **Eval results:** every report, failure and run, with comparisons, trends and live progress
  of a running eval.

```bash
uv run --group ui streamlit run ui/app.py
```

Then open http://localhost:8501. On the local 4B model, expect 1 to 3 minutes per question.
Model output is shown as plain text, never rendered as markdown or HTML.

How runs are graded:

- **Answer cases:** a cheap model pulls the numbers out of the prose answer (it never sees the
  expected values), then they're scored with the case's tolerance.
- **Underspecified:** asking passes. Otherwise a judge decides which listed reading the answer
  used and whether it said so, then scores that reading's ground truth.
- **No answer / adversarial:** a rubric judge, plus hard checks for adversarial cases: the DB
  file's hash is unchanged and no forbidden strings (keys, emails) appear in the answer.
- **Every failed run gets one root cause:** plan error, SQL error, schema hallucination, dirty
  data missed, critic false alarm, synthesis error, retry budget exhausted, other
  interpretation, unsafe output, or agent error. Critic misses are counted separately.

```bash
uv run python -m evals.run_evals --label baseline                      # 30 cases x 3
uv run python -m evals.run_evals --routing all-strong --label all-strong
uv run python -m evals.compare evals/reports/<all-strong>.json evals/reports/<baseline>.json
```

Every answer comes from a ground-truth SQL query that `make evals-verify` re-runs. Each
dirty-data case also carries the naive query, which is required to *fail*, so every trap
is proven to be real. The case files are hash-locked, so any edit shows up in review.

## Tools and safety layers

| Tool | What it does | What stops abuse |
|---|---|---|
| `sql_tool` | schema summary + read-only queries, row-capped | sqlglot AST guard (one SELECT, table allowlist, LIMIT), `mode=ro` + `query_only`, SQLite authorizer, 5s timeout, 1 MB value cap |
| `python` sandbox | pandas/numpy/matplotlib on query results | separate `python -I` process, clean env (no keys), temp dir, 10s timeout, audit hook (no network, processes, DB or outside files), import allowlist, rlimits on Linux |
| `chart_tool` | PNG from a declarative spec | fixed rendering code (no model code), size caps, mathtext disabled for data labels |
| `web_search` | optional external context | refuses emails, long numbers and values taken from company data; results labeled EXTERNAL and wrapped as untrusted |

Every layer is tested on its own in `tests/security/`; for example, writes are attempted
against the engine with the AST guard bypassed.

## How it corrects itself

```
START -> planner --plan--> executor -> critic --pass, more steps--> executor
             |                            |---pass, done-----------> synthesizer -> END
             |                            '---fail, budget left----> planner (replan)
             '--decline / clarify ------------------------------------> synthesizer
```

- **Critic, rules first:** step errors (with the real column names when a column doesn't
  exist), empty or truncated results, NULL-dominated columns, negative counts, rates
  outside 0–100, implausible magnitudes. Free and deterministic.
- **Critic, one model check:** does this result answer the step's goal under the business
  definitions? It catches what rules can't, such as the same region split across two spellings.
- **Replans are different by construction:** the planner sees every failed attempt with
  the critic's reason, and a plan that would execute the same thing again is rejected in code.
- **Capped:** at most 3 replans, 15 tool calls and $0.25 per question. Past that, the agent
  lists what it tried instead of answering.

Failure injection (`tests/test_replan_loop.py`) renames a column and empties a table in a
copy of the database and checks that the agent recovers with a different plan or declines,
and that the trace shows the planner → executor → critic → planner cycle.

## Ask a question

```bash
cp .env.example .env          # then add a free Gemini API key (or switch to OpenAI)
uv run python scripts/ask.py "Which region's churn rose most from Q3 to Q4 2025?"
uv run python -m evals.run_evals --categories simple_lookup,multi_step
```

## Layout

```
agent/        config, LLM router, logging; tools/ (sql, chart, web search); safety/ (guard, sandbox)
data/         schema.sql, generate.py, profile.py, DATA_CARD.md
evals/        frozen cases, held-out set, scoring, ground-truth verifier (harness: Phase 6)
tests/        unit tests; tests/security/ for malicious inputs
```
