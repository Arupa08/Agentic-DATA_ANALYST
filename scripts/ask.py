"""Ask the agent one question.

    uv run python scripts/ask.py "What was total revenue in Q3 2025?"
    uv run python scripts/ask.py "Revenue by month in 2025" --charts out/ --plan

Requires a filled-in .env (see .env.example).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from agent.config import get_settings
from agent.deps import build_deps
from agent.graph import Agent
from agent.logging_setup import configure_logging
from agent.safety.sandbox import warm_up


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("question")
    parser.add_argument("--charts", type=Path, help="directory to save chart PNGs into")
    parser.add_argument("--plan", action="store_true", help="print the full plan JSON")
    args = parser.parse_args()

    settings = get_settings()
    configure_logging("WARNING")
    warm_up()
    response = Agent(build_deps(settings)).ask(args.question, tags=("cli",))

    print(f"\n{response.answer}\n")
    for label, items in (("Assumptions", response.assumptions), ("Caveats", response.caveats)):
        if items:
            print(f"{label}:\n" + "\n".join(f"  - {i}" for i in items))
    print("\nHow I got here:")
    for n, attempt in enumerate(response.attempts, 1):
        print(f"  attempt {n} (abandoned at step {attempt.failed_step_id}): {attempt.reason}")
        if attempt.fix:
            print(f"    critic suggested: {attempt.fix}")
    if response.attempts:
        print(f"  final plan (after {response.replans} replan(s)):")
    for s in response.steps:
        mark = {True: "ok", False: "FAILED", None: "not run"}[s.ok]
        print(f"  {s.id}. [{s.tool}] {s.goal} -> {mark}" + (f": {s.error}" if s.error else ""))
    for i, sql in enumerate(response.sql, 1):
        print(f"\nSQL {i}:\n{sql}")
    if args.plan and response.plan:
        print("\nPlan:\n" + response.plan.model_dump_json(indent=2))
    if args.charts and response.charts:
        args.charts.mkdir(parents=True, exist_ok=True)
        for i, chart in enumerate(response.charts, 1):
            path = args.charts / f"{response.request_id}_{i}.png"
            path.write_bytes(chart.png)
            print(f"\nchart saved: {path}")
    cost = f"${response.cost_usd:.4f}" + (
        " (some calls unpriced)" if response.unpriced_calls else ""
    )
    print(
        f"\nstatus={response.status} cost={cost} tokens={response.total_tokens} "
        f"latency={response.latency_ms / 1000:.1f}s request_id={response.request_id}"
    )


if __name__ == "__main__":
    main()
