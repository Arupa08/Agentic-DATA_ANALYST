"""Phase 0 exit check: one traced LLM call, with tokens and cost logged.

    uv run python scripts/hello_trace.py

Requires a filled-in .env. With LANGSMITH_TRACING=true the call shows up in the
LangSmith project named by LANGSMITH_PROJECT, through the same masked tracer the agent uses.
"""

from __future__ import annotations

from agent.config import get_settings
from agent.llm import get_model, track_costs
from agent.logging_setup import configure_logging
from agent.tracing import build_tracer


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    tracer = build_tracer(settings)
    with track_costs() as tracker:
        reply = get_model("cheap").invoke(
            "Reply with one short sentence confirming you are reachable.",
            config={
                "run_name": "hello_world",
                "tags": ["phase-0"],
                "callbacks": [tracer] if tracer else [],
            },
        )
    print(reply.content)
    print(f"tokens={tracker.total_tokens} cost_usd={tracker.total_cost_usd:.6f}")
    print("traced to LangSmith" if tracer else "tracing off (LANGSMITH_TRACING=false)")


if __name__ == "__main__":
    main()
