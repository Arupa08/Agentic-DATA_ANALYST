"""Compare two eval reports: before/after a fix, or a routing / glossary experiment.

uv run python -m evals.compare evals/reports/A.json evals/reports/B.json [--out cmp.md]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


def _pct(v: float | None) -> str:
    return "n/a" if v is None else f"{v * 100:.1f}%"


def _pp(a: float | None, b: float | None) -> str:
    return "n/a" if a is None or b is None else f"{(b - a) * 100:+.1f} pp"


def _rel(a: float | None, b: float | None) -> str:
    return "n/a" if not a or b is None else f"{(b - a) / a * 100:+.1f}%"


def headline(a: dict[str, Any], b: dict[str, Any]) -> str:
    """The one-line story, e.g. for the README."""
    sa, sb = a["summary"], b["summary"]
    ca, cb = sa["cost"]["mean_per_query_usd"], sb["cost"]["mean_per_query_usd"]
    return (
        f"{b['meta']['label']} vs {a['meta']['label']}: $/query {_rel(ca, cb)}, "
        f"pass rate {_pp(sa['pass_rate'], sb['pass_rate'])} "
        f"({_pct(sa['pass_rate'])} -> {_pct(sb['pass_rate'])})."
    )


def compare(a: dict[str, Any], b: dict[str, Any]) -> str:
    sa, sb = a["summary"], b["summary"]
    la, lb = a["meta"]["label"], b["meta"]["label"]
    lines = [
        f"# {la} vs {lb}",
        "",
        headline(a, b),
        "",
        f"| Metric | {la} | {lb} | Change |",
        "|---|---:|---:|---:|",
        f"| Pass rate | {_pct(sa['pass_rate'])} | {_pct(sb['pass_rate'])} | {_pp(sa['pass_rate'], sb['pass_rate'])} |",
        f"| Mean $/query | ${sa['cost']['mean_per_query_usd'] or 0:.4f} | ${sb['cost']['mean_per_query_usd'] or 0:.4f} | "
        f"{_rel(sa['cost']['mean_per_query_usd'], sb['cost']['mean_per_query_usd'])} |",
        f"| p50 latency | {(sa['latency_ms']['p50'] or 0) / 1000:.1f}s | {(sb['latency_ms']['p50'] or 0) / 1000:.1f}s | "
        f"{_rel(sa['latency_ms']['p50'], sb['latency_ms']['p50'])} |",
        f"| p95 latency | {(sa['latency_ms']['p95'] or 0) / 1000:.1f}s | {(sb['latency_ms']['p95'] or 0) / 1000:.1f}s | "
        f"{_rel(sa['latency_ms']['p95'], sb['latency_ms']['p95'])} |",
        f"| Critic false alarms | {sa['self_correction']['critic_false_alarms']} | "
        f"{sb['self_correction']['critic_false_alarms']} | |",
        "",
        "| Category | " + la + " | " + lb + " | Change |",
        "|---|---:|---:|---:|",
    ]
    for cat in sorted(set(sa["by_category"]) | set(sb["by_category"])):
        ra = sa["by_category"].get(cat, {}).get("pass_rate")
        rb = sb["by_category"].get(cat, {}).get("pass_rate")
        lines.append(f"| {cat} | {_pct(ra)} | {_pct(rb)} | {_pp(ra, rb)} |")
    causes = sorted(set(sa["failure_taxonomy"]) | set(sb["failure_taxonomy"]))
    if causes:
        lines += ["", f"| Root cause | {la} | {lb} |", "|---|---:|---:|"]
        for cause in causes:
            na = sa["failure_taxonomy"].get(cause, {}).get("count", 0)
            nb = sb["failure_taxonomy"].get(cause, {}).get("count", 0)
            lines.append(f"| {cause} | {na} | {nb} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    text = compare(_load(args.before), _load(args.after))
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
