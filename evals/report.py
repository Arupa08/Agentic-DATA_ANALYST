"""Turn a list of RunGrades into the report metrics and a markdown page."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable
from typing import Any

from evals.grade import RunGrade
from evals.models import EvalCase


def percentile(values: Iterable[float], p: float) -> float | None:
    """Nearest-rank percentile; None for no data."""
    ordered = sorted(values)
    if not ordered:
        return None
    rank = max(1, math.ceil(p / 100 * len(ordered)))
    return ordered[rank - 1]


def _rate(passed: int, total: int) -> float | None:
    return round(passed / total, 3) if total else None


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 5) if values else None


def summarize(grades: list[RunGrade], cases: list[EvalCase]) -> dict[str, Any]:
    graded = [g for g in grades if g.graded]
    by_case: dict[str, list[RunGrade]] = defaultdict(list)
    for g in grades:
        by_case[g.case_id].append(g)

    categories: dict[str, dict[str, Any]] = {}
    for category in sorted({c.category for c in cases}):
        runs = [g for g in graded if g.category == category]
        categories[category] = {
            "runs": len(runs),
            "passed": sum(g.passed for g in runs),
            "pass_rate": _rate(sum(g.passed for g in runs), len(runs)),
            "mean_cost_usd": _mean([g.cost_usd for g in runs]),
            "p50_latency_ms": percentile((g.latency_ms for g in runs), 50),
        }

    causes: dict[str, list[str]] = defaultdict(list)
    for g in graded:
        if g.root_cause:
            causes[g.root_cause].append(f"{g.case_id}#{g.run}")

    adversarial = {}
    for case in cases:
        if case.expected_behavior != "blocked":
            continue
        runs = by_case.get(case.id, [])
        adversarial[case.id] = {
            "passed": f"{sum(g.passed for g in runs)}/{len(runs)}",
            "caught_by": sorted({layer for g in runs for layer in g.safety_layers}),
            "db_unchanged": all(g.db_unchanged is not False for g in runs),
            "leaked": sorted({s for g in runs for s in g.leaked}),
        }

    costs = [g.cost_usd for g in grades]
    latencies = [g.latency_ms for g in grades]
    return {
        "runs": len(grades),
        "graded": len(graded),
        "ungraded": len(grades) - len(graded),
        "not_run_quota": sum((g.grading_error or "").startswith("not run:") for g in grades),
        "passed": sum(g.passed for g in graded),
        "pass_rate": _rate(sum(g.passed for g in graded), len(graded)),
        "cases_passing_every_run": sum(
            all(g.passed for g in runs) for runs in by_case.values() if runs
        ),
        "cases": len(by_case),
        "by_category": categories,
        "failure_taxonomy": {k: {"count": len(v), "runs": v} for k, v in sorted(causes.items())},
        "self_correction": {
            "runs_with_replan": sum(g.replans > 0 for g in grades),
            "passed_after_replan": sum(g.passed_after_replan for g in graded),
            "critic_false_alarms": sum(g.critic_false_alarms for g in grades),
            "critic_misses": sum(g.critic_miss for g in grades),
        },
        "cost": {
            "agent_total_usd": round(sum(costs), 4),
            "mean_per_query_usd": _mean(costs),
            "grading_total_usd": round(sum(g.grading_cost_usd for g in grades), 4),
        },
        "latency_ms": {"p50": percentile(latencies, 50), "p95": percentile(latencies, 95)},
        "mean_tokens_per_query": _mean([float(g.tokens) for g in grades]),
        "adversarial": adversarial,
        "per_case": {
            cid: {
                "category": runs[0].category,
                "passed": f"{sum(g.passed for g in runs)}/{len(runs)}",
                "statuses": dict(Counter(g.status for g in runs)),
                "mean_cost_usd": _mean([g.cost_usd for g in runs]),
            }
            for cid, runs in sorted(by_case.items())
        },
    }


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def _usd(value: float | None) -> str:
    return "n/a" if value is None else f"${value:.4f}"


def _secs(value: float | None) -> str:
    return "n/a" if value is None else f"{value / 1000:.1f}s"


def to_markdown(meta: dict[str, Any], summary: dict[str, Any]) -> str:
    s = summary
    lines = [
        f"# Eval report: {meta['label']}",
        "",
        f"{meta['created']} · prompt `{meta['prompt_version']}` · cheap `{meta['models']['cheap']}` · "
        f"strong `{meta['models']['strong']}` · glossary {'on' if meta['use_glossary'] else 'off'} · "
        f"critic model check {'on' if meta['critic_llm'] else 'off'} · {meta['runs_per_case']} run(s) per case"
        + (" · **held-out set**" if meta.get("holdout") else ""),
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Pass rate (graded runs) | **{_pct(s['pass_rate'])}** ({s['passed']}/{s['graded']}) |",
        f"| Cases passing every run | {s['cases_passing_every_run']}/{s['cases']} |",
        f"| Mean cost per query | {_usd(s['cost']['mean_per_query_usd'])} |",
        f"| Latency p50 / p95 | {_secs(s['latency_ms']['p50'])} / {_secs(s['latency_ms']['p95'])} |",
        f"| Mean tokens per query | {s['mean_tokens_per_query'] or 0:,.0f} |",
        f"| Runs that replanned / passed after a replan | {s['self_correction']['runs_with_replan']} / "
        f"{s['self_correction']['passed_after_replan']} |",
        f"| Critic false alarms / misses | {s['self_correction']['critic_false_alarms']} / "
        f"{s['self_correction']['critic_misses']} |",
        f"| Ungraded runs (of which not run: quota) | {s['ungraded']} ({s['not_run_quota']}) |",
        f"| Agent cost (total) / grading cost | ${s['cost']['agent_total_usd']:.4f} / "
        f"${s['cost']['grading_total_usd']:.4f} |",
        "",
        "## By category",
        "",
        "| Category | Pass rate | Runs | Mean cost | p50 latency |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, c in s["by_category"].items():
        lines.append(
            f"| {name} | {_pct(c['pass_rate'])} | {c['passed']}/{c['runs']} | "
            f"{_usd(c['mean_cost_usd'])} | {_secs(c['p50_latency_ms'])} |"
        )
    lines += ["", "## Failure taxonomy", ""]
    if s["failure_taxonomy"]:
        lines += ["| Root cause | Runs | Examples |", "|---|---:|---|"]
        for cause, info in sorted(s["failure_taxonomy"].items(), key=lambda kv: -kv[1]["count"]):
            lines.append(f"| {cause} | {info['count']} | {', '.join(info['runs'][:6])} |")
    else:
        lines.append("No failures.")
    if s["adversarial"]:
        lines += [
            "",
            "## Safety",
            "",
            "| Case | Blocked | Caught by | DB unchanged | Leaks |",
            "|---|---:|---|---|---|",
        ]
        for cid, a in s["adversarial"].items():
            lines.append(
                f"| {cid} | {a['passed']} | {', '.join(a['caught_by']) or 'none recorded'} | "
                f"{'yes' if a['db_unchanged'] else '**NO**'} | {', '.join(a['leaked']) or 'none'} |"
            )
    lines += [
        "",
        "## Per case",
        "",
        "| Case | Category | Passed | Statuses | Mean cost |",
        "|---|---|---:|---|---:|",
    ]
    for cid, c in s["per_case"].items():
        statuses = ", ".join(f"{k} x{v}" for k, v in c["statuses"].items())
        lines.append(
            f"| {cid} | {c['category']} | {c['passed']} | {statuses} | {_usd(c['mean_cost_usd'])} |"
        )
    lines += [
        "",
        "Costs use the pricing table in `agent/llm.py`; verify it against the provider's "
        "current prices before quoting them.",
    ]
    return "\n".join(lines) + "\n"
