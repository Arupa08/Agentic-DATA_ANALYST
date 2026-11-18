"""Eval dashboard: results, failure analysis, safety, comparisons and live progress.

    uv run --group ui streamlit run ui/dashboard.py

Reads evals/reports/*.json (finished reports) and evals/reports/live/ (progress of a
running eval). Model-written text (answers, SQL) is rendered as plain text or code,
never as markdown, so a model-produced link or image can't make the browser fetch it.
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

# `streamlit run` puts ui/ on sys.path, not the project root
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evals.compare import headline  # noqa: E402
from evals.models import load_cases  # noqa: E402
from evals.verify import CASES_PATH  # noqa: E402
from ui.dashboard_data import (  # noqa: E402
    ReportRef,
    case_rows,
    case_runs,
    category_rows,
    compare_rows,
    hours_minutes,
    kpis,
    latest_live,
    layer_counts,
    list_reports,
    load_report,
    safety_rows,
    taxonomy_rows,
    trend_rows,
)

# validated default palette (dataviz skill): categorical slot 1 for single-series charts
SERIES_1 = "#2a78d6"
PASS, FAIL, SKIP = "✅ pass", "❌ fail", "⏸ ungraded"

st.set_page_config(page_title="Agent evals", page_icon="📊", layout="wide")


# ---------------------------------------------------------------- data access


@st.cache_data(show_spinner=False)
def _report(path: str, _mtime: float) -> dict[str, Any]:
    return load_report(Path(path))


def report_for(ref: ReportRef) -> dict[str, Any]:
    return _report(str(ref.path), ref.path.stat().st_mtime)


@st.cache_data(show_spinner=False)
def _questions() -> dict[str, str]:
    # main set only: held-out questions are never surfaced while the agent is being tuned
    return {c.id: c.question for c in load_cases(CASES_PATH).cases}


def outcome(run: dict[str, Any]) -> str:
    if not run.get("graded", True):
        return SKIP
    return PASS if run.get("passed") else FAIL


# ---------------------------------------------------------------- charts


def hbar(rows: list[dict[str, Any]], *, y: str, x: str, x_title: str, domain: list[float] | None = None,
         tooltip: list[str] | None = None) -> alt.Chart:  # fmt: skip
    df = pd.DataFrame(rows)
    scale = alt.Scale(domain=domain) if domain else alt.Undefined
    chart: alt.Chart = (
        alt.Chart(df)
        .mark_bar(color=SERIES_1, cornerRadiusEnd=4, height={"band": 0.7})
        .encode(
            # counts get whole-number ticks; percentages keep their 0-100 domain
            x=alt.X(
                f"{x}:Q",
                title=x_title,
                scale=scale,
                axis=alt.Axis(tickMinStep=1, format="d") if domain is None else alt.Undefined,
            ),
            y=alt.Y(
                f"{y}:N", sort="-x", title=None, axis=alt.Axis(labelOverlap=False, labelLimit=180)
            ),
            tooltip=tooltip or [y, x],
        )
        .properties(height=alt.Step(30))  # a fixed height per bar, so every label fits
    )
    return chart


def trend_chart(rows: list[dict[str, Any]]) -> alt.LayerChart:
    df = pd.DataFrame(rows)
    df["when"] = pd.to_datetime(df["created"], format="%Y-%m-%d_%H%M", errors="coerce")
    base = alt.Chart(df).encode(
        x=alt.X(
            "when:T", title=None, axis=alt.Axis(format="%b %d %H:%M", labelAngle=0, tickCount=6)
        ),
        y=alt.Y("pass_rate_pct:Q", title="Pass rate (%)", scale=alt.Scale(domain=[0, 100])),
        tooltip=["label", "created", "pass_rate_pct", "runs"],
    )
    layered: alt.LayerChart = (base.mark_line(color=SERIES_1, strokeWidth=2) + base.mark_point(
        color=SERIES_1, filled=True, size=80)).properties(height=260)  # fmt: skip
    return layered


# ---------------------------------------------------------------- sidebar

refs = list_reports()
st.sidebar.title("NL Analytics Agent")
st.sidebar.caption("Eval dashboard")
if st.sidebar.button("Reload reports", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

if not refs:
    st.title("No eval reports yet")
    st.write("Run an eval and its report will appear here:")
    st.code("uv run python -m evals.run_evals --label baseline", language="bash")
    st.stop()

ref = st.sidebar.selectbox("Report", refs, format_func=lambda r: r.title)
others = [r for r in refs if r.path != ref.path]
compare_ref = st.sidebar.selectbox(
    "Compare with", [None, *others], format_func=lambda r: "nothing" if r is None else r.title
)
report = report_for(ref)
base_report = report_for(compare_ref) if compare_ref else None
meta = report["meta"]
k = kpis(report)
kb = kpis(base_report) if base_report else None

st.sidebar.divider()
st.sidebar.markdown("**This report**")
st.sidebar.caption(
    f"Models: cheap `{meta['models']['cheap']}`, strong `{meta['models']['strong']}`  \n"
    f"Agent `{meta.get('agent_revision', 'initial')}`  \n"
    f"Prompt `{meta.get('prompt_version')}` · glossary {'on' if meta.get('use_glossary') else 'off'} · "
    f"critic model check {'on' if meta.get('critic_llm') else 'off'}  \n"
    f"{meta.get('runs_per_case')} run(s) per case"
    + (" · **held-out set**" if meta.get("holdout") else "")
)


def delta(key: str, scale: float = 1.0, unit: str = "") -> str | None:
    if kb is None or k.get(key) is None or kb.get(key) is None:
        return None
    return f"{(k[key] - kb[key]) * scale:+.1f}{unit}"


# ---------------------------------------------------------------- tabs

st.title(f"Eval report: {meta['label']}")
st.caption(f"{meta['created'].replace('_', ' ')} · {k['runs']} runs over {k['cases']} cases")
tab_overview, tab_cats, tab_cases, tab_safety, tab_compare, tab_live = st.tabs(
    ["Overview", "Categories & failures", "Cases", "Safety", "Compare & trend", "Live run"]
)

with tab_overview:
    rate = "n/a" if k["pass_rate"] is None else f"{k['pass_rate'] * 100:.1f}%"
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Pass rate", rate, delta("pass_rate", 100, " pp"), help="Passed / graded runs")
    c2.metric("Cases passing every run", f"{k['cases_passing_every_run']}/{k['cases']}")
    c3.metric("Graded runs passed", f"{k['passed']}/{k['graded']}")
    c4.metric(
        "Ungraded runs", k["ungraded"], help=f"of which not run (quota): {k['not_run_quota']}"
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Latency p50",
        f"{k['p50_latency_s']}s",
        delta("p50_latency_s", 1, "s"),
        delta_color="inverse",
    )
    c2.metric(
        "Latency p95",
        f"{k['p95_latency_s']}s",
        delta("p95_latency_s", 1, "s"),
        delta_color="inverse",
    )
    c3.metric("Tokens per query", f"{k['mean_tokens'] or 0:,.0f}")
    local = str(meta["models"]["cheap"]).startswith("ollama:")
    c4.metric(
        "Cost per query",
        f"${k['mean_cost_usd'] or 0:.4f}",
        help="Local models cost $0. Cloud costs use the list prices in agent/llm.py.",
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Runs that replanned", k["runs_with_replan"])
    c2.metric("Passed after a replan", k["passed_after_replan"])
    c3.metric(
        "Critic false alarms", k["critic_false_alarms"], help="A correct result the critic rejected"
    )
    c4.metric("Critic misses", k["critic_misses"], help="A wrong result the critic passed")
    if local:
        st.caption("Local model run: answers and grading both come from a small local model.")
    left, right = st.columns(2)
    with left:
        st.subheader("Pass rate by category")
        st.altair_chart(
            hbar(category_rows(report), y="category", x="pass_rate_pct", x_title="Pass rate (%)",
                 domain=[0, 100], tooltip=["category", "pass_rate_pct", "passed", "runs"]),
            use_container_width=True,
        )  # fmt: skip
    with right:
        st.subheader("Why runs failed")
        tax = taxonomy_rows(report)
        if tax:
            st.altair_chart(
                hbar(
                    tax,
                    y="root_cause",
                    x="runs",
                    x_title="Failed runs",
                    tooltip=["root_cause", "runs", "examples"],
                ),
                use_container_width=True,
            )
        else:
            st.success("No failed runs.")

with tab_cats:
    st.subheader("Categories")
    st.dataframe(
        pd.DataFrame(category_rows(report)),
        hide_index=True,
        use_container_width=True,
        column_config={
            "pass_rate_pct": st.column_config.ProgressColumn(
                "Pass rate", format="%.1f%%", min_value=0, max_value=100
            ),
            "mean_cost_usd": st.column_config.NumberColumn("Mean cost", format="$%.4f"),
            "p50_latency_s": st.column_config.NumberColumn("p50 latency", format="%.1fs"),
        },
    )
    st.subheader("Failure taxonomy")
    st.caption(
        "One root cause per failed run. Critic misses overlap with SQL errors and dirty-data misses, "
        "so they're counted on the Overview tab instead."
    )
    st.dataframe(pd.DataFrame(taxonomy_rows(report)), hide_index=True, use_container_width=True)

with tab_cases:
    rows = case_rows(report)
    cats = sorted({r["category"] for r in rows})
    f1, f2 = st.columns([3, 1])
    chosen_cats = f1.multiselect("Categories", cats, default=cats)
    only_failing = f2.toggle("Only cases with failures")
    shown = [
        r for r in rows
        if r["category"] in chosen_cats and (not only_failing or r["pass_fraction"] < 1)
    ]  # fmt: skip
    st.dataframe(
        pd.DataFrame(shown),
        hide_index=True,
        use_container_width=True,
        column_config={
            "pass_fraction": st.column_config.ProgressColumn(
                "Pass share", min_value=0, max_value=1, format="%.2f"
            ),
            "mean_latency_s": st.column_config.NumberColumn("Mean latency", format="%.1fs"),
        },
    )
    if shown:
        case_id = st.selectbox("Inspect a case", [r["case_id"] for r in shown])
        question = _questions().get(case_id)
        if question:
            st.markdown("**Question**")
            st.text(question)
        for run in case_runs(report, case_id):
            title = (
                f"Run {run['run']} · {outcome(run)} · {run.get('status')} · "
                f"{run.get('latency_ms', 0) / 1000:.1f}s"
                + (f" · {run['root_cause']}" if run.get("root_cause") else "")
            )
            with st.expander(title, expanded=not run.get("passed")):
                st.markdown("**Answer**")
                st.text(run.get("answer") or "(none)")
                if run.get("assumptions"):
                    st.markdown("**Stated assumptions**")
                    for a in run["assumptions"]:
                        st.text(f"• {a}")
                if run.get("fields"):
                    st.markdown("**Scored fields**")
                    st.dataframe(
                        pd.DataFrame(
                            [{"field": f["name"], "result": "✅ match" if f["passed"] else "❌ mismatch",
                              "expected": str(f["expected"]), "extracted": str(f["got"])} for f in run["fields"]]
                        ),
                        hide_index=True, use_container_width=True,
                    )  # fmt: skip
                if run.get("rubric"):
                    st.markdown("**Rubric**")
                    for item in run["rubric"]:
                        st.text(
                            f"{'✅' if item['passed'] else '❌'} {item['index']}. {item['reason']}"
                        )
                if run.get("interpretation"):
                    st.caption(f"Judge: answered under reading #{run['interpretation']}")
                if run.get("safety_layers"):
                    st.caption(
                        "Safety layers that stopped something: " + ", ".join(run["safety_layers"])
                    )
                if run.get("grading_error"):
                    st.warning(f"Not graded: {run['grading_error']}")
                for i, sql in enumerate(run.get("sql") or [], 1):
                    st.markdown(f"**SQL {i}** (exactly what ran)")
                    st.code(sql, language="sql", wrap_lines=True)
                st.caption(
                    f"tokens {run.get('tokens', 0):,} · replans {run.get('replans', 0)} · "
                    f"critic false alarms {run.get('critic_false_alarms', 0)} · request {run.get('request_id', '')}"
                )

with tab_safety:
    srows = safety_rows(report)
    if not srows:
        st.info("This report has no adversarial cases.")
    else:
        st.caption(
            "Blocked = refused the unsafe part, the database file's hash was unchanged, and no keys or "
            "customer emails appeared in the answer."
        )
        st.dataframe(
            pd.DataFrame(srows),
            hide_index=True,
            use_container_width=True,
            column_config={"db_unchanged": st.column_config.CheckboxColumn("DB unchanged")},
        )
        layers = layer_counts(report)
        if layers:
            st.subheader("Which layer stopped something")
            st.altair_chart(
                hbar(layers, y="layer", x="runs", x_title="Adversarial runs"),
                use_container_width=True,
            )

with tab_compare:
    if base_report is None or compare_ref is None:
        st.info(
            "Pick a report under **Compare with** in the sidebar to see a side-by-side comparison."
        )
    else:
        st.subheader(f"{compare_ref.label} → {meta['label']}")
        st.write(headline(base_report, report))
        st.dataframe(
            pd.DataFrame(compare_rows(base_report, report)).rename(
                columns={"base": compare_ref.label, "other": meta["label"]}
            ),
            hide_index=True,
            use_container_width=True,
        )
    trend = trend_rows(refs)
    if len(trend) >= 2:
        st.subheader("Pass rate over time (all reports)")
        st.altair_chart(trend_chart(trend), use_container_width=True)


@st.fragment(run_every=10)
def live_panel() -> None:
    status = latest_live()
    now = dt.datetime.now()
    if status is None:
        st.info("No live run. Progress appears here (refreshing every 10s) while an eval runs:")
        st.code("uv run python -m evals.run_evals --label baseline", language="bash")
        return
    state = status.state(now)
    st.subheader(f"{status.label}: {state}")
    st.progress(
        status.done / status.total if status.total else 0.0,
        text=f"{status.done} of {status.total} runs",
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Passed", f"{status.passed}/{status.graded}")
    running = status.running_pass_rate
    c2.metric("Running pass rate", "n/a" if running is None else f"{running * 100:.1f}%")
    elapsed = (now - status.started).total_seconds() if status.started else 0
    c3.metric("Elapsed", hours_minutes(elapsed))
    eta = status.eta_seconds(now)
    c4.metric("Est. remaining", "n/a" if eta is None else hours_minutes(eta))
    if status.ended and status.report:
        st.success(f"Finished. Report: {status.report}. Use **Reload reports** to open it.")
    if status.runs:
        recent = [
            {"run": f"{r['case_id']} #{r['run']}", "result": outcome(r), "status": r.get("status"),
             "root cause": r.get("root_cause") or "", "latency": f"{r.get('latency_ms', 0) / 1000:.1f}s",
             "at": str(r.get("at", ""))[11:]}
            for r in reversed(status.runs[-15:])
        ]  # fmt: skip
        st.dataframe(pd.DataFrame(recent), hide_index=True, use_container_width=True)


with tab_live:
    live_panel()
