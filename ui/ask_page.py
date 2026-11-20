"""Ask the agent: the live-demo page.

Type a question (or pick an example), watch the agent plan, run, check and replan, then
read the answer with everything behind it: the plan, the exact SQL that ran, result
tables, the critic's verdicts and any abandoned attempts.

Model-written text (answers, assumptions, caveats, reasons) is shown as plain text,
never markdown, so a model can't make the browser fetch an image or follow a link.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pydantic import ValidationError  # noqa: E402

from agent.config import get_settings  # noqa: E402
from agent.graph import MAX_QUESTION_CHARS, AgentResponse  # noqa: E402
from agent.llm import LOCAL_PROVIDERS, RateLimited, provider_of  # noqa: E402
from ui import ask_support as support  # noqa: E402

log = logging.getLogger(__name__)
REMOTE_URL = support.api_url()  # None: run the agent in-process


@st.cache_resource(show_spinner="Starting the agent (the first run loads the sandbox)...")
def _agent() -> Any:
    return support.make_agent()


# ---------------------------------------------------------------- running a question


def run_question(question: str) -> dict[str, Any] | None:
    """Run the agent in a worker thread; redraw its progress from this (the script) thread."""
    recorder = support.NodeRecorder()
    box: dict[str, Any] = {}
    # read here: Streamlit's context isn't available in the worker thread
    ip = st.context.ip_address if REMOTE_URL else None
    client_ip = ip if isinstance(ip, str) else None
    extra = {"forwarded_for": client_ip} if REMOTE_URL else {"extra_callbacks": [recorder]}

    def work() -> None:
        try:
            box["response"] = _agent().ask(question, tags=("ui",), **extra)
        except Exception as exc:  # reported below, generically
            box["error"] = exc

    thread = threading.Thread(target=work, daemon=True)
    thread.start()
    with st.status("Working on it...", expanded=True) as status:
        progress = st.empty()
        while thread.is_alive():
            lines = support.timeline(recorder.events, recorder.started)
            elapsed = time.monotonic() - recorder.started
            idle = f"waiting for the agent API ({elapsed:.0f}s)" if REMOTE_URL else "starting..."
            progress.code("\n".join(lines) or idle, language=None)
            thread.join(0.4)
        lines = support.timeline(recorder.events, recorder.started)
        progress.code("\n".join(lines) or "-", language=None)
        response: AgentResponse | None = box.get("response")
        if response is not None:
            status.update(
                label=f"Done in {response.latency_ms / 1000:.1f}s", state="complete", expanded=False
            )
        else:
            status.update(label="Couldn't answer", state="error")

    error = box.get("error")
    if isinstance(error, ValueError):
        st.warning(f"Questions must be 1 to {MAX_QUESTION_CHARS} characters.")
        return None
    if isinstance(error, RateLimited | support.RemoteBusy):
        st.error("The agent is busy or rate-limited right now. Try again in a minute.")
        return None
    if error is not None or response is None:
        log.error("ui_ask_failed", exc_info=error)  # details to the server log only
        st.error("Something went wrong. Please try again.")
        return None
    return {"question": question, "response": response, "timeline": lines}


# ---------------------------------------------------------------- showing an answer


def _table(columns: list[str], rows: list[list[Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=columns)


def render(entry: dict[str, Any], *, local: bool) -> None:
    r: AgentResponse = entry["response"]
    icon, label = support.STATUS_LABELS.get(r.status, ("•", r.status))
    st.markdown(f"#### {icon} {label}")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Time", f"{r.latency_ms / 1000:.1f}s")
    c2.metric("Tokens", f"{r.total_tokens:,}")
    c3.metric("Cost", "$0 (local)" if local or r.cost_usd == 0 else f"${r.cost_usd:.4f}")
    c4.metric("Replans", r.replans)
    c5.metric("Steps run", sum(s.ok is not None for s in r.steps))

    with st.container(border=True):
        st.text(r.answer)
    if r.assumptions:
        st.markdown("**Assumptions it made**")
        for a in r.assumptions:
            st.text(f"• {a}")
    if r.caveats:
        st.markdown("**Caveats**")
        for c in r.caveats:
            st.text(f"• {c}")
    for chart in r.charts:
        st.image(chart.png)
        st.text(chart.alt_text)

    with st.expander("How I got here", expanded=False):
        st.markdown("**Timeline**")
        st.code("\n".join(entry["timeline"]) or "-", language=None)
        for n, attempt in enumerate(r.attempts, 1):
            who = "a rule check" if attempt.source == "rule" else "the critic model"
            st.markdown(
                f"**Attempt {n}, abandoned at step {attempt.failed_step_id}** (rejected by {who})"
            )
            st.text(attempt.reason)
            if attempt.fix:
                st.text(f"Suggested fix: {attempt.fix}")
            for step in attempt.plan.steps:
                if step.sql:
                    st.code(step.sql, language="sql", wrap_lines=True)
        if r.plan is not None and r.plan.steps:
            st.markdown("**Final plan**")
            st.dataframe(
                pd.DataFrame(
                    [
                        {"step": s.id, "tool": s.tool, "goal": s.goal,
                         "result": {True: "✅ ok", False: "❌ failed", None: "not run"}[s.ok],
                         "rows": s.row_count, "error": s.error or ""}
                        for s in r.steps
                    ]
                ),
                hide_index=True,
                use_container_width=True,
            )  # fmt: skip
        for i, sql in enumerate(r.sql, 1):
            st.markdown(f"**SQL {i}** (exactly what ran, after the safety guard)")
            st.code(sql, language="sql", wrap_lines=True)
        for data in r.step_data:
            value = data.value
            if data.columns:
                st.markdown(f"**Step {data.step_id} result**")
                st.dataframe(
                    _table(data.columns, data.rows), hide_index=True, use_container_width=True
                )
            elif isinstance(value, dict) and "columns" in value:
                st.markdown(f"**Step {data.step_id} result**")
                st.dataframe(_table(value["columns"], value["rows"]), hide_index=True)
            elif value is not None:
                st.markdown(f"**Step {data.step_id} result**")
                st.text(str(value))
        if r.verdicts:
            st.markdown("**Critic verdicts**")
            st.dataframe(
                pd.DataFrame(
                    [
                        {"attempt": v.attempt + 1, "step": v.step_id,
                         "verdict": "✅ pass" if v.passed else "❌ fail",
                         "checked by": "rules" if v.source == "rule" else "critic model",
                         "reason": v.reason}
                        for v in r.verdicts
                    ]
                ),
                hide_index=True,
                use_container_width=True,
            )  # fmt: skip
        st.caption(f"request {r.request_id} · prompt {r.prompt_version}")


# ---------------------------------------------------------------- page

st.title("💬 Ask the agent")
st.caption(
    "Ask a business question about the shop's data. The agent plans, runs read-only SQL "
    "or sandboxed Python, checks every result, and replans when a check fails."
)

history: list[dict[str, Any]] = st.session_state.setdefault("history", [])

if REMOTE_URL:
    # the API holds the model settings and keys; this page holds none
    local = False
    with st.sidebar:
        st.markdown("**Agent**")
        st.caption("Answers come from the agent's HTTP API.")
        if isinstance(agent := _agent(), support.RemoteAgent) and agent.healthy():
            st.caption("🟢 The agent API is up")
        else:
            st.caption("🔴 The agent API isn't responding. Try again shortly.")
else:
    try:
        settings = get_settings()
    except ValidationError:
        st.error("The app's settings are incomplete. Compare your .env with .env.example.")
        st.stop()
    local = provider_of(settings.cheap_model) in LOCAL_PROVIDERS
    with st.sidebar:
        st.markdown("**Model**")
        for line in support.model_summary(settings):
            st.caption(line)
        reachable = support.local_model_reachable(settings)
        if reachable is True:
            st.caption("🟢 Local model server is running")
        elif reachable is False:
            st.caption("🔴 Ollama isn't responding. Start the Ollama app, then ask again.")

with st.sidebar:
    if history:
        st.divider()
        st.markdown("**This session**")
        answered = sum(e["response"].status == "answered" for e in history)
        mean_s = sum(e["response"].latency_ms for e in history) / len(history) / 1000
        st.caption(f"{len(history)} question(s) · {answered} answered · {mean_s:.0f}s on average")
        if st.button("Clear history", use_container_width=True):
            history.clear()
            st.rerun()

st.markdown("**Try an example**")
pending: str | None = None
for i, (label, question) in enumerate(support.EXAMPLES):
    if i % 4 == 0:
        columns = st.columns(4)  # one row at a time, so narrow screens keep the order
    if columns[i % 4].button(label, help=question, key=f"example_{i}", use_container_width=True):
        st.session_state["question_box"] = question
        pending = question

with st.form("ask", clear_on_submit=False):
    typed = st.text_area(
        "Your question",
        key="question_box",
        max_chars=MAX_QUESTION_CHARS,
        height=90,
        placeholder="e.g. What were the top 2 product categories by revenue in 2025?",
    )
    if st.form_submit_button("Ask", type="primary") and typed.strip():
        pending = typed.strip()

if local:
    st.caption("Running on a small local model: expect about 1 to 3 minutes per question.")

if pending:
    entry = run_question(pending)
    if entry is not None:
        history.append(entry)

for n, entry in enumerate(reversed(history)):
    if n == 0:
        st.divider()
        st.markdown("**Question**")
        st.text(entry["question"])
        render(entry, local=local)
    else:
        if n == 1:
            st.divider()
            st.markdown("**Earlier in this session**")
        icon, _ = support.STATUS_LABELS.get(entry["response"].status, ("•", ""))
        with st.expander(f"{icon} {entry['question'][:90]}"):
            render(entry, local=local)
