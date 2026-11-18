"""The project's UI: live Q&A with the agent, plus the eval results dashboard.

uv run --group ui streamlit run ui/app.py
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="NL Analytics Agent", page_icon="📊", layout="wide")

navigation = st.navigation(
    [
        st.Page("ask_page.py", title="Ask the agent", icon="💬", default=True),
        st.Page("dashboard.py", title="Eval results", icon="📊"),
    ]
)
navigation.run()
