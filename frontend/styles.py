"""Custom CSS and shared style helpers for the Streamlit app."""

from __future__ import annotations

import streamlit as st

APP_CSS = """
<style>
:root {
  --bg: #0b1020;
  --panel: #121a2e;
  --panel-2: #16213d;
  --border: rgba(148, 163, 184, 0.18);
  --text: #e6edf7;
  --muted: #94a3b8;
  --accent: #38bdf8;
  --accent-2: #818cf8;
  --good: #22c55e;
  --warn: #f59e0b;
}
.stApp { background: radial-gradient(1200px 600px at 10% -10%, #162544 0%, var(--bg) 45%), var(--bg); color: var(--text); }
.block-container { max-width: 1400px; padding-top: 1.2rem; padding-bottom: 2rem; }
section[data-testid="stSidebar"] { background: linear-gradient(180deg, #0f172a 0%, #0b1224 100%); border-right: 1px solid var(--border); }
section[data-testid="stSidebar"] * { color: var(--text); }
.hero {
  padding: 1.2rem 1.4rem; border: 1px solid var(--border); border-radius: 16px;
  background: linear-gradient(110deg, rgba(56,189,248,.12), rgba(129,140,248,.10));
  margin-bottom: 0.8rem;
}
.hero h1 { margin: 0 0 .2rem 0; font-size: 1.8rem; font-weight: 700; letter-spacing: -0.02em; }
.hero p { margin: 0; color: var(--muted); }
.card {
  border: 1px solid var(--border); border-radius: 14px; padding: 1rem 1.1rem;
  background: rgba(18, 26, 46, 0.75);
}
div[data-testid="stMetric"] {
  border: 1px solid var(--border); border-radius: 12px; padding: .6rem .8rem;
  background: rgba(18, 26, 46, 0.7);
}
.small-muted { color: var(--muted); font-size: 0.92rem; }
.pred-digit {
  font-size: 4.6rem; font-weight: 800; line-height: 1; margin: .1rem 0 .4rem 0;
  background: linear-gradient(90deg, #7dd3fc, #a5b4fc); -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.stTabs [data-baseweb="tab-list"] { gap: 6px; }
.stTabs [data-baseweb="tab"] { border-radius: 10px 10px 0 0; padding: 10px 16px; }
hr { border-color: var(--border); }
</style>
"""


def inject_css() -> None:
    st.markdown(APP_CSS, unsafe_allow_html=True)


def section_title(title: str, subtitle: str | None = None) -> None:
    st.subheader(title)
    if subtitle:
        st.markdown(f"<div class='small-muted'>{subtitle}</div>", unsafe_allow_html=True)
