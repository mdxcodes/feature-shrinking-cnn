from __future__ import annotations

import streamlit as st

PREMIUM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');
:root {
  --bg-0: #03050b;
  --bg-1: #060a16;
  --bg-2: #0a1022;
  --panel: rgba(10, 16, 34, 0.72);
  --panel-solid: #0b1224;
  --border: rgba(103, 232, 249, 0.14);
  --border-strong: rgba(103, 232, 249, 0.35);
  --text: #e6edf7;
  --muted: #94a3b8;
  --cyan: #22d3ee;
  --blue: #3b82f6;
  --violet: #a78bfa;
  --white: #f8fafc;
}
html, body, [class*="css"] { font-family: 'Inter', system-ui, -apple-system, Segoe UI, sans-serif; }
.stApp {
  color: var(--text);
  background:
    radial-gradient(1200px 600px at 50% -10%, #102046 0%, transparent 55%),
    radial-gradient(900px 500px at 0% 20%, rgba(59,130,246,0.12) 0%, transparent 50%),
    radial-gradient(900px 500px at 100% 10%, rgba(167,139,250,0.10) 0%, transparent 50%),
    linear-gradient(180deg, #050814 0%, #03050b 100%);
}
.stApp::before {
  content: ""; position: fixed; inset: 0; pointer-events: none; z-index: 0;
  background-image: linear-gradient(rgba(148,163,184,0.045) 1px, transparent 1px),
                    linear-gradient(90deg, rgba(148,163,184,0.045) 1px, transparent 1px);
  background-size: 34px 34px; opacity: 0.35;
}
.block-container { max-width: 1500px; padding-top: 1.0rem; padding-bottom: 2.5rem; position: relative; z-index: 1; }
section[data-testid="stSidebar"] { background: linear-gradient(180deg, #07101f 0%, #050a16 100%); border-right: 1px solid rgba(148,163,184,0.12); }
section[data-testid="stSidebar"] * { color: var(--text); }
.lab-header {
  padding: 1.3rem 1.5rem; border: 1px solid var(--border); border-radius: 18px;
  background: linear-gradient(120deg, rgba(34,211,238,0.10), rgba(59,130,246,0.08), rgba(167,139,250,0.08));
  box-shadow: 0 0 0 1px rgba(255,255,255,0.02) inset, 0 20px 50px rgba(0,0,0,0.25);
  margin-bottom: 0.8rem;
}
.lab-header .eyebrow { color: var(--cyan); font-size: 0.78rem; letter-spacing: .18em; text-transform: uppercase; font-weight: 600; }
.lab-header h1 { margin: .15rem 0 .35rem 0; font-size: 2.0rem; font-weight: 800; letter-spacing: -0.03em; }
.lab-header p { margin: 0; color: var(--muted); font-size: 1.0rem; }
.panel {
  border: 1px solid var(--border); border-radius: 16px; padding: 1rem 1.1rem;
  background: var(--panel); backdrop-filter: blur(6px);
}
.section-title { font-size: 1.15rem; font-weight: 700; margin: .2rem 0 .4rem 0; }
.subtle { color: var(--muted); font-size: .93rem; }
div[data-testid="stMetric"] {
  border: 1px solid rgba(148,163,184,0.14); border-radius: 12px; padding: .55rem .75rem;
  background: rgba(10, 16, 34, 0.65);
}
div[data-testid="stMetric"] label { color: var(--muted) !important; }
.stTabs [data-baseweb="tab-list"] { gap: 8px; background: transparent; }
.stTabs [data-baseweb="tab"] { border-radius: 10px 10px 0 0; padding: 10px 16px; color: var(--muted); }
.stTabs [aria-selected="true"] { color: var(--white) !important; }
hr { border-color: rgba(148,163,184,0.12); }
code, .mono { font-family: 'JetBrains Mono', ui-monospace, monospace !important; }
.pipeline-caption {
  text-align:center; color: #a5f3fc; letter-spacing:.12em; font-size:.78rem; text-transform:uppercase; margin-bottom:.3rem;
}
</style>
"""


def inject_premium_css() -> None:
    st.markdown(PREMIUM_CSS, unsafe_allow_html=True)
