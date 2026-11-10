from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

try:
    import plotly.graph_objects as go
except Exception:  # pragma: no cover
    go = None


def render_prediction_explorer(input_image, probs: np.ndarray) -> None:
    st.markdown("### Prediction Explorer")
    p = np.asarray(probs, dtype=np.float64).reshape(-1)
    pred = int(np.argmax(p))
    conf = float(p[pred])

    c1, c2 = st.columns([1, 1.6], gap="large")
    with c1:
        st.markdown("**Input image (28×28, as fed to model)**")
        if input_image is not None:
            st.image(input_image.resize((220, 220), __import__('PIL').Image.Resampling.NEAREST), width=220)
        st.markdown(f"<div style='font-size:4.8rem;font-weight:800;line-height:1;margin:.2rem 0;background:linear-gradient(90deg,#22d3ee,#3b82f6);-webkit-background-clip:text;-webkit-text-fill-color:transparent;'>{pred}</div>", unsafe_allow_html=True)
        st.metric("Predicted digit", str(pred))
        st.metric("Confidence", f"{conf*100:.4f}%")
        st.progress(min(max(conf, 0.0), 1.0))
        st.caption(f"Sum of probabilities: {p.sum()*100:.6f}% (≈ 100%)")

    with c2:
        st.markdown("**All ten class probabilities (actual model output)**")
        labels = [str(i) for i in range(10)]
        values = (p * 100.0)
        if go is not None:
            colors = ["#22d3ee" if i == pred else "#334155" for i in range(10)]
            fig = go.Figure(go.Bar(
                x=values, y=labels, orientation="h",
                marker_color=colors,
                text=[f"{v:.4f}%" for v in values],
                textposition="outside", cliponaxis=False,
            ))
            fig.update_layout(
                title="Softmax probabilities for digits 0–9",
                xaxis_title="Probability (%)", yaxis_title="Digit",
                xaxis=dict(range=[0, max(105.0, float(values.max()) + 10.0)]),
                yaxis=dict(autorange="reversed"),
                height=420, margin=dict(l=30, r=50, t=50, b=40),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#e6edf7"),
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        else:
            st.markdown("**Softmax probabilities for digits 0–9 (%)**")
            st.bar_chart(pd.DataFrame({"Probability (%)": values}, index=labels))
        # table
        df = pd.DataFrame({"Digit": list(range(10)), "Probability": p, "Percent": values})
        st.dataframe(df, use_container_width=True, hide_index=True)
