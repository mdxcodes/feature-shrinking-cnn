"""Prediction helpers and confidence visualizations."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from frontend.components import get_feature_extractor, load_inference_model


def predict(model_path: str, model_input: np.ndarray) -> np.ndarray:
    model = load_inference_model(model_path)
    probs = model.predict(model_input, verbose=0)
    return probs


def extract_feature_maps(model_path: str, model_input: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    extractor = get_feature_extractor(model_path)
    conv1, conv2 = extractor.predict(model_input, verbose=0)
    return conv1[0], conv2[0]


def render_prediction(probs: np.ndarray) -> int:
    probs = np.asarray(probs).reshape(-1)
    pred = int(np.argmax(probs))
    confidence = float(probs[pred]) * 100.0

    st.markdown(f"<div class='pred-digit'>{pred}</div>", unsafe_allow_html=True)
    st.markdown(f"**Predicted digit:** {pred} &nbsp;·&nbsp; **Confidence:** {confidence:.2f}%")

    if confidence >= 80.0:
        st.success(f"High confidence prediction ({confidence:.2f}%).")
    elif confidence >= 50.0:
        st.warning(f"Moderate confidence prediction ({confidence:.2f}%). Review the processed image/probabilities.")
    else:
        st.error(f"Low confidence prediction ({confidence:.2f}%). Try a clearer, centered digit.")

    st.progress(min(max(float(probs[pred]), 0.0), 1.0))
    return pred


def render_confidence_chart(probs: np.ndarray, pred: int) -> None:
    probs = np.asarray(probs).reshape(-1)
    labels = [str(i) for i in range(10)]
    values = (probs * 100.0).astype(float)
    colors = ["#38bdf8" if i == pred else "#334155" for i in range(10)]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color=colors,
            text=[f"{v:.2f}%" for v in values],
            textposition="outside",
            cliponaxis=False,
        )
    )
    fig.update_layout(
        title="Confidence scores for digits 0–9",
        xaxis_title="Probability (%)",
        yaxis_title="Digit class",
        xaxis=dict(range=[0, max(105.0, float(values.max()) + 8.0)]),
        yaxis=dict(autorange="reversed"),
        height=420,
        margin=dict(l=30, r=40, t=50, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e6edf7"),
    )
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    total = float(probs.sum()) * 100.0
    st.caption(f"Predicted digit highlighted in blue. Sum of probabilities: **{total:.2f}%** (expected ≈ 100%).")
