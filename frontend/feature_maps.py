"""Feature map extraction and visualization (conv_1 and conv_2)."""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from frontend.components import close_fig
from frontend.prediction import extract_feature_maps


def _render_feature_grid(maps: np.ndarray, title: str, cols: int, cmap: str = "viridis") -> None:
    # maps: (H, W, C)
    channels = maps.shape[-1]
    rows = math.ceil(channels / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))
    axes_arr = np.array(axes).reshape(rows, cols)
    for i in range(rows * cols):
        r, c = divmod(i, cols)
        ax = axes_arr[r, c]
        ax.axis("off")
        if i < channels:
            ax.imshow(maps[:, :, i], cmap=cmap)
            ax.set_title(f"F{i}", fontsize=7, pad=2)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    st.pyplot(fig, width="stretch")
    close_fig(fig)


def render_feature_maps_section(model_path: str, model_input: np.ndarray) -> None:
    st.markdown(
        "Feature maps are extracted from the **same input** used for prediction via an intermediate-output "
        "Keras model (original trained CNN unchanged)."
    )
    with st.spinner("Extracting convolution feature maps..."):
        conv1_maps, conv2_maps = extract_feature_maps(model_path, model_input)

    st.markdown("### First convolution feature maps (`conv_1`)")
    st.caption(
        f"Output shape: **(1, {conv1_maps.shape[0]}, {conv1_maps.shape[1]}, {conv1_maps.shape[2]})** → 16 maps at 14×14. "
        "These activations capture low-level visual features such as edges, strokes, and simple patterns."
    )
    _render_feature_grid(conv1_maps, "Conv1 feature maps — 14×14×16 (all 16 filters)", cols=8)

    st.markdown("### Second convolution feature maps (`conv_2`)")
    st.caption(
        f"Output shape: **(1, {conv2_maps.shape[0]}, {conv2_maps.shape[1]}, {conv2_maps.shape[2]})** → 32 maps at 12×12. "
        "These activations combine lower-level patterns into more complex structural representations of the digit."
    )
    _render_feature_grid(conv2_maps, "Conv2 feature maps — 12×12×32 (all 32 filters)", cols=8)

    c1, c2 = st.columns(2)
    c1.metric("Conv1 maps", f"{conv1_maps.shape[2]} filters", "14 × 14")
    c2.metric("Conv2 maps", f"{conv2_maps.shape[2]} filters", "12 × 12")
