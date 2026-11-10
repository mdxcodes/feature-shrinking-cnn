from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from frontend.components import close_fig


def _normalize_map_for_display(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32)
    mn, mx = float(m.min()), float(m.max())
    if mx - mn < 1e-12:
        return np.zeros_like(m)
    return (m - mn) / (mx - mn)


def render_feature_map_explorer(conv1: np.ndarray, conv2: np.ndarray) -> None:
    """Render dedicated feature-map explorer using actual activation tensors.
    conv1: (1,14,14,16) or (14,14,16), conv2: (1,12,12,32) or (12,12,32)
    """
    a1 = np.asarray(conv1)
    a2 = np.asarray(conv2)
    if a1.ndim == 4:
        a1 = a1[0]
    if a2.ndim == 4:
        a2 = a2[0]
    st.markdown("### Feature Map Explorer")
    st.caption("All feature maps shown are **actual activation tensors** from the loaded trained CNN for the current input. Display normalization is applied only for visibility; underlying values remain unchanged.")

    tab1, tab2 = st.tabs(["Conv1 — 16 maps (14×14×16)", "Conv2 — 32 maps (12×12×32)"])

    with tab1:
        st.markdown("**First convolution output** · 16 filters · 14×14 spatial size")
        cols_sel, cols_view = st.columns([1, 2])
        with cols_sel:
            idx = st.slider("Select filter/map", 0, 15, 0, key="fm_c1_idx")
            fmap = a1[:, :, idx]
            st.metric("Layer", "conv_1")
            st.metric("Map shape", "14 × 14")
            st.metric("Activation min", f"{float(fmap.min()):.6f}")
            st.metric("Activation max", f"{float(fmap.max()):.6f}")
            st.metric("Activation mean", f"{float(fmap.mean()):.6f}")
            st.markdown("**Numerical activations (14×14)**")
            st.dataframe(np.round(fmap, 5), use_container_width=True, height=260)
        with cols_view:
            st.markdown(f"**Enlarged feature map — Filter {idx}**")
            fig, ax = plt.subplots(figsize=(4.5, 4.5))
            im = ax.imshow(_normalize_map_for_display(fmap), cmap="viridis")
            ax.set_title(f"Conv1 Filter {idx} (display-normalized)")
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            st.pyplot(fig, use_container_width=True)
            close_fig(fig)
        st.markdown("**All 16 Conv1 feature maps (grid view)**")
        _render_grid(a1, "Conv1 actual activations (display-normalized) — 14×14×16", cols=8, key_prefix="c1")

    with tab2:
        st.markdown("**Second convolution output** · 32 filters · 12×12 spatial size")
        cols_sel, cols_view = st.columns([1, 2])
        with cols_sel:
            idx = st.slider("Select filter/map", 0, 31, 0, key="fm_c2_idx")
            fmap = a2[:, :, idx]
            st.metric("Layer", "conv_2")
            st.metric("Map shape", "12 × 12")
            st.metric("Activation min", f"{float(fmap.min()):.6f}")
            st.metric("Activation max", f"{float(fmap.max()):.6f}")
            st.metric("Activation mean", f"{float(fmap.mean()):.6f}")
            st.markdown("**Numerical activations (12×12)**")
            st.dataframe(np.round(fmap, 5), use_container_width=True, height=260)
        with cols_view:
            st.markdown(f"**Enlarged feature map — Filter {idx}**")
            fig, ax = plt.subplots(figsize=(4.5, 4.5))
            im = ax.imshow(_normalize_map_for_display(fmap), cmap="magma")
            ax.set_title(f"Conv2 Filter {idx} (display-normalized)")
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            st.pyplot(fig, use_container_width=True)
            close_fig(fig)
        st.markdown("**All 32 Conv2 feature maps (grid view)**")
        _render_grid(a2, "Conv2 actual activations (display-normalized) — 12×12×32", cols=8, key_prefix="c2")


def _render_grid(maps: np.ndarray, title: str, cols: int = 8, key_prefix: str = "") -> None:
    channels = maps.shape[-1]
    rows = math.ceil(channels / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.45, rows * 1.45))
    axes_arr = np.array(axes).reshape(rows, cols)
    for i in range(rows * cols):
        r, c = divmod(i, cols)
        ax = axes_arr[r, c]
        ax.axis("off")
        if i < channels:
            ax.imshow(_normalize_map_for_display(maps[:, :, i]), cmap="viridis" if key_prefix == "c1" else "magma")
            ax.set_title(f"{i}", fontsize=7, pad=1)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    close_fig(fig)
