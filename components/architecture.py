from __future__ import annotations

import streamlit as st

try:
    import plotly.graph_objects as go
except Exception:  # pragma: no cover
    go = None


def render_architecture_explorer(model) -> None:
    st.markdown("### Architecture Explorer")
    st.caption("Real architecture from the loaded trained Keras model (unchanged).")
    if model is None:
        st.warning("Model not loaded.")
        return

    # Actual layer metadata
    rows = []
    for layer in model.layers:
        out_shape = getattr(layer, "output", None)
        out_shape = tuple(out_shape.shape) if out_shape is not None else None
        r = {
            "Layer": layer.name,
            "Type": layer.__class__.__name__,
            "Output shape": str(out_shape),
            "Params": int(layer.count_params()),
        }
        if layer.__class__.__name__ == "Conv2D":
            r.update({
                "Filters/Units": layer.filters,
                "Kernel": str(layer.kernel_size),
                "Stride": str(layer.strides),
                "Padding": layer.padding,
                "Activation": getattr(layer.activation, "__name__", str(layer.activation)),
            })
        elif layer.__class__.__name__ == "Dense":
            r.update({
                "Filters/Units": layer.units,
                "Kernel": "—",
                "Stride": "—",
                "Padding": "—",
                "Activation": getattr(layer.activation, "__name__", str(layer.activation)),
            })
        else:
            r.update({"Filters/Units": "—", "Kernel": "—", "Stride": "—", "Padding": "—", "Activation": "—"})
        rows.append(r)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total parameters", f"{model.count_params():,}")
    c2.metric("Trainable", f"{sum(int(w.shape.num_elements()) for w in model.trainable_weights):,}")
    c3.metric("Non-trainable", f"{sum(int(w.shape.num_elements()) for w in model.non_trainable_weights):,}")
    c4.metric("Pooling layers", "0")

    st.dataframe(rows, use_container_width=True, hide_index=True)

    # Dimension pipeline
    st.markdown("#### Dimension changes through the network")
    stages = [
        ("Input", "28 × 28 × 1", "784 spatial values"),
        ("Conv1 (conv_1)", "14 × 14 × 16", "3×3, stride 2, SAME, ReLU"),
        ("Conv2 (conv_2)", "12 × 12 × 32", "3×3, stride 1, VALID, ReLU"),
        ("Flatten", "4,608", "12 × 12 × 32"),
        ("Dense (predictions)", "10", "Softmax probabilities"),
    ]
    for i, (name, shape, detail) in enumerate(stages):
        a, b = st.columns([1, 2.2])
        a.markdown(f"**{name}**")
        a.markdown(f"### `{shape}`")
        b.markdown(detail)
        if i < len(stages) - 1:
            st.markdown("<div style='text-align:center;color:#67e8f9;font-size:1.3rem;margin:-.2rem 0;'>↓</div>", unsafe_allow_html=True)
        st.markdown("---")

    # Shrinkage chart
    spatial = [28 * 28, 14 * 14, 12 * 12]
    labels = ["Input", "After Conv1", "After Conv2"]
    if go is not None:
        fig = go.Figure(go.Bar(
            x=labels,
            y=spatial,
            text=["28×28 = 784", "14×14 = 196", "12×12 = 144"],
            textposition="outside",
            marker_color=["#22d3ee", "#818cf8", "#34d399"],
        ))
        fig.update_layout(title="Spatial positions per feature map (H×W)", yaxis_title="Positions", height=360,
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#e6edf7"))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    else:
        import pandas as pd
        st.markdown("**Spatial positions per feature map (H×W)**")
        st.bar_chart(pd.DataFrame({"Positions": spatial}, index=labels))

    st.markdown("#### Why the dimensions change (no pooling)")
    st.markdown(
        "- **Stride 2 in Conv1:** the 3×3 kernel moves in steps of 2 across 28×28, halving spatial resolution to 14×14 while learning 16 filters. Downsampling is learned, not pooled.\n"
        "- **SAME padding in Conv1:** zero-padding is applied as needed so output size follows `ceil(input/stride)`; TensorFlow may pad asymmetrically (right/bottom) — receptive fields in the animation are derived from this rule.\n"
        "- **VALID padding in Conv2:** no padding is added, so a 3×3 kernel over 14×14 produces 12×12 (`14-3+1=12`) while depth increases to 32 feature maps.\n"
        "- **No pooling layer:** max/average pooling is absent. Spatial shrinkage comes entirely from strided convolution and valid padding.\n"
        "- **Flatten (4,608):** `12 × 12 × 32 = 4,608` activations are unfolded in Keras channels-last order into the Dense classifier input."
    )

    with st.expander("Keras model.summary()"):
        lines = []
        model.summary(print_fn=lambda x: lines.append(x))
        st.code("\n".join(lines), language="text")
