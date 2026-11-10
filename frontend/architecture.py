"""CNN architecture explorer for the Feature-Shrinking CNN."""

from __future__ import annotations

import streamlit as st

try:
    import plotly.graph_objects as go
except Exception:  # pragma: no cover
    go = None


PIPELINE = [
    ("Input", "28 × 28 × 1", "Grayscale image"),
    ("Conv2D 1", "14 × 14 × 16", "16 filters · 3×3 kernel · stride 2×2 · same padding · ReLU"),
    ("Conv2D 2", "12 × 12 × 32", "32 filters · 3×3 kernel · stride 1×1 · valid padding · ReLU"),
    ("Flatten", "4,608", "12 × 12 × 32 feature values"),
    ("Dense + Softmax", "10", "Probability for each digit (0–9)"),
]


def render_architecture_explorer(model) -> None:
    st.markdown(
        "This section describes the **existing, unchanged** Sequential CNN used for inference. "
        "Spatial downsampling is achieved by convolution stride and valid padding — **not** by pooling."
    )

    c1, c2, c3, c4 = st.columns(4)
    params = model.count_params() if model is not None else 50890
    c1.metric("Total trainable params", f"{int(params):,}")
    c2.metric("Pooling layers", "0")
    c3.metric("Input dimensions", "28 × 28 × 1")
    c4.metric("Output dimensions", "10 classes")

    st.markdown("#### Complete pipeline")
    for i, (name, shape, detail) in enumerate(PIPELINE):
        left, right = st.columns([1, 2])
        with left:
            st.markdown(f"**{name}**")
            st.markdown(f"#### `{shape}`")
        with right:
            st.markdown(detail)
            if name == "Conv2D 1":
                st.caption("Stride 2 downsamples 28×28 → 14×14; same padding preserves spatial alignment.")
            elif name == "Conv2D 2":
                st.caption("Valid padding (no zero-padding) shrinks 14×14 → 12×12 while depth increases to 32.")
            elif name == "Flatten":
                st.caption("12 × 12 × 32 = 4,608 learned features fed to the classifier.")
        if i < len(PIPELINE) - 1:
            st.markdown("<div style='text-align:center;color:#64748b;font-size:1.2rem;'>↓</div>", unsafe_allow_html=True)
        st.divider()

    st.markdown("#### Layer configuration (actual model)")
    if model is not None:
        rows = []
        for layer in model.layers:
            cfg = {"Layer": layer.name, "Type": layer.__class__.__name__, "Output shape": str(getattr(layer, "output", None).shape if getattr(layer, "output", None) is not None else "n/a")}
            if layer.__class__.__name__ == "Conv2D":
                cfg.update({
                    "Filters": layer.filters,
                    "Kernel": str(layer.kernel_size),
                    "Stride": str(layer.strides),
                    "Padding": layer.padding,
                    "Activation": getattr(layer.activation, "__name__", str(layer.activation)),
                })
            elif layer.__class__.__name__ == "Dense":
                cfg.update({"Units": layer.units, "Activation": getattr(layer.activation, "__name__", str(layer.activation))})
            rows.append(cfg)
        st.dataframe(rows, width="stretch", hide_index=True)
        # Also show Keras summary text
        lines = []
        model.summary(print_fn=lambda x: lines.append(x))
        st.code("\n".join(lines), language="text")
    else:
        st.info("Model not loaded; showing expected configuration.")
        st.code(
            "Input (28,28,1) -> Conv2D(16,3x3,stride2,same,relu) -> Conv2D(32,3x3,stride1,valid,relu) -> Flatten -> Dense(10,softmax)",
            language="text",
        )

    st.markdown("#### Spatial dimension shrinkage (visual comparison)")
    # Relative spatial areas/widths for 28,14,12 and depth intuition
    stages = ["Input", "After Conv1", "After Conv2"]
    spatial = [28 * 28, 14 * 14, 12 * 12]  # 784,196,144
    labels = ["28×28 (784)", "14×14 (196)", "12×12 (144)"]
    colors = ["#38bdf8", "#818cf8", "#22c55e"]
    if go is not None:
        fig = go.Figure(go.Bar(x=stages, y=spatial, text=labels, textposition="outside", marker_color=colors))
        fig.update_layout(title="Spatial positions per feature map across stages", yaxis_title="Spatial positions (H×W)", height=380, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#e6edf7"))
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    else:
        import pandas as pd
        st.markdown("**Spatial positions per feature map across stages**")
        st.bar_chart(pd.DataFrame({"Spatial positions (H×W)": spatial}, index=stages))

    a, b, c = st.columns(3)
    a.metric("Input spatial positions", "784")
    b.metric("After Conv1", "196", "-75.0% vs input")
    c.metric("After Conv2", "144", "-81.6% vs input")

    st.markdown("#### How downsampling works without pooling")
    st.markdown(
        "- **Strided convolution (Conv1):** a 3×3 kernel with stride 2 and `same` padding moves in steps of two, "
        "halving spatial resolution (28×28 → 14×14) while learning 16 feature detectors.\n"
        "- **Valid padding (Conv2):** a 3×3 kernel with stride 1 and `valid` padding applies no zero-padding, so each "
        "convolution drops border pixels (14×14 → 12×12) while expanding depth to 32 richer feature maps.\n"
        "- **No pooling layers:** max/average pooling are not used; spatial shrinkage is controlled entirely by learned "
        "convolutional filters, preserving task-relevant structure for the 4,608 flattened features and 10-class softmax output."
    )
