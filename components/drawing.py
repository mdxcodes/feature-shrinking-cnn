from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

from visualization.preprocessing import bytes_to_pil, canvas_to_pil, preprocess_digit_image

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"


def get_example_images() -> list[tuple[str, Image.Image]]:
    """Return example digit images from existing results if available."""
    examples: list[tuple[str, Image.Image]] = []
    sample_path = RESULTS_DIR / "mnist_samples.png"
    # We don't have individual example digits saved except maybe in arrays; keep simple and safe.
    # Try to load from cached mnist if present for a few examples.
    data_dir = ROOT / "data" / "mnist"
    x_path = data_dir / "x_test.npy"
    y_path = data_dir / "y_test.npy"
    if x_path.exists() and y_path.exists():
        try:
            x = np.load(x_path)
            y = np.load(y_path)
            # pick one per some digits
            wanted = [1, 7, 0, 3, 8]
            for d in wanted:
                idxs = np.where(y == d)[0]
                if len(idxs):
                    img = Image.fromarray(x[idxs[0]].astype(np.uint8), mode="L")
                    examples.append((f"Example: {d}", img))
        except Exception:
            pass
    return examples


def render_input_section():
    """Render drawing/upload UI and return (input_pil, source, canvas_result, uploaded)."""
    st.markdown("### Stage 0 — Input drawing")
    st.caption("Draw a handwritten digit (0–9) or upload an image. The processed 28×28 tensor shown is exactly what the CNN receives.")
    col1, col2 = st.columns([1.15, 1], gap="large")
    canvas_result = None
    uploaded = None
    example_choice = None

    with col1:
        st.markdown("**Interactive drawing canvas**")
        try:
            from streamlit_drawable_canvas import st_canvas
            if "canvas_nonce" not in st.session_state:
                st.session_state["canvas_nonce"] = 0
            canvas_result = st_canvas(
                fill_color="rgba(0, 0, 0, 0)",
                stroke_width=16,
                stroke_color="#FFFFFF",
                background_color="#000000",
                height=300,
                width=300,
                drawing_mode="freedraw",
                key=f"digit_canvas_{st.session_state['canvas_nonce']}",
                update_streamlit=True,
                return_image_data=True,
            )
            b1, b2 = st.columns(2)
            with b1:
                if st.button("Clear canvas", use_container_width=True):
                    st.session_state["canvas_nonce"] += 1
                    st.session_state.pop("viz_input", None)
                    st.rerun()
            with b2:
                if st.button("Reset input", use_container_width=True):
                    st.session_state["canvas_nonce"] += 1
                    st.session_state.pop("viz_input", None)
                    st.session_state.pop("viz_result", None)
                    st.rerun()
            st.caption("Mouse/touch supported. Use white strokes on dark background.")
        except Exception as e:
            st.warning(f"Drawing canvas unavailable. Use image upload. ({e})")

    with col2:
        st.markdown("**Upload image**")
        uploaded = st.file_uploader(
            "Upload a digit image (PNG/JPG/JPEG/BMP)",
            type=["png", "jpg", "jpeg", "bmp"],
            accept_multiple_files=False,
        )
        if uploaded is not None:
            try:
                preview = bytes_to_pil(uploaded.getvalue())
                if preview is not None:
                    st.image(preview, caption="Uploaded image", width=220)
            except Exception:
                st.error("Could not read uploaded image.")
        examples = get_example_images()
        if examples:
            labels = ["(none)"] + [e[0] for e in examples]
            sel = st.selectbox("Example digit (from cached MNIST test data)", labels, index=0)
            if sel != "(none)":
                for name, img in examples:
                    if name == sel:
                        example_choice = img
                        st.image(img.resize((140, 140), Image.Resampling.NEAREST), caption=name, width=140)
                        break

    # Determine source priority: upload > example selection > canvas
    input_pil = None
    source = ""
    if uploaded is not None:
        p = bytes_to_pil(uploaded.getvalue())
        if p is not None:
            input_pil = p.convert("RGB")
            source = f"Upload: {uploaded.name}"
    if input_pil is None and example_choice is not None:
        input_pil = example_choice.convert("RGB")
        source = "Example digit"
    if input_pil is None:
        p = canvas_to_pil(canvas_result)
        if p is not None:
            input_pil = p.convert("RGB")
            source = "Drawing canvas"
    return input_pil, source
