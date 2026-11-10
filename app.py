"""Feature-Shrinking CNN — interactive Streamlit demonstration app."""

from __future__ import annotations

from pathlib import Path

import streamlit as st
from PIL import Image

from frontend.architecture import render_architecture_explorer
from frontend.components import get_paths, load_inference_model, render_sidebar, resolve_model_path
from frontend.dashboard import render_dashboard
from frontend.feature_maps import render_feature_maps_section
from frontend.prediction import predict, render_confidence_chart, render_prediction
from frontend.preprocessing import bytes_to_pil, canvas_to_pil, preprocess_digit_image
from frontend.styles import inject_css

st.set_page_config(
    page_title="Feature-Shrinking CNN Lab",
    page_icon="◼",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_css()
paths = get_paths()

st.markdown(
    """
    <div class="hero">
      <h1>Feature-Shrinking CNN</h1>
      <p>Interactive MNIST digit recognition dashboard — draw or upload digits, inspect predictions, visualize convolution feature maps, and explore training/evaluation artifacts.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


def _get_input_image(uploaded_file, canvas_result):
    """Return (PIL image or None, source_label)."""
    if uploaded_file is not None:
        try:
            data = uploaded_file.getvalue()
            pil = bytes_to_pil(data)
            if pil is not None:
                return pil, f"Uploaded: {uploaded_file.name}"
        except Exception:
            pass
    pil = canvas_to_pil(canvas_result)
    if pil is not None:
        return pil, "Drawing canvas"
    return None, ""


def render_predict_tab(model_path: str):
    st.markdown("#### Interactive digit input")
    st.caption("Provide one handwritten digit (0–9). The image is preprocessed to 28×28 grayscale, normalized to [0,1], and fed as (1, 28, 28, 1).")

    col_draw, col_upload = st.columns([1.1, 1], gap="large")
    canvas_result = None
    uploaded = None

    with col_draw:
        st.markdown("**Draw a digit**")
        try:
            from streamlit_drawable_canvas import st_canvas

            if "canvas_nonce" not in st.session_state:
                st.session_state["canvas_nonce"] = 0

            canvas_result = st_canvas(
                fill_color="rgba(0, 0, 0, 0)",
                stroke_width=14,
                stroke_color="#FFFFFF",
                background_color="#000000",
                height=280,
                width=280,
                drawing_mode="freedraw",
                key=f"digit_canvas_{st.session_state['canvas_nonce']}",
                update_streamlit=True,
                return_image_data=True,
            )
            st.caption("Tip: draw a single centered digit with clear strokes.")
            if st.button("Clear canvas", key="clear_canvas_btn"):
                st.session_state["canvas_nonce"] += 1
                st.session_state.pop("last_inference", None)
                st.rerun()
        except Exception as e:
            st.warning(
                "Drawing canvas component is unavailable in this environment. "
                "Use image upload instead. "
                f"(Details: {e})"
            )

    with col_upload:
        st.markdown("**Upload a digit image**")
        uploaded = st.file_uploader(
            "PNG, JPG, JPEG, or BMP",
            type=["png", "jpg", "jpeg", "bmp"],
            accept_multiple_files=False,
            help="RGB images are converted to grayscale; backgrounds are auto-handled and the digit is cropped/centered.",
        )
        if uploaded is not None:
            try:
                preview = Image.open(uploaded).convert("RGB")
                st.image(preview, caption="Uploaded preview", width=220)
            except Exception:
                st.error("Could not read the uploaded file as an image.")

    input_img, source = _get_input_image(uploaded, canvas_result)
    if input_img is None:
        st.info("Draw a digit on the canvas or upload an image to run prediction and feature-map extraction.")
        st.session_state.pop("last_inference", None)
        return

    model_input, processed, meta = preprocess_digit_image(input_img)
    if not meta.get("ok") or model_input is None or processed is None:
        st.warning(meta.get("reason") or "Unable to preprocess the provided image.")
        st.session_state.pop("last_inference", None)
        return

    # Run inference for this input (single call per render/input state)
    try:
        probs = predict(model_path, model_input)[0]
    except Exception as e:
        st.error(f"Inference failed: {e}")
        st.session_state.pop("last_inference", None)
        return

    st.session_state["last_inference"] = {
        "source": source,
        "model_input": model_input,
        "processed": processed,
        "probs": probs,
        "meta": meta,
    }

    res_col, prob_col = st.columns([1, 1.4], gap="large")
    with res_col:
        st.markdown("#### Prediction")
        pred = render_prediction(probs)
        st.markdown("**Processed 28×28 model input**")
        st.image(processed.resize((196, 196), Image.Resampling.NEAREST), caption="Preprocessed (28×28 grayscale, normalized)", width=196)
        st.caption(f"Source: {source}")
        if meta.get("inverted"):
            st.caption("Note: image polarity was auto-inverted to match MNIST (bright digit on dark background).")
        st.caption(f"Input tensor shape: {tuple(model_input.shape)} · Output shape: (1, 10)")

    with prob_col:
        st.markdown("#### Confidence scores (all 10 classes)")
        render_confidence_chart(probs, pred)


def main() -> None:
    model_path = resolve_model_path()
    render_sidebar(active_section="Tabs")

    if model_path is None:
        st.error("No trained model found. Expected `models/best_mnist_cnn.keras` or `models/final_mnist_cnn.keras`.")
        return

    try:
        model = load_inference_model(str(model_path))
    except Exception as e:
        st.error(f"Failed to load model from `{model_path}`: {e}")
        st.caption("Ensure TensorFlow/Keras is installed in the active environment and the checkpoint is valid.")
        return

    tab_predict, tab_maps, tab_dashboard, tab_arch = st.tabs(
        [
            "1 · Predict",
            "2 · Feature Maps",
            "3 · Performance Dashboard",
            "4 · Architecture Explorer",
        ]
    )

    with tab_predict:
        render_predict_tab(str(model_path))

    with tab_maps:
        st.markdown("### Convolution feature-map explorer")
        st.caption("Visualizes all 16 `conv_1` maps (14×14) and all 32 `conv_2` maps (12×12) for the same input used in prediction.")
        last = st.session_state.get("last_inference")
        if not last or last.get("model_input") is None:
            st.info("No input processed yet. Go to **Predict**, draw or upload a digit, then return here to inspect its feature maps.")
        else:
            st.image(last["processed"].resize((140, 140), Image.Resampling.NEAREST), caption=f"Input used for maps — {last.get('source','')}", width=140)
            try:
                render_feature_maps_section(str(model_path), last["model_input"])
            except Exception as e:
                st.error(f"Failed to extract/render feature maps: {e}")

    with tab_dashboard:
        st.markdown("### Training and evaluation dashboard")
        st.caption("Metrics and charts are read from actual artifacts in `results/` (no fabricated values).")
        render_dashboard(paths["results"])

    with tab_arch:
        st.markdown("### CNN architecture explorer")
        render_architecture_explorer(model)


if __name__ == "__main__":
    main()
