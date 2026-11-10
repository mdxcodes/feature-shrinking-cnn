"""Inside the CNN
Interactive visualization for the Feature-Shrinking CNN (MNIST).
All ML computations, tensor extraction, and convolution math run in Python.
Browser JS/Canvas handles animation rendering only.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

from components.architecture import render_architecture_explorer
from components.dashboard import render_dashboard
from components.drawing import render_input_section
from components.feature_maps import render_feature_map_explorer
from components.prediction import render_prediction_explorer
from components.styles import inject_premium_css
from frontend.components import read_text_artifact
from visualization.animation_data import prepare_conv1_animation_data, prepare_conv2_animation_data
from visualization.convolution_math import compute_conv_step
from visualization.inference import get_intermediate_model, get_layer_params, load_trained_model, resolve_model_path, tensor_fingerprint, validate_architecture
from visualization.preprocessing import preprocess_digit_image
from visualization.rendering import render_cnn_animation
from visualization.tensor_extraction import extract_all, get_dense_logits

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"

st.set_page_config(
    page_title="Inside the CNN",
    page_icon="◼",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_premium_css()


def _hash_input(x: np.ndarray) -> str:
    return tensor_fingerprint(x)


@st.cache_data(show_spinner=False)
def _cached_inference(model_path: str, fp: str, x: np.ndarray):
    _ = fp
    return extract_all(model_path, x)


def _get_or_run_inference(model_path: str, x: np.ndarray):
    fp = _hash_input(x)
    return _cached_inference(model_path, fp, x)


def render_sidebar(model, model_path: Path | None):
    with st.sidebar:
        st.markdown("### Inside the CNN")
        st.caption("Feature-Shrinking CNN · MNIST · real trained model")
        st.markdown("---")
        st.markdown("**Model**")
        if model_path:
            st.success("Loaded")
            st.code(model_path.name)
        else:
            st.error("Not found")
        if model is not None:
            st.metric("Parameters", f"{model.count_params():,}")
            st.metric("Input", "28 × 28 × 1")
            st.metric("Output", "10 classes")
        st.markdown("**Pipeline**")
        st.markdown("`28×28×1 → 14×14×16 → 12×12×32 → 4608 → 10`")
        st.caption("No pooling · Stride/valid padding only")
        st.markdown("---")
        st.markdown("**Sections**")
        st.caption("1. Input & main forward-pass animation")
        st.caption("2. Feature map explorer")
        st.caption("3. Architecture explorer")
        st.caption("4. Prediction explorer")
        st.caption("5. Training & evaluation dashboard")
        st.markdown("---")
        st.caption("Inference runs once per input and is cached. Animations use precomputed actual tensors.")


def render_header():
    st.markdown(
        """
        <div class="lab-header">
          <div class="eyebrow">Real TensorFlow/Keras Inference</div>
          <h1>Inside the CNN</h1>
          <p>Watch a handwritten digit travel through the actual Feature-Shrinking CNN: input tensor → Conv1 → Conv2 → Flatten → Dense/Softmax prediction.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_conv1_inspector(model_path: str, model_input: np.ndarray, tensors: dict, filter_idx: int):
    st.markdown("#### Conv1 numerical inspection (actual weights, input, and activation)")
    st.caption("Select an output position to inspect element-wise multiplication, sum, bias, and ReLU for the chosen filter. Values are compared against the actual intermediate tensor.")
    c1, c2, c3 = st.columns(3)
    oh = c1.slider("Output row (0-13)", 0, 13, 0, key="c1_insp_oh")
    ow = c2.slider("Output col (0-13)", 0, 13, 0, key="c1_insp_ow")
    fidx = c3.slider("Filter index", 0, 15, filter_idx, key="c1_insp_f")
    model = load_trained_model(model_path)
    params = get_layer_params(model, "conv_1")
    step = compute_conv_step(model_input, params["kernel"], params["bias"], oh, ow, fidx, (2, 2), "same", actual_map=tensors["conv_1"])
    k = params["kernel"][:, :, 0, fidx]
    a, b = st.columns(2)
    with a:
        st.markdown("**3×3 kernel weights (actual)**")
        st.dataframe(np.round(k, 6))
        st.markdown("**3×3 input patch (with SAME padding as needed)**")
        st.dataframe(np.round(step.patch[:, :, 0], 6))
    with b:
        st.markdown("**Element-wise products**")
        st.dataframe(np.round(step.products[:, :, 0], 6))
        st.markdown("**Calculation**")
        st.code(
            f"sum(products) = {step.weighted_sum:.8f}\n"
            f"+ bias        = {step.bias:.8f}\n"
            f"pre-activation= {step.pre_activation:.8f}\n"
            f"ReLU          = {step.relu:.8f}\n"
            f"Actual tensor = {step.actual_activation:.8f}\n"
            f"abs diff      = {abs(step.relu - step.actual_activation):.2e}"
        )
    st.caption(f"Receptive field top-left in padded/input coordinates context: in_h_start={step.in_h_start}, in_w_start={step.in_w_start}, pad_top={step.pad_top}, pad_left={step.pad_left}")


def render_conv2_inspector(model_path: str, tensors: dict, filter_idx: int):
    st.markdown("#### Conv2 numerical inspection (3×3×16 actual kernel over all input channels)")
    st.caption("Conv2 uses VALID padding and stride 1 over the 14×14×16 Conv1 stack. The calculation sums across all 16 channels.")
    c1, c2, c3, c4 = st.columns(4)
    oh = c1.slider("Output row (0-11)", 0, 11, 0, key="c2_insp_oh")
    ow = c2.slider("Output col (0-11)", 0, 11, 0, key="c2_insp_ow")
    fidx = c3.slider("Filter index", 0, 31, filter_idx, key="c2_insp_f")
    ch = c4.slider("Inspect input channel", 0, 15, 0, key="c2_insp_ch")
    model = load_trained_model(model_path)
    params = get_layer_params(model, "conv_2")
    step = compute_conv_step(tensors["conv_1"], params["kernel"], params["bias"], oh, ow, fidx, (1, 1), "valid", actual_map=tensors["conv_2"])
    kf = params["kernel"][:, :, :, fidx]  # (3,3,16)
    a, b = st.columns(2)
    with a:
        st.markdown(f"**3×3 kernel weights — channel {ch} of 16**")
        st.dataframe(np.round(kf[:, :, ch], 6))
        st.markdown(f"**3×3 input patch — channel {ch} (from Conv1 14×14×16)**")
        st.dataframe(np.round(step.patch[:, :, ch], 6))
        st.markdown(f"**Products — channel {ch}**")
        st.dataframe(np.round(step.products[:, :, ch], 6))
    with b:
        ch_sums = step.products.reshape(9, 16).sum(axis=0)
        st.markdown("**Per-channel contribution sums (all 16 channels)**")
        st.dataframe({"channel": list(range(16)), "sum_3x3": np.round(ch_sums, 6)})
        st.markdown("**Full calculation (all channels)**")
        st.code(
            f"Σ spatial×channels = {step.weighted_sum:.8f}\n"
            f"+ bias             = {step.bias:.8f}\n"
            f"pre-activation     = {step.pre_activation:.8f}\n"
            f"ReLU               = {step.relu:.8f}\n"
            f"Actual tensor      = {step.actual_activation:.8f}\n"
            f"abs diff           = {abs(step.relu - step.actual_activation):.2e}"
        )
    st.caption("The complete activation uses all 16 input channels; per-channel views are provided for readability.")


def main():
    render_header()
    model_path = resolve_model_path()
    model = None
    if model_path is not None:
        try:
            model = load_trained_model(str(model_path))
            validate_architecture(model)
        except Exception as e:
            st.error(f"Failed to load/validate trained model: {e}")
            render_sidebar(None, model_path)
            return
    render_sidebar(model, model_path)

    if model_path is None or model is None:
        st.error("Trained model not found. Expected `models/best_mnist_cnn.keras` (preferred) or `models/final_mnist_cnn.keras`.")
        return

    # --- Input section ---
    input_pil, source = render_input_section()
    if input_pil is None:
        st.info("Draw a digit, select an example, or upload an image to start the CNN visualization.")
        st.markdown("---")
        st.markdown("## Existing training and evaluation dashboard")
        render_dashboard(RESULTS_DIR)
        return

    model_input, processed_pil, meta = preprocess_digit_image(input_pil)
    if not meta.get("ok") or model_input is None or processed_pil is None:
        st.warning(meta.get("reason") or "Unable to preprocess image.")
        return

    # Show processed input (exact tensor fed to model)
    st.markdown("#### Processed input (sent to CNN)")
    p1, p2, p3 = st.columns([1, 1, 2])
    with p1:
        st.image(processed_pil.resize((196, 196), Image.Resampling.NEAREST), caption="28×28 grayscale (model input)", width=196)
    with p2:
        arr = np.asarray(processed_pil, dtype=np.float32) / 255.0
        st.markdown(f"**Source:** {source}")
        st.markdown(f"**Tensor shape:** `(1, 28, 28, 1)`")
        st.markdown(f"**Dtype/range:** float32 in [0, 1]")
        st.markdown(f"**Min/Max:** {arr.min():.4f} / {arr.max():.4f}")
        if meta.get("inverted"):
            st.caption("Polarity auto-inverted to match MNIST (bright digit on dark background).")
        if meta.get("bbox"):
            st.caption(f"Detected content bbox: {meta['bbox']}")
    with p3:
        st.markdown("**28×28 normalized pixel grid (actual values)**")
        st.dataframe(np.round(arr, 3), use_container_width=True, height=220)
        st.caption("Each cell is the normalized pixel intensity received by the CNN.")

    # Inference once per input (cached)
    with st.spinner("Running actual CNN inference and extracting intermediate tensors..."):
        tensors = _get_or_run_inference(str(model_path), model_input)
    probs = np.asarray(tensors["predictions"]).reshape(-1)
    pred = int(np.argmax(probs))
    conf = float(probs[pred])

    st.markdown("---")
    st.markdown("## Inside the CNN")
    st.caption("Connected forward-pass pipeline. All activations, weights, and probabilities are from the loaded trained model. Animation renders precomputed data (no repeated inference).")

    # Controls for filter selection (affect precomputed animation payload)
    fc1, fc2, fc3, fc4 = st.columns([1, 1, 1, 2])
    conv1_filter = fc1.selectbox("Conv1 filter (for animation/inspection)", list(range(16)), index=0, key="sel_c1")
    conv2_filter = fc2.selectbox("Conv2 filter (for animation/inspection)", list(range(32)), index=0, key="sel_c2")
    with fc3:
        st.metric("Predicted digit", str(pred), f"{conf*100:.2f}% conf")
    with fc4:
        st.markdown("**Replay / controls:** use Play/Pause/Replay/Next/Prev/Speed inside the visualization. Changing filter selection rebuilds precomputed animation data from cached tensors (no re-inference).")

    # Build payload from cached tensors (fast; no TF inference)
    try:
        from visualization.animation_data import build_animation_payload
        payload = build_animation_payload(str(model_path), model_input, tensors, conv1_filter=int(conv1_filter), conv2_filter=int(conv2_filter))
        render_cnn_animation(payload, height=1000, key=f"cnn_anim_{_hash_input(model_input)[:8]}_{conv1_filter}_{conv2_filter}")
    except Exception as e:
        st.error(f"Animation renderer failed: {e}")
        st.exception(e)

    st.markdown("### Numerical convolution inspection")
    it1, it2 = st.tabs(["Conv1 step-by-step (actual)", "Conv2 step-by-step (actual, all 16 channels)"])
    with it1:
        render_conv1_inspector(str(model_path), model_input, tensors, int(conv1_filter))
    with it2:
        render_conv2_inspector(str(model_path), tensors, int(conv2_filter))

    st.markdown("---")
    st.markdown("## Feature Map Explorer")
    render_feature_map_explorer(tensors["conv_1"], tensors["conv_2"])

    st.markdown("---")
    st.markdown("## Architecture Explorer")
    render_architecture_explorer(model)

    st.markdown("---")
    st.markdown("## Prediction Explorer")
    render_prediction_explorer(processed_pil, probs)

    st.markdown("---")
    st.markdown("## Existing Training and Evaluation Dashboard")
    st.caption("Uses existing artifacts/history only. No retraining, no fabricated metrics.")
    render_dashboard(RESULTS_DIR)

    st.markdown("---")
    st.caption("Inside the CNN · Feature-Shrinking CNN (28×28×1 → 14×14×16 → 12×12×32 → 4608 → 10) · TensorFlow/Keras · Streamlit")


if __name__ == "__main__":
    main()
