from pathlib import Path
import json
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf
from tensorflow.keras import Model

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
MODEL_PATHS = [ROOT / "models/best_mnist_cnn.keras", ROOT / "models/final_mnist_cnn.keras"]

st.set_page_config(page_title="Feature-Shrinking CNN", page_icon="🧠", layout="wide")

st.markdown("""
<style>
.block-container{max-width:1400px;padding-top:1.4rem}
.hero{padding:1.3rem 1.5rem;border:1px solid rgba(128,128,128,.25);border-radius:16px;
background:linear-gradient(110deg,rgba(59,130,246,.13),rgba(139,92,246,.12))}
div[data-testid="stMetric"]{border:1px solid rgba(128,128,128,.2);padding:.8rem;border-radius:12px}
</style>
<div class="hero"><h1>🧠 Feature-Shrinking CNN</h1>
<p>Interactive MNIST digit recognition · TensorFlow/Keras · No pooling layers</p></div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model(path):
    return tf.keras.models.load_model(path, compile=False)

def image_to_tensor(image):
    gray = ImageOps.grayscale(image)
    gray = ImageOps.autocontrast(gray)
    a = np.asarray(gray, dtype=np.uint8)
    border = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
    if border.mean() > 127:
        a = 255 - a
    ys, xs = np.where(a > 24)
    canvas = Image.new("L", (28, 28), 0)
    if len(xs):
        crop = Image.fromarray(a[ys.min():ys.max()+1, xs.min():xs.max()+1])
        crop.thumbnail((20, 20), Image.Resampling.LANCZOS)
        canvas.paste(crop, ((28-crop.width)//2, (28-crop.height)//2))
    arr = np.asarray(canvas, dtype=np.float32) / 255.0
    return arr.reshape(1, 28, 28, 1), canvas

def feature_outputs(model, tensor):
    extractor = Model(model.inputs, [model.get_layer("conv_1").output,
                                    model.get_layer("conv_2").output])
    a, b = extractor.predict(tensor, verbose=0)
    return a[0], b[0]

def feature_grid(maps, title):
    n = maps.shape[-1]
    cols = 8
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(14, rows * 1.7))
    axes = np.asarray(axes).reshape(-1)
    for i, ax in enumerate(axes):
        ax.axis("off")
        if i < n:
            ax.imshow(maps[:, :, i], cmap="viridis")
            ax.set_title(f"Filter {i+1}", fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

def read_json(path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None

model_file = next((p for p in MODEL_PATHS if p.exists()), None)
if model_file is None:
    st.error("Model not found. Expected models/best_mnist_cnn.keras or models/final_mnist_cnn.keras.")
    st.stop()
try:
    model = load_model(str(model_file))
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()

with st.sidebar:
    st.success("Trained model loaded")
    st.caption(f"Model: {model_file.relative_to(ROOT)}")
    st.metric("Trainable parameters", f"{model.count_params():,}")
    st.metric("Input", "28 × 28 × 1")
    st.metric("Classes", "10")
    st.code("28×28×1\n↓ Conv2D 16, stride 2\n14×14×16\n↓ Conv2D 32, valid\n12×12×32\n↓ Flatten: 4608\n↓ Dense + Softmax\n10 classes")

predict_tab, maps_tab, dashboard_tab, architecture_tab = st.tabs(
    ["✍️ Draw / upload & predict", "🔬 Feature maps", "📊 Training dashboard", "🏗️ Architecture"])

with predict_tab:
    draw_col, upload_col = st.columns(2)
    with draw_col:
        st.subheader("1. Draw a digit")
        canvas = st_canvas(fill_color="rgba(255,255,255,0)", stroke_width=16,
            stroke_color="#000000", background_color="#FFFFFF", height=280, width=280,
            drawing_mode="freedraw", key="mnist_canvas", update_streamlit=True)
        st.caption("Draw one digit using black ink.")
    with upload_col:
        st.subheader("2. Or upload an image")
        uploaded = st.file_uploader("Upload a digit image", type=["png","jpg","jpeg","bmp"])
        if uploaded:
            st.image(uploaded, width=240)
    chosen = Image.open(uploaded).convert("RGB") if uploaded else None
    if chosen is None and canvas.image_data is not None:
        rgba = canvas.image_data.astype(np.uint8)
        if np.any(rgba[:, :, 3] > 0):
            chosen = Image.fromarray(rgba, "RGBA").convert("RGB")
    if chosen:
        tensor, processed = image_to_tensor(chosen)
        probs = model.predict(tensor, verbose=0)[0]
        pred = int(np.argmax(probs))
        c1, c2 = st.columns([1,2])
        with c1:
            st.subheader("Model input (28 × 28)")
            st.image(processed.resize((196,196), Image.Resampling.NEAREST), width=196)
        with c2:
            st.subheader("Prediction")
            st.markdown(f"<h1 style='font-size:4.5rem'>{pred}</h1>", unsafe_allow_html=True)
            st.metric("Confidence", f"{probs[pred]*100:.2f}%")
            st.progress(float(probs[pred]))
        st.subheader("Confidence for every digit")
        cols = st.columns(5)
        for d, p in enumerate(probs):
            cols[d % 5].metric(f"Digit {d}", f"{p*100:.2f}%")
    else:
        st.info("Draw a digit or upload an image to run inference.")

with maps_tab:
    st.subheader("Convolution feature-map explorer")
    st.caption("Draw or upload an image in the first tab. Feature maps are activations from the trained convolution filters.")
    source = Image.open(uploaded).convert("RGB") if uploaded else None
    if source is None and canvas.image_data is not None:
        rgba = canvas.image_data.astype(np.uint8)
        if np.any(rgba[:,:,3] > 0):
            source = Image.fromarray(rgba, "RGBA").convert("RGB")
    if source:
        tensor, processed = image_to_tensor(source)
        f1, f2 = feature_outputs(model, tensor)
        st.image(processed.resize((140,140), Image.Resampling.NEAREST), caption="Input: 28 × 28 × 1", width=140)
        st.markdown("**First convolution: 14 × 14 × 16**")
        feature_grid(f1, "Conv2D 1 · 16 feature maps")
        st.markdown("**Second convolution: 12 × 12 × 32**")
        feature_grid(f2, "Conv2D 2 · 32 feature maps")
    else:
        st.info("Draw or upload a digit in the first tab to inspect its feature maps.")

with dashboard_tab:
    st.subheader("Training and evaluation dashboard")
    metrics = read_json(RESULTS / "evaluation_metrics.json")
    history = read_json(RESULTS / "training_history.json")
    if metrics:
        acc = metrics.get("test_accuracy", metrics.get("accuracy"))
        loss = metrics.get("test_loss", metrics.get("loss"))
        f1 = metrics.get("macro_f1", metrics.get("f1_macro"))
        cols = st.columns(3)
        if acc is not None: cols[0].metric("Test accuracy", f"{float(acc)*100:.2f}%")
        if loss is not None: cols[1].metric("Test loss", f"{float(loss):.4f}")
        if f1 is not None: cols[2].metric("Macro F1", f"{float(f1)*100:.2f}%")
    else:
        st.warning("Evaluation metrics JSON not found. Run python evaluate.py.")
    if history:
        hist = history.get("history", history)
        if "accuracy" in hist and "val_accuracy" in hist:
            fig, ax = plt.subplots(figsize=(9,3.5))
            ax.plot(hist["accuracy"], label="Training accuracy")
            ax.plot(hist["val_accuracy"], label="Validation accuracy")
            ax.set(title="Training vs validation accuracy", xlabel="Epoch", ylabel="Accuracy")
            ax.grid(alpha=.25); ax.legend(); st.pyplot(fig); plt.close(fig)
        if "loss" in hist and "val_loss" in hist:
            fig, ax = plt.subplots(figsize=(9,3.5))
            ax.plot(hist["loss"], label="Training loss")
            ax.plot(hist["val_loss"], label="Validation loss")
            ax.set(title="Training vs validation loss", xlabel="Epoch", ylabel="Loss")
            ax.grid(alpha=.25); ax.legend(); st.pyplot(fig); plt.close(fig)
    for filename, caption in [("training_accuracy.png","Training accuracy"),
                              ("training_loss.png","Training loss"),
                              ("confusion_matrix.png","Confusion matrix"),
                              ("sample_predictions.png","Sample predictions")]:
        path = RESULTS / filename
        if path.exists():
            st.image(str(path), caption=caption, use_container_width=True)

with architecture_tab:
    st.subheader("CNN architecture")
    stages = [
        ("Input", "28 × 28 × 1", "Grayscale image"),
        ("Conv2D 1", "14 × 14 × 16", "16 filters · 3×3 · stride 2 · same · ReLU"),
        ("Conv2D 2", "12 × 12 × 32", "32 filters · 3×3 · stride 1 · valid · ReLU"),
        ("Flatten", "4,608", "12 × 12 × 32 feature values"),
        ("Dense + Softmax", "10", "Probability for each digit 0–9"),
    ]
    for name, shape, detail in stages:
        a, b = st.columns([1,2])
        a.markdown(f"**{name}**")
        a.markdown(f"### `{shape}`")
        b.write(detail)
        st.divider()
    a,b,c = st.columns(3)
    a.metric("Input spatial positions", "784")
    b.metric("After Conv1", "196", "75% fewer")
    c.metric("After Conv2", "144", "81.6% fewer than input")
    st.info("Spatial dimensions shrink through convolution strides and valid padding only. No pooling layer is used.")

st.caption("Feature-Shrinking CNN · MNIST educational demonstration · TensorFlow/Keras")
