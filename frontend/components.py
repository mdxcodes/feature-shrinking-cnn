"""Shared Streamlit components, model/data loading, and sidebar."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st
import tensorflow as tf

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
DATA_DIR = ROOT / "data" / "mnist"


def get_paths() -> dict[str, Path]:
    return {"root": ROOT, "models": MODELS_DIR, "results": RESULTS_DIR, "data": DATA_DIR}


def resolve_model_path() -> Path | None:
    best = MODELS_DIR / "best_mnist_cnn.keras"
    final = MODELS_DIR / "final_mnist_cnn.keras"
    if best.exists():
        return best
    if final.exists():
        return final
    return None


@st.cache_resource(show_spinner=False)
def load_inference_model(model_path: str) -> tf.keras.Model:
    return tf.keras.models.load_model(model_path, compile=False)


@st.cache_resource(show_spinner=False)
def get_feature_extractor(model_path: str) -> tf.keras.Model:
    model = load_inference_model(model_path)
    conv1 = model.get_layer("conv_1").output
    conv2 = model.get_layer("conv_2").output
    return tf.keras.Model(inputs=model.inputs, outputs=[conv1, conv2])


@st.cache_data(show_spinner=False)
def load_json_artifact(path_str: str) -> dict | None:
    path = Path(path_str)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def read_text_artifact(path_str: str) -> str | None:
    path = Path(path_str)
    if not path.exists():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return None


def render_sidebar(active_section: str) -> dict:
    model_path = resolve_model_path()
    st.sidebar.markdown("### Feature-Shrinking CNN")
    st.sidebar.caption("MNIST digit recognition · interactive lab dashboard")
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Navigation**")
    st.sidebar.markdown(
        "\n".join(
            [
                f"- {'**' if active_section == s else ''}{s}{'**' if active_section == s else ''}"
                for s in [
                    "Predict",
                    "Feature Maps",
                    "Performance Dashboard",
                    "Architecture Explorer",
                ]
            ]
        )
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Model status**")
    if model_path is None:
        st.sidebar.error("No trained model found")
    else:
        st.sidebar.success("Trained model available")
        st.sidebar.caption(f"Selected: `{model_path.name}`")
        try:
            model = load_inference_model(str(model_path))
            st.sidebar.metric("Parameters", f"{model.count_params():,}")
        except Exception as e:
            st.sidebar.error(f"Load error: {e}")

    st.sidebar.markdown("**Input / output**")
    st.sidebar.caption("Input: 28 × 28 × 1 (grayscale)")
    st.sidebar.caption("Output: 10 classes (digits 0–9)")
    st.sidebar.caption("Pooling layers: 0")
    st.sidebar.markdown("---")
    st.sidebar.caption("Inference runs locally from saved Keras checkpoints.")
    return {"model_path": model_path, "results_dir": RESULTS_DIR, "root": ROOT}


def close_fig(fig) -> None:
    try:
        plt.close(fig)
    except Exception:
        pass
