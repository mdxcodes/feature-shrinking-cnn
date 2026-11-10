from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st
import tensorflow as tf

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"


def resolve_model_path() -> Path | None:
    best = MODELS_DIR / "best_mnist_cnn.keras"
    final = MODELS_DIR / "final_mnist_cnn.keras"
    if best.exists():
        return best
    if final.exists():
        return final
    return None


@st.cache_resource(show_spinner=False)
def load_trained_model(model_path: str) -> tf.keras.Model:
    return tf.keras.models.load_model(model_path, compile=False)


@st.cache_resource(show_spinner=False)
def get_intermediate_model(model_path: str) -> tf.keras.Model:
    """Functional intermediate model reusing trained layers/weights."""
    model = load_trained_model(model_path)
    # Keras 3 / recent TF: prefer model.inputs; do not rely on model.input on un-called Sequential wrapper.
    inputs = model.inputs
    outputs = {
        "conv_1": model.get_layer("conv_1").output,
        "conv_2": model.get_layer("conv_2").output,
        "flatten": model.get_layer("flatten").output,
        "predictions": model.get_layer("predictions").output,
    }
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="intermediate_extractor")


def tensor_fingerprint(x: np.ndarray) -> str:
    arr = np.asarray(x, dtype=np.float32)
    return hashlib.sha256(arr.tobytes()).hexdigest()


@st.cache_data(show_spinner=False)
def run_full_inference(model_path: str, input_fingerprint: str, x: np.ndarray) -> dict[str, np.ndarray]:
    """Run inference once and return actual intermediate tensors (cached by input)."""
    _ = input_fingerprint
    extractor = get_intermediate_model(model_path)
    out = extractor(np.asarray(x, dtype=np.float32), training=False)
    result: dict[str, np.ndarray] = {}
    if isinstance(out, dict):
        for k, v in out.items():
            result[k] = v.numpy() if hasattr(v, "numpy") else np.asarray(v)
    else:
        keys = ["conv_1", "conv_2", "flatten", "predictions"]
        for k, v in zip(keys, out):
            result[k] = v.numpy() if hasattr(v, "numpy") else np.asarray(v)
    for k in list(result.keys()):
        result[k] = np.asarray(result[k])
    result["input"] = np.asarray(x, dtype=np.float32)
    return result


def get_layer_params(model: tf.keras.Model, layer_name: str) -> dict[str, np.ndarray]:
    layer = model.get_layer(layer_name)
    params: dict[str, np.ndarray] = {}
    weights = layer.get_weights()
    if layer_name.startswith("conv") or layer_name == "predictions":
        if len(weights) >= 1:
            params["kernel"] = np.asarray(weights[0])
        if len(weights) >= 2:
            params["bias"] = np.asarray(weights[1])
    return params


def validate_architecture(model: tf.keras.Model) -> dict[str, Any]:
    info = {
        "input_shape": tuple(model.input_shape),
        "output_shape": tuple(model.output_shape),
        "params": int(model.count_params()),
        "layers": [l.name for l in model.layers],
    }
    assert info["input_shape"] == (None, 28, 28, 1), info["input_shape"]
    assert info["output_shape"] == (None, 10), info["output_shape"]
    assert info["params"] == 50890, info["params"]
    c1 = model.get_layer("conv_1")
    c2 = model.get_layer("conv_2")
    assert c1.filters == 16 and c1.kernel_size == (3, 3) and c1.strides == (2, 2) and c1.padding == "same"
    assert c2.filters == 32 and c2.kernel_size == (3, 3) and c2.strides == (1, 1) and c2.padding == "valid"
    assert int(model.get_layer("flatten").output.shape[1]) == 4608
    assert model.get_layer("predictions").units == 10
    return info
