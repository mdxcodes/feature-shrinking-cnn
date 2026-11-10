from __future__ import annotations

import numpy as np
import streamlit as st

from visualization.animation_data import build_animation_payload
from visualization.inference import tensor_fingerprint
from visualization.tensor_extraction import extract_all


def ensure_inference_cached(model_path: str, model_input: np.ndarray) -> dict:
    """Run inference once per input and cache tensors/payload inputs in session_state."""
    fp = tensor_fingerprint(model_input)
    cache = st.session_state.get("viz_cache", {})
    if cache.get("fingerprint") == fp and cache.get("model_path") == model_path and "tensors" in cache:
        return cache
    tensors = extract_all(model_path, model_input)
    cache = {
        "fingerprint": fp,
        "model_path": model_path,
        "input": np.asarray(model_input, dtype=np.float32),
        "tensors": tensors,
    }
    st.session_state["viz_cache"] = cache
    return cache


def get_animation_payload(model_path: str, model_input: np.ndarray, conv1_filter: int, conv2_filter: int) -> dict:
    cache = ensure_inference_cached(model_path, model_input)
    # Build payload from cached tensors (no TF inference)
    payload = build_animation_payload(
        model_path,
        cache["input"],
        cache["tensors"],
        conv1_filter=conv1_filter,
        conv2_filter=conv2_filter,
    )
    return payload
