from __future__ import annotations

import numpy as np
import tensorflow as tf

from visualization.inference import get_intermediate_model, load_trained_model, run_full_inference, tensor_fingerprint


def extract_all(model_path: str, x: np.ndarray) -> dict[str, np.ndarray]:
    fp = tensor_fingerprint(x)
    tensors = run_full_inference(model_path, fp, x)
    # Ensure expected shapes exist (batch dimension kept as in model output)
    # conv outputs: (1,H,W,C)
    return tensors


def get_conv1_outputs(tensors: dict[str, np.ndarray]) -> np.ndarray:
    return np.asarray(tensors["conv_1"])  # (1,14,14,16)


def get_conv2_outputs(tensors: dict[str, np.ndarray]) -> np.ndarray:
    return np.asarray(tensors["conv_2"])  # (1,12,12,32)


def get_flatten_output(tensors: dict[str, np.ndarray]) -> np.ndarray:
    return np.asarray(tensors["flatten"])  # (1,4608)


def get_predictions(tensors: dict[str, np.ndarray]) -> np.ndarray:
    return np.asarray(tensors["predictions"])  # (1,10)


def get_dense_logits(model_path: str, flatten_vec: np.ndarray) -> np.ndarray:
    """Compute pre-softmax logits from actual dense weights/bias: z = xW + b."""
    model = load_trained_model(model_path)
    dense = model.get_layer("predictions")
    w, b = dense.get_weights()
    x = np.asarray(flatten_vec)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    logits = x.astype(np.float64) @ w.astype(np.float64) + b.astype(np.float64)
    return logits.astype(np.float32)
