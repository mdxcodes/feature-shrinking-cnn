from __future__ import annotations

import math
from typing import Any

import numpy as np

from visualization.convolution_math import compute_conv_step, get_receptive_field, iter_output_positions, same_padding_2d
from visualization.inference import get_layer_params, load_trained_model


def _f32(x):
    return np.asarray(x, dtype=np.float32)


def prepare_pixel_grid(input_tensor: np.ndarray) -> dict[str, Any]:
    x = _f32(input_tensor)
    if x.ndim == 4:
        x = x[0]
    img = x[:, :, 0]
    return {
        "shape": [28, 28, 1],
        "values": img.tolist(),
        "min": float(img.min()),
        "max": float(img.max()),
    }


def prepare_conv1_animation_data(
    model_path: str,
    input_tensor: np.ndarray,
    conv1_out: np.ndarray,
    filter_idx: int = 0,
    max_steps: int | None = None,
) -> dict[str, Any]:
    model = load_trained_model(model_path)
    params = get_layer_params(model, "conv_1")
    kernel = params["kernel"]  # (3,3,1,16)
    bias = params["bias"]
    x = _f32(input_tensor)
    if x.ndim == 4:
        x = x[0]
    out = _f32(conv1_out)
    if out.ndim == 4:
        out = out[0]
    oh_n, ow_n, f_n = out.shape
    sh, sw = (2, 2)
    kh, kw = (3, 3)
    pt, pb, pl, pr = same_padding_2d(28, 28, kh, kw, sh, sw)
    steps = []
    positions = list(iter_output_positions(oh_n, ow_n))
    if max_steps is not None:
        positions = positions[:max_steps]
    for (oh, ow) in positions:
        step = compute_conv_step(x, kernel, bias, oh, ow, filter_idx, (sh, sw), "same", actual_map=out)
        rf = get_receptive_field(oh, ow, sh, sw, kh, kw, "same", 28, 28)
        steps.append({
            "oh": oh, "ow": ow,
            "in_h_start": int(rf["in_h_start"]),
            "in_w_start": int(rf["in_w_start"]),
            "patch": step.patch[:, :, 0].tolist(),  # (3,3)
            "products": step.products[:, :, 0].tolist(),
            "kernel": _f32(kernel[:, :, 0, filter_idx]).tolist(),
            "sum": step.weighted_sum,
            "bias": step.bias,
            "pre": step.pre_activation,
            "relu": step.relu,
            "actual": step.actual_activation,
        })
    return {
        "layer": "conv_1",
        "input_shape": [28, 28, 1],
        "output_shape": [14, 14, 16],
        "kernel_shape": [3, 3],
        "stride": [2, 2],
        "padding": "SAME",
        "filters": 16,
        "filter_idx": filter_idx,
        "pad": [pt, pb, pl, pr],
        "kernel_weights": _f32(kernel[:, :, 0, filter_idx]).tolist(),
        "bias_value": float(bias[filter_idx]),
        "steps": steps,
        "output_map": _f32(out[:, :, filter_idx]).tolist(),
        "all_output_maps": _f32(out).transpose(2, 0, 1).tolist(),  # (16,14,14)
    }


def prepare_conv2_animation_data(
    model_path: str,
    conv1_out: np.ndarray,
    conv2_out: np.ndarray,
    filter_idx: int = 0,
    step_stride_sample: int = 1,
) -> dict[str, Any]:
    """Prepare conv2 animation data. Conv2 operates on 14x14x16 stack -> 12x12x32."""
    model = load_trained_model(model_path)
    params = get_layer_params(model, "conv_2")
    kernel = params["kernel"]  # (3,3,16,32)
    bias = params["bias"]
    x = _f32(conv1_out)
    if x.ndim == 4:
        x = x[0]
    out = _f32(conv2_out)
    if out.ndim == 4:
        out = out[0]
    oh_n, ow_n, f_n = out.shape
    sh, sw = (1, 1)
    kh, kw = (3, 3)
    steps = []
    # sample positions for animation if desired, but include grid info; default all 144
    positions = list(iter_output_positions(oh_n, ow_n))
    sampled = positions[::step_stride_sample]
    for (oh, ow) in sampled:
        step = compute_conv_step(x, kernel, bias, oh, ow, filter_idx, (sh, sw), "valid", actual_map=out)
        # per-channel contribution sums for readability
        ch_sums = step.products.reshape(kh * kw, 16).sum(axis=0).tolist()  # contribution per input channel
        steps.append({
            "oh": oh, "ow": ow,
            "in_h_start": step.in_h_start,
            "in_w_start": step.in_w_start,
            "channel_sums": ch_sums,
            "sum": step.weighted_sum,
            "bias": step.bias,
            "pre": step.pre_activation,
            "relu": step.relu,
            "actual": step.actual_activation,
        })
    # Provide full kernel for selected filter (3,3,16) and a compact view
    kf = _f32(kernel[:, :, :, filter_idx])  # (3,3,16)
    return {
        "layer": "conv_2",
        "input_shape": [14, 14, 16],
        "output_shape": [12, 12, 32],
        "kernel_shape": [3, 3, 16],
        "stride": [1, 1],
        "padding": "VALID",
        "filters": 32,
        "filter_idx": filter_idx,
        "kernel_weights": kf.tolist(),
        "bias_value": float(bias[filter_idx]),
        "steps": steps,
        "all_steps_count": len(positions),
        "output_map": _f32(out[:, :, filter_idx]).tolist(),
        "all_output_maps": _f32(out).transpose(2, 0, 1).tolist(),  # (32,12,12)
        "input_maps": _f32(x).transpose(2, 0, 1).tolist(),  # (16,14,14) for stack viz
    }


def prepare_flatten_data(conv2_out: np.ndarray, flatten_out: np.ndarray) -> dict[str, Any]:
    c2 = _f32(conv2_out)
    if c2.ndim == 4:
        c2 = c2[0]
    flat = _f32(flatten_out).reshape(-1)
    # Keras Flatten default is channels-last: iterates H,W,C in that order
    # c2 is (12,12,32)
    return {
        "input_shape": [12, 12, 32],
        "output_length": int(flat.shape[0]),
        "formula": "12 × 12 × 32 = 4,608",
        "vector": flat.tolist(),
        "vector_min": float(flat.min()),
        "vector_max": float(flat.max()),
        "vector_mean": float(flat.mean()),
        "nonzero": int(np.count_nonzero(flat > 0)),
        # map each of first N elements back to (h,w,c) for explanation
        "sample_index_map": [{"i": i, "h": i // (12 * 32), "w": (i // 32) % 12, "c": i % 32} for i in range(0, 4608, 256)],
    }


def prepare_dense_data(
    model_path: str,
    flatten_out: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, Any]:
    model = load_trained_model(model_path)
    params = get_layer_params(model, "predictions")
    w = _f32(params["kernel"])  # (4608,10)
    b = _f32(params["bias"])    # (10,)
    x = _f32(flatten_out).reshape(-1)
    probs = _f32(probabilities).reshape(-1)
    logits = (x.astype(np.float64) @ w.astype(np.float64) + b.astype(np.float64)).astype(np.float32)
    # For visualization, compute contribution summaries per class (top contributing input indices)
    contrib = (x[:, None] * w).astype(np.float32)  # (4608,10)
    top_k = 24
    top_contrib = {}
    for cls in range(10):
        vals = contrib[:, cls]
        idx = np.argsort(np.abs(vals))[::-1][:top_k]
        top_contrib[str(cls)] = {
            "indices": idx.tolist(),
            "values": vals[idx].tolist(),
        }
    return {
        "input_length": 4608,
        "output_classes": 10,
        "weights_shape": [4608, 10],
        "bias": b.tolist(),
        "logits": logits.tolist(),
        "probabilities": probs.tolist(),
        "predicted": int(np.argmax(probs)),
        "confidence": float(np.max(probs)),
        "top_contributions": top_contrib,
        "connection_count": 4608 * 10,
    }


def build_animation_payload(
    model_path: str,
    input_tensor: np.ndarray,
    tensors: dict[str, np.ndarray],
    conv1_filter: int = 0,
    conv2_filter: int = 0,
) -> dict[str, Any]:
    """Build full precomputed payload for browser animation (no inference in JS)."""
    probs = _f32(tensors["predictions"]).reshape(-1)
    return {
        "pixel_grid": prepare_pixel_grid(input_tensor),
        "conv1": prepare_conv1_animation_data(model_path, input_tensor, tensors["conv_1"], conv1_filter),
        "conv2": prepare_conv2_animation_data(model_path, tensors["conv_1"], tensors["conv_2"], conv2_filter),
        "flatten": prepare_flatten_data(tensors["conv_2"], tensors["flatten"]),
        "dense": prepare_dense_data(model_path, tensors["flatten"], probs),
        "input_image": _f32(input_tensor).reshape(28, 28).tolist(),
        "summary": {
            "pipeline": ["28×28×1", "14×14×16", "12×12×32", "4608", "10"],
            "predicted": int(np.argmax(probs)),
            "probabilities": probs.tolist(),
        },
    }
