from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import tensorflow as tf


@dataclass
class ConvStep:
    out_h: int
    out_w: int
    in_h_start: int
    in_w_start: int
    pad_top: int
    pad_left: int
    patch: np.ndarray  # (kh,kw,in_c) values used (padded zeros where outside)
    products: np.ndarray  # (kh,kw,in_c)
    weighted_sum: float
    bias: float
    pre_activation: float
    relu: float
    actual_activation: float


def same_padding_2d(in_h: int, in_w: int, kh: int, kw: int, sh: int, sw: int) -> tuple[int, int, int, int]:
    """Return (pad_top, pad_bottom, pad_left, pad_right) for SAME padding as TF."""
    out_h = math.ceil(in_h / sh)
    out_w = math.ceil(in_w / sw)
    pad_h = max((out_h - 1) * sh + kh - in_h, 0)
    pad_w = max((out_w - 1) * sw + kw - in_w, 0)
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    return pad_top, pad_bottom, pad_left, pad_right


def get_receptive_field(
    out_h: int,
    out_w: int,
    stride_h: int,
    stride_w: int,
    kernel_h: int,
    kernel_w: int,
    padding: str,
    in_h: int,
    in_w: int,
) -> dict:
    if padding == "same":
        pt, pb, pl, pr = same_padding_2d(in_h, in_w, kernel_h, kernel_w, stride_h, stride_w)
    elif padding == "valid":
        pt = pb = pl = pr = 0
    else:
        raise ValueError(f"Unsupported padding: {padding}")
    in_h_start = out_h * stride_h - pt
    in_w_start = out_w * stride_w - pl
    return {
        "in_h_start": in_h_start,
        "in_w_start": in_w_start,
        "kernel_h": kernel_h,
        "kernel_w": kernel_w,
        "pad_top": pt,
        "pad_bottom": pb,
        "pad_left": pl,
        "pad_right": pr,
        "stride_h": stride_h,
        "stride_w": stride_w,
    }


def compute_conv_step(
    input_tensor: np.ndarray,
    kernel: np.ndarray,
    bias: np.ndarray,
    out_h: int,
    out_w: int,
    filter_idx: int,
    stride: tuple[int, int],
    padding: str,
    actual_map: np.ndarray | None = None,
) -> ConvStep:
    """Compute one conv output value with full intermediate math.
    input_tensor: (H,W,C) or (1,H,W,C)
    kernel: (kh,kw,C,out_filters)
    bias: (out_filters,)
    actual_map: (1,OH,OW,F) or (OH,OW,F) optional for cross-check
    """
    x = np.asarray(input_tensor)
    if x.ndim == 4:
        x = x[0]
    k = np.asarray(kernel)
    b = np.asarray(bias).reshape(-1)
    kh, kw, in_c, out_f = k.shape
    sh, sw = int(stride[0]), int(stride[1])
    in_h, in_w, _ = x.shape
    rf = get_receptive_field(out_h, out_w, sh, sw, kh, kw, padding, in_h, in_w)
    h0, w0 = rf["in_h_start"], rf["in_w_start"]
    # build patch with zero padding
    patch = np.zeros((kh, kw, in_c), dtype=np.float64)
    for i in range(kh):
        ih = h0 + i
        if ih < 0 or ih >= in_h:
            continue
        for j in range(kw):
            iw = w0 + j
            if iw < 0 or iw >= in_w:
                continue
            patch[i, j, :] = x[ih, iw, :]
    kf = k[:, :, :, filter_idx].astype(np.float64)
    products = patch * kf
    weighted_sum = float(np.sum(products))
    bias_v = float(b[filter_idx]) if filter_idx < len(b) else 0.0
    pre = weighted_sum + bias_v
    relu_v = float(max(pre, 0.0))
    actual_v = relu_v
    if actual_map is not None:
        am = np.asarray(actual_map)
        if am.ndim == 4:
            am = am[0]
        actual_v = float(am[out_h, out_w, filter_idx])
    return ConvStep(
        out_h=out_h,
        out_w=out_w,
        in_h_start=h0,
        in_w_start=w0,
        pad_top=rf["pad_top"],
        pad_left=rf["pad_left"],
        patch=patch.astype(np.float32),
        products=products.astype(np.float32),
        weighted_sum=weighted_sum,
        bias=bias_v,
        pre_activation=pre,
        relu=relu_v,
        actual_activation=actual_v,
    )


def iter_output_positions(out_h: int, out_w: int):
    for oh in range(out_h):
        for ow in range(out_w):
            yield oh, ow


def manual_conv2d_layer(
    input_tensor: np.ndarray,
    kernel: np.ndarray,
    bias: np.ndarray,
    strides: tuple[int, int],
    padding: str,
) -> np.ndarray:
    """Compute full conv+ReLU for verification (small tensors)."""
    x = np.asarray(input_tensor)
    if x.ndim == 4:
        x = x[0]
    k = np.asarray(kernel)
    b = np.asarray(bias)
    kh, kw, in_c, out_f = k.shape
    sh, sw = strides
    in_h, in_w, _ = x.shape
    if padding == "same":
        out_h = math.ceil(in_h / sh)
        out_w = math.ceil(in_w / sw)
    elif padding == "valid":
        out_h = math.ceil((in_h - kh + 1) / sh)
        out_w = math.ceil((in_w - kw + 1) / sw)
    else:
        raise ValueError(padding)
    out = np.zeros((out_h, out_w, out_f), dtype=np.float64)
    for f in range(out_f):
        for oh in range(out_h):
            for ow in range(out_w):
                step = compute_conv_step(x, k, b, oh, ow, f, strides, padding)
                out[oh, ow, f] = step.relu
    return out.astype(np.float32)
