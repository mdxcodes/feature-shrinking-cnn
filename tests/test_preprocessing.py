from __future__ import annotations

import numpy as np
from PIL import Image

from visualization.preprocessing import preprocess_digit_image


def test_preprocess_empty_like_array_returns_none():
    arr = np.zeros((28, 28, 3), dtype=np.uint8)
    pil = Image.fromarray(arr, mode="RGB")
    x, proc, meta = preprocess_digit_image(pil)
    assert x is None or meta.get("ok") is False


def test_preprocess_valid_white_digit_on_black():
    # create a simple digit-like stroke
    arr = np.zeros((60, 60), dtype=np.uint8)
    arr[20:40, 28:32] = 255
    arr[28:32, 20:40] = 255
    pil = Image.fromarray(arr, mode="L")
    x, proc, meta = preprocess_digit_image(pil)
    assert meta.get("ok") is True
    assert x.shape == (1, 28, 28, 1)
    assert x.min() >= 0.0 and x.max() <= 1.0


def test_preprocess_invalid_bytes_handled():
    from visualization.preprocessing import bytes_to_pil
    pil = bytes_to_pil(b"not-an-image")
    assert pil is None
    x, proc, meta = preprocess_digit_image(pil)
    assert x is None
