from __future__ import annotations

import numpy as np

from visualization.inference import resolve_model_path
from visualization.tensor_extraction import extract_all, get_conv1_outputs, get_conv2_outputs, get_flatten_output, get_predictions, get_dense_logits


def test_tensor_extraction_from_zero_input():
    path = resolve_model_path()
    assert path is not None
    x = np.zeros((1, 28, 28, 1), dtype=np.float32)
    tensors = extract_all(str(path), x)
    c1 = get_conv1_outputs(tensors)
    c2 = get_conv2_outputs(tensors)
    f = get_flatten_output(tensors)
    p = get_predictions(tensors)
    assert c1.shape == (1, 14, 14, 16)
    assert c2.shape == (1, 12, 12, 32)
    assert f.shape == (1, 4608)
    assert p.shape == (1, 10)
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-5)
    logits = get_dense_logits(str(path), f)
    assert logits.shape == (1, 10)
