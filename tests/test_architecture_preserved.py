from __future__ import annotations

import numpy as np
import tensorflow as tf

from visualization.inference import resolve_model_path, load_trained_model, validate_architecture


def test_architecture_unchanged():
    path = resolve_model_path()
    assert path is not None
    model = load_trained_model(str(path))
    info = validate_architecture(model)
    assert info["params"] == 50890


def test_predictions_stable_on_fixed_input():
    path = resolve_model_path()
    assert path is not None
    model = load_trained_model(str(path))
    x = np.zeros((1, 28, 28, 1), dtype=np.float32)
    # simple deterministic pattern
    x[0, 5:23, 13:15, 0] = 1.0
    p1 = model.predict(x, verbose=0)
    p2 = model.predict(x, verbose=0)
    assert np.allclose(p1, p2)
    assert p1.shape == (1, 10)
    assert np.allclose(p1.sum(axis=1), 1.0, atol=1e-5)
