from __future__ import annotations

import numpy as np

from visualization.inference import resolve_model_path, load_trained_model, validate_architecture, get_intermediate_model


def test_model_loading_and_architecture():
    path = resolve_model_path()
    assert path is not None
    model = load_trained_model(str(path))
    info = validate_architecture(model)
    assert info["input_shape"] == (None, 28, 28, 1)
    assert info["output_shape"] == (None, 10)
    assert info["params"] == 50890


def test_intermediate_extractor_shapes():
    path = resolve_model_path()
    assert path is not None
    extractor = get_intermediate_model(str(path))
    x = np.zeros((1, 28, 28, 1), dtype=np.float32)
    out = extractor(x, training=False)
    # keys or list; handle both
    if isinstance(out, dict):
        conv1 = out["conv_1"]; conv2 = out["conv_2"]; flat = out["flatten"]; pred = out["predictions"]
    else:
        conv1, conv2, flat, pred = out[1], out[2], out[3], out[4]
    import numpy as _np
    conv1 = _np.asarray(conv1); conv2 = _np.asarray(conv2); flat = _np.asarray(flat); pred = _np.asarray(pred)
    assert conv1.shape[1:] == (14, 14, 16)
    assert conv2.shape[1:] == (12, 12, 32)
    assert flat.shape[1] == 4608
    assert pred.shape[1] == 10
    s = pred.sum(axis=1)
    assert _np.allclose(s, 1.0, atol=1e-5)
