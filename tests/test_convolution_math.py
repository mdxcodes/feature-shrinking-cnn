from __future__ import annotations

import numpy as np
import tensorflow as tf

from visualization.inference import resolve_model_path, load_trained_model, get_layer_params
from visualization.convolution_math import compute_conv_step, manual_conv2d_layer


def test_conv1_manual_matches_tensor_output():
    path = resolve_model_path()
    assert path is not None
    model = load_trained_model(str(path))
    x = np.zeros((1, 28, 28, 1), dtype=np.float32)
    # set a few pixels to have signal
    x[0, 10:15, 10:15, 0] = 1.0
    conv1_out = tf.keras.Model(model.inputs, model.get_layer("conv_1").output)(x, training=False).numpy()
    params = get_layer_params(model, "conv_1")
    # check a few positions and filters
    for f in [0, 5, 15]:
        for (oh, ow) in [(0, 0), (3, 4), (13, 13)]:
            step = compute_conv_step(x, params["kernel"], params["bias"], oh, ow, f, (2, 2), "same", actual_map=conv1_out)
            assert abs(step.relu - step.actual_activation) < 1e-4
    manual = manual_conv2d_layer(x, params["kernel"], params["bias"], (2, 2), "same")
    assert np.allclose(manual, conv1_out[0], atol=1e-4)


def test_conv2_manual_matches_tensor_output():
    path = resolve_model_path()
    assert path is not None
    model = load_trained_model(str(path))
    x = np.zeros((1, 28, 28, 1), dtype=np.float32)
    x[0, 8:18, 8:18, 0] = 1.0
    inter = tf.keras.Model(model.inputs, [model.get_layer("conv_1").output, model.get_layer("conv_2").output])(x, training=False)
    c1 = inter[0].numpy()
    c2 = inter[1].numpy()
    params = get_layer_params(model, "conv_2")
    for f in [0, 7, 31]:
        for (oh, ow) in [(0, 0), (5, 6), (11, 11)]:
            step = compute_conv_step(c1, params["kernel"], params["bias"], oh, ow, f, (1, 1), "valid", actual_map=c2)
            assert abs(step.relu - step.actual_activation) < 1e-4
    manual = manual_conv2d_layer(c1, params["kernel"], params["bias"], (1, 1), "valid")
    assert np.allclose(manual, c2[0], atol=1e-4)
