import pytest

tf = pytest.importorskip("tensorflow")

from model import build_model


def test_model_uses_required_output_shape():
    model = build_model()
    assert model.input_shape == (None, 28, 28, 1)
    assert model.output_shape == (None, 10)


def test_feature_map_dimensions_shrink_without_pooling():
    model = build_model()
    assert model.get_layer("conv_1").output.shape[1:] == (14, 14, 16)
    assert model.get_layer("conv_2").output.shape[1:] == (12, 12, 32)
    assert not any("Pooling" in layer.__class__.__name__ for layer in model.layers)


def test_required_layer_order_and_parameters():
    model = build_model()
    layer_types = [layer.__class__.__name__ for layer in model.layers]
    # Keras 2 includes InputLayer in model.layers; Keras 3 may omit it while
    # preserving input_shape. Accept either representation.
    assert layer_types in (
        ["InputLayer", "Conv2D", "Conv2D", "Flatten", "Dense"],
        ["Conv2D", "Conv2D", "Flatten", "Dense"],
    )

    first = model.get_layer("conv_1")
    assert first.filters == 16
    assert first.kernel_size == (3, 3)
    assert first.strides == (2, 2)
    assert first.padding == "same"
    assert first.activation.__name__ == "relu"

    second = model.get_layer("conv_2")
    assert second.filters == 32
    assert second.kernel_size == (3, 3)
    assert second.strides == (1, 1)
    assert second.padding == "valid"
    assert second.activation.__name__ == "relu"

    output = model.get_layer("predictions")
    assert output.units == 10
    assert output.activation.__name__ == "softmax"
