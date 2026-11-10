"""Feature-Shrinking CNN for 28x28 grayscale image classification."""

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, InputLayer


def build_model():
    """Build and compile the required CNN using the Sequential API."""
    model = Sequential(
        [
            InputLayer(shape=(28, 28, 1), name="image_input"),
            Conv2D(
                filters=16,
                kernel_size=(3, 3),
                strides=(2, 2),
                padding="same",
                activation="relu",
                name="conv_1",
            ),
            Conv2D(
                filters=32,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding="valid",
                activation="relu",
                name="conv_2",
            ),
            Flatten(name="flatten"),
            Dense(10, activation="softmax", name="predictions"),
        ],
        name="feature_shrinking_cnn",
    )

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


if __name__ == "__main__":
    cnn = build_model()
    cnn.summary()
