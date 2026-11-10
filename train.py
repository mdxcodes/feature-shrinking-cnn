"""Train the Feature-Shrinking CNN on the MNIST dataset."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.datasets import mnist

from model import build_model

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "mnist"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

SEED = 42
EPOCHS = 15
BATCH_SIZE = 128
VALIDATION_SPLIT = 0.1


def set_seeds(seed: int = SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def _mnist_files() -> dict[str, Path]:
    return {
        "x_train": DATA_DIR / "x_train.npy",
        "y_train": DATA_DIR / "y_train.npy",
        "x_test": DATA_DIR / "x_test.npy",
        "y_test": DATA_DIR / "y_test.npy",
    }


def download_and_store_mnist() -> dict[str, np.ndarray]:
    """Download MNIST if missing, store locally, and return raw arrays."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    files = _mnist_files()

    if all(path.exists() for path in files.values()):
        print("MNIST dataset found locally in data/mnist/; loading from disk.")
        data = {name: np.load(path, allow_pickle=False) for name, path in files.items()}
    else:
        print("MNIST dataset not found locally; downloading via tensorflow.keras.datasets.mnist...")
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        data = {
            "x_train": x_train,
            "y_train": y_train,
            "x_test": x_test,
            "y_test": y_test,
        }
        for name, path in files.items():
            np.save(path, data[name])
        print(f"Saved MNIST arrays to {DATA_DIR}")

    validate_raw_dataset(data)
    return data


def validate_raw_dataset(data: dict[str, np.ndarray]) -> None:
    expected = {
        "x_train": (60000, 28, 28),
        "y_train": (60000,),
        "x_test": (10000, 28, 28),
        "y_test": (10000,),
    }
    for key, shape in expected.items():
        actual = data[key].shape
        if actual != shape:
            raise ValueError(f"Unexpected shape for {key}: {actual} != {shape}")
    if data["x_train"].dtype != np.uint8 or data["x_test"].dtype != np.uint8:
        raise ValueError("Expected image arrays to have dtype uint8 before preprocessing.")
    print(
        "Validated raw MNIST split: "
        f"train={data['x_train'].shape}, test={data['x_test'].shape}, "
        f"labels={data['y_train'].shape}/{data['y_test'].shape}"
    )


def preprocess_images(images: np.ndarray) -> np.ndarray:
    images = images.astype("float32") / 255.0
    images = np.expand_dims(images, axis=-1)
    return images


def preprocess_data(data: dict[str, np.ndarray]):
    x_train = preprocess_images(data["x_train"])
    x_test = preprocess_images(data["x_test"])
    y_train = data["y_train"].astype(np.int64)
    y_test = data["y_test"].astype(np.int64)

    assert x_train.shape == (60000, 28, 28, 1)
    assert y_train.shape == (60000,)
    assert x_test.shape == (10000, 28, 28, 1)
    assert y_test.shape == (10000,)
    return x_train, y_train, x_test, y_test


def main() -> None:
    set_seeds(SEED)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    raw = download_and_store_mnist()
    x_train, y_train, x_test, y_test = preprocess_data(raw)
    _ = (x_test, y_test)  # test set is loaded/preprocessed but not used for training

    print("Preprocessed shapes:")
    print(f"  Training images: {x_train.shape}")
    print(f"  Training labels: {y_train.shape}")
    print(f"  Testing images:  {x_test.shape}")
    print(f"  Testing labels:  {y_test.shape}")

    model = build_model()
    model.summary()

    best_path = MODELS_DIR / "best_mnist_cnn.keras"
    final_path = MODELS_DIR / "final_mnist_cnn.keras"

    callbacks = [
        ModelCheckpoint(
            filepath=str(best_path),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    history = model.fit(
        x_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        callbacks=callbacks,
        verbose=2,
    )

    model.save(final_path)
    print(f"Saved final model to {final_path}")
    print(f"Best checkpoint saved to {best_path}")

    history_path = RESULTS_DIR / "training_history.json"
    serializable_history = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    serializable_history["epochs_completed"] = len(history.history.get("loss", []))
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(serializable_history, f, indent=2)
    print(f"Saved training history to {history_path}")
    print(f"Epochs completed: {serializable_history['epochs_completed']}")


if __name__ == "__main__":
    main()
