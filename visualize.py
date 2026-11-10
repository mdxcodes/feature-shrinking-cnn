"""Generate MNIST training and CNN visualizations for the project."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from train import download_and_store_mnist, preprocess_data

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
FEATURE_MAP_DIR = RESULTS_DIR / "feature_maps"


def load_model() -> tf.keras.Model:
    best_path = MODELS_DIR / "best_mnist_cnn.keras"
    final_path = MODELS_DIR / "final_mnist_cnn.keras"
    model_path = best_path if best_path.exists() else final_path
    if not model_path.exists():
        raise FileNotFoundError(
            f"No trained model found at {best_path} or {final_path}. Run train.py first."
        )
    print(f"Loading model from {model_path}")
    return tf.keras.models.load_model(model_path)


def plot_training_history(history_path: Path) -> None:
    if not history_path.exists():
        print(f"Training history not found at {history_path}; skipping training curves.")
        return
    with open(history_path, "r", encoding="utf-8") as f:
        history = json.load(f)

    if "accuracy" in history and "val_accuracy" in history:
        plt.figure(figsize=(8, 5))
        plt.plot(history["accuracy"], label="Training accuracy")
        plt.plot(history["val_accuracy"], label="Validation accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Training Accuracy vs Validation Accuracy")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "training_accuracy.png", dpi=150)
        plt.close()

    if "loss" in history and "val_loss" in history:
        plt.figure(figsize=(8, 5))
        plt.plot(history["loss"], label="Training loss")
        plt.plot(history["val_loss"], label="Validation loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss vs Validation Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "training_loss.png", dpi=150)
        plt.close()


def plot_sample_mnist_images(x_raw: np.ndarray, y_raw: np.ndarray) -> None:
    idxs = np.arange(16)
    fig, axes = plt.subplots(4, 4, figsize=(6, 6))
    for ax, idx in zip(axes.flat, idxs):
        ax.imshow(x_raw[idx], cmap="gray")
        ax.set_title(f"Label: {y_raw[idx]}")
        ax.axis("off")
    fig.suptitle("MNIST Sample Images", fontsize=14)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "mnist_samples.png", dpi=150)
    plt.close(fig)


def plot_sample_predictions(model: tf.keras.Model, x_test: np.ndarray, y_test: np.ndarray) -> None:
    rng = np.random.default_rng(42)
    idxs = rng.choice(len(x_test), size=12, replace=False)
    images = x_test[idxs]
    labels = y_test[idxs]
    probs = model.predict(images, verbose=0)
    preds = np.argmax(probs, axis=1)
    confs = np.max(probs, axis=1)

    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    for ax, img, actual, pred, conf in zip(axes.flat, images, labels, preds, confs):
        ax.imshow(img.squeeze(), cmap="gray")
        color = "green" if actual == pred else "red"
        ax.set_title(f"Actual: {actual}\nPred: {pred} ({conf:.2f})", color=color, fontsize=10)
        ax.axis("off")
    fig.suptitle("Sample Predictions (image, actual, predicted, confidence)", fontsize=14)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "sample_predictions.png", dpi=150)
    plt.close(fig)


def _plot_feature_maps(feature_maps: np.ndarray, title: str, out_path: Path, cols: int = 8) -> None:
    # feature_maps: (H, W, C)
    channels = feature_maps.shape[-1]
    rows = math.ceil(channels / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.4, rows * 1.4))
    axes_arr = np.array(axes).reshape(rows, cols)
    for i in range(rows * cols):
        r, c = divmod(i, cols)
        ax = axes_arr[r, c]
        if i < channels:
            fmap = feature_maps[:, :, i]
            ax.imshow(fmap, cmap="viridis")
            ax.set_title(f"Ch {i}", fontsize=7)
        ax.axis("off")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def visualize_feature_maps(model: tf.keras.Model, x_test: np.ndarray) -> None:
    FEATURE_MAP_DIR.mkdir(parents=True, exist_ok=True)
    sample = x_test[0:1]  # (1,28,28,1)

    # Save original input
    plt.figure(figsize=(3, 3))
    plt.imshow(sample[0, :, :, 0], cmap="gray")
    plt.title("Original input (28x28)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(FEATURE_MAP_DIR / "input_28x28.png", dpi=150)
    plt.close()

    conv1 = model.get_layer("conv_1").output
    conv2 = model.get_layer("conv_2").output
    feature_model = tf.keras.Model(inputs=model.inputs, outputs=[conv1, conv2])
    fmap1, fmap2 = feature_model.predict(sample, verbose=0)

    _plot_feature_maps(
        fmap1[0],
        "Conv1 feature maps (14x14x16)",
        FEATURE_MAP_DIR / "conv1_14x14x16.png",
        cols=8,
    )
    _plot_feature_maps(
        fmap2[0],
        "Conv2 feature maps (12x12x32)",
        FEATURE_MAP_DIR / "conv2_12x12x32.png",
        cols=8,
    )

    explanation = (
        "Feature Shrinking in the CNN\n"
        "============================\n"
        "Original input: 28x28x1 (784 spatial values).\n\n"
        "Conv1 (16 filters, 3x3, stride 2, padding 'same', ReLU):\n"
        "- Stride 2 downsamples spatial dimensions by ~2 while 'same' padding preserves alignment.\n"
        "- Output: 14x14x16 feature maps, capturing low-level edges/strokes at reduced resolution.\n\n"
        "Conv2 (32 filters, 3x3, stride 1, padding 'valid', ReLU):\n"
        "- No downsampling by stride, but 'valid' padding removes border pixels (no zero-padding),\n"
        "  shrinking 14x14 -> 12x12 while increasing depth to 32 channels.\n"
        "- Output: 12x12x32 feature maps, capturing higher-level combinations/structures.\n\n"
        "Flatten: 12x12x32 = 4608 features fed to the classifier.\n"
        "No pooling layers are used; spatial reduction comes from strided convolution and valid padding,\n"
        "which lets learned filters control how information is downsampled while extracting structure.\n"
    )
    with open(FEATURE_MAP_DIR / "feature_shrinking_explanation.txt", "w", encoding="utf-8") as f:
        f.write(explanation)

    print(f"Saved feature map visualizations to {FEATURE_MAP_DIR}")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FEATURE_MAP_DIR.mkdir(parents=True, exist_ok=True)

    raw = download_and_store_mnist()
    _, _, x_test, y_test = preprocess_data(raw)

    plot_training_history(RESULTS_DIR / "training_history.json")
    plot_sample_mnist_images(raw["x_test"], raw["y_test"])

    model = load_model()
    plot_sample_predictions(model, x_test, y_test)
    visualize_feature_maps(model, x_test)

    # Copy/ensure confusion matrix exists if evaluate already produced it
    cm_path = RESULTS_DIR / "confusion_matrix.png"
    if not cm_path.exists():
        print("confusion_matrix.png not found (run evaluate.py to generate metrics/confusion matrix).")

    print("Visualization generation complete.")


if __name__ == "__main__":
    main()
