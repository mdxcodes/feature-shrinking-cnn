"""Evaluate the trained Feature-Shrinking CNN on the MNIST test set."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score

from train import download_and_store_mnist, preprocess_data

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"


def load_evaluation_model() -> tf.keras.Model:
    best_path = MODELS_DIR / "best_mnist_cnn.keras"
    final_path = MODELS_DIR / "final_mnist_cnn.keras"
    model_path = best_path if best_path.exists() else final_path
    if not model_path.exists():
        raise FileNotFoundError(
            f"No trained model found. Expected {best_path} or {final_path}. Run train.py first."
        )
    print(f"Loading model from {model_path}")
    return tf.keras.models.load_model(model_path)


def plot_confusion_matrix(cm: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    classes = list(range(10))
    ax.set(
        xticks=np.arange(len(classes)),
        yticks=np.arange(len(classes)),
        xticklabels=classes,
        yticklabels=classes,
        ylabel="True label",
        xlabel="Predicted label",
        title="MNIST Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=0)
    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=8,
            )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    raw = download_and_store_mnist()
    _, _, x_test, y_test = preprocess_data(raw)

    model = load_evaluation_model()
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)

    probs = model.predict(x_test, batch_size=256, verbose=0)
    if probs.shape != (10000, 10):
        raise ValueError(f"Unexpected prediction shape: {probs.shape}")
    row_sums = probs.sum(axis=1)
    if not np.all(np.isfinite(probs)):
        raise ValueError("Predictions contain NaN or infinite values.")
    if not np.allclose(row_sums, 1.0, atol=1e-5):
        raise ValueError(f"Prediction probabilities do not sum to 1: {row_sums[:5]}")

    y_pred = np.argmax(probs, axis=1)
    precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=list(range(10)))
    cls_report = classification_report(y_test, y_pred, digits=4)

    per_class_acc = {}
    for digit in range(10):
        idx = y_test == digit
        per_class_acc[str(digit)] = float(np.mean(y_pred[idx] == y_test[idx])) if np.any(idx) else 0.0

    eval_path = RESULTS_DIR / "evaluation_report.txt"
    with open(eval_path, "w", encoding="utf-8") as f:
        f.write("MNIST Test Evaluation Report\n")
        f.write("============================\n")
        f.write(f"Test samples: {len(y_test)}\n")
        f.write(f"Test loss: {test_loss:.6f}\n")
        f.write(f"Test accuracy: {test_accuracy:.6f}\n")
        f.write(f"Macro precision: {precision:.6f}\n")
        f.write(f"Macro recall: {recall:.6f}\n")
        f.write(f"Macro F1-score: {f1:.6f}\n")
        f.write("\nPer-class accuracy:\n")
        for digit in range(10):
            f.write(f"  Digit {digit}: {per_class_acc[str(digit)]:.6f}\n")
        f.write("\nConfusion matrix (rows=true, cols=pred):\n")
        f.write(np.array2string(cm))

    cls_path = RESULTS_DIR / "classification_report.txt"
    with open(cls_path, "w", encoding="utf-8") as f:
        f.write(cls_report)

    plot_confusion_matrix(cm, RESULTS_DIR / "confusion_matrix.png")

    summary = {
        "test_samples": int(len(y_test)),
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "per_class_accuracy": per_class_acc,
        "prediction_shape": list(probs.shape),
        "probabilities_sum_to_one": bool(np.allclose(row_sums, 1.0, atol=1e-5)),
    }
    with open(RESULTS_DIR / "evaluation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Test loss: {test_loss:.6f}")
    print(f"Test accuracy: {test_accuracy:.6f}")
    print(f"Macro precision: {precision:.6f}")
    print(f"Macro recall: {recall:.6f}")
    print(f"Macro F1-score: {f1:.6f}")
    print(f"Saved evaluation report to {eval_path}")
    print(f"Saved classification report to {cls_path}")
    print(f"Saved confusion matrix image to {RESULTS_DIR / 'confusion_matrix.png'}")


if __name__ == "__main__":
    main()
