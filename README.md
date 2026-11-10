# Feature-Shrinking CNN (MNIST)

A lightweight TensorFlow/Keras Convolutional Neural Network for classifying 28×28 grayscale handwritten digits (MNIST, classes 0–9). Spatial dimensions are reduced using **strided convolution and valid padding**—**no pooling layers** are used.

## Project overview
- **Model:** Sequential CNN defined in `model.py` (`build_model()`), compiled with Adam, sparse categorical cross-entropy, and accuracy.
- **Dataset:** MNIST handwritten digits loaded via `tensorflow.keras.datasets.mnist`, cached locally under `data/mnist/` to avoid repeated downloads.
- **Pipeline:** `train.py` (download/cache/preprocess/train/save), `evaluate.py` (test metrics/reports), `visualize.py` (curves, samples, predictions, feature maps).
- **Artifacts:** trained models in `models/`, metrics/plots in `results/`.

## CNN architecture (unchanged)
| Layer | Configuration | Output shape (batch excluded) |
|---|---|---:|
| Input | Grayscale image | `28 × 28 × 1` |
| Conv2D `conv_1` | 16 filters, `3×3`, stride `2×2`, padding `same`, ReLU | `14 × 14 × 16` |
| Conv2D `conv_2` | 32 filters, `3×3`, stride `1×1`, padding `valid`, ReLU | `12 × 12 × 32` |
| Flatten `flatten` | Convert feature maps to vector | `4608` |
| Dense `predictions` | 10 units, softmax | `10` |

- **API:** `tf.keras.Sequential` only. **No pooling, BatchNormalization, or Dropout.**
- **Model I/O:** input `(None, 28, 28, 1)` → output `(None, 10)`.
- **Parameters:** Total/trainable **50,890** (198.79 KB), non-trainable **0**.

### Feature-map dimensions (spatial shrinking)
- Original input: **28×28×1**
- After Conv1 (stride 2, same): **14×14×16**
- After Conv2 (stride 1, valid): **12×12×32** (border reduction without pooling)
- Flattened features: **12 × 12 × 32 = 4,608**

## Dataset information and source
- **Source:** Official MNIST via `tensorflow.keras.datasets.mnist` (loaded with `mnist.load_data()`).
- **Total:** 70,000 images (28×28 grayscale, 1 channel), 10 classes (digits 0–9).
- **Split (preserved):** 60,000 train / 10,000 test.
- **Local storage (cached, not committed):** `data/mnist/x_train.npy`, `y_train.npy`, `x_test.npy`, `y_test.npy`.

## Installation instructions
Use the existing virtual environment (`.venv`) when present.

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

Dependencies (see `requirements.txt`): `tensorflow`, `pytest`, `numpy`, `matplotlib`, `scikit-learn`.

> Note: in the verified environment, TensorFlow resolved to a CPU build (`tensorflow-cpu 2.22.0rc0`, Python 3.14). Use a compatible stable TensorFlow for your platform if preferred; keep the CNN architecture/tests unchanged.

## Dataset download instructions
`train.py` (and evaluation/visualization scripts) will **download MNIST automatically only if local `.npy` files are missing**, then cache them in `data/mnist/`. No repeated downloads when files already exist.

```bash
source .venv/bin/activate
python train.py   # downloads/caches MNIST on first run if needed
```

## Preprocessing
- Images converted to `float32` and normalized to `[0, 1]`.
- Channel dimension added: `(28, 28) -> (28, 28, 1)`.
- Labels kept as integers for `sparse_categorical_crossentropy`.
- Expected shapes: train `(60000, 28, 28, 1)/(60000,)`, test `(10000, 28, 28, 1)/(10000,)`. No resizing.

## Training instructions
```bash
source .venv/bin/activate

python train.py
```

**Configuration (as implemented):**
- Epochs: 15 (max), batch size: 128, validation split: 0.1
- Optimizer: Adam, loss: `sparse_categorical_crossentropy`, metrics: `accuracy`
- Fixed random seed: 42 (reproducibility)
- Callbacks: `ModelCheckpoint` (best by `val_accuracy` → `models/best_mnist_cnn.keras`), `EarlyStopping` (monitor `val_loss`, patience 3, restore best weights), `ReduceLROnPlateau` (monitor `val_loss`, factor 0.5, patience 2, min_lr 1e-6)
- Test set is **not** used during training/selection. Saves `models/final_mnist_cnn.keras` and `results/training_history.json`.

## Evaluation instructions
```bash
source .venv/bin/activate

python evaluate.py
```

Evaluates the trained model (prefers best checkpoint) on the untouched 10,000-image MNIST test set. Computes test loss/accuracy, macro precision/recall/F1, confusion matrix, and per-class accuracy (scikit-learn). Saves:
- `results/evaluation_report.txt`
- `results/classification_report.txt`
- `results/confusion_matrix.png`
- `results/evaluation_metrics.json`

## Visualization instructions
```bash
source .venv/bin/activate

python visualize.py
```

Generates (in `results/`): training accuracy/loss curves, MNIST sample images with labels, sample predictions (image/actual/predicted/confidence), confusion matrix (if produced), and feature-map visualizations in `results/feature_maps/` (input 28×28, Conv1 14×14×16, Conv2 12×12×32) plus `feature_shrinking_explanation.txt`.

## Actual training results (executed run)
From `results/training_history.json`:
- **Epochs completed:** 9 / 15 (early stopping triggered; best weights restored from epoch 6)
- **Final recorded epoch (9) metrics:** train accuracy **0.99572**, train loss **0.01491**, val accuracy **0.98750**, val loss **0.05571**
- **Best validation accuracy observed:** **0.98683** (epoch 6, saved as best checkpoint)

| Epoch | Train acc | Train loss | Val acc | Val loss | LR |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.91063 | 0.32161 | 0.97467 | 0.09472 | 0.001 |
| 2 | 0.97324 | 0.08808 | 0.98317 | 0.06430 | 0.001 |
| 3 | 0.98156 | 0.06140 | 0.98500 | 0.05697 | 0.001 |
| 4 | 0.98572 | 0.04797 | 0.98633 | 0.05552 | 0.001 |
| 5 | 0.98876 | 0.03898 | 0.98667 | 0.05499 | 0.001 |
| 6 | 0.99072 | 0.03195 | 0.98683 | 0.05422 | 0.001 |
| 7 | 0.99244 | 0.02605 | 0.98633 | 0.05462 | 0.001 |
| 8 | 0.99387 | 0.02114 | 0.98600 | 0.05578 | 0.001 |
| 9 | 0.99572 | 0.01491 | 0.98750 | 0.05571 | 0.0005 |

## Actual test results (executed evaluation on 10,000 MNIST test images)
From `results/evaluation_report.txt` / `classification_report.txt` (model: best checkpoint):
- **Test accuracy:** **0.988600**
- **Test loss:** **0.042186**
- **Macro precision:** **0.988566**
- **Macro recall:** **0.988410**
- **Macro F1-score:** **0.988474**

Per-class accuracy (0–9): 0.992857, 0.996476, 0.982558, 0.994059, 0.991853, 0.980942, 0.986430, 0.991245, 0.982546, 0.985134.

## Generated artifacts
- **Models:** `models/best_mnist_cnn.keras`, `models/final_mnist_cnn.keras`
- **Data cache:** `data/mnist/x_train.npy`, `y_train.npy`, `x_test.npy`, `y_test.npy`
- **Results:** `results/training_history.json`, `results/evaluation_metrics.json`, `results/evaluation_report.txt`, `results/classification_report.txt`, `results/training_accuracy.png`, `results/training_loss.png`, `results/confusion_matrix.png`, `results/mnist_samples.png`, `results/sample_predictions.png`
- **Feature maps:** `results/feature_maps/input_28x28.png`, `conv1_14x14x16.png`, `conv2_12x12x32.png`, `feature_shrinking_explanation.txt`

## Testing
```bash
source .venv/bin/activate
pytest -v
```
Existing architecture tests in `tests/test_model.py` must remain passing (Sequential API, required layer order/parameters, shapes, and no pooling).

## Project structure
```text
feature-shrinking-cnn/
├── model.py
├── train.py
├── evaluate.py
├── visualize.py
├── data/mnist/           # cached MNIST .npy (not committed)
├── models/
├── results/
│   └── feature_maps/
├── tests/
├── requirements.txt
├── README.md
└── .gitignore
```
