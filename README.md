# Inside the CNN
## Feature-Shrinking CNN Interactive Visualization (MNIST)

A Python-first interactive visualization of a **real trained TensorFlow/Keras CNN** classifying handwritten digits (MNIST, 0–9). The central experience animates how a drawn/uploaded digit travels through the actual network: **input tensor → Conv1 → Conv2 → Flatten → Dense/Softmax**.

- **No retraining. No architecture changes. No fabricated outputs.**
- All feature maps, convolution math, activations, logits, and probabilities come from the existing trained model at `models/best_mnist_cnn.keras` (fallback `models/final_mnist_cnn.keras`).
- Animation is rendered in-browser (HTML5 Canvas/CSS via a lightweight Streamlit component); **all ML inference and tensor math run in Python**.

## Project overview
This project extends the existing Feature-Shrinking CNN MNIST repository with an interactive visualization:
- **Main experience:** connected forward-pass animation showing the digit moving through every CNN stage with actual tensors and filter sweeps.
- **Feature Map Explorer:** inspect all 16 Conv1 (14×14) and 32 Conv2 (12×12) real activation maps.
- **Architecture Explorer:** real layer metadata, shapes, parameters, and dimension-shrink explanations from the loaded model.
- **Prediction Explorer:** actual softmax probabilities for all 10 classes.
- **Training & Evaluation Dashboard:** preserved results loaded from `results/` artifacts (no regenerated/fabricated metrics).

## CNN architecture (preserved, unchanged)
| Stage | Configuration | Output shape |
|---|---|---:|
| Input | Grayscale image | `28 × 28 × 1` |
| Conv2D `conv_1` | 16 filters, 3×3, stride 2, padding **same**, ReLU | `14 × 14 × 16` |
| Conv2D `conv_2` | 32 filters, 3×3, stride 1, padding **valid**, ReLU | `12 × 12 × 32` |
| Flatten `flatten` | 12 × 12 × 32 | `4,608` |
| Dense `predictions` | 10 units, softmax | `10` |

- **Sequential API (Keras/TensorFlow)**, **50,890 trainable parameters**, **0 pooling layers**, no BatchNorm/Dropout.
- Spatial downsampling comes from **strided convolution** (Conv1) and **valid padding** (Conv2).

## Technology stack
- **Python, TensorFlow/Keras**: model loading, inference, intermediate tensor extraction, convolution math verification.
- **NumPy, Pillow**: preprocessing and tensor/image operations.
- **Streamlit**: application UI, layout, session state, caching.
- **streamlit-drawable-canvas**: interactive digit drawing (mouse/touch).
- **Plotly / Matplotlib**: charts and feature-map visualizations.
- **HTML5 Canvas + CSS + requestAnimationFrame** (custom Streamlit component): smooth browser-side animation of precomputed data.
- **pytest**: automated tests.

## Repository layout (key files)
- `app.py` — main Streamlit app (“Inside the CNN”).
- `model.py` — CNN definition (`build_model()`), architecture preserved.
- `visualization/` — inference & extraction (`inference.py`, `tensor_extraction.py`), manual/verified convolution math (`convolution_math.py`), animation payload preparation (`animation_data.py`), preprocessing (`preprocessing.py`), browser renderer (`rendering.py`).
- `components/` — UI modules: drawing/input, feature maps, architecture, prediction, dashboard, styles.
- `frontend/` — shared helpers reused for artifact/model loading and dashboard utilities.
- `models/` — trained checkpoints (`best_mnist_cnn.keras`, `final_mnist_cnn.keras`).
- `results/` — training/evaluation artifacts and figures.
- `data/mnist/` — cached MNIST arrays (`.npy`, git-ignored).
- `tests/` — architecture and visualization/inference/math tests.
- `train.py`, `evaluate.py`, `visualize.py` — existing training/evaluation/visualization scripts (preserved).

## Installation
Use the existing virtual environment when present.

```bash
cd /secondary/feature-shrinking-cnn
source .venv/bin/activate        # bash/zsh
# or (fish): source .venv/bin/activate.fish
pip install -r requirements.txt
```

Verified environment includes TensorFlow (CPU build), Streamlit, streamlit-drawable-canvas, Plotly, Matplotlib, scikit-learn, NumPy, Pillow, pandas, pytest. Keep TensorFlow version compatible with your platform; do **not** change the CNN architecture or weights.

## Launch the application
```bash
cd /secondary/feature-shrinking-cnn
.venv/bin/streamlit run app.py
# or with activated env: streamlit run app.py
```

A `.streamlit/config.toml` is included to avoid inotify watcher overflow in large environments (file watcher disabled; restart Streamlit after code edits).

## How to draw and upload digits
1. **Draw:** use the interactive canvas (mouse/touch), white strokes on dark background. Use **Clear canvas** or **Reset input** as needed.
2. **Upload:** PNG/JPG/JPEG/BMP supported (grayscale/RGB; inverted backgrounds handled).
3. **Examples:** sample digits can be loaded from cached MNIST test data when available.

The app preprocesses input to match MNIST conventions used by the model: grayscale, polarity normalization (bright digit on dark), crop/center/scale, resize to 28×28, normalize to `[0,1]`, shape **(1, 28, 28, 1)**. The **exact processed 28×28 tensor** sent to the model is always displayed.

## How the animation works
- **Inference once per input:** when you provide a digit, Python runs the trained CNN once and extracts actual intermediate tensors: input, `conv_1` (14×14×16), `conv_2` (12×12×32), `flatten` (4608), and `predictions` (10). Results are cached in Streamlit session/data cache (no repeated inference during animation or filter/stage changes beyond payload preparation from cached tensors).
- **Precomputed animation data:** Python prepares step-by-step convolution data (receptive-field coordinates from TF stride/padding rules, actual kernel weights/bias, input patches, element-wise products, sums, ReLU, and actual activation values) and a browser payload.
- **Browser rendering:** a lightweight HTML5 Canvas component animates the forward pass with `requestAnimationFrame` and UI controls (Play/Pause/Replay/Next/Prev, speed Slow/Normal/Fast). **JavaScript performs rendering/animation only** — it does not run TensorFlow or invent outputs.
- **Conv1:** animated 3×3 receptive field sweeps 28×28 with stride 2 and SAME padding; each step highlights the input patch and corresponding 14×14 output cell and shows actual math.
- **Conv2:** operates on the 14×14×16 stack with 3×3×16 kernels, stride 1, VALID padding; full calculation sums across all spatial positions and 16 channels (channel contributions shown for readability).
- **Flatten:** visualizes unfolding of 12×12×32 into 4,608 values in Keras channels-last order.
- **Dense/Output:** shows logits `xW+b` (actual weights/bias) and softmax probabilities for 10 classes; connections are visually aggregated for clarity (46,080 weights) while calculations remain exact.

## How feature maps are extracted
An intermediate Keras functional model is built from the **loaded trained model** (`model.inputs` → `conv_1`, `conv_2`, `flatten`, `predictions` outputs), reusing original layers/weights (no second trained model). For the current preprocessed input tensor, actual activations are:
- Conv1: **(1, 14, 14, 16)** — 16 real feature maps
- Conv2: **(1, 12, 12, 32)** — 32 real feature maps
- Flatten: **(1, 4608)** — actual flattened vector
- Predictions: **(1, 10)** — actual softmax probabilities (sum ≈ 1)

Display normalization may be applied for visibility in grids; underlying numeric activations shown in explorers/inspectors are the real values.

## How to interpret convolution visualizations
- **Receptive field:** highlighted region in the input (or input stack) that contributes to one output cell.
- **Kernel/weights:** trained 3×3 (Conv1) or 3×3×16 (Conv2) filter values for the selected output filter.
- **Element-wise products → sum → +bias → ReLU:** manual step values computed in Python from actual tensors/weights and compared to the actual intermediate activation.
- **SAME padding (Conv1):** TF-compatible padding/receptive-field coordinates are derived so the animated position matches real computation.
- **VALID padding (Conv2):** no padding; spatial size reduces 14×14 → 12×12.
- **Stride:** Conv1 moves by 2 pixels (downsampling), Conv2 by 1.

## Training and evaluation (preserved)
Existing scripts and artifacts are preserved:
```bash
python train.py      # download/cache MNIST if missing, train (no retraining required for the app)
python evaluate.py   # evaluate on test set and write results/*
python visualize.py  # generate plots/feature-map figures in results/*
```
The app’s dashboard reads actual files from `results/` (e.g. `training_history.json`, `evaluation_metrics.json`, reports, confusion matrix, curves). If an artifact is missing, the UI reports it as unavailable.

## Run tests
```bash
cd /secondary/feature-shrinking-cnn
PYTHONPATH=. .venv/bin/python -m pytest -q
# or: pytest -v
```

Test coverage includes: architecture preservation (`tests/test_model.py`, `tests/test_architecture_preserved.py`), preprocessing/input handling, model loading & intermediate extraction, tensor dimensions (Conv1/Conv2/Flatten/predictions), probability sums, and numerical Conv1/Conv2 calculations verified against TensorFlow intermediate activations.

## Notes
- `streamlit-drawable-canvas` behavior varies by version; the app requests `image_data` and handles canvas/input fallbacks defensively.
- Headless run example: `streamlit run app.py --server.headless true --server.port 8501 --server.address 0.0.0.0`
