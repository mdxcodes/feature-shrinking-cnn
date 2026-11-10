from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image


def to_pil(image) -> Image.Image | None:
    if image is None:
        return None
    if isinstance(image, Image.Image):
        return image.copy()
    try:
        arr = np.asarray(image)
        if arr.size == 0:
            return None
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        if arr.ndim == 2:
            return Image.fromarray(arr, mode="L")
        if arr.ndim == 3 and arr.shape[2] == 4:
            alpha = arr[:, :, 3]
            if not np.any(alpha > 0):
                return None
            return Image.fromarray(arr, mode="RGBA")
        if arr.ndim == 3 and arr.shape[2] == 3:
            return Image.fromarray(arr, mode="RGB")
    except Exception:
        return None
    return None


def canvas_to_pil(canvas_result) -> Image.Image | None:
    if canvas_result is None:
        return None
    try:
        image_data = canvas_result.image_data
    except Exception:
        image_data = None
    if image_data is None:
        return None
    return to_pil(image_data)


def bytes_to_pil(data: bytes) -> Image.Image | None:
    try:
        return Image.open(BytesIO(data))
    except Exception:
        return None


def pil_to_grayscale_array(pil: Image.Image) -> np.ndarray:
    if pil.mode == "RGBA":
        bg = Image.new("RGBA", pil.size, (255, 255, 255, 255))
        pil = Image.alpha_composite(bg, pil).convert("L")
    elif pil.mode != "L":
        pil = pil.convert("L")
    return np.asarray(pil, dtype=np.uint8)


def preprocess_digit_image(image, target_size: int = 28):
    """Preprocess drawing/upload into MNIST-style model input.
    Returns (model_input, processed_pil, meta).
    model_input: (1,28,28,1) float32 in [0,1] or None.
    """
    meta = {"ok": False, "reason": "", "inverted": False, "bbox": None, "had_content": False}
    pil = to_pil(image)
    if pil is None:
        meta["reason"] = "No valid image provided."
        return None, None, meta

    try:
        arr = pil_to_grayscale_array(pil)
    except Exception:
        meta["reason"] = "Invalid or corrupted image."
        return None, None, meta

    h, w = arr.shape
    if h == 0 or w == 0:
        meta["reason"] = "Unexpected image dimensions."
        return None, None, meta

    # Match MNIST polarity: bright digit on dark background.
    border = np.concatenate([arr[0, :], arr[h - 1, :], arr[:, 0], arr[:, w - 1]])
    border_mean = float(border.mean()) if border.size else 255.0
    global_mean = float(arr.mean())
    invert = border_mean > 127 or global_mean > 127
    if invert:
        arr = 255 - arr
        meta["inverted"] = True

    threshold = max(20, int(np.percentile(arr, 98) * 0.2)) if arr.size else 20
    mask = arr > threshold
    if not np.any(mask):
        meta["reason"] = "No digit strokes detected (blank or too faint)."
        return None, None, meta
    meta["had_content"] = True

    ys, xs = np.where(mask)
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    meta["bbox"] = (x0, y0, x1, y1)

    pad = max(2, int(0.08 * max(y1 - y0 + 1, x1 - x0 + 1)))
    y0m = max(0, y0 - pad)
    y1m = min(h, y1 + pad + 1)
    x0m = max(0, x0 - pad)
    x1m = min(w, x1 + pad + 1)
    cropped = arr[y0m:y1m, x0m:x1m]
    ch, cw = cropped.shape
    if ch == 0 or cw == 0:
        meta["reason"] = "Cropping failed."
        return None, None, meta

    # Fit into 20x20 box, center in 28x28 (MNIST-like centering)
    box = 20
    scale = min(box / ch, box / cw)
    nh = max(1, int(round(ch * scale)))
    nw = max(1, int(round(cw * scale)))
    digit_img = Image.fromarray(cropped, mode="L").resize((nw, nh), Image.Resampling.LANCZOS)

    canvas = Image.new("L", (target_size, target_size), 0)
    left = (target_size - nw) // 2
    top = (target_size - nh) // 2
    canvas.paste(digit_img, (left, top))

    proc_arr = np.asarray(canvas, dtype=np.float32) / 255.0
    model_input = proc_arr.reshape(1, target_size, target_size, 1).astype(np.float32)
    meta["ok"] = True
    meta["reason"] = "ok"
    return model_input, canvas, meta
