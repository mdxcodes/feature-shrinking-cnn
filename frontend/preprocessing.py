"""Image preprocessing for canvas drawings and uploaded digit images."""

from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image, ImageOps


def _to_pil(image) -> Image.Image | None:
    if image is None:
        return None
    if isinstance(image, Image.Image):
        return image
    try:
        arr = np.asarray(image)
        if arr.size == 0:
            return None
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        if arr.ndim == 2:
            return Image.fromarray(arr, mode="L")
        if arr.ndim == 3 and arr.shape[2] == 4:
            # RGBA canvas
            alpha = arr[:, :, 3]
            if not np.any(alpha > 0):
                return None
            return Image.fromarray(arr, mode="RGBA").convert("RGB")
        if arr.ndim == 3 and arr.shape[2] == 3:
            return Image.fromarray(arr, mode="RGB")
    except Exception:
        return None
    return None


def canvas_to_pil(canvas_result) -> Image.Image | None:
    if canvas_result is None:
        return None
    # streamlit-drawable-canvas exposes image_data (RGBA numpy); accessing it may
    # raise if not requested or unavailable in the installed component version.
    try:
        image_data = canvas_result.image_data
    except Exception:
        image_data = None
    if image_data is None:
        # Fallback for component result shapes that provide raw data/url
        try:
            raw = getattr(canvas_result, "json_data", None)
            _ = raw  # reserved for future fallback handling
        except Exception:
            pass
        return None
    return _to_pil(image_data)


def bytes_to_pil(data: bytes) -> Image.Image | None:
    try:
        return Image.open(BytesIO(data)).convert("RGB")
    except Exception:
        return None


def preprocess_digit_image(image, target_size: int = 28) -> tuple[np.ndarray | None, Image.Image | None, dict]:
    """Return (model_input, processed_pil, meta) or (None, None, meta) on failure.

    Model input shape: (1, 28, 28, 1), float32 in [0, 1].
    """
    meta = {"ok": False, "reason": "", "inverted": False, "bbox": None}
    pil = _to_pil(image)
    if pil is None:
        meta["reason"] = "No valid image provided."
        return None, None, meta

    # Convert to grayscale
    if pil.mode == "RGBA":
        # composite over white, then grayscale
        bg = Image.new("RGBA", pil.size, (255, 255, 255, 255))
        pil = Image.alpha_composite(bg, pil).convert("L")
    else:
        pil = pil.convert("L")

    arr = np.asarray(pil, dtype=np.uint8)

    # Determine background and invert if needed to match MNIST (bright digit on dark bg)
    # Use border statistics; if border is bright, assume dark ink on light background -> invert.
    h, w = arr.shape
    border = np.concatenate([arr[0, :], arr[h - 1, :], arr[:, 0], arr[:, w - 1]])
    border_mean = float(border.mean()) if border.size else 255.0
    global_mean = float(arr.mean())
    invert = border_mean > 127 or global_mean > 127
    if invert:
        arr = 255 - arr
        meta["inverted"] = True

    # Binarize-ish mask for cropping (robust to faint strokes)
    threshold = max(20, int(np.percentile(arr, 98) * 0.2))
    mask = arr > threshold
    if not np.any(mask):
        meta["reason"] = "No digit strokes detected (blank or too faint)."
        return None, None, meta

    ys, xs = np.where(mask)
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    meta["bbox"] = (x0, y0, x1, y1)

    # Crop with small margin
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

    # Preserve aspect ratio: fit into 20x20 box (MNIST-like), center on 28x28
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
