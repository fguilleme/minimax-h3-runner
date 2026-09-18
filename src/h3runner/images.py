from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps


_CANVAS_MULTIPLE = 32
_BASE_SHORT_EDGE = 768
_MAX_PIXELS = 768 * 1344


def aspect_preserving_canvas(width: int, height: int) -> tuple[int, int]:
    """Return an H3 canvas close to the source aspect ratio.

    H3 requires dimensions divisible by 32. The canvas follows the same
    768-short-edge and 768*1344-pixel cap used by ComfyUI's MiniMax node.
    """
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    ratio = width / height
    if ratio >= 1.0:
        nominal_width, nominal_height = _BASE_SHORT_EDGE * ratio, _BASE_SHORT_EDGE
    else:
        nominal_width, nominal_height = _BASE_SHORT_EDGE, _BASE_SHORT_EDGE / ratio
    if nominal_width * nominal_height > _MAX_PIXELS:
        scale = (_MAX_PIXELS / (nominal_width * nominal_height)) ** 0.5
        nominal_width *= scale
        nominal_height *= scale
    return (
        max(_CANVAS_MULTIPLE, round(nominal_width / _CANVAS_MULTIPLE) * _CANVAS_MULTIPLE),
        max(_CANVAS_MULTIPLE, round(nominal_height / _CANVAS_MULTIPLE) * _CANVAS_MULTIPLE),
    )


def fit_aspect_preserving_canvas(
    source_width: int,
    source_height: int,
    max_width: int,
    max_height: int,
) -> tuple[int, int]:
    """Fit a source ratio inside a bounded H3 canvas without upscaling it."""
    for value, name in ((source_width, "source width"), (source_height, "source height"),
                        (max_width, "max width"), (max_height, "max height")):
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    max_pixels = max_width * max_height
    source_ratio = source_width / source_height
    width_limit = min(source_width, int((max_pixels * source_ratio) ** 0.5))
    width = max(_CANVAS_MULTIPLE, width_limit // _CANVAS_MULTIPLE * _CANVAS_MULTIPLE)
    height = max(
        _CANVAS_MULTIPLE,
        int(round((width / source_ratio) / _CANVAS_MULTIPLE)) * _CANVAS_MULTIPLE,
    )
    if width * height > max_pixels:
        width -= _CANVAS_MULTIPLE
    return width, height


def image_dimensions(path: str | Path) -> tuple[int, int]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"keyframe image not found: {path}")
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)
        return image.size


def load_image_tensor(path: str | Path) -> torch.Tensor:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"keyframe image not found: {path}")
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        pixels = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(pixels.copy()).unsqueeze(0)
