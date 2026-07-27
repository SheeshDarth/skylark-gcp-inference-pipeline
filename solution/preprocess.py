"""Deterministic raster-to-model preprocessing.

The supplied model consumes a centered 640x640 letterbox.  Keeping the
letterbox metadata alongside the tensor is important: model coordinates must
be mapped back through the same scale and padding before source-window
offsets are applied.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class LetterboxMeta:
    """The forward transform applied to one source window."""

    scale: float
    pad_x: int
    pad_y: int
    source_width: int
    source_height: int
    target_size: int


@dataclass(frozen=True)
class PreprocessedTile:
    tensor: np.ndarray
    letterbox: LetterboxMeta


def _to_float_unit(array: Any) -> np.ndarray:
    """Convert a 3-band array to float32 [0, 1] using explicit dtype rules.

    The assignment requires source inspection rather than filename assumptions.
    Integer rasters use their dtype range; common uint8/float images use 255
    when values exceed one; other float data uses its finite tile range as a
    documented fallback. The selected policy is logged by the caller.
    """

    values = np.asanyarray(array)
    if np.ma.isMaskedArray(values):
        values = values.filled(0)
    values = np.asarray(values)
    if values.ndim != 3 or values.shape[0] != 3:
        raise ValueError(f"expected [3,H,W] raster data, got {values.shape}")

    result = values.astype(np.float32, copy=False)
    if np.issubdtype(values.dtype, np.integer):
        info = np.iinfo(values.dtype)
        scale = float(info.max) if info.max > 0 else 1.0
        result = result / scale
    else:
        finite = result[np.isfinite(result)]
        if finite.size == 0:
            result = np.zeros_like(result, dtype=np.float32)
        else:
            low = float(finite.min())
            high = float(finite.max())
            if low >= 0.0 and high <= 1.0:
                pass
            elif low >= 0.0 and high <= 255.0:
                result = result / 255.0
            elif high > low:
                result = (result - low) / (high - low)
            else:
                result = np.zeros_like(result, dtype=np.float32)

    return np.clip(np.nan_to_num(result, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)


def _resize_bilinear(chw: np.ndarray, height: int, width: int) -> np.ndarray:
    """Resize a [C,H,W] array with pixel-center bilinear interpolation."""

    channels, source_height, source_width = chw.shape
    if (source_height, source_width) == (height, width):
        return chw.astype(np.float32, copy=True)

    y = (np.arange(height, dtype=np.float32) + 0.5) * source_height / height - 0.5
    x = (np.arange(width, dtype=np.float32) + 0.5) * source_width / width - 0.5
    y = np.clip(y, 0.0, source_height - 1.0)
    x = np.clip(x, 0.0, source_width - 1.0)
    y0 = np.floor(y).astype(np.intp)
    x0 = np.floor(x).astype(np.intp)
    y1 = np.minimum(y0 + 1, source_height - 1)
    x1 = np.minimum(x0 + 1, source_width - 1)
    wy = (y - y0).reshape(1, height, 1)
    wx = (x - x0).reshape(1, 1, width)

    top_left = chw[:, y0[:, None], x0[None, :]]
    top_right = chw[:, y0[:, None], x1[None, :]]
    bottom_left = chw[:, y1[:, None], x0[None, :]]
    bottom_right = chw[:, y1[:, None], x1[None, :]]
    top = top_left * (1.0 - wx) + top_right * wx
    bottom = bottom_left * (1.0 - wx) + bottom_right * wx
    return (top * (1.0 - wy) + bottom * wy).astype(np.float32, copy=False)


def letterbox_rgb(rgb: np.ndarray, target_size: int = 640) -> tuple[np.ndarray, LetterboxMeta]:
    """Center-letterbox a [3,H,W] RGB array into the model input size."""

    values = np.asarray(rgb, dtype=np.float32)
    if values.ndim != 3 or values.shape[0] != 3:
        raise ValueError(f"expected [3,H,W] RGB data, got {values.shape}")
    source_height, source_width = values.shape[1:]
    if source_height <= 0 or source_width <= 0 or target_size <= 0:
        raise ValueError("source and target dimensions must be positive")

    scale = min(target_size / source_width, target_size / source_height)
    resized_width = max(1, int(round(source_width * scale)))
    resized_height = max(1, int(round(source_height * scale)))
    delta_x = target_size - resized_width
    delta_y = target_size - resized_height
    pad_x = int(round(delta_x / 2.0 - 0.1))
    pad_y = int(round(delta_y / 2.0 - 0.1))
    resized = _resize_bilinear(values, resized_height, resized_width)

    output = np.full((3, target_size, target_size), 114.0 / 255.0, dtype=np.float32)
    output[:, pad_y : pad_y + resized_height, pad_x : pad_x + resized_width] = resized
    return output, LetterboxMeta(
        scale=float(scale),
        pad_x=pad_x,
        pad_y=pad_y,
        source_width=source_width,
        source_height=source_height,
        target_size=target_size,
    )


def preprocess_tile(tile) -> PreprocessedTile:
    """Convert a raster window to the verified model tensor contract."""

    if tile.valid_height <= 0 or tile.valid_width <= 0:
        raise ValueError("tile has no valid source pixels")
    source = tile.rgb[:, : tile.valid_height, : tile.valid_width]
    tensor, letterbox = letterbox_rgb(source, tile.tile_size)
    return PreprocessedTile(tensor=tensor[np.newaxis, ...], letterbox=letterbox)
