"""Deterministic raster-to-model preprocessing."""

from __future__ import annotations

from typing import Any

import numpy as np


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

