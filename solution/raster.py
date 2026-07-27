"""Bounded-memory GeoTIFF/COG metadata and window access."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from .preprocess import _to_float_unit


@dataclass(frozen=True)
class RasterInfo:
    width: int
    height: int
    count: int
    dtypes: tuple[str, ...]
    crs: str | None
    transform: tuple[float, float, float, float, float, float]
    resolution: tuple[float, float]
    nodata: float | int | None
    color_interpretation: tuple[str, ...]
    block_shapes: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class WindowTile:
    rgb: np.ndarray
    valid_mask: np.ndarray
    row_off: int
    col_off: int
    valid_height: int
    valid_width: int
    tile_size: int


def _require_rasterio():
    try:
        import rasterio  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised in the container
        raise RuntimeError("rasterio is required for GeoTIFF inference") from exc


def describe_dataset(dataset) -> RasterInfo:
    crs = dataset.crs.to_string() if dataset.crs else None
    transform = dataset.transform
    colorinterp = tuple(str(value) for value in dataset.colorinterp)
    block_shapes = tuple(tuple(shape) for shape in dataset.block_shapes)
    return RasterInfo(
        width=int(dataset.width),
        height=int(dataset.height),
        count=int(dataset.count),
        dtypes=tuple(str(value) for value in dataset.dtypes),
        crs=crs,
        transform=(transform.a, transform.b, transform.c, transform.d, transform.e, transform.f),
        resolution=(float(dataset.res[0]), float(dataset.res[1])),
        nodata=dataset.nodata,
        color_interpretation=colorinterp,
        block_shapes=block_shapes,
    )


def _starts(length: int, tile_size: int, overlap: int) -> list[int]:
    if length <= tile_size:
        return [0]
    stride = tile_size - overlap
    if stride <= 0:
        raise ValueError("overlap must be smaller than tile_size")
    starts = list(range(0, length - tile_size + 1, stride))
    final = length - tile_size
    if starts[-1] != final:
        starts.append(final)
    return starts


def iter_window_specs(width: int, height: int, tile_size: int = 640, overlap: int = 128):
    """Yield complete raster coverage as (row, col, valid_height, valid_width)."""

    if width <= 0 or height <= 0:
        raise ValueError("raster dimensions must be positive")
    for row in _starts(height, tile_size, overlap):
        for col in _starts(width, tile_size, overlap):
            yield row, col, min(tile_size, height - row), min(tile_size, width - col)


def _rgb_indexes(dataset) -> list[int]:
    """Choose RGB bands from inspected color interpretation when available."""

    names = [str(value).lower() for value in dataset.colorinterp]
    wanted = ["red", "green", "blue"]
    indexes: list[int] = []
    for name in wanted:
        if name in names:
            indexes.append(names.index(name) + 1)
    if len(indexes) == 3:
        return indexes
    if dataset.count >= 3:
        return [1, 2, 3]
    if dataset.count == 2:
        return [1, 2, 1]
    if dataset.count == 1:
        return [1, 1, 1]
    raise ValueError("raster contains no readable bands")


def read_tile(dataset, row_off: int, col_off: int, valid_height: int, valid_width: int, tile_size: int = 640) -> WindowTile:
    _require_rasterio()
    from rasterio.windows import Window

    if not (0 <= row_off < dataset.height and 0 <= col_off < dataset.width):
        raise ValueError("window origin is outside the raster")
    window = Window(col_off, row_off, valid_width, valid_height)
    raw = dataset.read(indexes=_rgb_indexes(dataset), window=window, masked=True)
    masked = np.ma.getmaskarray(raw)
    valid_mask = ~np.any(masked, axis=0)
    rgb = _to_float_unit(raw)

    padded = np.zeros((3, tile_size, tile_size), dtype=np.float32)
    padded[:, :valid_height, :valid_width] = rgb
    padded_mask = np.zeros((tile_size, tile_size), dtype=bool)
    padded_mask[:valid_height, :valid_width] = valid_mask
    return WindowTile(
        rgb=padded,
        valid_mask=padded_mask,
        row_off=row_off,
        col_off=col_off,
        valid_height=valid_height,
        valid_width=valid_width,
        tile_size=tile_size,
    )


def iter_tiles(dataset, tile_size: int = 640, overlap: int = 128) -> Iterator[WindowTile]:
    for row, col, height, width in iter_window_specs(dataset.width, dataset.height, tile_size, overlap):
        yield read_tile(dataset, row, col, height, width, tile_size)

