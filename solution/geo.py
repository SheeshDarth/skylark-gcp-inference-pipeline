"""Pixel-to-geographic conversion using raster affine and CRS metadata."""

from __future__ import annotations


def pixel_to_wgs84(dataset, pixel_x: float, pixel_y: float) -> tuple[float, float]:
    if dataset.crs is None:
        raise ValueError("raster has no CRS; cannot produce WGS84 coordinates")
    try:
        from pyproj import Transformer
    except ImportError as exc:  # pragma: no cover - exercised in the container
        raise RuntimeError("pyproj is required for WGS84 conversion") from exc

    source_x, source_y = dataset.transform * (float(pixel_x), float(pixel_y))
    transformer = Transformer.from_crs(dataset.crs, "EPSG:4326", always_xy=True)
    longitude, latitude = transformer.transform(source_x, source_y)
    return float(longitude), float(latitude)

