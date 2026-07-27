import pytest
import rasterio
from rasterio.io import MemoryFile
from rasterio.transform import from_origin

from solution.geo import pixel_to_wgs84


def test_continuous_pixel_coordinate_uses_pixel_center_convention():
    profile = {
        "driver": "GTiff",
        "height": 2,
        "width": 2,
        "count": 1,
        "dtype": "uint8",
        "crs": "EPSG:32643",
        "transform": from_origin(500000.0, 2000000.0, 2.0, 2.0),
    }
    with MemoryFile() as memory:
        with memory.open(**profile) as dataset:
            longitude, latitude = pixel_to_wgs84(dataset, 0.5, 0.5)
            source_x, source_y = dataset.transform * (0.5, 0.5)
            assert source_x == pytest.approx(500001.0)
            assert source_y == pytest.approx(1999999.0)
            assert longitude != 0.0
            assert latitude != 0.0


def test_missing_crs_is_rejected():
    profile = {
        "driver": "GTiff",
        "height": 1,
        "width": 1,
        "count": 1,
        "dtype": "uint8",
        "transform": from_origin(0.0, 1.0, 1.0, 1.0),
    }
    with MemoryFile() as memory:
        with memory.open(**profile) as dataset:
            with pytest.raises(ValueError, match="no CRS"):
                pixel_to_wgs84(dataset, 0.5, 0.5)

