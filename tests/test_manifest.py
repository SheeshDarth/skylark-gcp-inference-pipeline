from pathlib import Path

from solution.manifest import load_manifest


def test_manifest_resolves_raster_paths_relative_to_manifest(tmp_path: Path):
    manifest_path = tmp_path / "data" / "manifest.json"
    manifest_path.parent.mkdir()
    manifest_path.write_text(
        '{"schema_version":"1.0","scenes":[{"scene_id":"sample_001","raster_path":"rasters/sample.tif"}]}',
        encoding="utf-8",
    )

    manifest = load_manifest(manifest_path)

    assert manifest.scenes[0].scene_id == "sample_001"
    assert manifest.scenes[0].raster_path == manifest_path.parent / "rasters" / "sample.tif"

