"""Input manifest parsing with strict, path-relative scene resolution."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


class ManifestError(ValueError):
    """Raised when an input manifest violates the published contract."""


@dataclass(frozen=True)
class Scene:
    scene_id: str
    raster_path: Path


@dataclass(frozen=True)
class Manifest:
    path: Path
    schema_version: str
    scenes: tuple[Scene, ...]


def _require_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"manifest field {field!r} must be a non-empty string")
    return value


def load_manifest(path: str | Path) -> Manifest:
    manifest_path = Path(path).expanduser().resolve()
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ManifestError(f"manifest does not exist: {manifest_path}") from exc
    except json.JSONDecodeError as exc:
        raise ManifestError(f"manifest is not valid JSON: {manifest_path}: {exc}") from exc

    if not isinstance(payload, dict):
        raise ManifestError("manifest root must be an object")

    schema_version = _require_string(payload.get("schema_version", "1.0"), "schema_version")
    raw_scenes = payload.get("scenes")
    if not isinstance(raw_scenes, list):
        raise ManifestError("manifest field 'scenes' must be an array")

    scenes: list[Scene] = []
    seen: set[str] = set()
    for index, raw_scene in enumerate(raw_scenes):
        if not isinstance(raw_scene, dict):
            raise ManifestError(f"scene {index} must be an object")
        scene_id = _require_string(raw_scene.get("scene_id"), f"scenes[{index}].scene_id")
        raster_rel = _require_string(raw_scene.get("raster_path"), f"scenes[{index}].raster_path")
        if scene_id in seen:
            raise ManifestError(f"duplicate scene_id in manifest: {scene_id}")
        seen.add(scene_id)
        scenes.append(Scene(scene_id=scene_id, raster_path=manifest_path.parent / raster_rel))

    return Manifest(path=manifest_path, schema_version=schema_version, scenes=tuple(scenes))

