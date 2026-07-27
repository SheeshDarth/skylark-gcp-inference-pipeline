"""CLI entrypoint for bounded-memory GCP inference."""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np

from .geo import pixel_to_wgs84
from .manifest import ManifestError, load_manifest
from .model import ModelContractError, OnnxRunner
from .postprocess import RasterDetection, deduplicate_detections, nms_candidates
from .raster import describe_dataset, iter_tiles


LOGGER = logging.getLogger("skylark.infer")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run manifest-driven GCP inference")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-spec", default=None)
    parser.add_argument("--confidence-threshold", type=float, default=0.35)
    parser.add_argument("--nms-iou", type=float, default=0.50)
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument("--dedup-radius", type=float, default=16.0)
    return parser.parse_args(argv)


def _map_candidate(candidate, tile) -> RasterDetection | None:
    if not (0.0 <= candidate.keypoint_x < tile.valid_width and 0.0 <= candidate.keypoint_y < tile.valid_height):
        return None
    mask_x = min(tile.tile_size - 1, max(0, int(math.floor(candidate.keypoint_x))))
    mask_y = min(tile.tile_size - 1, max(0, int(math.floor(candidate.keypoint_y))))
    if not bool(tile.valid_mask[mask_y, mask_x]):
        return None
    return RasterDetection(
        pixel_x=float(tile.col_off + candidate.keypoint_x),
        pixel_y=float(tile.row_off + candidate.keypoint_y),
        confidence=float(candidate.confidence),
        box_x=float(tile.col_off + candidate.center_x - candidate.width / 2.0),
        box_y=float(tile.row_off + candidate.center_y - candidate.height / 2.0),
        box_width=float(candidate.width),
        box_height=float(candidate.height),
    )


def _validate_scene_output(scene_id: str, detections: list[dict[str, float]], width: int, height: int) -> None:
    for index, detection in enumerate(detections):
        required = ("pixel_x", "pixel_y", "longitude", "latitude", "confidence")
        if any(field not in detection for field in required):
            raise ValueError(f"{scene_id} detection {index} is missing a required field")
        if not all(math.isfinite(float(detection[field])) for field in required):
            raise ValueError(f"{scene_id} detection {index} contains a non-finite value")
        if not (0.0 <= detection["pixel_x"] < width and 0.0 <= detection["pixel_y"] < height):
            raise ValueError(f"{scene_id} detection {index} is outside raster bounds")
        if not (0.0 <= detection["confidence"] <= 1.0):
            raise ValueError(f"{scene_id} detection {index} has invalid confidence")


def run_inference(args: argparse.Namespace) -> dict[str, Any]:
    if not 0.0 <= args.confidence_threshold <= 1.0:
        raise ValueError("confidence threshold must be in [0,1]")
    if not 0.0 <= args.nms_iou <= 1.0:
        raise ValueError("NMS IoU must be in [0,1]")
    manifest = load_manifest(args.manifest)
    runner = OnnxRunner(args.model, args.model_spec)
    LOGGER.info("model contract: %s", runner.describe())

    scene_outputs: list[dict[str, Any]] = []
    try:
        import rasterio
    except ImportError as exc:  # pragma: no cover - exercised in the container
        raise RuntimeError("rasterio is required for inference") from exc

    for scene in manifest.scenes:
        if not scene.raster_path.exists():
            raise FileNotFoundError(f"raster for scene {scene.scene_id} does not exist: {scene.raster_path}")
        detections: list[RasterDetection] = []
        with rasterio.open(scene.raster_path) as dataset:
            info = describe_dataset(dataset)
            if info.crs is None:
                raise ValueError(f"scene {scene.scene_id} has no CRS")
            LOGGER.info("scene=%s metadata=%s", scene.scene_id, info)
            for tile in iter_tiles(dataset, tile_size=640, overlap=args.overlap):
                if not bool(tile.valid_mask.any()):
                    continue
                tensor = tile.rgb[np.newaxis, ...].astype(np.float32, copy=False)
                candidates = runner.predict(tensor)
                candidates = [item for item in candidates if item.confidence >= args.confidence_threshold]
                candidates = nms_candidates(candidates, args.nms_iou)
                for candidate in candidates:
                    mapped = _map_candidate(candidate, tile)
                    if mapped is not None:
                        detections.append(mapped)

            detections = deduplicate_detections(detections, args.dedup_radius)
            scene_predictions: list[dict[str, float]] = []
            for detection in detections:
                longitude, latitude = pixel_to_wgs84(dataset, detection.pixel_x, detection.pixel_y)
                scene_predictions.append(
                    {
                        "pixel_x": detection.pixel_x,
                        "pixel_y": detection.pixel_y,
                        "longitude": longitude,
                        "latitude": latitude,
                        "confidence": detection.confidence,
                    }
                )
            _validate_scene_output(scene.scene_id, scene_predictions, dataset.width, dataset.height)
            scene_outputs.append({"scene_id": scene.scene_id, "detections": scene_predictions})

    return {"schema_version": "1.0", "scenes": scene_outputs}


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args(argv)
    try:
        payload = run_inference(args)
        output_path = Path(args.output).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    except (ManifestError, ModelContractError, OSError, ValueError, RuntimeError) as exc:
        LOGGER.error("inference failed: %s", exc)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
