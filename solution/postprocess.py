"""Deterministic suppression and cross-window duplicate fusion."""

from __future__ import annotations

from dataclasses import dataclass
import math

from .model import Candidate


@dataclass(frozen=True)
class RasterDetection:
    pixel_x: float
    pixel_y: float
    confidence: float
    box_x: float
    box_y: float
    box_width: float
    box_height: float


def _iou(left: Candidate, right: Candidate) -> float:
    left_x1 = left.center_x - left.width / 2.0
    left_y1 = left.center_y - left.height / 2.0
    left_x2 = left.center_x + left.width / 2.0
    left_y2 = left.center_y + left.height / 2.0
    right_x1 = right.center_x - right.width / 2.0
    right_y1 = right.center_y - right.height / 2.0
    right_x2 = right.center_x + right.width / 2.0
    right_y2 = right.center_y + right.height / 2.0
    inter_w = max(0.0, min(left_x2, right_x2) - max(left_x1, right_x1))
    inter_h = max(0.0, min(left_y2, right_y2) - max(left_y1, right_y1))
    intersection = inter_w * inter_h
    left_area = max(0.0, left.width) * max(0.0, left.height)
    right_area = max(0.0, right.width) * max(0.0, right.height)
    union = left_area + right_area - intersection
    return intersection / union if union > 0.0 else 0.0


def nms_candidates(candidates: list[Candidate], iou_threshold: float = 0.5) -> list[Candidate]:
    ordered = sorted(candidates, key=lambda item: item.confidence, reverse=True)
    kept: list[Candidate] = []
    for candidate in ordered:
        if all(_iou(candidate, previous) <= iou_threshold for previous in kept):
            kept.append(candidate)
    return kept


def deduplicate_detections(detections: list[RasterDetection], radius: float = 16.0) -> list[RasterDetection]:
    if radius <= 0.0:
        raise ValueError("deduplication radius must be positive")
    ordered = sorted(detections, key=lambda item: item.confidence, reverse=True)
    grid: dict[tuple[int, int], list[RasterDetection]] = {}
    kept: list[RasterDetection] = []

    for detection in ordered:
        cell = (math.floor(detection.pixel_x / radius), math.floor(detection.pixel_y / radius))
        duplicate = False
        for gx in range(cell[0] - 1, cell[0] + 2):
            for gy in range(cell[1] - 1, cell[1] + 2):
                for existing in grid.get((gx, gy), []):
                    distance = math.hypot(detection.pixel_x - existing.pixel_x, detection.pixel_y - existing.pixel_y)
                    if distance <= radius:
                        duplicate = True
                        break
                if duplicate:
                    break
            if duplicate:
                break
        if not duplicate:
            kept.append(detection)
            grid.setdefault(cell, []).append(detection)
    return kept

