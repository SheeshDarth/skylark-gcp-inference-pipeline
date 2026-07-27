from solution.model import Candidate
from solution.postprocess import RasterDetection, deduplicate_detections, nms_candidates


def test_nms_keeps_highest_confidence_overlapping_box():
    candidates = [
        Candidate(10, 10, 10, 10, 10, 10, 0.9),
        Candidate(10.5, 10.5, 10, 10, 10.5, 10.5, 0.8),
    ]
    kept = nms_candidates(candidates, iou_threshold=0.5)
    assert len(kept) == 1
    assert kept[0].confidence == 0.9


def test_cross_window_dedup_keeps_highest_confidence():
    detections = [
        RasterDetection(100, 100, 0.9, 95, 95, 10, 10),
        RasterDetection(103, 102, 0.8, 98, 97, 10, 10),
        RasterDetection(200, 200, 0.7, 195, 195, 10, 10),
    ]
    kept = deduplicate_detections(detections, radius=8)
    assert [(item.pixel_x, item.pixel_y) for item in kept] == [(100, 100), (200, 200)]

