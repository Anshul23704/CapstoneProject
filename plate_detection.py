"""
plate_detection.py — Stage 2.5: Full-Frame Plate Detection.

WHY THIS STAGE EXISTS
──────────────────────
Every previous version of this pipeline ran the plate detector LATE and
LOCALLY: inside Stage 6's worker threads, on a small padded crop around a
single vehicle, independently for every one of that vehicle's top-k
selected frames (up to 20 separate detector calls per vehicle). This had
three compounding costs:

1. Each crop was resized/letterboxed to the detector's imgsz independently
   of the others, so small crops and large crops landed at different
   effective resolutions relative to the plate — the exact opposite of
   consistent, model's-full-potential detection.
2. The detector never saw the plate in the context it was almost certainly
   trained on (full traffic scenes, multiple plates per image, real scale
   and perspective) — it saw an isolated, padded, single-vehicle crop.
3. It re-ran the (expensive) detector many times over frames of the same
   vehicle instead of once per frame — wasted GPU time that could instead
   go toward a single higher-resolution full-frame pass.

This stage replaces all of that: the plate detector runs EXACTLY ONCE per
incoming frame, on the full frame, at a fixed resolution matched to the
frame size (no shrink, no per-crop guessing). Every plate box it finds
that frame — for every vehicle in view — comes out of one detector call.
Association to a specific vehicle track happens downstream in Stage 3,
entirely in full-frame coordinates, so there is no ROI-offset arithmetic
anywhere in the pipeline anymore (the class of bug that caused the last
two runs' regressions).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from ultralytics import YOLO

import config

logger = logging.getLogger(__name__)

BBox = Tuple[int, int, int, int]


@dataclass(frozen=True)
class PlateDetection:
    bbox: BBox      # full-frame coordinates
    conf: float


@dataclass
class PlateDetectionConfig:
    model_path:        str   = config.PLATE_MODEL_PATH
    conf_threshold:    float = config.PLATE_CONF_THRESHOLD
    device:            str   = config.DEVICE
    imgsz:             int   = config.PLATE_DETECT_IMGSZ
    min_plate_area:    int   = config.MIN_PLATE_AREA
    min_aspect_ratio:  float = config.MIN_PLATE_ASPECT_RATIO
    max_aspect_ratio:  float = config.MAX_PLATE_ASPECT_RATIO


class PlateDetectionStage:
    """
    Stage 2.5 — full-frame plate detection, one YOLO call per frame.

    Runs on the main thread right after Stage 2's vehicle detection, using
    the SAME frame (no re-crop, no re-resize). Loaded once, not once per
    worker thread — previously each of Stage 6's N worker threads loaded
    its own copy of this exact model, which was pure duplicated GPU memory
    for a model that only ever needs one instance since detection is no
    longer split across threads.
    """

    def __init__(self, config: PlateDetectionConfig = PlateDetectionConfig()) -> None:
        self.cfg = config
        logger.info("PlateDetectionStage: loading plate detector from %s", config.model_path)
        self._model = YOLO(config.model_path)
        self._model.to(config.device)

    def process(self, frame: np.ndarray) -> List[PlateDetection]:
        """
        Full-frame plate detection. imgsz is fixed to config.PLATE_DETECT_IMGSZ
        (sized to comfortably cover the ingestion frame's long side — see
        config.py) so every frame is detected at the same, known resolution
        instead of a per-crop-dependent one.
        """
        results = self._model(
            frame,
            conf=self.cfg.conf_threshold,
            imgsz=self.cfg.imgsz,
            verbose=False,
        )[0]

        detections: List[PlateDetection] = []
        if results.boxes is None or len(results.boxes) == 0:
            return detections

        for i in range(len(results.boxes)):
            x1, y1, x2, y2 = map(int, results.boxes.xyxy[i])
            conf = float(results.boxes.conf[i])
            w = max(1, x2 - x1)
            h = max(1, y2 - y1)
            area = w * h
            if area < self.cfg.min_plate_area:
                continue

            aspect_ratio = w / float(h)
            if not (self.cfg.min_aspect_ratio <= aspect_ratio <= self.cfg.max_aspect_ratio):
                # Rejects square QR codes (~1.0), tall vertical ads (<1.0), or ultra-wide strips
                continue

            detections.append(PlateDetection(bbox=(x1, y1, x2, y2), conf=conf))

        return detections
