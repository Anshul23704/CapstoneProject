from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Set, Tuple
import numpy as np
import cv2
from ultralytics import YOLO
import supervision as sv

import config

logger = logging.getLogger(__name__)

BBox = Tuple[int, int, int, int]


@dataclass
class DetectionConfig:
    model_path: str = config.VEHICLE_MODEL_PATH
    conf_threshold: float = config.DETECTION_CONF_THRESHOLD
    iou_threshold: float = config.DETECTION_IOU_THRESHOLD
    track_buffer: int = config.TRACK_BUFFER
    match_threshold: float = config.TRACK_MATCH_THRESHOLD
    device: str = config.DEVICE
    max_frame_size: int = config.MAX_FRAME_SIZE
    # FIX: this used to be hardcoded to {2, 5, 7} inside _detect() while
    # main_pipeline.py independently hardcoded {2, 3, 7} (including
    # motorcycles) — the two silently disagreed on what counts as a
    # "vehicle". Both now read from config.VEHICLE_CLASS_IDS.
    vehicle_class_ids: Set[int] = field(default_factory=lambda: set(config.VEHICLE_CLASS_IDS))


class DetectionTrackingStage:
    """
    Stage 2 — Detection & Tracking (main GPU thread).

    Per the architecture diagram (Slide 9), this is the ONLY stage that runs
    on the main thread's GPU path during real-time ingestion. Plate detection
    and OCR happen later, asynchronously, in Stage 6's worker threads — this
    stage must stay cheap per-frame.
    """

    def __init__(self, config: DetectionConfig) -> None:
        self.cfg = config

        self._model = YOLO(config.model_path)
        self._model.to(config.device)

        self._model.to(config.device)

    def process(self, frame: np.ndarray) -> Dict[int, BBox]:
        orig_h, orig_w = frame.shape[:2]
        detect_frame, scale = self._maybe_resize(frame)

        tracked_detections = self._detect(detect_frame)

        if tracked_detections is None:
            return {}

        logger.debug("Tracker IDs: %s", tracked_detections.tracker_id)

        return self._to_dict(tracked_detections, scale=scale, orig_w=orig_w, orig_h=orig_h)

    def _maybe_resize(self, frame: np.ndarray) -> Tuple[np.ndarray, float]:
        h, w = frame.shape[:2]
        if w > self.cfg.max_frame_size:
            scale = self.cfg.max_frame_size / float(w)
            frame = cv2.resize(
                frame,
                (self.cfg.max_frame_size, int(h * scale)),
            )
            return frame, scale
        return frame, 1.0

    def _detect(self, frame: np.ndarray):
        try:
            results = self._model.track(
                frame,
                conf=self.cfg.conf_threshold,
                iou=self.cfg.iou_threshold,
                tracker="botsort.yaml",
                persist=True,
                verbose=False,
            )[0]

            detections = sv.Detections.from_ultralytics(results)

            mask = np.isin(detections.class_id, list(self.cfg.vehicle_class_ids))
            detections = detections[mask]
            return detections

        except RuntimeError as exc:
            logger.error("Detection failed: %s", exc)
            return None

    @staticmethod
    def _to_dict(
        detections: sv.Detections,
        scale: float = 1.0,
        orig_w: int = 99999,
        orig_h: int = 99999,
    ) -> Dict[int, BBox]:
        result: Dict[int, BBox] = {}

        if detections.tracker_id is None:
            return result

        inv_scale = 1.0 / scale if scale > 0 else 1.0

        for bbox, tid in zip(detections.xyxy, detections.tracker_id):
            if tid is None:
                continue

            x1, y1, x2, y2 = bbox
            rx1 = max(0, min(orig_w, int(round(x1 * inv_scale))))
            ry1 = max(0, min(orig_h, int(round(y1 * inv_scale))))
            rx2 = max(0, min(orig_w, int(round(x2 * inv_scale))))
            ry2 = max(0, min(orig_h, int(round(y2 * inv_scale))))

            if rx2 > rx1 and ry2 > ry1:
                result[int(tid)] = (rx1, ry1, rx2, ry2)

        return result
