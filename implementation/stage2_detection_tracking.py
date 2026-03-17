from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np
import cv2
from ultralytics import YOLO
import supervision as sv

logger = logging.getLogger(__name__)

BBox = Tuple[int, int, int, int]


@dataclass
class DetectionConfig:
    model_path: str = "D:\\Sem6_Subjects\\Capstone\\implementation\\models\\best.pt"
    conf_threshold: float = 0.50
    iou_threshold: float = 0.45
    track_buffer: int = 30
    match_threshold: float = 0.80
    device: str = "cuda"
    max_frame_size: int = 1920


class DetectionTrackingStage:

    def __init__(self, config: DetectionConfig) -> None:
        self.cfg = config

        self._model = YOLO(config.model_path)
        self._model.to(config.device)

        self._tracker = sv.ByteTrack(
            track_activation_threshold=config.conf_threshold,
            lost_track_buffer=config.track_buffer,
            minimum_matching_threshold=config.match_threshold,
        )

    def process(self, frame: np.ndarray) -> Dict[int, BBox]:

        frame = self._maybe_resize(frame)

        raw_detections = self._detect(frame)

        if raw_detections is None:
            return {}

        tracked = self._tracker.update_with_detections(raw_detections)

        return self._to_dict(tracked)

    def _maybe_resize(self, frame: np.ndarray) -> np.ndarray:

        h, w = frame.shape[:2]

        if w > self.cfg.max_frame_size:

            scale = self.cfg.max_frame_size / w

            frame = cv2.resize(
                frame,
                (self.cfg.max_frame_size, int(h * scale)),
            )

        return frame

    def _detect(self, frame: np.ndarray):

        try:

            results = self._model(
                frame,
                conf=self.cfg.conf_threshold,
                iou=self.cfg.iou_threshold,
                verbose=False,
            )[0]

            return sv.Detections.from_ultralytics(results)

        except RuntimeError as exc:
            logger.error("Detection failed: %s", exc)
            return None

    @staticmethod
    def _to_dict(detections: sv.Detections):

        result = {}

        if detections.tracker_id is None:
            return result

        for bbox, tid in zip(detections.xyxy, detections.tracker_id):

            if tid is None:
                continue

            x1, y1, x2, y2 = map(int, bbox)

            result[int(tid)] = (x1, y1, x2, y2)

        return result