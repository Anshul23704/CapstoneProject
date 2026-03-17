from __future__ import annotations
import copy
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import cv2

BBox = Tuple[int, int, int, int]


class VehicleBufferState(Enum):
    ACTIVE = auto()
    FINALIZING = auto()
    DONE = auto()


@dataclass
class FrameEntry:
    crop: np.ndarray
    bbox: BBox
    frame_idx: int
    timestamp: float


@dataclass
class VehicleBuffer:
    track_id: int
    state: VehicleBufferState = VehicleBufferState.ACTIVE
    frames: List[FrameEntry] = field(default_factory=list)
    first_seen: int = 0
    last_seen_frame: int = 0
    last_seen_time: float = 0.0
    finalize_reason: Optional[str] = None

    def is_active(self):
        return self.state == VehicleBufferState.ACTIVE

    def to_snapshot(self):
        snap = copy.copy(self)
        snap.frames = list(self.frames)
        return snap


@dataclass
class BufferingConfig:
    max_buffer_size: int = 30
    timeout_frames: int = 30
    force_finalize_at: int = 50
    max_vehicles: int = 64
    roi_polygon: Optional[np.ndarray] = None


class ActiveBufferingStage:

    def __init__(self, config: BufferingConfig):
        self.cfg = config
        self._registry: Dict[int, VehicleBuffer] = {}
        self._finalised_ids: Set[int] = set()
        self._new_track_ids: List[int] = []

    def update(self, track_map, frame, frame_idx, timestamp):

        ready_for_finalization = []
        self._new_track_ids.clear()

        for tid, bbox in track_map.items():

            buf = self._get_or_create(tid, frame_idx, timestamp)

            if not buf.is_active():
                continue

            crop = self._safe_crop(frame, bbox)

            if crop.size == 0:
                continue

            self._append(buf, crop, bbox, frame_idx, timestamp)

        return ready_for_finalization

    def _get_or_create(self, tid, frame_idx, timestamp):

        if tid not in self._registry:

            buf = VehicleBuffer(
                track_id=tid,
                first_seen=frame_idx,
                last_seen_frame=frame_idx,
                last_seen_time=timestamp,
            )

            self._registry[tid] = buf

        return self._registry[tid]

    def _append(self, buf, crop, bbox, frame_idx, timestamp):

        buf.frames.append(FrameEntry(crop, bbox, frame_idx, timestamp))

        buf.last_seen_frame = frame_idx
        buf.last_seen_time = timestamp

        if len(buf.frames) > self.cfg.max_buffer_size:
            buf.frames.pop(0)

    @staticmethod
    def _safe_crop(frame, bbox):

        h, w = frame.shape[:2]

        x1 = max(0, bbox[0])
        y1 = max(0, bbox[1])
        x2 = min(w, bbox[2])
        y2 = min(h, bbox[3])

        if x2 <= x1 or y2 <= y1:
            return np.empty((0, 0, 3))

        return frame[y1:y2, x1:x2].copy()