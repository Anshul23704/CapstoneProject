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
    crop: np.ndarray        # vehicle crop (small, for sharpness ranking)
    bbox: BBox              # vehicle bbox in full-frame coordinates
    frame_idx: int
    timestamp: float
    full_frame: np.ndarray = None  # full BGR frame — used by worker to run plate detector at full resolution


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

   

    def update(
        self,
        track_map: Dict[int, BBox],
        frame: np.ndarray,
        frame_idx: int,
        timestamp: float,
    ) -> List[VehicleBuffer]:
       
        
        ready_for_finalization: List[VehicleBuffer] = []
        self._new_track_ids.clear()

        
        for tid, bbox in track_map.items():
            buf = self._get_or_create(tid, frame_idx, timestamp)

            if not buf.is_active():
                continue

            crop = self._safe_crop(frame, bbox)
            if crop.size == 0:
                continue

            self._append(buf, crop, bbox, frame_idx, timestamp, frame)

        
        for tid, buf in list(self._registry.items()):
            if not buf.is_active():
                continue

            frames_since_seen = frame_idx - buf.last_seen_frame

            # 2a. Force-finalize after collecting enough frames
            if len(buf.frames) >= self.cfg.force_finalize_at:
                self._finalize(buf, "force_finalize_at reached")
                ready_for_finalization.append(buf.to_snapshot())
                continue

            # 2b. Time-out: vehicle has not appeared for too long
            if frames_since_seen > self.cfg.timeout_frames:
                # Only worth keeping if we actually collected something
                if buf.frames:
                    self._finalize(buf, "timeout")
                    ready_for_finalization.append(buf.to_snapshot())
                else:
                    # No crops at all – just discard silently
                    buf.state = VehicleBufferState.DONE
                    self._finalised_ids.add(tid)

        return ready_for_finalization

    @property
    def active_count(self) -> int:
        """Number of vehicle buffers currently in ACTIVE state."""
        return sum(1 for b in self._registry.values() if b.is_active())

    @property
    def finalized_count(self) -> int:
        """Total number of buffers that have been finalized so far."""
        return len(self._finalised_ids)

    

    def _finalize(self, buf: VehicleBuffer, reason: str) -> None:
        buf.state = VehicleBufferState.DONE
        buf.finalize_reason = reason
        self._finalised_ids.add(buf.track_id)

    def _get_or_create(self, tid: int, frame_idx: int, timestamp: float) -> VehicleBuffer:
        if tid not in self._registry:
            buf = VehicleBuffer(
                track_id=tid,
                first_seen=frame_idx,
                last_seen_frame=frame_idx,
                last_seen_time=timestamp,
            )
            self._registry[tid] = buf
            self._new_track_ids.append(tid)

        return self._registry[tid]

    def _append(
        self,
        buf: VehicleBuffer,
        crop: np.ndarray,
        bbox: BBox,
        frame_idx: int,
        timestamp: float,
        full_frame: np.ndarray,
    ) -> None:
        buf.frames.append(FrameEntry(crop, bbox, frame_idx, timestamp, full_frame=full_frame))
        buf.last_seen_frame = frame_idx
        buf.last_seen_time = timestamp

        
        if len(buf.frames) > self.cfg.max_buffer_size:
            buf.frames.pop(0)

    @staticmethod
    def _safe_crop(frame: np.ndarray, bbox: BBox) -> np.ndarray:
        h, w = frame.shape[:2]
        x1 = max(0, bbox[0])
        y1 = max(0, bbox[1])
        x2 = min(w, bbox[2])
        y2 = min(h, bbox[3])

        if x2 <= x1 or y2 <= y1:
            return np.empty((0, 0, 3))

        return frame[y1:y2, x1:x2].copy()