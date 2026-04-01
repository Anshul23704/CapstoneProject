from __future__ import annotations
import copy
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import cv2

BBox = Tuple[int, int, int, int]

# ── Padding added around the vehicle bbox when saving the ROI.
# Large enough to capture a plate that sticks out of the bbox slightly,
# small enough to not balloon memory.
_ROI_PAD = 80   # pixels


class VehicleBufferState(Enum):
    ACTIVE     = auto()
    FINALIZING = auto()
    DONE       = auto()


@dataclass
class FrameEntry:
    crop:       np.ndarray   # vehicle crop — used for sharpness ranking
    bbox:       BBox         # vehicle bbox in ORIGINAL full-frame coords
    frame_idx:  int
    timestamp:  float
    # BUG FIX (memory): instead of storing the raw full frame (~6 MB each)
    # we now store a padded ROI around the vehicle (~0.1-0.3 MB each).
    # roi_offset records the (x0, y0) of the ROI so that bbox coords inside
    # the ROI can be reconstructed by the worker.
    full_frame:  Optional[np.ndarray] = None   # padded ROI
    roi_offset:  Tuple[int, int]      = (0, 0) # (x_offset, y_offset) in original frame


@dataclass
class VehicleBuffer:
    track_id:        int
    state:           VehicleBufferState = VehicleBufferState.ACTIVE
    frames:          List[FrameEntry]   = field(default_factory=list)
    first_seen:      int   = 0
    last_seen_frame: int   = 0
    last_seen_time:  float = 0.0
    finalize_reason: Optional[str] = None

    def is_active(self) -> bool:
        return self.state == VehicleBufferState.ACTIVE

    def to_snapshot(self):
        snap        = copy.copy(self)
        snap.frames = list(self.frames)
        return snap


@dataclass
class BufferingConfig:
    max_buffer_size:  int = 20
    timeout_frames:   int = 30
    force_finalize_at: int = 20
    max_vehicles:     int = 20
    roi_polygon:      Optional[np.ndarray] = None


class ActiveBufferingStage:

    def __init__(self, config: BufferingConfig):
        self.cfg             = config
        self._registry:       Dict[int, VehicleBuffer] = {}
        self._finalised_ids:  Set[int]                 = set()
        self._new_track_ids:  List[int]                = []

    # ── Public API ────────────────────────────────────────────────────────────

    def update(
        self,
        track_map:  Dict[int, BBox],
        frame:      np.ndarray,
        frame_idx:  int,
        timestamp:  float,
    ) -> List[VehicleBuffer]:

        ready_for_finalization: List[VehicleBuffer] = []
        self._new_track_ids.clear()

        # ── 1. Ingest detections ──────────────────────────────────────────────
        for tid, bbox in track_map.items():
            buf = self._get_or_create(tid, frame_idx, timestamp)

            if not buf.is_active():
                continue

            crop = self._safe_crop(frame, bbox)
            if crop.size == 0:
                continue

            # BUG FIX: extract a padded ROI instead of the full frame
            roi, offset = self._safe_roi(frame, bbox, pad=_ROI_PAD)
            self._append(buf, crop, bbox, frame_idx, timestamp, roi, offset)

        # ── 2. Finalization sweep ─────────────────────────────────────────────
        for tid, buf in list(self._registry.items()):
            if not buf.is_active():
                continue

            frames_since_seen = frame_idx - buf.last_seen_frame

            if len(buf.frames) >= self.cfg.force_finalize_at:
                self._finalize(buf, "force_finalize_at reached")
                ready_for_finalization.append(buf.to_snapshot())
                del self._registry[tid]
                continue

            if frames_since_seen > self.cfg.timeout_frames:
                if buf.frames:
                    self._finalize(buf, "timeout")
                    ready_for_finalization.append(buf.to_snapshot())
                    del self._registry[tid]  
                else:
                    buf.state = VehicleBufferState.DONE
                    self._finalised_ids.add(tid)

        return ready_for_finalization

    @property
    def active_count(self) -> int:
        return sum(1 for b in self._registry.values() if b.is_active())

    @property
    def finalized_count(self) -> int:
        return len(self._finalised_ids)

    # ── Internals ─────────────────────────────────────────────────────────────

    def _finalize(self, buf: VehicleBuffer, reason: str) -> None:
        buf.state           = VehicleBufferState.DONE
        buf.finalize_reason = reason
        self._finalised_ids.add(buf.track_id)

    def _get_or_create(self, tid: int, frame_idx: int, timestamp: float) -> VehicleBuffer:
        if tid not in self._registry:
            buf = VehicleBuffer(
                track_id        = tid,
                first_seen      = frame_idx,
                last_seen_frame = frame_idx,
                last_seen_time  = timestamp,
            )
            self._registry[tid] = buf
            self._new_track_ids.append(tid)
        return self._registry[tid]

    def _append(
        self,
        buf:        VehicleBuffer,
        crop:       np.ndarray,
        bbox:       BBox,
        frame_idx:  int,
        timestamp:  float,
        roi:        np.ndarray,
        roi_offset: Tuple[int, int],
    ) -> None:
        buf.frames.append(
            FrameEntry(
                crop       = crop,
                bbox       = bbox,
                frame_idx  = frame_idx,
                timestamp  = timestamp,
                full_frame = roi,
                roi_offset = roi_offset,
            )
        )
        buf.last_seen_frame = frame_idx
        buf.last_seen_time  = timestamp

        if len(buf.frames) > self.cfg.max_buffer_size:
            buf.frames.pop(0)

    # ── Static helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _safe_crop(frame: np.ndarray, bbox: BBox) -> np.ndarray:
        h, w = frame.shape[:2]
        x1 = max(0, bbox[0]);  y1 = max(0, bbox[1])
        x2 = min(w, bbox[2]);  y2 = min(h, bbox[3])
        if x2 <= x1 or y2 <= y1:
            return np.empty((0, 0, 3))
        return frame[y1:y2, x1:x2].copy()

    @staticmethod
    def _safe_roi(
        frame: np.ndarray,
        bbox:  BBox,
        pad:   int = _ROI_PAD,
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Return a padded region around bbox and its (x0,y0) offset."""
        h, w = frame.shape[:2]
        x1 = max(0, bbox[0] - pad);  y1 = max(0, bbox[1] - pad)
        x2 = min(w, bbox[2] + pad);  y2 = min(h, bbox[3] + pad)
        return frame[y1:y2, x1:x2].copy(), (x1, y1)
