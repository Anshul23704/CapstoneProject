from __future__ import annotations
import copy
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Sequence, Set, Tuple
import numpy as np
import cv2

import config
from plate_utils import expand_bbox, intersection_over_area
from plate_detection import PlateDetection

BBox = Tuple[int, int, int, int]

# Padding added around the vehicle bbox when saving the ROI. Large enough to
# capture a plate that sticks out of the bbox slightly, small enough not to
# balloon memory.
_ROI_PAD = config.ROI_PAD

_IOA_THRESHOLD = config.PLATE_VEHICLE_IOA_THRESHOLD
_ASSOC_EXPAND  = config.VEHICLE_BBOX_ASSOC_EXPAND


def _match_plate(
    vehicle_bbox: BBox,
    plate_detections: Sequence[PlateDetection],
    frame_shape,
) -> Tuple[Optional[BBox], float]:
    """
    Associate the best plate detection (already in full-frame coordinates,
    same as vehicle_bbox — see plate_detection.py) to this vehicle via IoA,
    exactly as Slide 5 specifies. Everything here is in ONE coordinate
    system; there is no ROI offset to get wrong.
    """
    if not plate_detections:
        return None, 0.0

    assoc_box = expand_bbox(vehicle_bbox, _ASSOC_EXPAND, frame_shape)

    best_bbox: Optional[BBox] = None
    best_conf = -1.0
    best_ioa  = 0.0
    for det in plate_detections:
        ioa = intersection_over_area(det.bbox, assoc_box)
        if ioa < _IOA_THRESHOLD:
            continue
        if det.conf > best_conf or (det.conf == best_conf and ioa > best_ioa):
            best_conf = det.conf
            best_bbox = det.bbox
            best_ioa  = ioa

    return best_bbox, (best_conf if best_bbox is not None else 0.0)


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
    # Instead of storing the raw full frame (~6 MB each) we store a padded
    # ROI around the vehicle (~0.1-0.3 MB each). roi_offset records the
    # (x0, y0) of the ROI so bbox coords inside it can be reconstructed.
    full_frame:  Optional[np.ndarray] = None   # padded ROI
    roi_offset:  Tuple[int, int]      = (0, 0) # (x_offset, y_offset) in original frame
    # Plate box matched to this vehicle THIS FRAME by Stage 2.5's full-frame
    # detection + Stage 3's IoA association — full-frame coordinates, same
    # space as `bbox`. None if no plate was detected/associated this frame
    # (that's expected on many frames; Stage 5 ranks selection toward frames
    # where this IS set).
    plate_bbox:  Optional[BBox] = None
    plate_conf:  float          = 0.0


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
    max_buffer_size:  Optional[int] = config.BUFFER_MAX_SIZE
    timeout_frames:   int = config.BUFFER_TIMEOUT_FRAMES
    force_finalize_at: Optional[int] = config.BUFFER_FORCE_FINALIZE_AT
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
        plate_detections: Sequence[PlateDetection] = (),
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

            roi, offset = self._safe_roi(frame, bbox, pad=_ROI_PAD)

            plate_bbox, plate_conf = _match_plate(bbox, plate_detections, frame.shape)

            self._append(buf, crop, bbox, frame_idx, timestamp, roi, offset,
                         plate_bbox, plate_conf)

        # ── 2. Finalization sweep ─────────────────────────────────────────────
        for tid, buf in list(self._registry.items()):
            if not buf.is_active():
                continue

            frames_since_seen = frame_idx - buf.last_seen_frame

            if self.cfg.force_finalize_at is not None and len(buf.frames) >= self.cfg.force_finalize_at:
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

    def flush_all(self) -> List[VehicleBuffer]:
        """
        FIX: previously there was no way to drain vehicles that were still
        ACTIVE (not yet timed out, not yet at force_finalize_at) when the
        video stream ended — main_pipeline.py would just exit its ingestion
        loop and every such vehicle's buffer was silently discarded, along
        with any plates on it. This is easy to hit in practice: any vehicle
        still visible in the last `timeout_frames` frames of the video, or
        any vehicle that never accumulates `force_finalize_at` frames,
        never gets a ready_for_finalization callback.

        Call this once after the ingestion loop ends to force-finalize
        every remaining active buffer that has at least one frame.
        """
        flushed: List[VehicleBuffer] = []
        for tid, buf in list(self._registry.items()):
            if not buf.is_active():
                continue
            if buf.frames:
                self._finalize(buf, "stream_end")
                flushed.append(buf.to_snapshot())
            else:
                buf.state = VehicleBufferState.DONE
                self._finalised_ids.add(tid)
            del self._registry[tid]
        return flushed

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
        plate_bbox: Optional[BBox] = None,
        plate_conf: float = 0.0,
    ) -> None:
        buf.frames.append(
            FrameEntry(
                crop       = crop,
                bbox       = bbox,
                frame_idx  = frame_idx,
                timestamp  = timestamp,
                full_frame = roi,
                roi_offset = roi_offset,
                plate_bbox = plate_bbox,
                plate_conf = plate_conf,
            )
        )
        buf.last_seen_frame = frame_idx
        buf.last_seen_time  = timestamp

        if self.cfg.max_buffer_size is not None and len(buf.frames) > self.cfg.max_buffer_size:
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
