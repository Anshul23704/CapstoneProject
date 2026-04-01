from __future__ import annotations
import hashlib
import logging
import time
from dataclasses import dataclass
from typing import Optional, Set
import numpy as np
import cv2

from stage3_active_buffering import VehicleBuffer, FrameEntry

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FinalizedVehicle:
    track_id:           int
    frames:             tuple
    first_seen_frame:   int
    last_seen_frame:    int
    track_duration:     int
    avg_bbox_area:      float
    finalize_reason:    str
    low_diversity:      bool
    possible_id_switch: bool
    creation_time:      float


@dataclass
class FinalizationConfig:
    min_frames:            int   = 1
    area_std_threshold:    float = 5000.0
    diversity_ratio_limit: float = 0.95


class VehicleFinalizationStage:

    def __init__(self, config: FinalizationConfig = FinalizationConfig()) -> None:
        self.cfg = config
        self._finalized_ids: Set[int] = set()

    def process(self, buf: VehicleBuffer) -> Optional[FinalizedVehicle]:

        if buf.track_id in self._finalized_ids:
            return None   

        visible_frames = [f for f in buf.frames if f.crop.size > 0]

        if len(visible_frames) < self.cfg.min_frames:
            return None

        seen_idxs = set()
        unique_frames = []
        for fe in visible_frames:
            if fe.frame_idx not in seen_idxs:
                unique_frames.append(fe)
                seen_idxs.add(fe.frame_idx)

        low_diversity = self._check_low_diversity(unique_frames)

        areas = [(f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]) for f in unique_frames]
        avg_area = float(np.mean(areas))
        area_std = float(np.std(areas))
        possible_id_switch = area_std > self.cfg.area_std_threshold

        duration = max(0, buf.last_seen_frame - buf.first_seen)

        self._finalized_ids.add(buf.track_id)

        return FinalizedVehicle(
            track_id=buf.track_id,
            frames=tuple(unique_frames),
            first_seen_frame=buf.first_seen,
            last_seen_frame=buf.last_seen_frame,
            track_duration=duration,
            avg_bbox_area=avg_area,
            finalize_reason=buf.finalize_reason or "unknown",
            low_diversity=low_diversity,
            possible_id_switch=possible_id_switch,
            creation_time=time.time(),
        )

    def _avg_hash(self, crop):
        small = cv2.resize(crop, (8, 8))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        mean = gray.mean()
        bits = (gray >= mean).flatten()
        return hashlib.md5(bits.tobytes()).hexdigest()

    def _check_low_diversity(self, frames):
        if len(frames) < 2:
            return True
        hashes = [self._avg_hash(f.crop) for f in frames]
        diversity = len(set(hashes)) / len(hashes)
        return (1.0 - diversity) > self.cfg.diversity_ratio_limit