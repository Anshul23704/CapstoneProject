from __future__ import annotations
import logging
import uuid
from dataclasses import dataclass
from queue import Queue, Full
from typing import Optional, Set
import numpy as np
import cv2

from stage3_active_buffering import FrameEntry
from stage4_vehicle_finalization import FinalizedVehicle

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProcessingJob:
    job_id:             str
    track_id:           int
    selected_frames:    tuple      
    track_duration:     int
    avg_bbox_area:      float
    finalize_reason:    str
    low_quality:        bool
    low_diversity:      bool
    possible_id_switch: bool
    creation_time:      float


@dataclass
class JobCreationConfig:
    top_k: int = 5
    blur_threshold: float = 80.0    # Laplacian variance below this → low quality


class JobCreationStage:

    def __init__(self, queue: Queue, config: JobCreationConfig = JobCreationConfig()):
        self.queue = queue
        self.cfg = config
        self._dispatched_ids: Set[int] = set()

    def dispatch(self, fv: FinalizedVehicle) -> Optional[ProcessingJob]:

        if fv.track_id in self._dispatched_ids:
            return None

        # Sort all frames by sharpness descending, keep top-k
        ranked = sorted(
            fv.frames,
            key=lambda f: self._sharpness(f.crop),
            reverse=True,
        )
        selected = ranked[: self.cfg.top_k]

        # low_quality: True when even the sharpest selected frame is blurry.
        # Previously this was always hardcoded False — the blur_threshold
        # config was declared but never applied.
        best_sharpness = self._sharpness(selected[0].crop) if selected else 0.0
        low_quality = best_sharpness < self.cfg.blur_threshold

        job = ProcessingJob(
            job_id=str(uuid.uuid4()),
            track_id=fv.track_id,
            selected_frames=tuple(selected),
            track_duration=fv.track_duration,
            avg_bbox_area=fv.avg_bbox_area,
            finalize_reason=fv.finalize_reason,
            low_quality=low_quality,
            low_diversity=fv.low_diversity,
            possible_id_switch=fv.possible_id_switch,
            creation_time=fv.creation_time,
        )

        try:
            self.queue.put(job)
            self._dispatched_ids.add(fv.track_id)
            logger.debug(
                "Dispatched job=%s track=%s low_quality=%s",
                job.job_id, job.track_id, low_quality,
            )
            return job
        except Full:
            logger.warning("Job queue full — dropped track_id=%s", fv.track_id)
            return None

    @staticmethod
    def _sharpness(crop: np.ndarray) -> float:
        """Laplacian variance — higher means sharper."""
        if crop is None or crop.size == 0:
            return 0.0
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())