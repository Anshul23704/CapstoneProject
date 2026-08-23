from __future__ import annotations
import logging
import uuid
from dataclasses import dataclass
from queue import Queue, Full
from typing import List, Optional, Set
import numpy as np

import config
from plate_utils import laplacian_sharpness
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
    top_k: int = config.TOP_K_FRAMES
    blur_threshold: float = config.BLUR_THRESHOLD    # Laplacian variance below this → low quality


class JobCreationStage:

    def __init__(self, queue: Queue, config: JobCreationConfig = JobCreationConfig()):
        self.queue = queue
        self.cfg = config
        self._dispatched_ids: Set[int] = set()

    def dispatch(self, fv: FinalizedVehicle) -> Optional[ProcessingJob]:

        if fv.track_id in self._dispatched_ids:
            return None

        # FIX: Stage 2.5 now runs plate detection once per frame, full-frame,
        # during buffering — so by the time a vehicle reaches this stage we
        # already know exactly which of its frames have a matched plate_bbox
        # (frame_entry.plate_bbox is not None) and which don't. Selecting by
        # sharpness alone (the old behaviour) routinely sent frames with NO
        # detected plate at all to OCR, wasting Stage 6 work on crops that
        # were never going to produce a reading. We now rank
        # (has_plate, plate_conf, sharpness) so frames with a real detection
        # are always preferred, and only fall back to plate-less frames if a
        # vehicle's ENTIRE track never got one (so it still gets *a* result
        # rather than being silently dropped).
        def _rank_key(f: FrameEntry):
            has_plate = f.plate_bbox is not None
            if has_plate and f.plate_bbox is not None:
                pw = max(1, f.plate_bbox[2] - f.plate_bbox[0])
                ph = max(1, f.plate_bbox[3] - f.plate_bbox[1])
                plate_area = float(pw * ph)
            else:
                plate_area = 1.0
            sharpness = self._sharpness(f.crop)
            # Rank by plate presence first, then combined resolution (plate area) & sharpness score
            score = (np.sqrt(plate_area) * sharpness * (f.plate_conf if has_plate else 0.5))
            return (has_plate, score)

        scored = sorted(fv.frames, key=_rank_key, reverse=True)

        plated_frames: List[FrameEntry] = [f for f in scored if f.plate_bbox is not None]

        # Within the plated frames, still respect the blur gate (Slide 5/17's
        # "actively drop blurry/occluded frames before OCR" commitment) —
        # fall back to the full plated set if every plated frame is blurry,
        # and fall back further to the unplated set only if there was no
        # plate detection anywhere in the track.
        sharp_plated = [f for f in plated_frames if self._sharpness(f.crop) >= self.cfg.blur_threshold]
        if sharp_plated:
            pool = sharp_plated
        elif plated_frames:
            pool = plated_frames
        else:
            pool = scored  # no plate ever detected on this track — best-effort fallback

        selected = pool[: self.cfg.top_k]

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
                "Dispatched job=%s track=%s frames_selected=%d/%d low_quality=%s",
                job.job_id, job.track_id, len(selected), len(fv.frames), low_quality,
            )
            return job
        except Full:
            logger.warning("Job queue full — dropped track_id=%s", fv.track_id)
            return None

    @staticmethod
    def _sharpness(crop: np.ndarray) -> float:
        return laplacian_sharpness(crop)
