"""
stage6_worker_pool.py — Worker Pool (TrOCR).

Integrated custom fine-tuned TrOCR license plate recognition engine.
Deprecates legacy CPU-heavy EasyOCR bilateral & adaptive contrast filters.
Directly ingests raw BGR crops in track-level batches for fast inference.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from enum import Enum, auto
from queue import Queue
from typing import List, Optional, Tuple

import cv2
import numpy as np

import config as app_config
config = app_config
from ocr_engine.trocr_engine import TrOCRInferenceEngine
from plate_utils import (
    clean_ocr_text,
    laplacian_sharpness,
    soft_format_indian_plate,
    plate_edge_density,
)

logger = logging.getLogger(__name__)

BBox = Tuple[int, int, int, int]


class _DiagnosticCounters:
    """Tallies, across all worker threads, how many frames survive each step."""
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.frames_seen         = 0
        self.skipped_blurry      = 0
        self.no_plate_bbox       = 0
        self.crop_empty          = 0
        self.ocr_empty           = 0
        self.ocr_format_rejected = 0
        self.ocr_forwarded       = 0

    def bump(self, field: str) -> None:
        with self._lock:
            setattr(self, field, getattr(self, field) + 1)

    def report(self) -> str:
        return (
            "\n" + "=" * 60 +
            "\n  TrOCR PIPELINE DIAGNOSTIC REPORT\n" +
            "=" * 60 +
            f"\n  Frames attempted (selected) : {self.frames_seen}"
            f"\n  Skipped (too blurry)        : {self.skipped_blurry}"
            f"\n  No plate_bbox from Stage 3  : {self.no_plate_bbox}"
            f"\n  Empty/invalid crop          : {self.crop_empty}"
            f"\n  OCR returned 0 detections   : {self.ocr_empty}"
            f"\n  OCR too short/low conf      : {self.ocr_format_rejected}"
            f"\n  OCR forwarded to fusion     : {self.ocr_forwarded}"
            "\n" + "=" * 60
        )


DIAG = _DiagnosticCounters()


class RecognitionStatus(Enum):
    SUCCESS         = auto()
    PARTIAL_SUCCESS = auto()
    FAILED          = auto()
    NO_PLATE        = auto()


@dataclass
class RecognitionResult:
    job_id:                   str
    track_id:                 int
    plate_text:               str
    confidence:               float
    status:                   RecognitionStatus
    plate_bbox:               Optional[BBox] = None
    best_crop_path:           str = ""
    # Per-frame OCR readings forwarded to Stage 7 temporal fusion:
    # tuple of (formatted_text, conf, vehicle_bbox, plate_bbox_full, frame_idx)
    frame_readings:           tuple = ()
    frame_partials:           tuple = ()
    frame_readings_bilateral: tuple = ()
    frame_readings_adaptive:  tuple = ()
    # Raw bounding boxes for Stage 10 video rendering
    raw_detections:           tuple = ()
    winner_branch:            str = "trocr"
    avg_sharpness:            float = 0.0


@dataclass
class WorkerConfig:
    plate_padding:       int   = config.PLATE_PADDING
    use_gpu:             bool  = (config.DEVICE in ("cuda", "mps"))
    upscale_factor:      float = config.PLATE_UPSCALE_FACTOR
    blur_threshold:      float = config.BLUR_THRESHOLD
    save_crops:          bool  = True
    crops_dir:           Optional[str] = None
    high_conf_crops_dir: Optional[str] = None
    partial_crops_dir:   Optional[str] = None
    high_conf_threshold: float = config.HIGH_CONF_CROP_THRESHOLD
    trocr_model_path:    str   = config.TROCR_MODEL_PATH
    min_conf:            float = config.TROCR_MIN_CONF_THRESHOLD


class Worker(threading.Thread):

    def __init__(self, job_q: Queue, result_q: Queue, config: WorkerConfig = None, config_obj: WorkerConfig = None):
        super().__init__(daemon=True)
        self.job_q    = job_q
        self.result_q = result_q
        self.cfg      = config or config_obj or WorkerConfig()

        logger.info("Worker init: loading TrOCR engine (model=%s, device=%s)",
                    self.cfg.trocr_model_path, app_config.DEVICE)
        self.engine = TrOCRInferenceEngine(
            model_path=self.cfg.trocr_model_path,
            device=app_config.DEVICE,
            use_fp16=True,
        )

    def run(self) -> None:
        while True:
            job = self.job_q.get()
            if job is None:
                self.job_q.task_done()
                break
            try:
                result = self._process(job)
                self.result_q.put(result)
            except Exception as exc:
                logger.error("Processing failed job=%s track=%s: %s",
                             job.job_id, job.track_id, exc, exc_info=True)
                self.result_q.put(RecognitionResult(
                    job_id=job.job_id, track_id=job.track_id,
                    plate_text="", confidence=0.0, status=RecognitionStatus.FAILED,
                ))
            finally:
                self.job_q.task_done()

    # ── Per-job processing ────────────────────────────────────────────────────

    def _process(self, job) -> RecognitionResult:
        """
        Runs track-level batched TrOCR inference across selected frames.
        1. Filters out frames that are too blurry or have invalid crops.
        2. Crops the plate with standard padding.
        3. Batches all crops for the vehicle track in a single forward pass.
        4. Validates and syntactically formats predicted plate strings.
        5. Saves crops and context images.
        """
        valid_candidates = []  # tuples of (frame_entry, plate_crop, sharpness)
        raw_detections: List[Tuple[BBox, Optional[BBox], int, str, float]] = []
        frame_readings: List[Tuple[str, float, BBox, Optional[BBox], int]] = []
        sharpness_values = []

        for frame_entry in job.selected_frames:
            sharpness = laplacian_sharpness(frame_entry.crop)
            if sharpness < self.cfg.blur_threshold:
                DIAG.bump("skipped_blurry")
                continue

            DIAG.bump("frames_seen")

            if frame_entry.plate_bbox is None:
                DIAG.bump("no_plate_bbox")
                continue

            plate_crop = self._crop_plate(frame_entry)
            if plate_crop is None or plate_crop.size == 0:
                DIAG.bump("crop_empty")
                continue

            if plate_edge_density(plate_crop) < 0.02:
                DIAG.bump("crop_empty")
                continue

            valid_candidates.append((frame_entry, plate_crop, sharpness))

        if valid_candidates:
            # Batched TrOCR GPU inference across all crops for this track
            crops_batch = [c for (_, c, _) in valid_candidates]
            batch_predictions = self.engine.predict_batch(crops_batch)

            for (frame_entry, plate_crop, sharpness), (pred_text, avg_conf, char_confs) in zip(valid_candidates, batch_predictions):
                clean = clean_ocr_text(pred_text)
                if clean.startswith("IND") and len(clean) > 10:
                    clean = clean[3:]

                formatted = soft_format_indian_plate(clean)

                # Plausibility gate: standard plate length >= 6 and confidence threshold
                if len(formatted) >= 6 and avg_conf >= self.cfg.min_conf:
                    DIAG.bump("ocr_forwarded")
                    sharpness_values.append(sharpness)
                    frame_readings.append(
                        (formatted, avg_conf, frame_entry.bbox, frame_entry.plate_bbox, frame_entry.frame_idx)
                    )
                    f_text, f_conf = formatted, avg_conf
                else:
                    if formatted:
                        DIAG.bump("ocr_format_rejected")
                    else:
                        DIAG.bump("ocr_empty")
                    f_text, f_conf = "", 0.0

                raw_detections.append((
                    frame_entry.bbox, frame_entry.plate_bbox, frame_entry.frame_idx,
                    f_text, f_conf,
                ))

                # Save enhanced crop and comparison context image
                if self.cfg.save_crops and (self.cfg.crops_dir or self.cfg.high_conf_crops_dir):
                    try:
                        target_dir = self.cfg.high_conf_crops_dir if (
                            f_text and f_conf >= self.cfg.high_conf_threshold and self.cfg.high_conf_crops_dir
                        ) else self.cfg.crops_dir

                        if target_dir:
                            os.makedirs(target_dir, exist_ok=True)
                            prefix = f"track_{job.track_id:03d}_frame_{frame_entry.frame_idx:04d}"
                            
                            # Save plate image (and backwards-compatible _plate_bilateral.png)
                            cv2.imwrite(os.path.join(target_dir, f"{prefix}_plate.png"), plate_crop)
                            cv2.imwrite(os.path.join(target_dir, f"{prefix}_plate_bilateral.png"), plate_crop)

                            roi = frame_entry.full_frame
                            if roi is not None and roi.size > 0:
                                context_img = roi.copy()
                                ox, oy = frame_entry.roi_offset
                                px1, py1, px2, py2 = frame_entry.plate_bbox
                                lx1 = max(0, px1 - ox)
                                ly1 = max(0, py1 - oy)
                                lx2 = min(context_img.shape[1], px2 - ox)
                                ly2 = min(context_img.shape[0], py2 - oy)
                                cv2.rectangle(context_img, (lx1, ly1), (lx2, ly2), (0, 255, 0), 2)

                                lbl = f"TrOCR: {f_text} ({f_conf:.2f})" if f_text else "No Plate Detected"
                                cv2.putText(context_img, lbl, (lx1, max(18, ly1 - 18)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
                                cv2.imwrite(os.path.join(target_dir, f"{prefix}_context.png"), context_img)
                    except Exception as exc:
                        logger.debug("Failed saving crop image: %s", exc)

        if frame_readings:
            status = RecognitionStatus.SUCCESS
            best_entry = max(frame_readings, key=lambda x: x[1])
            best_text, best_conf, _, best_bbox, _ = best_entry
        else:
            status = RecognitionStatus.NO_PLATE
            best_text, best_conf, best_bbox = "", 0.0, None

        avg_sharpness = float(sum(sharpness_values) / len(sharpness_values)) if sharpness_values else 0.0

        return RecognitionResult(
            job_id                   = job.job_id,
            track_id                 = job.track_id,
            plate_text               = best_text,
            confidence               = best_conf,
            status                   = status,
            plate_bbox               = best_bbox,
            frame_readings           = tuple(frame_readings),
            frame_partials           = (),
            frame_readings_bilateral = tuple(frame_readings),
            frame_readings_adaptive  = (),
            raw_detections           = tuple(raw_detections),
            winner_branch            = "trocr",
            avg_sharpness            = avg_sharpness,
        )

    # ── Cropping ─────────────────────────────────────────────────────────────

    @staticmethod
    def _crop_plate(frame_entry) -> Optional[np.ndarray]:
        roi = frame_entry.full_frame
        if roi is None or roi.size == 0:
            return None

        ox, oy = frame_entry.roi_offset
        px1, py1, px2, py2 = frame_entry.plate_bbox
        rh, rw = roi.shape[:2]

        lx1 = max(0, px1 - ox)
        ly1 = max(0, py1 - oy)
        lx2 = min(rw, px2 - ox)
        ly2 = min(rh, py2 - oy)

        if lx2 <= lx1 or ly2 <= ly1:
            return None

        p = config.PLATE_PADDING
        lx1 = max(0, lx1 - p); ly1 = max(0, ly1 - p)
        lx2 = min(rw, lx2 + p); ly2 = min(rh, ly2 + p)

        return roi[ly1:ly2, lx1:lx2].copy()


class WorkerPoolStage:

    def __init__(self, job_q: Queue, result_q: Queue,
                 num_workers: int = config.NUM_WORKERS, config: WorkerConfig = None, config_obj: WorkerConfig = None):
        self.job_q    = job_q
        self.result_q = result_q
        cfg           = config or config_obj or WorkerConfig()
        self.workers  = [Worker(job_q, result_q, config=cfg) for _ in range(num_workers)]

    def start(self) -> None:
        for w in self.workers:
            w.start()

    def shutdown(self) -> None:
        for _ in self.workers:
            self.job_q.put(None)
        for w in self.workers:
            w.join()
        logger.info("All TrOCR workers shut down cleanly.")
        logger.info(DIAG.report())
