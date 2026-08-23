"""
stage6_worker_pool.py — Worker Pool (OCR only).

ARCHITECTURE CHANGE vs previous version
─────────────────────────────────────────
Plate detection no longer happens here. It used to run per-worker, per
selected frame, on a small padded per-vehicle ROI — meaning up to
TOP_K_FRAMES separate detector calls per vehicle, each at a different
effective resolution, AND every worker thread loading its own full copy of
the plate-detector model. Two rounds of tuning that in isolation (imgsz,
IoA slack) both backfired, because the fundamental problem wasn't the
threshold — it was that the detector was seeing small, inconsistent,
context-free crops instead of full frames.

Plate detection now runs exactly once per frame, full-frame, in
plate_detection.py (Stage 2.5), and Stage 3 already matched each vehicle's
frames to a plate_bbox (full-frame coordinates) via IoA at buffering time.
By the time a job reaches this stage, every selected frame either already
has a known plate_bbox or it doesn't — this stage's only job is: crop it,
enhance it, OCR it. No detector model is loaded here anymore.

This stage also now forwards a raw per-frame detection record for EVERY
frame with a matched plate_bbox — regardless of whether OCR produced valid
text — via RecognitionResult.raw_detections. main_pipeline.py uses this to
give Stage 10 a box to draw on every frame with a real detection, not only
frames whose OCR happened to fully validate.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from enum import Enum, auto
from queue import Queue
from typing import List, Optional, Tuple

import cv2
import easyocr
import numpy as np

import config
from plate_utils import (
    clean_ocr_text,
    enhance_plate_crop,
    laplacian_sharpness,
    license_complies_format,
    format_license,
)

logger = logging.getLogger(__name__)

BBox = Tuple[int, int, int, int]


class _DiagnosticCounters:
    """
    DIAGNOSTIC (temporary): tallies, across all worker threads, how many
    frames survive each step of the OCR pipeline. Printed once by
    WorkerPoolStage.shutdown().
    """
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.frames_seen        = 0
        self.skipped_blurry     = 0
        self.no_plate_bbox      = 0   # frame selected but Stage 3 never matched a plate to it
        self.crop_empty         = 0   # matched plate_bbox produced an empty/invalid crop
        self.ocr_empty          = 0   # frames where EasyOCR returned zero detections
        self.ocr_format_rejected = 0  # every OCR detection too short/long to plausibly be a plate
        self.ocr_forwarded_plausible = 0  # forwarded to fusion without being independently valid
        self.ocr_format_passed  = 0   # already a fully valid plate on its own

    def bump(self, field: str) -> None:
        with self._lock:
            setattr(self, field, getattr(self, field) + 1)

    def report(self) -> str:
        return (
            "\n" + "=" * 60 +
            "\n  PLATE PIPELINE DIAGNOSTIC (temporary instrumentation)\n" +
            "=" * 60 +
            f"\n  Frames attempted (selected) : {self.frames_seen}"
            f"\n  Skipped (too blurry)        : {self.skipped_blurry}"
            f"\n  No plate_bbox from Stage 3  : {self.no_plate_bbox}"
            f"\n  Empty/invalid crop          : {self.crop_empty}"
            f"\n  OCR returned 0 detections   : {self.ocr_empty}"
            f"\n  OCR too short/long (noise)  : {self.ocr_format_rejected}"
            f"\n  OCR forwarded to fusion     : {self.ocr_forwarded_plausible}  (plausible length, not independently valid)"
            f"\n  OCR fully valid on its own  : {self.ocr_format_passed}"
            "\n" + "=" * 60
        )


DIAG = _DiagnosticCounters()

# DIAGNOSTIC (temporary): saves a capped sample of the actual crops fed to
# EasyOCR at each failure point, plus the raw text EasyOCR returned.
import os as _os
_DEBUG_DIR = _os.path.join(_os.getcwd(), "debug_plate_crops")
_DEBUG_MAX_PER_CATEGORY = 25
_DEBUG_LOCK = threading.Lock()
_DEBUG_COUNTS: dict = {}


def _debug_save(category: str, img: np.ndarray, tag: str = "") -> None:
    with _DEBUG_LOCK:
        if _DEBUG_COUNTS.get(category, 0) >= _DEBUG_MAX_PER_CATEGORY:
            return
        _DEBUG_COUNTS[category] = _DEBUG_COUNTS.get(category, 0) + 1
        n = _DEBUG_COUNTS[category]
    try:
        cat_dir = _os.path.join(_DEBUG_DIR, category)
        _os.makedirs(cat_dir, exist_ok=True)
        fname = f"{n:03d}{('_' + tag) if tag else ''}.png"
        cv2.imwrite(_os.path.join(cat_dir, fname), img)
    except Exception:
        logger.exception("debug_save failed for category=%s", category)


class RecognitionStatus(Enum):
    SUCCESS  = auto()
    FAILED   = auto()
    NO_PLATE = auto()


@dataclass
class RecognitionResult:
    job_id:         str
    track_id:       int
    plate_text:     str
    confidence:     float
    status:         RecognitionStatus
    plate_bbox:     Optional[BBox] = None
    best_crop_path: str = ""
    # Per-frame OCR readings forwarded to Stage 7 temporal fusion AND used by
    # main_pipeline.py to emit one CSV row per successful frame. Each entry
    # is (formatted_text, conf, vehicle_bbox, plate_bbox_full, frame_idx) —
    # text/conf already validated & formatted; only frames with plausible OCR
    # text are included here.
    frame_readings: tuple = ()
    # NEW: every frame with a matched plate_bbox, regardless of OCR outcome
    # — (vehicle_bbox, plate_bbox, frame_idx, ocr_text_or_empty, ocr_conf).
    # Lets Stage 10 draw a box on every frame with a real detection, not
    # only frames whose OCR happened to fully validate.
    raw_detections: tuple = ()


@dataclass
class WorkerConfig:
    plate_padding:    int   = config.PLATE_PADDING
    use_gpu:          bool  = (config.DEVICE == "cuda")
    upscale_factor:   float = config.PLATE_UPSCALE_FACTOR
    blur_threshold:   float = config.BLUR_THRESHOLD


class Worker(threading.Thread):

    def __init__(self, job_q: Queue, result_q: Queue, config: WorkerConfig):
        super().__init__(daemon=True)
        self.job_q    = job_q
        self.result_q = result_q
        self.cfg      = config

        logger.info("Worker init: loading EasyOCR (gpu=%s)", config.use_gpu)
        self.ocr = easyocr.Reader(['en'], gpu=config.use_gpu, verbose=False)

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
        For each selected frame:
          1. Skip the frame outright if its vehicle crop is too blurry to be
             worth an OCR call (Slide 5/17 commitment).
          2. Use the plate_bbox Stage 3 already matched (full-frame coords)
             — no detection happens here anymore.
          3. Crop it out of the stored ROI, enhance it (CLAHE + sharpen +
             upscale), OCR it with strict format validation + correction.
          4. Record it in raw_detections (always, if a plate_bbox existed)
             and in frame_readings (only if OCR produced plausible text).
        """
        best_text  = ""
        best_conf  = 0.0
        best_bbox: Optional[BBox] = None
        frame_readings: List[Tuple[str, float, BBox, Optional[BBox], int]] = []
        raw_detections: List[Tuple[BBox, Optional[BBox], int, str, float]] = []

        for frame_entry in job.selected_frames:

            if laplacian_sharpness(frame_entry.crop) < self.cfg.blur_threshold:
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

            plate_ready = enhance_plate_crop(plate_crop, upscale_factor=self.cfg.upscale_factor)

            text, conf = self._run_ocr_validated(
                plate_ready, tag=f"t{job.track_id}_f{frame_entry.frame_idx}"
            )

            raw_detections.append((
                frame_entry.bbox, frame_entry.plate_bbox, frame_entry.frame_idx,
                text, conf,
            ))

            if text:
                frame_readings.append(
                    (text, conf, frame_entry.bbox, frame_entry.plate_bbox, frame_entry.frame_idx)
                )
                if conf > best_conf:
                    best_conf = conf
                    best_text = text
                    best_bbox = frame_entry.plate_bbox

        status = RecognitionStatus.SUCCESS if best_text else RecognitionStatus.NO_PLATE

        return RecognitionResult(
            job_id         = job.job_id,
            track_id       = job.track_id,
            plate_text     = best_text,
            confidence     = best_conf,
            status         = status,
            plate_bbox     = best_bbox,
            frame_readings = tuple(frame_readings),
            raw_detections = tuple(raw_detections),
        )

    # ── Cropping (no detection — plate_bbox is already known) ─────────────────

    @staticmethod
    def _crop_plate(frame_entry) -> Optional[np.ndarray]:
        """
        plate_bbox is in FULL-FRAME coordinates (set by Stage 3 from Stage
        2.5's full-frame detection). frame_entry.full_frame is the padded
        ROI stored for this frame, offset by frame_entry.roi_offset — so we
        translate plate_bbox into ROI-local coordinates once, here, and
        nowhere else in the pipeline.
        """
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

    # ── OCR with strict validation ─────────────────────────────────────────────

    _MIN_PLATE_LEN = 8
    _MAX_PLATE_LEN = 10

    def _run_ocr_validated(self, plate_img: np.ndarray, tag: str = "") -> Tuple[str, float]:
        """
        Iterates every EasyOCR detection, applies OCR-confusion correction
        (format_license), and returns the best candidate. Correct first,
        then return the first ALREADY-valid corrected candidate immediately;
        if none of this frame's detections corrects to a fully valid plate,
        still forward the highest-confidence length-plausible candidate
        (8-10 chars) so it can contribute to Stage 7 fusion. Only true noise
        (too short/too long to plausibly be a plate) is dropped here — final
        validity is enforced on the FUSED result by Stage 7, not per-frame.
        """
        detections = self.ocr.readtext(
            plate_img,
            detail          = 1,
            paragraph       = False,
            width_ths       = 0.9,
            contrast_ths    = 0.05,
            adjust_contrast = 0.6,
            min_size        = 10,
            allowlist       = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
        )

        if not detections:
            DIAG.bump("ocr_empty")
            _debug_save("ocr_empty", plate_img, tag=tag)
            return "", 0.0

        plausible: List[Tuple[str, float]] = []
        raw_seen: List[str] = []
        for (_bbox, text, conf) in detections:
            clean = clean_ocr_text(text)
            if not clean:
                continue
            corrected = format_license(clean)
            raw_seen.append(f"'{text}'->'{corrected}'(conf={conf:.2f})")

            if license_complies_format(corrected):
                logger.debug("OCR accepted (fully valid): '%s' -> '%s' conf=%.3f",
                              clean, corrected, conf)
                DIAG.bump("ocr_format_passed")
                return corrected, float(conf)

            if self._MIN_PLATE_LEN <= len(corrected) <= self._MAX_PLATE_LEN:
                plausible.append((corrected, float(conf)))

        if plausible:
            DIAG.bump("ocr_forwarded_plausible")
            best_text, best_conf = max(plausible, key=lambda x: x[1])
            logger.debug(
                "OCR forwarded (length-plausible, not independently valid): '%s' conf=%.3f",
                best_text, best_conf,
            )
            return best_text, best_conf

        if raw_seen:
            DIAG.bump("ocr_format_rejected")
            logger.info("OCR true-reject (too short/long) [%s]: %s", tag, "; ".join(raw_seen))
            _debug_save("ocr_format_rejected", plate_img, tag=tag)
        else:
            DIAG.bump("ocr_empty")
            _debug_save("ocr_empty", plate_img, tag=tag)
        return "", 0.0


class WorkerPoolStage:

    def __init__(self, job_q: Queue, result_q: Queue,
                 num_workers: int = config.NUM_WORKERS, config: WorkerConfig = None):
        self.job_q    = job_q
        self.result_q = result_q
        cfg           = config or WorkerConfig()
        self.workers  = [Worker(job_q, result_q, cfg) for _ in range(num_workers)]

    def start(self) -> None:
        for w in self.workers:
            w.start()

    def shutdown(self) -> None:
        for _ in self.workers:
            self.job_q.put(None)
        for w in self.workers:
            w.join()
        logger.info("All workers shut down cleanly.")
        logger.info(DIAG.report())
