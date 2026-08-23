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
import os
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
    deskew_plate_crop,
    enhance_plate_crop_bilateral,
    enhance_plate_crop_adaptive,
    laplacian_sharpness,
    license_complies_format,
    format_license,
    soft_format_indian_plate,
    plate_edge_density,
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
        self.ocr_forwarded_bilateral = 0
        self.ocr_forwarded_adaptive = 0

    def bump(self, field: str) -> None:
        with self._lock:
            setattr(self, field, getattr(self, field) + 1)

    def report(self) -> str:
        return (
            "\n" + "=" * 60 +
            "\n  PLATE PIPELINE DIAGNOSTIC & BRANCH COMPARISON\n" +
            "=" * 60 +
            f"\n  Frames attempted (selected) : {self.frames_seen}"
            f"\n  Skipped (too blurry)        : {self.skipped_blurry}"
            f"\n  No plate_bbox from Stage 3  : {self.no_plate_bbox}"
            f"\n  Empty/invalid crop          : {self.crop_empty}"
            f"\n  OCR returned 0 detections   : {self.ocr_empty}"
            f"\n  OCR too short/long (noise)  : {self.ocr_format_rejected}"
            f"\n  OCR forwarded to fusion     : {self.ocr_forwarded_plausible}"
            f"\n    └─ Bilateral branch reads : {self.ocr_forwarded_bilateral}"
            f"\n    └─ Adaptive branch reads  : {self.ocr_forwarded_adaptive}"
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
    # Specific per-branch readings for side-by-side comparison:
    frame_readings_bilateral: tuple = ()
    frame_readings_adaptive:  tuple = ()
    # Raw bounding boxes for Stage 10 video rendering
    raw_detections:           tuple = ()
    winner_branch:            str = "none"


@dataclass
class WorkerConfig:
    plate_padding:    int   = config.PLATE_PADDING
    use_gpu:          bool  = (config.DEVICE == "cuda")
    upscale_factor:   float = config.PLATE_UPSCALE_FACTOR
    blur_threshold:   float = config.BLUR_THRESHOLD
    save_crops:       bool  = True
    crops_dir:        Optional[str] = None


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
          1. Skip if vehicle crop is too blurry.
          2. Use plate_bbox already matched in Stage 3.
          3. Crop plate and apply geometric perspective deskewing.
          4. Parallely execute Branch A (Bilateral) and Branch B (Adaptive Contrast).
          5. Save both enhanced crops (_plate_bilateral.png & _plate_adaptive.png)
             and the context image with comparison overlay.
          6. Run OCR on both and record comparative performance.
        """
        best_text  = ""
        best_conf  = 0.0
        best_bbox: Optional[BBox] = None
        frame_readings: List[Tuple[str, float, BBox, Optional[BBox], int]] = []
        readings_bilateral: List[Tuple[str, float, BBox, Optional[BBox], int]] = []
        readings_adaptive: List[Tuple[str, float, BBox, Optional[BBox], int]] = []
        raw_detections: List[Tuple[BBox, Optional[BBox], int, str, float]] = []

        bilateral_wins = 0
        adaptive_wins = 0

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

            if plate_edge_density(plate_crop) < 0.02:
                DIAG.bump("crop_empty")
                continue

            # 1. Perspective Deskewing
            plate_deskewed = deskew_plate_crop(plate_crop)

            # 2. Parallel Preprocessing Branches
            plate_bilateral = enhance_plate_crop_bilateral(
                plate_deskewed, upscale_factor=self.cfg.upscale_factor
            )
            plate_adaptive = enhance_plate_crop_adaptive(
                plate_deskewed, upscale_factor=self.cfg.upscale_factor
            )

            # 3. Parallel OCR Recognition
            tag_b = f"t{job.track_id}_f{frame_entry.frame_idx}_bilateral"
            tag_a = f"t{job.track_id}_f{frame_entry.frame_idx}_adaptive"
            text_b, conf_b = self._run_ocr_validated(plate_bilateral, tag=tag_b)
            text_a, conf_a = self._run_ocr_validated(plate_adaptive, tag=tag_a)

            if text_b:
                DIAG.bump("ocr_forwarded_bilateral")
                readings_bilateral.append(
                    (text_b, conf_b, frame_entry.bbox, frame_entry.plate_bbox, frame_entry.frame_idx)
                )
            if text_a:
                DIAG.bump("ocr_forwarded_adaptive")
                readings_adaptive.append(
                    (text_a, conf_a, frame_entry.bbox, frame_entry.plate_bbox, frame_entry.frame_idx)
                )

            # Select best candidate for frame
            if conf_b >= conf_a and text_b:
                frame_text, frame_conf = text_b, conf_b
                bilateral_wins += 1
            elif text_a:
                frame_text, frame_conf = text_a, conf_a
                adaptive_wins += 1
            elif text_b:
                frame_text, frame_conf = text_b, conf_b
                bilateral_wins += 1
            else:
                frame_text, frame_conf = "", 0.0

            # 4. Save both enhanced crops and comparison context image
            if self.cfg.save_crops and self.cfg.crops_dir:
                try:
                    os.makedirs(self.cfg.crops_dir, exist_ok=True)
                    prefix = f"track_{job.track_id:03d}_frame_{frame_entry.frame_idx:04d}"
                    cv2.imwrite(os.path.join(self.cfg.crops_dir, f"{prefix}_plate_bilateral.png"), plate_bilateral)
                    cv2.imwrite(os.path.join(self.cfg.crops_dir, f"{prefix}_plate_adaptive.png"), plate_adaptive)

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
                        
                        # Comparison Label Overlay
                        lbl_b = f"Bilateral: {text_b} ({conf_b:.2f})" if text_b else "Bilateral: --"
                        lbl_a = f"Adaptive:  {text_a} ({conf_a:.2f})" if text_a else "Adaptive:  --"
                        cv2.putText(context_img, lbl_b, (lx1, max(18, ly1 - 18)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
                        cv2.putText(context_img, lbl_a, (lx1, max(32, ly1 - 4)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1, cv2.LINE_AA)
                        cv2.imwrite(os.path.join(self.cfg.crops_dir, f"{prefix}_context.png"), context_img)
                except Exception as exc:
                    logger.debug("Failed saving crop image: %s", exc)

            raw_detections.append((
                frame_entry.bbox, frame_entry.plate_bbox, frame_entry.frame_idx,
                frame_text, frame_conf,
            ))

            if frame_text:
                frame_readings.append(
                    (frame_text, frame_conf, frame_entry.bbox, frame_entry.plate_bbox, frame_entry.frame_idx)
                )
                if frame_conf > best_conf:
                    best_conf = frame_conf
                    best_text = frame_text
                    best_bbox = frame_entry.plate_bbox

        status = RecognitionStatus.SUCCESS if best_text else RecognitionStatus.NO_PLATE
        winner = "bilateral" if bilateral_wins > adaptive_wins else ("adaptive" if adaptive_wins > bilateral_wins else ("tie" if bilateral_wins > 0 else "none"))

        return RecognitionResult(
            job_id                   = job.job_id,
            track_id                 = job.track_id,
            plate_text               = best_text,
            confidence               = best_conf,
            status                   = status,
            plate_bbox               = best_bbox,
            frame_readings           = tuple(frame_readings),
            frame_readings_bilateral = tuple(readings_bilateral),
            frame_readings_adaptive  = tuple(readings_adaptive),
            raw_detections           = tuple(raw_detections),
            winner_branch            = winner,
        )

    # ── Cropping (no detection — plate_bbox is already known) ─────────────────

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

    # ── OCR with strict validation & Dual-Pass Whitelisting ────────────────────

    _MIN_PLATE_LEN = 8
    _MAX_PLATE_LEN = 10

    def _run_ocr_validated(self, plate_img: np.ndarray, tag: str = "") -> Tuple[str, float]:
        """
        Iterates EasyOCR detections and supports:
        1. Single-line plate candidate extraction and correction.
        2. Multi-line / vertically stacked candidate merging (e.g., 2-line Indian plates).
        3. Dual-pass region split OCR (Left: State+Series, Right: 4-digit registration number).
        4. Automatic stripping of accidental 'IND' country identifier holograms.
        5. Soft positional heuristic formatting for Indian vehicle layout.
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

        plausible: List[Tuple[str, float]] = []
        raw_seen: List[str] = []

        if detections:
            # 1. Single detection candidates
            for (_bbox, text, conf) in detections:
                clean = clean_ocr_text(text)
                if not clean:
                    continue

                if clean.startswith("IND") and len(clean) > 10:
                    clean = clean[3:]

                corrected = soft_format_indian_plate(clean)
                raw_seen.append(f"'{text}'->'{corrected}'(conf={conf:.2f})")

                if self._MIN_PLATE_LEN <= len(corrected) <= self._MAX_PLATE_LEN:
                    plausible.append((corrected, float(conf)))

            # 2. Multi-line / vertically stacked detection merging (for 2-line plates)
            if len(detections) >= 2:
                sorted_dets = sorted(detections, key=lambda d: min(p[1] for p in d[0]))
                clean_parts = []
                confs = []
                for (_bbox, text, conf) in sorted_dets:
                    c = clean_ocr_text(text)
                    if not c or c == "IND":
                        continue
                    if c.startswith("IND") and len(c) > 4:
                        c = c[3:]
                    clean_parts.append(c)
                    confs.append(float(conf))

                if len(clean_parts) >= 2:
                    merged_text = "".join(clean_parts)
                    if merged_text.startswith("IND") and len(merged_text) > 10:
                        merged_text = merged_text[3:]

                    corrected_merged = soft_format_indian_plate(merged_text)
                    merged_conf = float(sum(confs) / len(confs)) if confs else 0.0
                    raw_seen.append(f"stacked[{'+'.join(clean_parts)}]->'{corrected_merged}'(conf={merged_conf:.2f})")

                    if self._MIN_PLATE_LEN <= len(corrected_merged) <= self._MAX_PLATE_LEN:
                        plausible.append((corrected_merged, merged_conf))

        # 3. Dual-Pass Region Split (for wide single-line plates)
        h, w = plate_img.shape[:2]
        if w / float(max(1, h)) >= 2.2:
            try:
                left_crop = plate_img[:, :int(w * 0.58)]
                right_crop = plate_img[:, int(w * 0.48):]

                left_res = self.ocr.readtext(
                    left_crop,
                    detail=1,
                    paragraph=False,
                    allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
                )
                right_res = self.ocr.readtext(
                    right_crop,
                    detail=1,
                    paragraph=False,
                    allowlist="0123456789",  # Strict numeric for registration digits
                )

                if left_res and right_res:
                    l_text = clean_ocr_text("".join(d[1] for d in left_res))
                    r_text = clean_ocr_text("".join(d[1] for d in right_res))
                    if l_text.startswith("IND") and len(l_text) > 4:
                        l_text = l_text[3:]

                    split_combined = l_text + r_text
                    if self._MIN_PLATE_LEN <= len(split_combined) <= self._MAX_PLATE_LEN:
                        l_conf = sum(d[2] for d in left_res) / len(left_res)
                        r_conf = sum(d[2] for d in right_res) / len(right_res)
                        split_conf = float((l_conf + r_conf) / 2.0)
                        corrected_split = soft_format_indian_plate(split_combined)
                        raw_seen.append(f"split[{l_text}+{r_text}]->'{corrected_split}'(conf={split_conf:.2f})")
                        plausible.append((corrected_split, split_conf))
            except Exception:
                pass

        if plausible:
            DIAG.bump("ocr_forwarded_plausible")
            best_text, best_conf = max(plausible, key=lambda x: x[1])
            logger.debug(
                "OCR forwarded (soft-corrected): '%s' conf=%.3f",
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
