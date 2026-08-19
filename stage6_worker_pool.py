"""
stage6_worker_pool.py — Worker Pool (Plate Detection + OCR)

INTEGRATION CHANGES (vs previous version)
──────────────────────────────────────────
1. Imported `license_complies_format` and `format_license` from the reference
   pipeline's util logic (inlined here so no extra module dependency).
   These replace the loose regex-only check that was accepting garbage like
   "DFAMGEO1VISI" and "5" as valid plates.

2. `_run_ocr` now mirrors the reference `read_license_plate` flow:
   - Iterates every EasyOCR detection independently (not joined).
   - Uppercases and strips spaces/dashes before validation.
   - Calls `license_complies_format` (strict 7-char positional check).
   - Calls `format_license` (positional char-swap) only on passing text.
   - Returns the first detection that passes — same as reference pipeline.

3. `_detect_plate_in_frame` now performs a spatial containment check
   (plate bbox must lie inside vehicle bbox) identical to `get_car` in the
   reference pipeline. This prevents plates from neighbouring vehicles
   being assigned to the wrong track.

4. `frame_readings` now stores (formatted_text, conf) pairs so Stage 7
   temporal fusion receives already-validated, formatted strings.
"""

from __future__ import annotations

import logging
import string
import threading
from dataclasses import dataclass
from enum import Enum, auto
from queue import Queue
from typing import List, Optional, Tuple

import cv2
import easyocr
import numpy as np
from ultralytics import YOLO

logger = logging.getLogger(__name__)

BBox = Tuple[int, int, int, int]


# ── Licence-plate format helpers (ported from reference util.py) ──────────────

# Characters that OCR confuses in digit positions
_CHAR_TO_INT = {
    'O': '0', 'I': '1', 'J': '3',
    'A': '4', 'G': '6', 'S': '5',
}
# Characters that OCR confuses in letter positions
_INT_TO_CHAR = {v: k for k, v in _CHAR_TO_INT.items()}


def license_complies_format(text: str) -> bool:
    """
    Strict 7-character positional check (ported from reference util.py).
    Layout: LL DD LLL  (L=letter, D=digit)
    Positions 0,1,4,5,6 must be letters (or OCR-confusable letters).
    Positions 2,3 must be digits (or OCR-confusable digits).
    """
    if len(text) != 7:
        return False
    alpha_ok = set(string.ascii_uppercase) | set(_INT_TO_CHAR.keys())
    digit_ok  = set('0123456789')         | set(_CHAR_TO_INT.keys())
    checks = [
        text[0] in alpha_ok,
        text[1] in alpha_ok,
        text[2] in digit_ok,
        text[3] in digit_ok,
        text[4] in alpha_ok,
        text[5] in alpha_ok,
        text[6] in alpha_ok,
    ]
    return all(checks)


def format_license(text: str) -> str:
    """
    Positional character correction (ported from reference util.py).
    Applies char-swap only in the correct position class.
    """
    mapping = {
        0: _INT_TO_CHAR, 1: _INT_TO_CHAR,
        2: _CHAR_TO_INT, 3: _CHAR_TO_INT,
        4: _INT_TO_CHAR, 5: _INT_TO_CHAR, 6: _INT_TO_CHAR,
    }
    out = []
    for j in range(7):
        ch = text[j]
        out.append(mapping[j].get(ch, ch))
    return "".join(out)


# ─────────────────────────────────────────────────────────────────────────────

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
    # Per-frame OCR readings forwarded to Stage 7 temporal fusion.
    # Each entry is (formatted_text, conf) — already validated & formatted.
    frame_readings: tuple = ()


@dataclass
class WorkerConfig:
    plate_model_path: str   = "E:\\Capstone\\implementation\\models\\license_plate_detector.pt"
    plate_conf:       float = 0.25
    plate_padding:    int   = 6
    use_gpu:          bool  = True
    upscale_factor:   float = 2.0
    min_plate_area:   int   = 200


class Worker(threading.Thread):

    def __init__(self, job_q: Queue, result_q: Queue, config: WorkerConfig):
        super().__init__(daemon=True)
        self.job_q    = job_q
        self.result_q = result_q
        self.cfg      = config

        logger.info("Worker init: loading plate detector from %s", config.plate_model_path)
        self.plate_detector = YOLO(config.plate_model_path)

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
          1. Run plate detector on the padded ROI.
          2. Perform spatial containment check (plate must be inside vehicle bbox)
             — mirrors get_car() from the reference pipeline.
          3. Preprocess the plate crop.
          4. Run OCR with strict format validation + positional correction.
          5. Collect validated (text, conf) pairs for Stage 7.
        """
        best_text  = ""
        best_conf  = 0.0
        best_bbox: Optional[BBox] = None
        frame_readings: List[Tuple[str, float]] = []

        for frame_entry in job.selected_frames:
            roi = getattr(frame_entry, 'full_frame', None)

            if roi is None or roi.size == 0:
                roi = frame_entry.crop
                roi_offset = (0, 0)
            else:
                roi_offset = getattr(frame_entry, 'roi_offset', (0, 0))

            ox, oy = roi_offset
            vx1, vy1, vx2, vy2 = frame_entry.bbox
            vehicle_bbox_roi = (
                max(0, vx1 - ox),
                max(0, vy1 - oy),
                max(0, vx2 - ox),
                max(0, vy2 - oy),
            )

            plate_crop, plate_bbox_roi = self._detect_plate_in_frame(
                roi, vehicle_bbox_roi
            )

            if plate_crop is None:
                logger.debug("No plate in frame %d for track %d",
                             frame_entry.frame_idx, job.track_id)
                continue

            if plate_bbox_roi is not None:
                px1, py1, px2, py2 = plate_bbox_roi
                plate_bbox_full = (px1 + ox, py1 + oy, px2 + ox, py2 + oy)
            else:
                plate_bbox_full = None

            plate_ready = self._preprocess_plate(plate_crop)

            # INTEGRATION: use validated OCR (reference read_license_plate flow)
            text, conf = self._run_ocr_validated(plate_ready)

            if text:
                frame_readings.append((text, conf))
                if conf > best_conf:
                    best_conf = conf
                    best_text = text
                    best_bbox = plate_bbox_full

        if best_text:
            return RecognitionResult(
                job_id         = job.job_id,
                track_id       = job.track_id,
                plate_text     = best_text,
                confidence     = best_conf,
                status         = RecognitionStatus.SUCCESS,
                plate_bbox     = best_bbox,
                frame_readings = tuple(frame_readings),
            )

        return RecognitionResult(
            job_id         = job.job_id,
            track_id       = job.track_id,
            plate_text     = "",
            confidence     = 0.0,
            status         = RecognitionStatus.NO_PLATE,
            frame_readings = tuple(frame_readings),
        )

    # ── Plate detection with containment check ────────────────────────────────

    def _detect_plate_in_frame(
        self,
        frame: np.ndarray,
        vehicle_bbox_roi: Tuple[int, int, int, int],
    ) -> Tuple[Optional[np.ndarray], Optional[BBox]]:
        """
        Runs the plate detector and returns the best plate crop that is
        spatially contained within vehicle_bbox_roi.

        INTEGRATION: containment check mirrors get_car() from reference
        pipeline — the plate bbox (x1,y1,x2,y2) must lie strictly inside
        the vehicle bbox. This stops plates from adjacent vehicles being
        assigned to the wrong track.
        """
        results = self.plate_detector(frame, conf=self.cfg.plate_conf, verbose=False)[0]

        if results.boxes is None or len(results.boxes) == 0:
            return None, None

        vx1, vy1, vx2, vy2 = vehicle_bbox_roi
        best_conf = -1.0
        best_box  = None

        for i in range(len(results.boxes)):
            px1, py1, px2, py2 = map(int, results.boxes.xyxy[i])
            conf       = float(results.boxes.conf[i])
            plate_area = max(1, (px2 - px1) * (py2 - py1))

            if plate_area < self.cfg.min_plate_area:
                continue

            # INTEGRATION: containment check (from get_car in reference util.py)
            # Plate must be fully inside the vehicle bbox in ROI coordinates.
            contained = (px1 > vx1 and py1 > vy1 and px2 < vx2 and py2 < vy2)
            if not contained:
                logger.debug(
                    "Plate bbox (%d,%d,%d,%d) not contained in vehicle bbox "
                    "(%d,%d,%d,%d) — skipped",
                    px1, py1, px2, py2, vx1, vy1, vx2, vy2,
                )
                continue

            if conf > best_conf:
                best_conf = conf
                best_box  = (px1, py1, px2, py2)

        if best_box is None:
            return None, None

        px1, py1, px2, py2 = best_box
        p  = self.cfg.plate_padding
        h, w = frame.shape[:2]
        px1 = max(0, px1 - p);  py1 = max(0, py1 - p)
        px2 = min(w, px2 + p);  py2 = min(h, py2 + p)

        return frame[py1:py2, px1:px2].copy(), (px1, py1, px2, py2)

    # ── Preprocessing ─────────────────────────────────────────────────────────

    def _preprocess_plate(self, plate_crop: np.ndarray) -> np.ndarray:
        h, w = plate_crop.shape[:2]
        if w < 120 or h < 30:
            scale = max(self.cfg.upscale_factor, 120 / max(w, 1), 30 / max(h, 1))
            plate_crop = cv2.resize(plate_crop,
                                    (int(w * scale), int(h * scale)),
                                    interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 64, 255, cv2.THRESH_BINARY_INV)
        return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

    # ── OCR with strict validation (reference pipeline flow) ──────────────────

    def _run_ocr_validated(self, plate_img: np.ndarray) -> Tuple[str, float]:
        """
        INTEGRATION: mirrors read_license_plate() from reference util.py.

        Key differences from the old _run_ocr:
        - Does NOT join all detections into one string.
        - Validates each detection independently with license_complies_format.
        - Applies positional format_license correction only on passing text.
        - Returns the first detection that passes the format check.

        This is why "DFAMGEO1VISI" (13 chars) and "5" (1 char) were
        previously returned — they were never length-checked. Now they
        are silently rejected and the result is NO_PLATE rather than a
        garbage INVALID plate.
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
            return "", 0.0

        for (_bbox, text, conf) in detections:
            clean = text.upper().replace(" ", "").replace("-", "")
            if not clean:
                continue
            if license_complies_format(clean):
                formatted = format_license(clean)
                logger.debug("OCR accepted: '%s' → '%s' conf=%.3f", clean, formatted, conf)
                return formatted, float(conf)

        # No detection passed the format check
        return "", 0.0


class WorkerPoolStage:

    def __init__(self, job_q: Queue, result_q: Queue,
                 num_workers: int = 2, config: WorkerConfig = None):
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
