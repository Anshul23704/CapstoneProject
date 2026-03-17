from __future__ import annotations
import logging
import threading
from dataclasses import dataclass
from enum import Enum, auto
from queue import Queue
from typing import List, Optional, Tuple
import numpy as np
import cv2
import easyocr
from ultralytics import YOLO

logger = logging.getLogger(__name__)

BBox = Tuple[int, int, int, int]


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


@dataclass
class WorkerConfig:
    plate_model_path: str   = "D:\\Sem6_Subjects\\Capstone\\implementation\\models\\license_plate_detector.pt"
    plate_conf:       float = 0.25
    plate_padding:    int   = 6
    use_gpu:          bool  = True
    upscale_factor:   float = 2.0
    min_plate_area:   int   = 200     # lowered — plates in full frames are larger


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
        KEY FIX vs previous version:
        The plate detector is now run on the FULL FRAME stored in each
        FrameEntry, not on the small vehicle crop.  This matches what the
        GitHub reference does: detect plates in the full frame at full
        resolution, then use the vehicle bbox to confirm the plate belongs
        to this track, then crop the plate at full resolution for OCR.

        FrameEntry fields used:
          .full_frame  — the complete BGR frame (added to FrameEntry below)
          .bbox        — vehicle bbox in the full frame (x1,y1,x2,y2)
          .frame_idx   — for debug logging
        """
        best_text = ""
        best_conf = 0.0
        best_bbox: Optional[BBox] = None

        for frame_entry in job.selected_frames:

            # ── Use full_frame if available, fall back to crop ────────────
            # After the FrameEntry change (see stage3_active_buffering.py),
            # each entry carries full_frame.  On older entries it may be None.
            full_frame = getattr(frame_entry, 'full_frame', None)

            if full_frame is None or full_frame.size == 0:
                # Fallback: try plate detector on crop directly (low res, may fail)
                logger.debug("No full_frame on entry %d, falling back to crop",
                             frame_entry.frame_idx)
                full_frame = frame_entry.crop

            vehicle_bbox = frame_entry.bbox   # (x1,y1,x2,y2) in full-frame coords

            # ── Step 1: run plate detector on the full frame ──────────────
            plate_crop, plate_bbox_full = self._detect_plate_in_frame(
                full_frame, vehicle_bbox
            )

            if plate_crop is None:
                logger.debug("No plate in frame %d for track %d",
                             frame_entry.frame_idx, job.track_id)
                continue

            # ── Step 2: preprocess ────────────────────────────────────────
            plate_ready = self._preprocess_plate(plate_crop)

            # ── Step 3: OCR ───────────────────────────────────────────────
            text, conf = self._run_ocr(plate_ready)

            if text and conf > best_conf:
                best_conf = conf
                best_text = text
                best_bbox = plate_bbox_full

        if best_text:
            return RecognitionResult(
                job_id=job.job_id, track_id=job.track_id,
                plate_text=best_text, confidence=best_conf,
                status=RecognitionStatus.SUCCESS, plate_bbox=best_bbox,
            )

        return RecognitionResult(
            job_id=job.job_id, track_id=job.track_id,
            plate_text="", confidence=0.0, status=RecognitionStatus.NO_PLATE,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _detect_plate_in_frame(
        self,
        frame: np.ndarray,
        vehicle_bbox: BBox,
    ) -> Tuple[Optional[np.ndarray], Optional[BBox]]:
        """
        Run the plate detector on the FULL frame.
        Only accept plate detections whose bbox falls INSIDE vehicle_bbox —
        this is the same containment check that get_car() does in the GitHub
        reference code.
        Returns (plate_crop_at_full_res, plate_bbox) or (None, None).
        """
        results = self.plate_detector(frame, conf=self.cfg.plate_conf, verbose=False)[0]

        if results.boxes is None or len(results.boxes) == 0:
            return None, None

        vx1, vy1, vx2, vy2 = vehicle_bbox
        best_conf  = -1.0
        best_box   = None

        for i in range(len(results.boxes)):
            px1, py1, px2, py2 = map(int, results.boxes.xyxy[i])
            conf = float(results.boxes.conf[i])

            # IoA check: require >=50% of plate area overlaps vehicle bbox.
            # Strict containment fails when YOLO clips the bumper (very common).
            inter_x1 = max(px1, vx1)
            inter_y1 = max(py1, vy1)
            inter_x2 = min(px2, vx2)
            inter_y2 = min(py2, vy2)
            inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
            plate_area = max(1, (px2 - px1) * (py2 - py1))
            ioa = inter_area / plate_area

            if ioa >= 0.5:
                if plate_area < self.cfg.min_plate_area:
                    continue
                if conf > best_conf:
                    best_conf = conf
                    best_box  = (px1, py1, px2, py2)

        if best_box is None:
            return None, None

        px1, py1, px2, py2 = best_box
        p = self.cfg.plate_padding
        h, w = frame.shape[:2]
        px1 = max(0, px1 - p)
        py1 = max(0, py1 - p)
        px2 = min(w, px2 + p)
        py2 = min(h, py2 + p)

        plate_crop = frame[py1:py2, px1:px2].copy()
        return plate_crop, (px1, py1, px2, py2)

    def _preprocess_plate(self, plate_crop: np.ndarray) -> np.ndarray:
        """Upscale, CLAHE, sharpen — same as before."""
        h, w = plate_crop.shape[:2]

        if w < 120 or h < 30:
            scale = max(self.cfg.upscale_factor, 120 / max(w, 1), 30 / max(h, 1))
            plate_crop = cv2.resize(
                plate_crop, (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_CUBIC,
            )

        gray    = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
        clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        gray    = clahe.apply(gray)
        blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.0)
        sharp   = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
        return cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR)

    def _run_ocr(self, plate_img: np.ndarray) -> Tuple[str, float]:
        """EasyOCR on preprocessed plate image."""
        detections = self.ocr.readtext(
            plate_img, detail=1, paragraph=False,
            width_ths=0.9, contrast_ths=0.1, adjust_contrast=0.5,
        )
        if not detections:
            return "", 0.0

        texts, confs = [], []
        for (_bbox, text, conf) in detections:
            clean = text.strip()
            if clean:
                texts.append(clean)
                confs.append(conf)

        if not texts:
            return "", 0.0

        return " ".join(texts), float(np.mean(confs))


class WorkerPoolStage:

    def __init__(self, job_q: Queue, result_q: Queue,
                 num_workers: int = 2, config: WorkerConfig = None):
        self.job_q    = job_q
        self.result_q = result_q
        cfg = config or WorkerConfig()
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