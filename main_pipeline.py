
from __future__ import annotations

import csv
import logging
import os
import sys
from datetime import datetime
from queue import Empty, Queue

import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
import easyocr

from stage1_frame_ingestion      import FrameIngestionStage, FrameIngestionConfig
from stage3_active_buffering     import ActiveBufferingStage, BufferingConfig
from stage4_vehicle_finalization import VehicleFinalizationStage
from stage5_job_creation         import JobCreationStage
from stage8_database_analytics   import DatabaseAnalyticsStage, DatabaseConfig
from stage9_interpolation        import interpolate_csv



logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt = "%H:%M:%S",
)
logger = logging.getLogger(__name__)


RUN_ID       = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR   = os.path.join("output", RUN_ID)
VIDEO_SOURCE = "D:\\Sem6_Subjects\\Capstone\\implementation\\traffic.mp4"
MODEL_VEHICLE = "D:\\Sem6_Subjects\\Capstone\\implementation\\models\\yolo26s.pt"
MODEL_PLATE   = "D:\\Sem6_Subjects\\Capstone\\implementation\\models\\license_plate_detector.pt"

RICH_CSV_PATH = os.path.join(OUTPUT_DIR, "results_rich.csv")
DB_PATH       = os.path.join(OUTPUT_DIR, "results.db")
os.makedirs(OUTPUT_DIR, exist_ok=True)


VEHICLE_CLASSES = {2, 3, 5, 7}


import string

_CHAR_TO_INT = {'O': '0', 'I': '1', 'J': '3', 'A': '4', 'G': '6', 'S': '5'}
_INT_TO_CHAR = {v: k for k, v in _CHAR_TO_INT.items()}


def license_complies_format(text: str) -> bool:
    if len(text) != 7:
        return False
    alpha_ok = set(string.ascii_uppercase) | set(_INT_TO_CHAR.keys())
    digit_ok  = set('0123456789')          | set(_CHAR_TO_INT.keys())
    return all([
        text[0] in alpha_ok,
        text[1] in alpha_ok,
        text[2] in digit_ok,
        text[3] in digit_ok,
        text[4] in alpha_ok,
        text[5] in alpha_ok,
        text[6] in alpha_ok,
    ])


def format_license(text: str) -> str:
    mapping = {
        0: _INT_TO_CHAR, 1: _INT_TO_CHAR,
        2: _CHAR_TO_INT, 3: _CHAR_TO_INT,
        4: _INT_TO_CHAR, 5: _INT_TO_CHAR, 6: _INT_TO_CHAR,
    }
    return "".join(mapping[j].get(text[j], text[j]) for j in range(7))


def read_license_plate(license_plate_crop: np.ndarray, ocr_reader) -> tuple:

    detections = ocr_reader.readtext(license_plate_crop)
    for detection in detections:
        _, text, score = detection
        text = text.upper().replace(' ', '').replace('-', '')
        if license_complies_format(text):
            return format_license(text), score
    return None, None


def get_car(license_plate_bbox, track_ids_array: np.ndarray):

    x1, y1, x2, y2 = license_plate_bbox[:4]
    for j in range(len(track_ids_array)):
        xcar1, ycar1, xcar2, ycar2, car_id = track_ids_array[j]
        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            return track_ids_array[j]
    return np.array([-1, -1, -1, -1, -1])



print("Loading models...")
vehicle_model = YOLO(MODEL_VEHICLE)
plate_model   = YOLO(MODEL_PLATE)
ocr_reader    = easyocr.Reader(['en'], gpu=True, verbose=False)
print("Models loaded.\n")


tracker = sv.ByteTrack(
    track_activation_threshold=0.25,
    lost_track_buffer=50,
    minimum_matching_threshold=0.60,
)


processing_queue: Queue = Queue()
result_queue:     Queue = Queue()

ingestion    = FrameIngestionStage(FrameIngestionConfig(source=VIDEO_SOURCE,target_resolution=(1280, 720)))
buffering    = ActiveBufferingStage(BufferingConfig())
finalization = VehicleFinalizationStage()
job_creator  = JobCreationStage(processing_queue)
db           = DatabaseAnalyticsStage(DatabaseConfig(db_path=DB_PATH))


_RICH_FIELDS = [
    "frame_nmr", "car_id",
    "car_bbox", "license_plate_bbox",
    "license_plate_bbox_score",
    "license_number", "license_number_score",
]


all_frame_results: dict = {}

detection_count = 0
finalized_count = 0
ocr_success     = 0


print(f"Starting pipeline...  (outputs → {OUTPUT_DIR})\n")

with ingestion:
    total_frames = int(ingestion._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for frame, frame_idx, ts in ingestion.frames():

        all_frame_results[frame_idx] = {}

        # ── Stage 2a: Vehicle detection ───────────────────────────────────────
        raw = vehicle_model(frame, conf=0.25, iou=0.45, verbose=False)[0]
        sv_dets = sv.Detections.from_ultralytics(raw)

        # Keep only vehicle classes
        mask = np.isin(sv_dets.class_id, list(VEHICLE_CLASSES))
        sv_dets = sv_dets[mask]

        # ByteTrack update
        tracked = tracker.update_with_detections(sv_dets)
        detection_count += len(tracked)

        # Build track_map {track_id: bbox} and track_ids_array for get_car()
        track_map: dict = {}
        track_ids_rows = []
        if tracked.tracker_id is not None:
            for bbox, tid in zip(tracked.xyxy, tracked.tracker_id):
                if tid is None:
                    continue
                x1, y1, x2, y2 = map(float, bbox)
                track_map[int(tid)] = (int(x1), int(y1), int(x2), int(y2))
                track_ids_rows.append([x1, y1, x2, y2, float(tid)])

        track_ids_array = np.array(track_ids_rows) if track_ids_rows else np.empty((0, 5))

        
        plate_results = plate_model(frame, conf=0.25, verbose=False)[0]

        for lp in plate_results.boxes.data.tolist():
            px1, py1, px2, py2, score, class_id = lp

            # Assign to vehicle using get_car (reference pipeline)
            car_row = get_car(lp, track_ids_array)
            car_id = int(car_row[4])
            if car_id == -1:
                continue

            xcar1, ycar1, xcar2, ycar2 = map(int, car_row[:4])

            # Crop and preprocess plate
            lp_crop = frame[int(py1):int(py2), int(px1):int(px2)]
            if lp_crop.size == 0:
                continue

            
            lp_gray = cv2.cvtColor(lp_crop, cv2.COLOR_BGR2GRAY)
            _, lp_thresh = cv2.threshold(lp_gray, 64, 255, cv2.THRESH_BINARY_INV)

            # OCR
            plate_text, plate_score = read_license_plate(lp_thresh, ocr_reader)

            if plate_text is not None:
                ocr_success += 1
                all_frame_results[frame_idx][car_id] = {
                    "car":   {"bbox": [xcar1, ycar1, xcar2, ycar2]},
                    "license_plate": {
                        "bbox":       [px1, py1, px2, py2],
                        "text":       plate_text,
                        "bbox_score": score,
                        "text_score": plate_score,
                    },
                }

                # Write to DB
                db.insert_result(
                    run_id=RUN_ID,
                    track_id=car_id,
                    job_id=f"{frame_idx}_{car_id}",
                    plate_text=plate_text,
                    confidence=plate_score,
                    status="OK",
                    is_valid=True,
                )

        
        ready_buffers = buffering.update(track_map, frame, frame_idx, ts)
        for buf in ready_buffers:
            fv = finalization.process(buf)
            if fv:
                finalized_count += 1

        sys.stdout.write(
            f"\rFrame {frame_idx + 1}/{total_frames} | "
            f"Det:{detection_count} | "
            f"Buf:{buffering.active_count} | "
            f"Fin:{finalized_count} | "
            f"OCR:{ocr_success}"
        )
        sys.stdout.flush()


print("\n\nWriting results CSV...")

with open(RICH_CSV_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=_RICH_FIELDS)
    writer.writeheader()

    for frame_nmr, frame_data in sorted(all_frame_results.items()):
        for car_id, data in frame_data.items():
            if "car" not in data or "license_plate" not in data:
                continue
            if "text" not in data["license_plate"]:
                continue
            cb = data["car"]["bbox"]
            pb = data["license_plate"]["bbox"]
            writer.writerow({
                "frame_nmr":              frame_nmr,
                "car_id":                 car_id,
                "car_bbox":               "[{} {} {} {}]".format(*cb),
                "license_plate_bbox":     "[{} {} {} {}]".format(
                                              pb[0], pb[1], pb[2], pb[3]),
                "license_plate_bbox_score": data["license_plate"]["bbox_score"],
                "license_number":         data["license_plate"]["text"],
                "license_number_score":   data["license_plate"]["text_score"],
            })

print(f"  Rich CSV saved → {RICH_CSV_PATH}")


unique_plates: dict = {}
for fd in all_frame_results.values():
    for cid, data in fd.items():
        if "license_plate" in data and "text" in data["license_plate"]:
            plate = data["license_plate"]["text"].strip().upper()
            score = data["license_plate"]["text_score"]
            if plate and (plate not in unique_plates or score > unique_plates[plate]):
                unique_plates[plate] = score

print("\n" + "=" * 60)
print(f"  PIPELINE COMPLETE — {RUN_ID}")
print("=" * 60)
print(f"  Frames processed : {ingestion.total_frames}")
print(f"  Vehicles tracked : {finalized_count}")
print(f"  OCR successes    : {ocr_success}")
print(f"  Unique plates    : {len(unique_plates)}")
if unique_plates:
    print("\n  Detected plates:")
    for plate, conf in sorted(unique_plates.items(), key=lambda x: -x[1]):
        print(f"    {plate:<20}  conf={conf * 100:.1f}%")
else:
    print("  No plates detected.")
print()
print(f"  Rich CSV → {RICH_CSV_PATH}")
print(f"  DB       → {DB_PATH}")
print("=" * 60)

# ── Stage 8: analytics report ─────────────────────────────────────────────────
report_path = db.generate_report(run_id=RUN_ID, output_dir=OUTPUT_DIR)
if report_path:
    print(f"  Analytics → {report_path}")
db.close()

# ── Stage 9: interpolation ────────────────────────────────────────────────────
print("\nRunning Stage 9 — interpolation...")
try:
    interpolated_csv = interpolate_csv(RICH_CSV_PATH)
    print(f"  Interpolated CSV → {interpolated_csv}")
except Exception as exc:
    logger.error("Stage 9 failed: %s", exc, exc_info=True)
    interpolated_csv = None

