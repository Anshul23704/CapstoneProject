from __future__ import annotations
import csv
import logging
import os
import sys
from datetime import datetime
from queue import Empty, Queue

import cv2

from stage1_frame_ingestion import FrameIngestionStage, FrameIngestionConfig
from stage2_detection_tracking import DetectionTrackingStage, DetectionConfig
from stage3_active_buffering import ActiveBufferingStage, BufferingConfig
from stage4_vehicle_finalization import VehicleFinalizationStage
from stage5_job_creation import JobCreationStage
from stage6_worker_pool import WorkerPoolStage
from stage6_worker_pool import RecognitionStatus


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)



RUN_ID     = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = os.path.join("output", RUN_ID)
CSV_PATH   = os.path.join(OUTPUT_DIR, "results.csv")
os.makedirs(OUTPUT_DIR, exist_ok=True)



processing_queue: Queue = Queue()
result_queue:     Queue = Queue()

ingestion    = FrameIngestionStage(FrameIngestionConfig(
    source="D:\\Sem6_Subjects\\Capstone\\implementation\\traffic.mp4"
))
detection    = DetectionTrackingStage(DetectionConfig())
buffering    = ActiveBufferingStage(BufferingConfig())
finalization = VehicleFinalizationStage()
job_creator  = JobCreationStage(processing_queue)
worker_pool  = WorkerPoolStage(processing_queue, result_queue)

worker_pool.start()


detection_count = 0
finalized_count = 0
job_count       = 0
result_count    = 0
all_results: list = []



def _collect_result(result, csv_writer) -> None:
    """
    Called every time a RecognitionResult arrives.
    Prints it live, writes it to CSV, appends to all_results.
    """
    all_results.append(result)

    status_str = "OK"   if result.status == RecognitionStatus.SUCCESS else "FAIL"
    plate      = result.plate_text.strip() if result.plate_text else "(no text)"
    conf_pct   = f"{result.confidence * 100:.1f}%"

    # Print on its own line so it doesn't collide with the \r progress bar
    print(f"\n  → track={result.track_id:>5}  plate={plate:<20}  conf={conf_pct}  [{status_str}]")

    csv_writer.writerow({
        "track_id":   result.track_id,
        "plate_text": plate,
        "confidence": f"{result.confidence:.4f}",
        "status":     status_str,
        "job_id":     result.job_id,
    })



print(f"Starting pipeline...  (outputs → {OUTPUT_DIR})\n")

with open(CSV_PATH, "w", newline="", encoding="utf-8") as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=["track_id", "plate_text", "confidence", "status", "job_id"])
    writer.writeheader()

    with ingestion:
        total_frames = int(ingestion._cap.get(cv2.CAP_PROP_FRAME_COUNT))

        for frame, frame_idx, ts in ingestion.frames():

            # Stage 2 — detect & track
            track_map = detection.process(frame)
            detection_count += len(track_map)

            # Stage 3 — buffer crops
            ready_buffers = buffering.update(track_map, frame, frame_idx, ts)

            # Stage 4 + 5 — finalize & dispatch
            for buf in ready_buffers:
                fv = finalization.process(buf)
                if fv:
                    finalized_count += 1
                    job = job_creator.dispatch(fv)
                    if job:
                        job_count += 1

            # Drain completed OCR results
            while not result_queue.empty():
                try:
                    result = result_queue.get_nowait()
                    result_count += 1
                    _collect_result(result, writer)
                    csv_file.flush()        # write to disk immediately
                except Empty:
                    break

            # Progress bar
            sys.stdout.write(
                f"\rFrame {frame_idx + 1}/{total_frames} | "
                f"Det:{detection_count} | "
                f"Buf:{buffering.active_count} | "
                f"Fin:{finalized_count} | "
                f"Jobs:{job_count} | "
                f"Res:{result_count}"
            )
            sys.stdout.flush()

    # End-of-video: flush remaining jobs
    print("\n\nVideo finished — waiting for remaining OCR jobs...")
    worker_pool.shutdown()

    while True:
        try:
            result = result_queue.get_nowait()
            result_count += 1
            _collect_result(result, writer)
            csv_file.flush()
        except Empty:
            break


successful = [r for r in all_results if r.status == RecognitionStatus.SUCCESS]
failed     = [r for r in all_results if r.status != RecognitionStatus.SUCCESS]


seen: dict = {}
for r in successful:
    plate = r.plate_text.strip().upper()
    if plate not in seen or r.confidence > seen[plate].confidence:
        seen[plate] = r
unique_plates = sorted(seen.keys())

print("\n" + "=" * 60)
print(f"  PIPELINE COMPLETE — {RUN_ID}")
print("=" * 60)
print(f"  Frames processed : {ingestion.total_frames}")
print(f"  Vehicles tracked : {finalized_count}")
print(f"  Jobs dispatched  : {job_count}")
print(f"  OCR results      : {result_count}  (success={len(successful)}, failed={len(failed)})")
print(f"  Unique plates    : {len(unique_plates)}")
print()
if unique_plates:
    print("  Detected plates:")
    for plate in unique_plates:
        r = seen[plate]
        print(f"    {plate:<20}  conf={r.confidence * 100:.1f}%  track={r.track_id}")
else:
    print("  No plates detected.")
print()
print(f"  CSV saved  → {CSV_PATH}")
print("=" * 60)