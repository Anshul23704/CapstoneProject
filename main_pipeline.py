

from __future__ import annotations

import csv
import logging
import os
import sys
import threading
from datetime import datetime
from queue import Empty, Queue
from typing import List

import config
from stage1_frame_ingestion      import FrameIngestionStage, FrameIngestionConfig
from stage2_detection_tracking   import DetectionTrackingStage, DetectionConfig
from plate_detection             import PlateDetectionStage, PlateDetectionConfig
from stage3_active_buffering     import ActiveBufferingStage, BufferingConfig
from stage4_vehicle_finalization import VehicleFinalizationStage, FinalizationConfig
from stage5_job_creation         import JobCreationStage, JobCreationConfig
from stage6_worker_pool          import WorkerPoolStage, WorkerConfig, RecognitionStatus
from stage7_temporal_fusion      import TemporalFusionStage, FusionConfig
from stage8_database_analytics   import DatabaseAnalyticsStage, DatabaseConfig
from stage9_interpolation        import interpolate_csv
from stage10_visualize           import render_annotated_video


logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt = "%H:%M:%S",
)
logger = logging.getLogger(__name__)


RUN_ID     = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = os.path.join(config.OUTPUT_ROOT, RUN_ID)
os.makedirs(OUTPUT_DIR, exist_ok=True)

RICH_CSV_PATH   = os.path.join(OUTPUT_DIR, "results_rich.csv")
# NEW: every frame with a matched plate detection, regardless of whether OCR
# fully validated that track — this is what Stage 10 now draws boxes from,
# so "detect as many plates as possible" isn't gated by full OCR validation.
RAW_CSV_PATH    = os.path.join(OUTPUT_DIR, "results_raw_detections.csv")
DB_PATH         = os.path.join(OUTPUT_DIR, "results.db")
ANNOTATED_VIDEO = os.path.join(OUTPUT_DIR, "annotated_output.mp4")

_RICH_FIELDS = [
    "frame_nmr", "car_id",
    "car_bbox", "license_plate_bbox",
    "license_plate_bbox_score",
    "license_number", "license_number_score",
]


def _result_consumer(
    result_q: Queue,
    stop_event: threading.Event,
    fusion: TemporalFusionStage,
    db: DatabaseAnalyticsStage,
    run_id: str,
    rows: List[dict],
    raw_rows: List[dict],
    stats: dict,
) -> None:
    """
    Runs on its own thread. Drains RecognitionResults from Stage 6's worker
    pool, applies Stage 7 temporal fusion across each vehicle's
    frame_readings, writes the fused result to Stage 8's DB, and stages CSV
    rows for two separate outputs:

    - `rows` (-> results_rich.csv): only frames from tracks whose FUSED
      result is fully valid — used for Stage 9 interpolation and the
      "confirmed plate" analytics/count.
    - `raw_rows` (-> results_raw_detections.csv): EVERY frame with a
      matched plate detection, regardless of whether that track's fusion
      ever validated — this is what Stage 10 draws boxes from now, so the
      video shows every real detection instead of only fully-validated
      ones. Label uses the per-frame OCR text if present, else the track's
      fused text once known, else blank (box drawn, no label).

    Exits once stop_event is set AND the queue has been fully drained —
    stop_event alone isn't enough because workers may still be flushing
    their last results when shutdown begins.
    """
    while True:
        try:
            result = result_q.get(timeout=0.5)
        except Empty:
            if stop_event.is_set():
                break
            continue

        try:
            fused_text, fused_conf, is_valid = "", 0.0, False

            if result.status == RecognitionStatus.SUCCESS and result.frame_readings:
                readings_for_fusion = [(t, c) for (t, c, _vb, _pb, _fi) in result.frame_readings]
                fused_text, fused_conf, is_valid = fusion.process(readings_for_fusion)

                db.insert_result(
                    run_id=run_id,
                    track_id=result.track_id,
                    job_id=result.job_id,
                    plate_text=fused_text,
                    confidence=fused_conf,
                    status=result.status.name,
                    is_valid=is_valid,
                    plate_bbox=result.plate_bbox,
                    num_readings=len(result.frame_readings),
                )

                # --- STRICT REGEX VALIDATION (DISABLED FOR TESTING) ---
                # To re-enable strict Indian license plate format validation before outputting to CSV,
                # uncomment the following lines and ensure `license_complies_format` is imported from plate_utils.
                #
                # from plate_utils import license_complies_format
                # if is_valid and fused_text and not license_complies_format(fused_text):
                #     logger.info("Plate rejected by strict validation in final stage: '%s'", fused_text)
                #     is_valid = False
                #     fused_text = ""
                # --------------------------------------------------------

                if is_valid and fused_text:
                    for (_t, _c, vehicle_bbox, plate_bbox_full, frame_idx) in result.frame_readings:
                        if plate_bbox_full is None:
                            continue
                        rows.append({
                            "frame_nmr":                frame_idx,
                            "car_id":                   result.track_id,
                            "car_bbox":                 "[{} {} {} {}]".format(*vehicle_bbox),
                            "license_plate_bbox":       "[{} {} {} {}]".format(*plate_bbox_full),
                            "license_plate_bbox_score": f"{fused_conf:.4f}",
                            "license_number":           fused_text,
                            "license_number_score":     f"{fused_conf:.4f}",
                        })
                    stats["fused_success"] += 1
            else:
                db.insert_result(
                    run_id=run_id,
                    track_id=result.track_id,
                    job_id=result.job_id,
                    plate_text="",
                    confidence=0.0,
                    status=result.status.name,
                    is_valid=False,
                    num_readings=len(result.frame_readings),
                )
                stats["no_plate"] += 1

            # NEW: raw stream — every real detection, not just validated ones.
            for (vehicle_bbox, plate_bbox_full, frame_idx, ocr_text, ocr_conf) in result.raw_detections:
                if plate_bbox_full is None:
                    continue
                label = ocr_text or (fused_text if is_valid else "")
                score = ocr_conf if ocr_text else (fused_conf if is_valid else 0.0)
                raw_rows.append({
                    "frame_nmr":                frame_idx,
                    "car_id":                   result.track_id,
                    "car_bbox":                 "[{} {} {} {}]".format(*vehicle_bbox),
                    "license_plate_bbox":       "[{} {} {} {}]".format(*plate_bbox_full),
                    "license_plate_bbox_score": f"{score:.4f}",
                    "license_number":           label,
                    "license_number_score":     f"{score:.4f}",
                })
        except Exception:
            logger.exception("result_consumer failed on job=%s track=%s",
                              result.job_id, result.track_id)
        finally:
            result_q.task_done()


def run() -> None:
    logger.info("Loading models (vehicle=%s, plate=%s, device=%s)...",
                config.VEHICLE_MODEL_PATH, config.PLATE_MODEL_PATH, config.DEVICE)

    detection_tracking = DetectionTrackingStage(DetectionConfig())
    plate_detector      = PlateDetectionStage(PlateDetectionConfig())
    buffering           = ActiveBufferingStage(BufferingConfig())
    finalization        = VehicleFinalizationStage(FinalizationConfig())

    processing_queue: Queue = Queue()
    result_queue:     Queue = Queue()

    job_creator = JobCreationStage(processing_queue, JobCreationConfig())
    crops_dir   = os.path.join(OUTPUT_DIR, "plate_crops")
    os.makedirs(crops_dir, exist_ok=True)
    worker_pool = WorkerPoolStage(
        processing_queue, result_queue,
        num_workers=config.NUM_WORKERS,
        config=WorkerConfig(save_crops=True, crops_dir=crops_dir),
    )
    fusion = TemporalFusionStage(FusionConfig())
    db     = DatabaseAnalyticsStage(DatabaseConfig(db_path=DB_PATH))

    rows: List[dict] = []
    raw_rows: List[dict] = []
    stats = {"fused_success": 0, "no_plate": 0}
    stop_event = threading.Event()

    worker_pool.start()
    consumer_thread = threading.Thread(
        target=_result_consumer,
        args=(result_queue, stop_event, fusion, db, RUN_ID, rows, raw_rows, stats),
        daemon=True,
    )
    consumer_thread.start()

    ingestion = FrameIngestionStage(
        FrameIngestionConfig(source=config.VIDEO_SOURCE, target_resolution=None)
    )

    finalized_count = 0
    logger.info("Starting pipeline... (outputs -> %s)", OUTPUT_DIR)

    with ingestion:
        total_frames = ingestion.source_frame_count

        for frame, frame_idx, ts in ingestion.frames():

            # ── Main thread, GPU: Stage 2 detect + track ────────────────────
            track_map = detection_tracking.process(frame)

            # ── Stage 2.5, GPU: full-frame plate detection (once/frame) ─────
            plate_detections = plate_detector.process(frame)

            # ── Stage 3: buffer per-vehicle frames, match plates by IoA ─────
            ready_buffers = buffering.update(
                track_map, frame, frame_idx, ts, plate_detections=plate_detections
            )

            # ── Stage 4 + 5: finalize -> dispatch async job ─────────────────
            for buf in ready_buffers:
                fv = finalization.process(buf)
                if fv is None:
                    continue
                finalized_count += 1
                job_creator.dispatch(fv)   # non-blocking: workers pick this up

            sys.stdout.write(
                f"\rFrame {frame_idx + 1}/{max(total_frames, 1)} | "
                f"Active:{buffering.active_count} | "
                f"Finalized:{finalized_count} | "
                f"OK:{stats['fused_success']} | "
                f"NoPlate:{stats['no_plate']}"
            )
            sys.stdout.flush()

    print()
    logger.info("Ingestion complete. Flushing vehicles still active at stream end...")

    # FIX: previously any vehicle still ACTIVE when the video ended was
    # silently dropped — never finalized, never dispatched, never OCR'd.
    for buf in buffering.flush_all():
        fv = finalization.process(buf)
        if fv is None:
            continue
        finalized_count += 1
        job_creator.dispatch(fv)

    logger.info("Waiting for worker pool to finish %d queued jobs...",
                processing_queue.qsize())
    processing_queue.join()
    worker_pool.shutdown()

    stop_event.set()
    consumer_thread.join(timeout=30)

    logger.info(
        "Pipeline complete — finalized=%d fused_success=%d no_plate=%d",
        finalized_count, stats["fused_success"], stats["no_plate"],
    )

    # ── Write rich CSV (fully-validated fused plates only) ──────────────────
    rows.sort(key=lambda r: (r["frame_nmr"], r["car_id"]))
    with open(RICH_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_RICH_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Rich CSV saved -> %s", RICH_CSV_PATH)

    # ── Write raw detections CSV (every real detection, any track) ──────────
    raw_rows.sort(key=lambda r: (r["frame_nmr"], r["car_id"]))
    with open(RAW_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_RICH_FIELDS)
        writer.writeheader()
        writer.writerows(raw_rows)
    logger.info("Raw detections CSV saved -> %s (%d rows)", RAW_CSV_PATH, len(raw_rows))

    # ── Stage 8: analytics report ───────────────────────────────────────────
    report_path = db.generate_report(run_id=RUN_ID, output_dir=OUTPUT_DIR)
    if report_path:
        logger.info("Analytics -> %s", report_path)
    db.close()

    # ── Stage 9: interpolation (of the validated/fused CSV only) ─────────────
    interpolated_csv = None
    try:
        interpolated_csv = interpolate_csv(RICH_CSV_PATH)
        logger.info("Interpolated CSV -> %s", interpolated_csv)
    except Exception:
        logger.exception("Stage 9 (interpolation) failed")

    # ── Stage 10: annotated video ────────────────────────────────────────────
    # FIX: previously rendered from the interpolated *validated-fusion-only*
    # CSV, so with only a handful of fully-validated tracks the entire video
    # showed almost no boxes. Now renders from RAW_CSV_PATH — every frame
    # with a real plate detection, for every track, whether or not that
    # track's OCR ever fully validated — so the output actually reflects
    # everything the model detected, with tight per-frame boxes (no
    # interpolation needed for these: each row is a real detection, not a
    # guess between two real ones).
    if raw_rows:
        try:
            out = render_annotated_video(RAW_CSV_PATH, config.VIDEO_SOURCE, ANNOTATED_VIDEO)
            if out:
                logger.info("Annotated video -> %s", out)
        except Exception:
            logger.exception("Stage 10 (visualization) failed")
    else:
        logger.warning("No raw detections at all — skipping Stage 10 video render")

    unique_plates = {row["license_number"] for row in rows}
    print("\n" + "=" * 60)
    print(f"  PIPELINE COMPLETE — {RUN_ID}")
    print("=" * 60)
    print(f"  Vehicles finalized      : {finalized_count}")
    print(f"  Valid fused plates      : {stats['fused_success']}")
    print(f"  No-plate vehicles       : {stats['no_plate']}")
    print(f"  Unique plate texts      : {len(unique_plates)}")
    print(f"  Raw detection boxes     : {len(raw_rows)}  (drawn in video regardless of OCR validity)")
    print(f"  Rich CSV                : {RICH_CSV_PATH}")
    print(f"  Raw detections CSV      : {RAW_CSV_PATH}")
    print(f"  DB                      : {DB_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    run()
