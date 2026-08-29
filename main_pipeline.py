

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
import cv2
import pandas as pd
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
    crops_dir: str,
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
    import cv2
    import pandas as pd
    
    # Initialize incremental state
    rich_file = open(RICH_CSV_PATH, "a", newline="", encoding="utf-8")
    raw_file = open(RAW_CSV_PATH, "a", newline="", encoding="utf-8")
    rich_writer = csv.DictWriter(rich_file, fieldnames=_RICH_FIELDS)
    raw_writer = csv.DictWriter(raw_file, fieldnames=_RICH_FIELDS)
    
    best_detections = {}
    valid_count = 0

    try:
        cap = cv2.VideoCapture(config.VIDEO_SOURCE)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()
    except Exception:
        fps = 30.0

    while True:
        try:
            result = result_q.get(timeout=0.5)
        except Empty:
            if stop_event.is_set():
                break
            continue

        try:
            fused_text, fused_conf, is_valid = "", 0.0, False
            fused_b, conf_b = "", 0.0

            if result.status == RecognitionStatus.SUCCESS and result.frame_readings:
                readings_for_fusion = [(t, c) for (t, c, _vb, _pb, _fi) in result.frame_readings]
                fused_text, fused_conf, is_valid = fusion.process(readings_for_fusion)

                if result.frame_readings_bilateral:
                    r_b = [(t, c) for (t, c, _vb, _pb, _fi) in result.frame_readings_bilateral]
                    fused_b, conf_b, _ = fusion.process(r_b)



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
                    plate_text_bilateral=fused_b,
                    conf_bilateral=conf_b,

                    winner_branch=result.winner_branch,
                )

                if is_valid and fused_text:
                    for (_t, _c, vehicle_bbox, plate_bbox_full, frame_idx) in result.frame_readings:
                        if plate_bbox_full is None:
                            continue
                        row = {
                            "frame_nmr":                frame_idx,
                            "car_id":                   result.track_id,
                            "car_bbox":                 "[{} {} {} {}]".format(*vehicle_bbox),
                            "license_plate_bbox":       "[{} {} {} {}]".format(*plate_bbox_full),
                            "license_plate_bbox_score": f"{fused_conf:.4f}",
                            "license_number":           fused_text,
                            "license_number_score":     f"{fused_conf:.4f}",
                        }
                        rows.append(row)
                        rich_writer.writerow(row)
                    rich_file.flush()
                    stats["fused_success"] += 1
                    valid_count += 1
                    
                    # Track high confidence outputs live
                    if fused_conf >= config.FINAL_OUTPUT_CONF_THRESHOLD:
                        current_best = best_detections.get(result.track_id)
                        if not current_best or fused_conf > float(current_best["Confidence"]):
                            # get frame of best detection
                            best_frame = result.frame_readings[0][4]
                            sec = best_frame / fps
                            ts = f"{int(sec//3600):02d}:{int((sec%3600)//60):02d}:{sec%60:06.3f}"
                            
                            # Determine target directory
                            target_dir = crops_dir
                            high_conf_crops_dir = os.path.join(OUTPUT_DIR, "plate_crops_high_confidence")
                            if fused_conf >= config.HIGH_CONF_CROP_THRESHOLD and os.path.exists(high_conf_crops_dir):
                                target_dir = high_conf_crops_dir
                            
                            context_f = os.path.join(target_dir, f"track_{int(result.track_id):03d}_frame_{best_frame:04d}_context.png")
                            plate_b_f = os.path.join(target_dir, f"track_{int(result.track_id):03d}_frame_{best_frame:04d}_plate_bilateral.png")
                            
                            best_detections[result.track_id] = {
                                "Track_ID":                  int(result.track_id),
                                "License_Plate":             fused_text,
                                "Confidence":                f"{fused_conf:.4f}",
                                "Avg_Sharpness":             f"{result.avg_sharpness:.2f}",
                                "Timestamp":                 ts,
                                "Frame_Number":              best_frame,
                                "Context_Image_Path":        os.path.abspath(context_f) if os.path.exists(context_f) else context_f,
                                "Plate_Bilateral_Path":      os.path.abspath(plate_b_f) if os.path.exists(plate_b_f) else plate_b_f,
                            }
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
                    winner_branch=result.winner_branch,
                    avg_sharpness=result.avg_sharpness,
                )
                stats["no_plate"] += 1

            # Raw stream — every real detection, not just validated ones.
            for (vehicle_bbox, plate_bbox_full, frame_idx, ocr_text, ocr_conf) in result.raw_detections:
                if plate_bbox_full is None:
                    continue
                label = ocr_text or (fused_text if is_valid else "")
                score = ocr_conf if ocr_text else (fused_conf if is_valid else 0.0)
                raw_row = {
                    "frame_nmr":                frame_idx,
                    "car_id":                   result.track_id,
                    "car_bbox":                 "[{} {} {} {}]".format(*vehicle_bbox),
                    "license_plate_bbox":       "[{} {} {} {}]".format(*plate_bbox_full),
                    "license_plate_bbox_score": f"{score:.4f}",
                    "license_number":           label,
                    "license_number_score":     f"{score:.4f}",
                }
                raw_rows.append(raw_row)
                raw_writer.writerow(raw_row)
            raw_file.flush()
            
            # Periodically write Final_outputs and Preprocessing comparison
            if valid_count > 0 and valid_count % 5 == 0:
                final_outputs_csv = os.path.join(OUTPUT_DIR, "Final_outputs.csv")
                final_outputs_md  = os.path.join(OUTPUT_DIR, "Final_outputs.md")
                
                high_conf_outputs = list(best_detections.values())
                high_conf_outputs.sort(key=lambda x: (x["Frame_Number"], x["Track_ID"]))
                
                if high_conf_outputs:
                    with open(final_outputs_csv, "w", newline="", encoding="utf-8") as f:
                        writer = csv.DictWriter(f, fieldnames=list(high_conf_outputs[0].keys()))
                        writer.writeheader()
                        writer.writerows(high_conf_outputs)
                    
                    with open(final_outputs_md, "w", encoding="utf-8") as f:
                        f.write(f"# High-Confidence License Plate Outputs — Run {RUN_ID}\\n\\n")
                        f.write(f"**Confidence Threshold:** $\\\\ge {config.FINAL_OUTPUT_CONF_THRESHOLD:.2f}$ | **Total Verified Vehicles:** {len(high_conf_outputs)}\\n\\n")
                        f.write("| Track ID | License Plate | Confidence | Avg Sharpness | Timestamp | Frame | Context Image | Plate Crop |\\n")
                        f.write("| :---: | :---: | :---: | :---: | :---: | :---: | :--- | :--- |\\n")
                        for item in high_conf_outputs:
                            ctx_link = f"[View Context ROI](file://{item['Context_Image_Path']})"
                            plate_link = f"[View Plate](file://{item['Plate_Bilateral_Path']})"
                            f.write(f"| **{item['Track_ID']}** | `{item['License_Plate']}` | **{item['Confidence']}** | {item['Avg_Sharpness']} | `{item['Timestamp']}` | {item['Frame_Number']} | {ctx_link} | {plate_link} |\\n")
                
                db.dump_preprocessing_csv(run_id=run_id, output_dir=OUTPUT_DIR)
                valid_count += 1  # prevent triggering multiple times for the same count
                
        except Exception:
            logger.exception("result_consumer failed on job=%s track=%s",
                              result.job_id, result.track_id)
        finally:
            result_q.task_done()
            
    # Cleanup files
    rich_file.close()
    raw_file.close()


def run() -> None:
    import time
    import json
    import subprocess
    
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
    high_conf_crops_dir = os.path.join(OUTPUT_DIR, "plate_crops_high_confidence")
    os.makedirs(crops_dir, exist_ok=True)
    os.makedirs(high_conf_crops_dir, exist_ok=True)
    worker_pool = WorkerPoolStage(
        processing_queue, result_queue,
        num_workers=config.NUM_WORKERS,
        config=WorkerConfig(
            save_crops=True, 
            crops_dir=crops_dir,
            high_conf_crops_dir=high_conf_crops_dir,
            high_conf_threshold=config.HIGH_CONF_CROP_THRESHOLD
        ),
    )
    fusion = TemporalFusionStage(FusionConfig())
    db     = DatabaseAnalyticsStage(DatabaseConfig(db_path=DB_PATH))

    rows: List[dict] = []
    raw_rows: List[dict] = []
    stats = {"fused_success": 0, "no_plate": 0}
    stop_event = threading.Event()
    
    # Initialize CSV headers for live streaming
    with open(RICH_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_RICH_FIELDS)
        writer.writeheader()
    with open(RAW_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_RICH_FIELDS)
        writer.writeheader()

    worker_pool.start()
    consumer_thread = threading.Thread(
        target=_result_consumer,
        args=(result_queue, stop_event, fusion, db, RUN_ID, rows, raw_rows, stats, crops_dir),
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
            
            # Write status.json for GUI
            try:
                status_dict = {
                    "frame_idx": frame_idx + 1,
                    "total_frames": total_frames,
                    "active_count": buffering.active_count,
                    "finalized_count": finalized_count,
                    "fused_success": stats["fused_success"],
                    "no_plate": stats["no_plate"],
                    "status": "running"
                }
                with open(os.path.join(OUTPUT_DIR, "status.json"), "w") as f:
                    json.dump(status_dict, f)
            except Exception:
                pass


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

    # ── Stage 8 Post-Processing: Plate-Guided Trajectory Stitching ──────────
    stitched_rows, alias_map = DatabaseAnalyticsStage.stitch_fragmented_rows(rows, max_frame_gap=300)
    if alias_map:
        logger.info("Stitched %d fragmented track segments: %s", len(alias_map), alias_map)
        # Apply alias map to raw_rows so video and visualizers show unified vehicle IDs
        stitched_raw_rows = []
        for r in raw_rows:
            r_copy = dict(r)
            tid = int(float(r_copy["car_id"]))
            if tid in alias_map:
                r_copy["car_id"] = alias_map[tid]
            stitched_raw_rows.append(r_copy)
        raw_rows = stitched_raw_rows
    else:
        stitched_rows = rows

    # ── Write rich CSV (fully-validated fused plates only) ──────────────────
    stitched_rows.sort(key=lambda r: (int(r["frame_nmr"]), int(float(r["car_id"]))))
    with open(RICH_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_RICH_FIELDS)
        writer.writeheader()
        writer.writerows(stitched_rows)
    logger.info("Rich CSV saved -> %s (%d rows)", RICH_CSV_PATH, len(stitched_rows))

    # ── Write raw detections CSV (every real detection, any track) ──────────
    raw_rows.sort(key=lambda r: (int(r["frame_nmr"]), int(float(r["car_id"]))))
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

    # ── High-Confidence Final Outputs Export ──────────────────────────────────
    final_outputs_csv = os.path.join(OUTPUT_DIR, "Final_outputs.csv")
    final_outputs_md  = os.path.join(OUTPUT_DIR, "Final_outputs.md")
    
    # Calculate video FPS for timestamp calculation
    try:
        cap = cv2.VideoCapture(config.VIDEO_SOURCE)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()
    except Exception:
        fps = 30.0

    # Group stitched detections by car_id to get highest confidence entry per vehicle
    high_conf_outputs = []
    if stitched_rows:
        df_stitched = pd.DataFrame(stitched_rows)
        for car_id, group in df_stitched.groupby("car_id"):
            best_row = group.sort_values(by="license_number_score", ascending=False).iloc[0]
            conf_val = float(best_row["license_number_score"])
            if conf_val >= config.FINAL_OUTPUT_CONF_THRESHOLD:
                f_idx = int(best_row["frame_nmr"])
                plate_txt = str(best_row["license_number"])
                sec = f_idx / fps
                ts = f"{int(sec//3600):02d}:{int((sec%3600)//60):02d}:{sec%60:06.3f}"
                
                # Check for existing crop images
                context_f = os.path.join(crops_dir, f"track_{int(car_id):03d}_frame_{f_idx:04d}_context.png")
                plate_b_f = os.path.join(crops_dir, f"track_{int(car_id):03d}_frame_{f_idx:04d}_plate_bilateral.png")
                
                high_conf_outputs.append({
                    "Track_ID":                  int(car_id),
                    "License_Plate":             plate_txt,
                    "Confidence":                f"{conf_val:.4f}",
                    "Timestamp":                 ts,
                    "Frame_Number":              f_idx,
                    "Context_Image_Path":        os.path.abspath(context_f) if os.path.exists(context_f) else context_f,
                    "Plate_Bilateral_Path":      os.path.abspath(plate_b_f) if os.path.exists(plate_b_f) else plate_b_f,
                })

    high_conf_outputs.sort(key=lambda x: (x["Frame_Number"], x["Track_ID"]))

    # Write Final_outputs.csv
    if high_conf_outputs:
        with open(final_outputs_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(high_conf_outputs[0].keys()))
            writer.writeheader()
            writer.writerows(high_conf_outputs)
        logger.info("High-confidence Final Outputs CSV saved -> %s (%d vehicles)", final_outputs_csv, len(high_conf_outputs))

        # Write easy-to-read Final_outputs.md
        with open(final_outputs_md, "w", encoding="utf-8") as f:
            f.write(f"# High-Confidence License Plate Outputs — Run {RUN_ID}\n\n")
            f.write(f"**Confidence Threshold:** $\\ge {config.FINAL_OUTPUT_CONF_THRESHOLD:.2f}$ | **Total Verified Vehicles:** {len(high_conf_outputs)}\n\n")
            f.write("| Track ID | License Plate | Confidence | Timestamp | Frame | Context Image | Plate Crop |\n")
            f.write("| :---: | :---: | :---: | :---: | :---: | :--- | :--- |\n")
            for item in high_conf_outputs:
                ctx_link = f"[View Context ROI](file://{item['Context_Image_Path']})"
                plate_link = f"[View Plate](file://{item['Plate_Bilateral_Path']})"
                f.write(f"| **{item['Track_ID']}** | `{item['License_Plate']}` | **{item['Confidence']}** | `{item['Timestamp']}` | {item['Frame_Number']} | {ctx_link} | {plate_link} |\n")
        logger.info("Human-readable Final Outputs MD saved -> %s", final_outputs_md)

    # ── Stage 9: interpolation (of the validated/fused CSV only) ─────────────
    interpolated_csv = None
    try:
        interpolated_csv = interpolate_csv(RICH_CSV_PATH)
        logger.info("Interpolated CSV -> %s", interpolated_csv)
    except Exception:
        logger.exception("Stage 9 (interpolation) failed")

    # ── Stage 10: annotated video ────────────────────────────────────────────
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

    try:
        with open(os.path.join(OUTPUT_DIR, "status.json"), "r+") as f:
            status_dict = json.load(f)
            status_dict["status"] = "complete"
            f.seek(0)
            json.dump(status_dict, f)
            f.truncate()
    except Exception:
        pass

if __name__ == "__main__":
    run()
