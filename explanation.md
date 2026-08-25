# Automatic License Plate Recognition (ALPR) Pipeline Explanation

This document explains the architecture and flow of the ALPR project. The system is designed as a modular, multi-stage pipeline that processes a raw video stream, detects and tracks vehicles, smartly buffers frames, asynchronously reads license plates, fuses data over time for accuracy, and generates visualizations and analytics.

## Overall Pipeline Flow

The pipeline transforms a raw video into a rich dataset of tracked vehicles and recognized license plates. The flow from stage to stage is as follows:

1. **Video Ingestion:** The video stream is read frame-by-frame.
2. **Detection & Tracking:** Vehicles are detected and assigned persistent tracking IDs across frames.
3. **Smart Buffering:** As vehicles move through the frame, crops of their regions are temporarily buffered in memory.
4. **Finalization:** When a vehicle leaves the scene, its track is finalized and evaluated for quality.
5. **Job Dispatch:** The sharpest frames from the vehicle's buffer are packaged into a processing job.
6. **Worker Pool (OCR):** Background threads detect license plates on these sharp frames and run Optical Character Recognition (OCR) to read the text.
7. **Temporal Fusion:** OCR readings from multiple frames of the same vehicle are merged (voted on) to produce a single, highly accurate license plate string.
8. **Analytics & Database:** Results are saved to a local database, and visual reports are generated.
9. **Interpolation:** Any missing frames in a vehicle's tracking trajectory are smoothed over using mathematical interpolation.
10. **Visualization:** The final tracked bounding boxes and license plate text are drawn onto a new output video.

---

## Detailed Stage Breakdown

### Stage 1: Frame Ingestion (`stage1_frame_ingestion.py`)
- **What it does:** Reads the raw video stream using OpenCV.
- **Image/Video specific:** It handles resizing frames to a target resolution (e.g., 1280x720) to maintain consistent processing speeds. It manages video FPS, gracefully handles corrupted or dropped frames, and yields a continuous stream of `(frame, frame_idx, timestamp)` tuples to the rest of the pipeline.

### Stage 2: Detection and Tracking (`stage2_detection_tracking.py`)
- **What it does:** Identifies vehicles in the current frame and tracks them over time.
- **Image/Video specific:** Uses a YOLO object detection model to find vehicles (cars, buses, trucks). It then passes these bounding boxes to a `ByteTrack` tracker, which assigns a persistent `track_id` to each vehicle so the system knows it's the same physical car across consecutive video frames.

### Stage 3: Active Buffering (`stage3_active_buffering.py`)
- **What it does:** Accumulates frames of a tracked vehicle while it remains in the scene.
- **Image/Video specific:** Instead of saving full-resolution video frames into memory (which would cause massive memory bloat), it extracts a "padded Region of Interest (ROI)" around the vehicle's bounding box. It keeps an active list of these ROI crops until the vehicle times out or leaves the camera's view.

### Stage 4: Vehicle Finalization (`stage4_vehicle_finalization.py`)
- **What it does:** Wraps up a vehicle's track once it is no longer visible.
- **In general:** It analyzes the buffered frames to ensure the track represents a valid vehicle. It computes the track duration, checks for abnormal bounding box variations (which might indicate ID switching errors), and checks visual diversity (using image hashing) to discard static false positives. 

### Stage 5: Job Creation (`stage5_job_creation.py`)
- **What it does:** Prepares the finalized vehicle data for OCR processing.
- **Image/Video specific:** Not all frames are good for reading text. This stage calculates the sharpness (Laplacian variance) of every cropped frame in the buffer. It selects the top-K sharpest frames, bundles them into a `ProcessingJob`, and dispatches it to a queue for the worker pool.

### Stage 6: Worker Pool (Plate Detection & OCR) (`stage6_worker_pool.py`)
- **What it does:** Extracts and reads the license plate text asynchronously.
- **Image/Video specific:** Background worker threads pick up jobs from the queue. For each sharp frame:
  1. A specific YOLO license plate detector finds the plate.
  2. A spatial containment check ensures the plate is physically inside the vehicle's bounding box (preventing plates from being assigned to neighboring cars).
  3. The plate crop is preprocessed (resized, converted to grayscale, thresholded).
  4. EasyOCR extracts the raw text.
  5. The text undergoes strict format validation (e.g., ensuring characters align with a standard license plate format) and positional correction (e.g., fixing an 'O' that should be a '0').

### Stage 7: Temporal Fusion (`stage7_temporal_fusion.py`)
- **What it does:** Combines multiple readings of the same plate into one highly accurate result.
- **In general:** Because OCR is not perfect on single frames, a vehicle might yield readings like `AB12CDE` on one frame and `A812CDE` on another. This stage aligns the readings by length and performs a confidence-weighted character-level majority vote to deduce the correct plate. The fused string is strictly validated one last time.

### Stage 8: Database & Analytics (`stage8_database_analytics.py`)
- **What it does:** Persists results and generates insights.
- **In general:** It safely writes every vehicle's OCR result, confidence score, and status to an SQLite database (`ocr_results.db`) using thread locks. It can also generate a matplotlib analytics report (saved as a PNG) visualizing confidence distributions, pipeline status breakdowns, and plates detected over time.

### Stage 9: Interpolation (`stage9_interpolation.py`)
- **What it does:** Smooths out the tracking data.
- **In general:** Object detectors occasionally miss a vehicle for a frame or two. This stage reads the generated CSV data and uses linear interpolation (`scipy.interpolate.interp1d`) to mathematically estimate and fill in the missing bounding box coordinates for frames where the vehicle was not explicitly detected.

### Stage 10: Visualization (`stage10_visualize.py`)
- **What it does:** Renders the final output video.
- **Image/Video specific:** It re-reads the original video and the interpolated CSV results. Frame-by-frame, it uses OpenCV to draw clean bounding boxes around the vehicles, highlights the license plates, and draws an overlaid graphic showing the recognized license plate text and a zoomed-in crop of the plate itself. The annotated frames are written to an output `out.mp4` video.
