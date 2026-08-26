# Pipeline Changes & Reversion Guide

This document tracks major architectural and processing changes made to the pipeline, detailing the previous state, the new state, and instructions on how to revert cleanly if necessary.

## 1. Tracking Algorithm Upgrade (ByteTrack -> BoT-SORT)

**Previous State:** 
`stage2_detection_tracking.py` used `supervision.ByteTrack` fed by YOLO detection results. 
**New State:** 
`stage2_detection_tracking.py` uses Ultralytics YOLO's native built-in tracker running BoT-SORT (`tracker="botsort.yaml"`). This uses appearance-based matching (ReID) to bridge gaps when bounding boxes shift rapidly.
**How to Revert:**
1. In `stage2_detection_tracking.py`, re-add `self._tracker = sv.ByteTrack(...)` in `__init__`.
2. Revert the `self._model.track(...)` call back to `self._model(...)`.
3. Re-add `tracked = self._tracker.update_with_detections(raw_detections)` and pass `tracked` to `_to_dict` instead of passing `detections` directly.

## 2. Adaptive Filter Removal

**Previous State:** 
The pipeline maintained a dual-branch processing system (Bilateral Filter vs Adaptive Filter) all the way through Stage 6 (Worker Pool), Stage 7 (Fusion), and Stage 8 (Analytics).
**New State:** 
The Adaptive Filter branch has been completely removed to halve the OCR processing time and simplify the pipeline.
**How to Revert:**
Reverting this requires restoring code across 5 files: `plate_utils.py`, `stage6_worker_pool.py`, `main_pipeline.py`, `stage8_database_analytics.py`, and `GUI/app.py`. 
*Use git to revert this specific commit if needed, as manually adding back the dual-branch logic is complex.*

## 3. Morphological Erosion for M/N/H Confusion

**Previous State:** 
`plate_utils.py:enhance_plate_crop_bilateral` simply applied a bilateral filter. Thick character strokes caused the OCR model to frequently confuse `M`, `N`, and `H`.
**New State:** 
Added a slight morphological erosion (using a 2x2 or 3x3 kernel) to thin out the character strokes in `enhance_plate_crop_bilateral` before passing to OCR. This helps preserve the white space inside letters like M and H.
**How to Revert:**
In `plate_utils.py`, go to `enhance_plate_crop_bilateral` and comment out or remove the `cv2.erode(...)` lines that are applied to the grayscale image.
