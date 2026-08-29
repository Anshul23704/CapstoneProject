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

## 4. OCR Engine Switch (PaddleOCR -> EasyOCR Reversion)

**Previous State:** 
We briefly switched to `paddleocr.PaddleOCR` to try and improve accuracy on slanted/complex texts.
**New State:** 
Reverted back to `easyocr.Reader`. We discovered that loading both PaddlePaddle (PaddleOCR) and PyTorch (Ultralytics/YOLO) into the exact same Python process on a Mac causes a fatal C++ symbol collision in the underlying tensor libraries (PyTorch NMS throws a `PreconditionNotMet` error originating from Paddle's `dense_tensor_impl.cc`).
**Note for Windows Migration:** When running this project on a Windows PC with an NVIDIA GPU (e.g., RTX 4080), this collision does not happen because the frameworks will correctly allocate via CUDA.
**How to Switch Back to PaddleOCR on Windows:**
1. In `requirements.txt`, swap `easyocr` for `paddlepaddle-gpu` and `paddleocr`. Run `pip install -r requirements.txt`.
2. In `stage6_worker_pool.py`, replace `import easyocr` with `from paddleocr import PaddleOCR`.
3. In `Worker.__init__`, initialize with `self.ocr = PaddleOCR(use_textline_orientation=True, lang='en')`.
4. In `Worker._run_ocr_validated`, update the OCR call to translate PaddleOCR's nested list output back into the expected `(bbox, text, conf)` tuple format that the rest of the pipeline expects.

## 5. Re-Implemented Adaptive Thresholding (Branch B)

**Previous State:** 
The pipeline was only running a Bilateral Filter (which preserves edges but doesn't handle uneven lighting well).
**New State:** 
Re-introduced `enhance_plate_crop_adaptive` using CLAHE and Gaussian Adaptive Thresholding. Both Bilateral and Adaptive filters are now run in parallel during Stage 6, and the OCR engine chooses whichever one yields the highest confidence for that specific frame.
**How to Revert:**
In `stage6_worker_pool.py`, remove the `plate_adaptive` preprocessing step and revert the comparison logic to only use the bilateral filter branch.

## 6. Average Sharpness Metric

**Previous State:** 
The pipeline output included confidence scores, but no metrics on the actual image quality / blurriness of the plates that fed into those scores. This resulted in false positives from plates that were extremely far away and heavily pixelated/blurry but still triggered high OCR confidence on noise.
**New State:** 
The pipeline now calculates the Laplacian variance (sharpness) of every plate crop before feeding it into the OCR engine. The average sharpness across all processed frames for a given vehicle track is now exported to `ocr_results.db`, displayed in the Streamlit GUI (`Avg Sharpness`), and logged as a new column in `Final_outputs.csv` and `Final_outputs.md`. This allows for easy filtering of false positives caused by low-resolution / far-away plate captures.
**How to Revert:**
1. In `stage6_worker_pool.py`, remove `avg_sharpness` from the `RecognitionResult` dataclass and the `_process` return statement.
2. In `main_pipeline.py`, remove `Avg_Sharpness` from the `best_detections` dictionary.
3. In `stage8_database_analytics.py`, remove the `avg_sharpness` column from `_CREATE_TABLE` and the `insert_result` logic.
