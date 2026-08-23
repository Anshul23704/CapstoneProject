# Automatic License Plate Recognition (ALPR) Pipeline — Progress & State Checkpoint

> **Purpose of this Document:**
> This file acts as a comprehensive progress log, technical audit trail, and resume checkpoint for the ALPR Capstone Project codebase. It details all modifications made from baseline, stage-by-stage improvements, test run results, diagnosed bottlenecks, and future recommendations.

---

## 1. System Architecture Overview

The pipeline processes video input through 10 modular stages:
* **Stage 1 (Ingestion):** Multi-threaded frame decoding (`FrameIngestionStage`) resizing to a uniform $1280 \times 720$.
* **Stage 2 (Vehicle Tracking):** YOLOv8 + ByteTrack (`DetectionTrackingStage`) assigning unique track IDs to vehicles.
* **Stage 2.5 (Plate Detection):** YOLO plate detection model running **full-frame** at native $1280 \times 720$ resolution once per frame.
* **Stage 3 (Active Buffering & IoA Association):** Vehicle trajectory accumulation with Intersection-over-Area ($\text{IoA} \ge 0.35$) matching of plates to vehicles.
* **Stage 4 (Vehicle Finalization):** Finalization of car tracks upon scene exit or frame timeout.
* **Stage 5 (Job Creation & Quality Gate):** Laplacian variance sharpness filtering and selection of top-$k$ frames per vehicle.
* **Stage 6 (Worker Pool & OCR):** Multi-threaded image enhancement (bilateral filter + upscale) and EasyOCR text extraction.
* **Stage 7 (Temporal Fusion):** Multi-frame character-level majority voting across frame readings.
* **Stage 8 (Database & Analytics):** SQLite database persistence (`ocr_results`) and metric visualization.
* **Stage 9 (Interpolation):** Linear bounding-box interpolation across missing intermediate frames.
* **Stage 10 (Visualization):** Annotated video rendering (`annotated_output.mp4`) displaying tracked cars and plate boxes.

---

## 2. Iteration Log & Chronological Run Analysis

```
Baseline Run (0 Plates) ──► Teammate Merge ──► Run 1 (319 Cars / 2 Plates) ──► Run 2 (178 Cars / 10 Plates) ──► Soft Positional & Crop Audit
```

---

### [Phase 0] Baseline Setup & Initial Diagnosis

#### Initial Problem:
* Running the pipeline on fresh environments failed or produced **0 detected plates** despite vehicles passing through the FOV.

#### Root Causes Diagnosed:
1. Missing environment dependencies (`pandas`, `easyocr`, `ultralytics`, etc.).
2. Stage 6 ran plate detection on small, padded vehicle crops at dynamic resolutions, causing the plate detector to fail due to lack of full-frame context.
3. Strict hardcoded 7-character Brazilian regex template (`LL DD LLL`) rejected all Indian plates (9–10 characters) 100% of the time.

---

### [Phase 1] Teammate Code Integration & Central Configuration

#### Implemented Changes (User Requested):
* Merged teammate's refactored architecture with Stage 2.5 full-frame plate detection.
* Centralized paths and hyperparameters into `config.py`.
* Configured dynamic platform support (`mps` on Apple Silicon macOS, `cuda` on Nvidia, `cpu` fallback).

---

### [Phase 2] Test Run 1 (`20260823_104434`) — Diagnosis & Bottleneck Analysis

#### Run Results:
* **Finalized Vehicles:** 319 (Massive over-count for test video).
* **Fused Plates:** 2.
* **Plate Detections:** Only a handful.

#### Root Cause Analysis:
1. **Vehicle Over-Segmentation (319 Vehicles):**
   * Multi-class tracking tracked buses, motorcycles, trucks, and background clutter.
   * Hardcoded `BUFFER_FORCE_FINALIZE_AT = 20` forcibly severed vehicle tracks every 20 frames, fragmenting a single car passing through the camera into 10–15 duplicate track IDs.
2. **Low OCR Conversion (2 Plates):**
   * Preprocessing used CLAHE + unsharp masking, amplifying high-frequency noise/glare on reflective plates.
   * Strict Indian regex check in intermediate workers (`stage6_worker_pool.py`) dropped any read that had a single OCR character confusion before it could reach Stage 7 multi-frame fusion.

---

### [Phase 3] Test Run 2 (`20260823_112933`) — Noise Reduction & Bilateral Preprocessing

#### Implemented Changes (User Requested):
1. **Targeted Vehicle Filtering:**
   * Configured `config.py` with `VEHICLE_CLASS_IDS = {2}` (Cars only) with visible warning comments for re-enabling bus/truck classes.
2. **Continuous Buffering:**
   * Updated `config.py` and `stage3_active_buffering.py` with `BUFFER_MAX_SIZE = None` and `BUFFER_FORCE_FINALIZE_AT = None`.
3. **Bilateral Preprocessing:**
   * Added `enhance_plate_crop_bilateral` in `plate_utils.py` (Grayscale $\rightarrow$ Bilateral Filter $d=11, \sigma=17$ $\rightarrow$ Cubic Upscale). Old CLAHE function kept commented out in `stage6_worker_pool.py`.
4. **Loosened Intermediate Regex:**
   * Removed intermediate `license_complies_format` filter in `stage6_worker_pool.py` and `stage7_temporal_fusion.py`.
   * Staged strict validation as a disabled comment block in `main_pipeline.py` right before CSV writing.

#### Run Results:
| Metric | Previous Run | Test Run 2 | Result |
| :--- | :--- | :--- | :--- |
| **Total Finalized Vehicles** | 319 | **178** | **44% noise reduction** |
| **Raw Plate Bounding Boxes** | ~2 | **739 detections** across **60 cars** | Clean plate localization |
| **Fused Plates (OCR)** | 2 | **10 cars** | **5x increase in OCR reads** |
| **Rich CSV Rows** | 12 | **58 rows** | Smooth multi-frame readings |

#### Detected Plate Readings:
* Track 1: `KAO1MPL882` (20 readings, conf: 0.5057)
* Track 44: `KAZ5NE0BZ` (3 readings, conf: 0.1530)
* Track 355: `FK42VE8879` (5 readings, conf: 0.0850)
* Track 400: `E53406692` (9 readings, conf: 0.1139)
* Track 548: `4P23JL3969` (8 readings, conf: 0.1734)
* Track 568: `451497037` (4 readings, conf: 0.2086)
* Track 418 (`W1406593`), Track 446 (`352406695`), Track 526 (`K425E0670`), Track 541 (`00451205`)

---

### [Phase 4] Image Saving Audit & Soft Positional Correction

#### Implemented Changes (User Requested):
1. **Automated Plate Crop & Context Export:**
   * Added automated image saving into `output/<RUN_ID>/plate_crops/`:
     * `track_<ID>_frame_<NUM>_plate.png`: Isolated enhanced plate crop fed to OCR.
     * `track_<ID>_frame_<NUM>_context.png`: Vehicle ROI with green bounding box and OCR prediction label overlay.
2. **Soft Positional Heuristic Correction:**
   * Implemented `soft_format_indian_plate` in `plate_utils.py` and connected to `stage6_worker_pool.py`.
   * **Rule Layout (`SS DD L{1,3} DDDD`):**
     * **Indices `0, 1` (State Code):** Must be letters. Digits/confusions converted (`4` $\rightarrow$ `A`, `0` $\rightarrow$ `O`, `1` $\rightarrow$ `I`, `5` $\rightarrow$ `S`, `8` $\rightarrow$ `B`).
     * **Indices `2, 3` (RTO Code):** Must be digits. Letters converted (`O` $\rightarrow$ `0`, `I`/`L` $\rightarrow$ `1`, `Z` $\rightarrow$ `2`, `S` $\rightarrow$ `5`, `B` $\rightarrow$ `8`).
     * **Indices `4, 5` (Series Code):** Must be letters for 10-char plates.
     * **Indices `6..9` (Registration Number):** Must be digits (`L` $\rightarrow$ `1`, `O`/`D` $\rightarrow$ `0`, `B` $\rightarrow$ `8`, `Z` $\rightarrow$ `2`).
   * **Correction Results:**
     * `KAO1MPL882` $\rightarrow$ **`KA01MP1882`**
     * `4P23JL3969` $\rightarrow$ **`AP23JL3969`**
     * `KAZ5NE0BZ` $\rightarrow$ **`KA25NE082`**
     * `451497037` $\rightarrow$ **`AS1497037`**
   * *Non-destructive:* Never drops/rejects strings that don't conform.

---

## 3. Comprehensive Stage Audit & Current File Map

| File Path | Primary Responsibility | Key Recent Edits |
| :--- | :--- | :--- |
| `config.py` | Centralized paths, thresholds, devices | `VEHICLE_CLASS_IDS={2}`, `BUFFER_FORCE_FINALIZE_AT=None` |
| `plate_utils.py` | Positional heuristics, geometry, filters | Added `enhance_plate_crop_bilateral`, `soft_format_indian_plate`, expanded confusion maps |
| `stage2_detection_tracking.py` | Vehicle detection & ByteTrack | Filters by `VEHICLE_CLASS_IDS` |
| `plate_detection.py` | Stage 2.5 full-frame plate detector | YOLOv8 plate inference at $1280 \times 720$ |
| `stage3_active_buffering.py` | Trajectory buffer & IoA association | Unlimited buffer mode, IoA matching |
| `stage4_vehicle_finalization.py` | Quality checks & track closure | Drops short tracks ($<3$ frames) |
| `stage5_job_creation.py` | Frame selection for OCR workers | Sharpness sorting, top-$k$ frame selection |
| `stage6_worker_pool.py` | Image enhancement, OCR, crop export | Saves plate/context crops, bilateral filtering, soft correction |
| `stage7_temporal_fusion.py` | Multi-frame character voting | Modal length alignment, voting without hard rejection |
| `stage8_database_analytics.py` | SQLite DB logging & summary plots | Logs `ocr_results` with tracking metadata |
| `stage9_interpolation.py` | Bounding box smoothing | Interpolates validated vehicle detections |
| `stage10_visualize.py` | Annotated video creation | Renders bounding boxes from `results_raw_detections.csv` |
| `main_pipeline.py` | Threading orchestrator & CSV writer | Wires `plate_crops/` export, deferred strict validation |

---

## 4. Suggested Future Enhancements

These are recommended optimizations that can be selectively enabled in future iterations:

### Suggested Change 1: State Code Dictionary Fuzzy Match
* **Concept:** Match the first 2 letters against the 36 valid Indian State/UT abbreviations (`KA`, `MH`, `DL`, `TS`, `AP`, `TN`, `KL`, `HR`, `UP`, `MP`, `GJ`, `WB`, `RJ`, etc.) using Levenshtein edit distance:
  * `ES` / `3S` $\rightarrow$ `TS`
  * `FK` $\rightarrow$ `KA`
  * `OO` $\rightarrow$ `OD` / `DL`

### Suggested Change 2: Region-Split Dual-Pass OCR Whitelisting
* **Concept:** Crop the plate into a left portion (alphabetic state + series) and right portion (numeric registration digits), and invoke EasyOCR with strict whitelists:
  * Left: `allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ"`
  * Right: `allowlist="0123456789"`

### Suggested Change 3: Plate Perspective Deskewing
* **Concept:** Use edge detection and contour fitting to compute an affine transform that rectifies angled plates horizontally before OCR.

### Suggested Change 4: Fine-Tuned License Plate OCR (TrOCR / CRNN)
* **Concept:** Replace or augment EasyOCR with a compact CRNN or Microsoft TrOCR fine-tuned specifically on Indian vehicle plate fonts (FE-Schrift / Mandated IND font).
