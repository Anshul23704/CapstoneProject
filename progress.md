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

### [Phase 5] Resolution Restoration & False-Positive Hardening (All 4 Fixes Implemented)

#### Issues Diagnosed from Visual Crop Audit:
1. **Low Pixel Resolution & Blurry Plates:** Stage 1's downscaling to $1280 \times 720$ shrank plate bounding boxes by $>50\%$, rendering small font characters illegible.
2. **False Positives (QR Codes & Square Stickers):** Low plate detector confidence ($0.25$) triggered on auto-rickshaw ads and square QR codes.
3. **Empty / Hallucinated Bounding Boxes:** Smooth bumper recesses without text were marked as plates.

#### Implemented Changes (User Requested):
1. **Native Video Resolution Ingestion (Fix 1):**
   * Modified `main_pipeline.py` to `target_resolution=None` in `FrameIngestionConfig`. Frames stream at full native source resolution ($1080\text{p}/4\text{K}$), instantly multiplying character pixel density $2\times\text{–}3\times$.
2. **Plate Confidence & Aspect Ratio Filtering (Fix 2):**
   * In `config.py` and `plate_detection.py`, raised `PLATE_CONF_THRESHOLD` to `0.40`.
   * Enforced geometric bounds `MIN_PLATE_ASPECT_RATIO = 1.3` and `MAX_PLATE_ASPECT_RATIO = 5.5`, automatically rejecting square QR codes ($w/h \approx 1.0$), tall vertical stickers, and extreme horizontal trim.
3. **Resolution & Sharpness Aware Frame Selection (Fix 3):**
   * In `stage5_job_creation.py`, modified ranking formula to $\text{Score} = \sqrt{\text{Plate Bounding Box Area}} \times \text{Sharpness} \times \text{Confidence}$. OCR workers now prioritize frames where the car is closest and plate resolution is maximal.
4. **Canny Edge Density Verification & Lanczos4 Upscaling (Fix 4):**
   * In `plate_utils.py` and `stage6_worker_pool.py`, implemented `plate_edge_density(plate_crop)` ($<0.02 \rightarrow \text{drop empty bodywork}$).
   * Upgraded plate upscaling interpolation from Bicubic to `cv2.INTER_LANCZOS4` for sharper text stroke edges.

---

### [Phase 6] Coordinate Realignment, 2-Line Plate OCR Merging & 90-Frame Tracker Persistence (`20260823_130547`)

#### Implemented Changes:
1. **Stage 2 Coordinate Inverse Scaling:**
   * Updated `DetectionTrackingStage` in `stage2_detection_tracking.py` to map $1920\text{p}$-downscaled bounding boxes back to the native input frame coordinate space ($2592 \times 1944$) using inverse scaling (`rx = int(round(x * inv_scale))`).
   * Bounding boxes passed to Stage 3 now align with Stage 2.5 full-frame plate detections ($1.00\text{ IoA}$).
2. **Two-Line Indian Plate EasyOCR Merging:**
   * Updated `_run_ocr_validated()` in `stage6_worker_pool.py` to sort vertically stacked EasyOCR text boxes, filter out `IND` hologram badges, and concatenate top/bottom lines (e.g. `KA01` + `AB1234` $\rightarrow$ `KA01AB1234`).
3. **90-Frame Longevity & Match Relaxation:**
   * Increased `TRACK_BUFFER = 90` frames (~3 seconds at 30 fps) and synchronized `BUFFER_TIMEOUT_FRAMES = 90` in `config.py`.
   * Relaxed `TRACK_MATCH_THRESHOLD = 0.45` to maintain track continuity across fast motion and camera panning.
4. **Plate Detector Aspect Ratio & Threshold Tuning:**
   * Lowered `MIN_PLATE_ASPECT_RATIO = 1.10` to admit square/stacked 2-line plates.
   * Adjusted `PLATE_CONF_THRESHOLD = 0.30`.

#### Run Results Comparison:
| Metric | Baseline | Run 1 (104434) | Run 2 (112933) | Run 3 (122935) | **Phase 6 Run (130547)** | Result |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Total Finalized Vehicles** | 0 | 319 | 178 | 148 | **136** | Clean tracking, reduced noise |
| **Raw Plate Bounding Boxes** | 0 | Handful | 739 | 52 (bugged) | **759** | Complete full-frame localization |
| **Fused Plates (SUCCESS)** | 0 | 2 | 10 | 1 (bugged) | **33** | **3.3x increase over previous best** |
| **Unique Plate Strings** | 0 | 2 | 10 | 1 | **29** | High diversity across vehicles |
| **Interpolated CSV Rows** | 0 | 12 | 58 | 15 | **1,113 rows** | Smooth multi-frame trajectory |

#### Sample Plate Readings in Phase 6:
* Track 896/906/886: **`AP39JL3969`** (Conf: 0.9810 across multiple readings)
* Track 629/699: **`KA53AC6692`** (Conf: 0.5514 across 20 readings)
* Track 20/543: **`KA25HE8870`** (Conf: 0.4549 across 13 readings)
* Track 4/3: **`KA01MP4882`** / **`KA01HP4882`** (Conf: 0.4467 across 13 readings)
* Track 830: **`KA04OS1206`** (Conf: 0.2263 across 16 readings)
* Track 847: **`KA01MN1413`** (Conf: 0.3814)

---

## 3. Comprehensive Stage Audit & Current File Map

| File Path | Primary Responsibility | Key Recent Edits |
| :--- | :--- | :--- |
| `config.py` | Centralized paths, thresholds, devices | `PLATE_CONF_THRESHOLD=0.30`, `MIN_ASPECT_RATIO=1.10`, `TRACK_BUFFER=90`, `BUFFER_TIMEOUT=90`, `TRACK_MATCH=0.45` |
| `plate_utils.py` | Positional heuristics, geometry, filters | `plate_edge_density`, `INTER_LANCZOS4`, `enhance_plate_crop_bilateral`, `soft_format_indian_plate` |
| `stage2_detection_tracking.py` | Vehicle detection & ByteTrack | Inverse coordinate scaling (`orig_w, orig_h`), `VEHICLE_CLASS_IDS={2}` |
| `plate_detection.py` | Stage 2.5 full-frame plate detector | Aspect ratio filtering $[1.10, 5.5]$, native frame processing |
| `stage3_active_buffering.py` | Trajectory buffer & IoA association | 90-frame timeout synchronization, IoA matching |
| `stage4_vehicle_finalization.py` | Quality checks & track closure | Drops short tracks ($<3$ frames) |
| `stage5_job_creation.py` | Frame selection for OCR workers | Combined $\text{Area} \times \text{Sharpness}$ ranking |
| `stage6_worker_pool.py` | Image enhancement, OCR, crop export | 2-line vertically stacked EasyOCR merging, `IND` hologram filtering, crop export |
| `stage7_temporal_fusion.py` | Multi-frame character voting | Modal length alignment, voting without hard rejection |
| `stage8_database_analytics.py` | SQLite DB logging & summary plots | Logs `ocr_results` with tracking metadata |
| `stage9_interpolation.py` | Bounding box smoothing | Interpolates validated vehicle detections |
| `stage10_visualize.py` | Annotated video creation | Renders bounding boxes from `results_raw_detections.csv` |
| `main_pipeline.py` | Threading orchestrator & CSV writer | Native resolution (`target_resolution=None`), raw detection streaming |

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

