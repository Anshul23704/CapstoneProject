# OCR Custom Model Architecture & Integration Context

## 1. System Objective & Role of the Document
This document provides a highly detailed architectural, algorithmic, and data-flow breakdown of the existing Automatic License Plate Recognition (ALPR) pipeline. It is specifically designed to be read by an AI model tasked with designing and training a custom OCR model on a remote, more powerful system. The custom OCR model will eventually replace or supplement the existing `EasyOCR`-based Stage 6 module in this pipeline. 

This document explicitly details how data is curated before reaching the OCR stage, what the exact inputs to the OCR model look like, how the OCR output is formatted, and how temporal fusion validates the OCR output across multiple frames.

---

## 2. Pre-OCR Data Flow: Detection and Spatial Association

Before the OCR model sees any image, the pipeline carefully isolates the license plate through a series of full-frame detection and association stages.

### 2.1 Stage 2 & 2.5: Vehicle and Plate Detection
- **Vehicle Detection & Tracking (Stage 2):** A YOLO model detects vehicles (`COCO class 2: car`). A `BotSORT` tracker assigns a persistent `track_id`. The frame is dynamically resized to a `MAX_FRAME_SIZE` (1920) before detection to prevent GPU memory exhaustion, but bounding boxes are inverse-scaled back to native full-frame coordinates.
- **Plate Detection (Stage 2.5):** A secondary YOLO model detects license plates on the **full, uncropped original frame** (fixed `imgsz=1280`). 
  - **Geometric Sanity Checks:** 
    - `MIN_PLATE_AREA = 80` px^2.
    - Aspect Ratio bounds: `MIN_PLATE_ASPECT_RATIO = 1.10` (allows 2-line plates), `MAX_PLATE_ASPECT_RATIO = 5.5` (filters wide trim strips).

### 2.2 Stage 3: Spatial Association (IoA)
- Plates are matched to vehicles using **Intersection-over-Area (IoA)**, not Intersection-over-Union (IoU). 
- The vehicle bounding box is expanded by `VEHICLE_BBOX_ASSOC_EXPAND = 0.12` (12% of height on all sides) to catch plates on bumpers.
- A plate belongs to a vehicle if `IoA >= 0.35` (35% of the plate's area falls within the expanded vehicle box).
- **Storage:** Instead of storing full frames in memory, the pipeline extracts a padded Region of Interest (ROI) crop around the vehicle (`ROI_PAD = 80` px). The offset of this ROI is saved so full-frame plate coordinates can be mapped into the ROI.

---

## 3. Data Curation & Job Creation (Input Selection)

Not every frame of a tracked vehicle is sent to the OCR model. 

### 3.1 Stage 4: Vehicle Finalization
- A vehicle track must exist for at least `MIN_FRAMES_FOR_RELIABLE = 3` frames to be considered valid for OCR.
- The pipeline flags tracks with extremely low visual diversity (static objects) or sudden bounding box area variance (ID switches).

### 3.2 Stage 5: Job Creation (Top-K Selection)
- A heuristic ranks all frames in a vehicle's track. Up to `TOP_K_FRAMES = 15` are selected and bundled into an OCR Job.
- **Ranking Function:** 
  `Score = sqrt(plate_area) * laplacian_sharpness * plate_detector_confidence`
- **Blur Gating:** Frames with a `laplacian_sharpness < BLUR_THRESHOLD (80.0)` are heavily penalized and skipped if sharper frames exist. 
- **Consequence for OCR Model Design:** The custom OCR model will primarily see the sharpest, highest-resolution plates available for a given vehicle, bounded to a maximum of 15 frames per vehicle.

---

## 4. Current OCR Stage Architecture (Stage 6)

The OCR stage runs asynchronously in a worker pool (`NUM_WORKERS = 3`). The custom OCR model must either replace this entire stage or act as the inference engine within it.

### 4.1 Plate Cropping & Deskewing
- The plate is cropped from the vehicle ROI using the mapped coordinates, expanded by `PLATE_PADDING = 10` pixels.
- **Edge Density Check:** If Canny edge density is `< 0.02`, the crop is discarded as a false positive (e.g., empty bumper).
- **Perspective Rectification:** The pipeline finds the largest bounding quadrilateral (via Canny edges and contours) and applies a `cv2.warpAffine` rotation to horizontally align the text if the tilt angle is between 1.5° and 35.0°.

### 4.2 Image Preprocessing Branches
Currently, two parallel image enhancement algorithms run, and OCR is executed on both to find the best read. 
- **Branch A (Bilateral):** 
  - Upscaled via Lanczos4 (`PLATE_UPSCALE_FACTOR = 2.0`).
  - Morphological Dilation (1 iteration, 2x2 kernel) to thin thick black characters on white backgrounds.
  - Bilateral Filtering (`d=11, sigmaColor=17, sigmaSpace=17`).
- **Branch B (Adaptive):**
  - Upscaled via Lanczos4 (2.0x).
  - CLAHE (Contrast Limited Adaptive Histogram Equalization, `clipLimit=2.0`, `tileGridSize=(8,8)`).
  - Adaptive Gaussian Thresholding (`blockSize=15, C=5`) to binarize shadows/glares.

*Architectural Note for Custom OCR:* If the custom OCR model is inherently robust to lighting variations, shadows, and low resolution, these preprocessing branches can be bypassed entirely, saving CPU overhead.

### 4.3 Advanced OCR Heuristics (Fallback handling)
The current OCR (EasyOCR) struggles with specific layouts, necessitating custom handling that the new model should ideally natively solve:
- **Two-Line Plates (Motorcycles/Commercial):** Current pipeline sorts bounding boxes vertically and merges them if $\ge 2$ text lines are detected.
- **Ultra-Wide Plates:** If Aspect Ratio $\ge 2.2$, the pipeline splits the crop horizontally. Left half (58%) is read with an alphanumeric allowlist; right half (52%) is read with a strict digit-only allowlist.
- **"IND" Hologram:** The current system hard-strips any leading text starting with "IND" to avoid incorporating the hologram into the license number.

---

## 5. Output Constraints & Post-OCR Temporal Fusion (Stage 7)

The custom OCR model's output must integrate with the Stage 7 Temporal Fusion algorithm, which synthesizes a single plate string from the ~15 per-frame OCR readings.

### 5.1 Indian Plate Formatting Rules
- Standard Format: `[State: 2 Letters] [RTO: 1-2 Digits] [Series: 1-3 Letters] [Number: 4 Digits]` (Total length: 8-10 chars).
- BH Series: `[Year: 2 Digits] BH [Number: 4 Digits] [Letters: 1-2]`
- **Positional Canonicalization (`soft_format_indian_plate`):**
  - Indices 0, 1: Forced to Letters (e.g., '0' -> 'O'). Snapped to valid Indian State Codes via Levenshtein distance (e.g., "MH", "DL").
  - Indices 2, 3: Forced to Digits (for standard plates).
  - Last 4 indices: Forced to Digits.

### 5.2 Temporal Fusion Algorithm
- **Input:** A list of `(text, confidence)` pairs for a single track.
- **Modal Alignment:** The system finds the most frequent string length (e.g., 10 chars) across the 15 frames. Any reading that is not exactly this length is discarded.
- **Character-Level Majority Voting:** For each character index, the system tallies the `confidence` scores of the proposed characters. The character with the highest summed confidence weight wins that position.
- **Consequence for Custom OCR:** The custom model **must** output a confidence score along with the text. Ideally, it should output a per-character confidence score, or at least a highly calibrated bounding-box level confidence, as this directly dictates the outcome of the temporal fusion vote.

### 5.3 Partial Stitching
If no frames yield a valid 8-10 character string, the pipeline falls back to partials. It bins characters by their relative X-coordinate `(x_min / width)` into 10 spatial bins and attempts to vote and stitch a plate across frames.

---

## 6. Design Mandates for the Custom OCR Model

Based on this architecture, the custom OCR model must be designed with the following constraints and objectives:

1. **Input Distribution:** 
   - The model will receive tightly cropped license plate images (`PLATE_PADDING = 10`), potentially perspective-deskewed.
   - Resolutions will vary significantly. The custom model architecture (e.g., a CRNN with a flexible CNN backbone, or a Vision Transformer) must handle dynamic input widths, or the pipeline must pad/resize to a fixed tensor size without distorting aspect ratio.
2. **Native Dual-Line Support:** 
   - The model must natively recognize and correctly order characters in 2-line plates (common in India). This removes the need for the fragile bounding-box sorting heuristic currently in use.
3. **Character-Level Confidence:** 
   - The model's loss function (e.g., CTC Loss or Cross-Entropy in an Attention decoder) should yield calibrated softmax probabilities per character. Stage 7's Temporal Fusion relies entirely on confidence-weighted voting.
4. **Vocabulary / Classes:** 
   - `0-9, A-Z`. The model must be highly discriminative between common confusions (0/O, 8/B, 1/I, 5/S) without solely relying on the downstream `soft_format_indian_plate` heuristic.
5. **Contextual Robustness:**
   - The model should be trained with augmentations mimicking the harsh conditions that currently require Branch A (Bilateral) and Branch B (Adaptive CLAHE) so that these CPU-bound preprocessing steps can be deprecated. (e.g., simulate shadows, glare, motion blur, and low contrast).
