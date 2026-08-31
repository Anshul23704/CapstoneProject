"""
config.py — Single source of truth for paths, thresholds and constants.

WHY THIS FILE EXISTS
─────────────────────
Every stage previously hardcoded its own copy of model paths
(E:\\Capstone\\...), class-id sets, and thresholds. That's how stage6 and
main_pipeline.py ended up with two different, silently-diverging copies of
license_complies_format/format_license, and how VEHICLE_CLASSES in
main_pipeline.py (which included motorcycles) drifted from the {car,bus,truck}
set used in stage2. Centralizing these here means every stage config
dataclass now *derives its defaults* from one place instead of restating them.

All paths are resolvable from environment variables so the pipeline runs on
any machine/CI box, not just the one with an E:\\ drive.
"""
from __future__ import annotations

import os

try:
    import torch
    if torch.cuda.is_available():
        DEVICE = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        DEVICE = "mps"
    else:
        DEVICE = "cpu"
except Exception:
    DEVICE = "cpu"

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

VEHICLE_MODEL_PATH = os.getenv("VEHICLE_MODEL_PATH", "yolov8s.pt")

PLATE_MODEL_PATH = os.getenv(
    "PLATE_MODEL_PATH",
    os.path.join(MODEL_DIR, "license_plate_yolo26m-3", "weights", "best.pt")
    if os.path.exists(os.path.join(MODEL_DIR, "license_plate_yolo26m-3", "weights", "best.pt"))
    else os.path.join(MODEL_DIR, "best.pt")
)

VIDEO_SOURCE = os.getenv(
    "VIDEO_SOURCE",
    os.path.join(PROJECT_ROOT, "dataset", "test3.mp4")
)

OUTPUT_ROOT = os.getenv(
    "OUTPUT_ROOT",
    os.path.join(PROJECT_ROOT, "output")
)

# ── Vehicle classes (COCO ids) ────────────────────────────────────────────────
# ==============================================================================
# REMINDER: Currently configured to track CARS ONLY (COCO class 2).
# If you ever want to re-enable buses and trucks in the future, change this back to:
# VEHICLE_CLASS_IDS = {2, 5, 7}   # 2: car, 5: bus, 7: truck
# (Motorcycle is COCO class 3)
# ==============================================================================
VEHICLE_CLASS_IDS = {2}   # CARS ONLY

# ── Detection / tracking ──────────────────────────────────────────────────────
DETECTION_CONF_THRESHOLD = 0.25
DETECTION_IOU_THRESHOLD  = 0.45
MAX_FRAME_SIZE            = 1920   # resize cap before detection (GPU memory)

TRACK_BUFFER            = 90     # 90 frames (~3.0s at 30fps) persistence buffer
TRACK_MATCH_THRESHOLD   = 0.45   # relaxed threshold to maintain tracks during fast motion / pan

# ── Buffering (Stage 3) ───────────────────────────────────────────────────────
# Unlimited buffering: a vehicle keeps all its frames until it leaves the scene.
BUFFER_MAX_SIZE          = None   # None = keep all frames (no sliding window pop)
BUFFER_TIMEOUT_FRAMES    = 90     # synchronized with TRACK_BUFFER (90 frames)
BUFFER_FORCE_FINALIZE_AT = None   # None = do not force finalize; let vehicle track complete naturally
ROI_PAD                  = 80     # px padding around vehicle bbox for the stored ROI

# ── Finalization (Stage 4) ────────────────────────────────────────────────────
# Slide 16: "Ensure minimum frame count for reliable processing". Abstract
# (Slide 4) promises tracking "across 5-7 frames". The old default (1) enforced
# nothing. We don't require the full 5-7 (short tracks near frame edges are
# common and still useful), but we do require enough frames for temporal
# fusion to mean something.
MIN_FRAMES_FOR_RELIABLE = 3
AREA_STD_THRESHOLD      = 5000.0
DIVERSITY_RATIO_LIMIT   = 0.95

# ── Job creation (Stage 5) ────────────────────────────────────────────────────
TOP_K_FRAMES    = 15
BLUR_THRESHOLD  = 80.0   # Laplacian variance below this = too blurry/occluded

# ── Plate detection (Stage 2.5) ────────────────────────────────────────────────
# ARCHITECTURE CHANGE: plate detection used to run inside Stage 6's worker
# threads, on a small padded per-vehicle crop, independently for every one
# of a vehicle's top-k selected frames (up to TOP_K_FRAMES detector calls
# PER VEHICLE, each at a different effective resolution depending on that
# crop's size). Two runs of tuning that in isolation both regressed
# (imgsz guesses that shrank small crops further; IoA slack that let in
# non-plate rectangles) — the crop-based approach fights itself no matter
# how the knobs are set, because the detector never sees a plate at the
# scale/context it's actually good at.
#
# Plate detection now runs ONCE PER FRAME, on the FULL frame, at a fixed
# resolution sized to the ingestion frame itself — the same way your
# reference full-frame OCR tool sees it. One detector call finds every
# plate in that frame for every vehicle at once. Association to a specific
# track happens afterward in Stage 3, in full-frame coordinates only — no
# ROI offsets anywhere, which removes the entire class of coordinate bugs
# the last two tuning passes were fighting.
PLATE_CONF_THRESHOLD = 0.30
# Ingestion frames run at native resolution (target_resolution=None).
# imgsz=1280 or 1920 covers native resolution cleanly.
PLATE_DETECT_IMGSZ   = 1280
MIN_PLATE_AREA        = 80   # px^2
MIN_PLATE_ASPECT_RATIO = 1.10  # allows 2-line square plates (~1.1-1.3) while rejecting extreme square/vertical artifacts
MAX_PLATE_ASPECT_RATIO = 5.5   # filters out overly wide trim strips

PLATE_PADDING        = 10   # px padding added around the matched plate box before crop/enhance
PLATE_UPSCALE_FACTOR = 2.0

# Slide 5 panel-feedback commitment: association is by Intersection-over-Area
# (of the plate box), not strict containment. Threshold kept moderate (not
# loosened further) now that both boxes live in the same full-frame
# coordinate space — the earlier IoA rejections were partly a byproduct of
# ROI-local coordinate drift, not the threshold itself.
PLATE_VEHICLE_IOA_THRESHOLD = 0.35

# The vehicle (COCO) detector's box is frequently a few % too tight around
# the visible body and clips the rear bumper/plate region, especially at an
# angle. Expand the vehicle box by this fraction of its own height *only*
# for the plate-association test (never for cropping/storage/CSV output).
# Applied once, in full-frame coordinates, to every vehicle — unlike the
# previous per-ROI version this can't accidentally pull in a NEIGHBORING
# vehicle's plate, because association now happens against the specific
# frame's specific vehicle box, at the same scale the detector actually ran.
VEHICLE_BBOX_ASSOC_EXPAND = 0.12

# ── Worker pool / OCR (Stage 6) ────────────────────────────────────────────────
# Workers now only run OCR (plate detection moved to Stage 2.5 above), so
# NUM_WORKERS is sized for OCR throughput, not shared with plate-model load.
NUM_WORKERS = 3     # matches "Worker Thread 1/2/3" in the architecture diagram

# ── Temporal fusion (Stage 7) ─────────────────────────────────────────────────
MIN_READINGS_FOR_FUSION = 2

# ── Final Outputs Filtering ───────────────────────────────────────────────────
FINAL_OUTPUT_CONF_THRESHOLD = 0.20  # Minimum confidence required for high-confidence final output table
HIGH_CONF_CROP_THRESHOLD = 0.40     # Threshold for routing plate crops to the high confidence folder
