"""
plate_utils.py — Shared plate-text formatting, geometry, and image-quality
helpers used by Stage 5, Stage 6, and Stage 7.

WHY THIS FILE EXISTS
─────────────────────
license_complies_format / format_license previously existed as THREE
independent copies (stage6_worker_pool.py, stage7_temporal_fusion.py,
main_pipeline.py) — already showing drift (stage7's format_license used a
dynamic `len(text)` loop while stage6/main used a hardcoded `range(7)`).
Sharpness (Laplacian variance) was duplicated between stage5 and stage6.
IoA-based plate/vehicle association is new (previously stage6 used strict
containment, contradicting Slide 5's "IoA ≥ 50%" panel-response promise).
Centralizing avoids the two copies silently disagreeing again.
"""
from __future__ import annotations

import re
import string
from typing import Tuple

import cv2
import numpy as np

BBox = Tuple[int, int, int, int]

# ── OCR confusion tables ──────────────────────────────────────────────────────
CHAR_TO_INT = {'O': '0', 'D': '0', 'Q': '0', 'I': '1', 'L': '1', 'Z': '2', 'J': '3', 'A': '4', 'S': '5', 'G': '6', 'B': '8'}
INT_TO_CHAR = {'0': 'O', '1': 'I', '2': 'Z', '3': 'J', '4': 'A', '5': 'S', '6': 'G', '7': 'T', '8': 'B'}

_ALPHA_OK = set(string.ascii_uppercase) | set(INT_TO_CHAR.keys())
_DIGIT_OK = set('0123456789') | set(CHAR_TO_INT.keys())

# ── Indian vehicle registration plate formats ──────────────────────────────────
# FIX: the previous version hardcoded a 7-char "LL DD LLL" layout (the
# Brazilian/Mercosur format this codebase was originally templated from).
# That meant every real Indian plate — which is 9-10 chars — failed
# validation 100% of the time regardless of OCR quality: this, not OCR
# accuracy, was why fused_success stayed at 0 across an entire run.
#
# Standard format: SS DD L{1,3} DDDD
#   SS = state code (2 letters), DD = RTO code (1-2 digits),
#   L{1,3} = series (1-3 letters), DDDD = registration number (4 digits)
#   e.g. KA01AB1234, MH12DE1433, DL3CAB1234, KA51MJ7743
# BH-series (2021+): DD BH DDDD L{1,2}
#   e.g. 21BH1234AB
_STANDARD_RE = re.compile(r'^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}$')
_BH_RE       = re.compile(r'^[0-9]{2}BH[0-9]{4}[A-Z]{1,2}$')


def license_complies_format(text: str) -> bool:
    """Validates against Indian standard-series or BH-series plate formats."""
    if not text or not (8 <= len(text) <= 10):
        return False
    return bool(_STANDARD_RE.match(text) or _BH_RE.match(text))


def format_license(text: str) -> str:
    """
    Best-effort OCR-confusion correction for near-miss reads.

    If `text` already validates, it's returned unchanged. Otherwise this
    tries every plausible segmentation of the standard SS-DD-L{1,3}-DDDD
    layout (series length can be 1, 2, or 3 letters, so the split point
    isn't fixed the way the old 7-char template assumed) and, for each
    candidate segmentation, forces state+series positions to their
    letter-equivalent and RTO+number positions to their digit-equivalent
    using the confusion table, then returns the first corrected candidate
    that validates. Falls back to the original text if nothing works.
    """
    if license_complies_format(text):
        return text

    n = len(text)
    for series_len in (1, 2, 3):
        # SS(2) + DD(1 or 2) + series_len + DDDD(4)
        for rto_len in (1, 2):
            total = 2 + rto_len + series_len + 4
            if total != n:
                continue
            state   = text[0:2]
            rto     = text[2:2 + rto_len]
            series  = text[2 + rto_len: 2 + rto_len + series_len]
            number  = text[2 + rto_len + series_len:]

            fixed = (
                "".join(INT_TO_CHAR.get(c, c) for c in state)
                + "".join(CHAR_TO_INT.get(c, c) for c in rto)
                + "".join(INT_TO_CHAR.get(c, c) for c in series)
                + "".join(CHAR_TO_INT.get(c, c) for c in number)
            )
            if license_complies_format(fixed):
                return fixed

    return text


def soft_format_indian_plate(text: str) -> str:
    """
    Applies heuristic positional character correction without ever discarding
    or rejecting the plate string.
    - Position 0..1 (State Code): forces digits/confusions to letters.
    - Position -4..end (Registration number): forces letters/confusions to digits.
    - Position 2..3 (RTO code): forces letters/confusions to digits.
    - Remaining middle characters (Series): forces digits/confusions to letters.
    """
    if not text or len(text) < 7:
        return text

    chars = list(text)
    n = len(chars)

    # 1. State code (First 2 characters -> Letters)
    chars[0] = INT_TO_CHAR.get(chars[0], chars[0])
    chars[1] = INT_TO_CHAR.get(chars[1], chars[1])

    # 2. Last 4 characters (Registration number -> Digits)
    for i in range(max(2, n - 4), n):
        chars[i] = CHAR_TO_INT.get(chars[i], chars[i])

    # 3. If standard 9 or 10 character plate:
    if n in (9, 10):
        # Position 2: RTO code first digit -> Digit
        chars[2] = CHAR_TO_INT.get(chars[2], chars[2])
        if n == 10:
            # Position 3: RTO code second digit -> Digit
            chars[3] = CHAR_TO_INT.get(chars[3], chars[3])
            # Positions 4..5: Series -> Letters
            for i in range(4, n - 4):
                chars[i] = INT_TO_CHAR.get(chars[i], chars[i])
        elif n == 9:
            # 9-char format: either 2-digit RTO + 1-letter series, or 1-digit RTO + 2-letter series
            if chars[3] in '0123456789' or chars[3] in CHAR_TO_INT:
                chars[3] = CHAR_TO_INT.get(chars[3], chars[3])
                for i in range(4, n - 4):
                    chars[i] = INT_TO_CHAR.get(chars[i], chars[i])
            else:
                for i in range(3, n - 4):
                    chars[i] = INT_TO_CHAR.get(chars[i], chars[i])

    return "".join(chars)


def clean_ocr_text(raw_text: str) -> str:
    return raw_text.upper().replace(" ", "").replace("-", "")


# ── Image quality ─────────────────────────────────────────────────────────────

def laplacian_sharpness(crop: np.ndarray) -> float:
    """Higher = sharper. Used to drop blurry/occluded frames before OCR."""
    if crop is None or crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def enhance_plate_crop(
    plate_crop: np.ndarray,
    upscale_factor: float = 2.0,
    min_width: int = 120,
    min_height: int = 30,
) -> np.ndarray:
    """
    CLAHE + unsharp-mask sharpening + upscaling, per Slide 18's promised
    preprocessing ("Enhance images using CLAHE + sharpening + upscaling").

    Replaces the previous crude global cv2.threshold(..., THRESH_BINARY_INV)
    binarization, which destroys OCR signal on colored/reflective/shadowed
    plates (a fixed global threshold has no way to adapt to per-plate
    lighting — exactly the "weather/occlusion degradation" scenario Slide 5
    claims to address).
    """
    if plate_crop is None or plate_crop.size == 0:
        return plate_crop

    h, w = plate_crop.shape[:2]
    scale = max(upscale_factor, min_width / max(w, 1), min_height / max(h, 1))
    if scale > 1.0:
        plate_crop = cv2.resize(
            plate_crop,
            (max(1, int(w * scale)), max(1, int(h * scale))),
            interpolation=cv2.INTER_CUBIC,
        )

    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)

    # CLAHE: adaptive local contrast — robust to uneven lighting/shadow/glare.
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    contrast_enhanced = clahe.apply(gray)

    # Unsharp mask sharpening.
    blurred = cv2.GaussianBlur(contrast_enhanced, (0, 0), sigmaX=3)
    sharpened = cv2.addWeighted(contrast_enhanced, 1.5, blurred, -0.5, 0)

    return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)


def enhance_plate_crop_bilateral(
    plate_crop: np.ndarray,
    upscale_factor: float = 2.0,
    min_width: int = 120,
    min_height: int = 30,
) -> np.ndarray:
    """
    Grayscale + bilateral filter + upscaling, as requested to reduce noise before OCR.
    """
    if plate_crop is None or plate_crop.size == 0:
        return plate_crop

    h, w = plate_crop.shape[:2]
    scale = max(upscale_factor, min_width / max(w, 1), min_height / max(h, 1))
    if scale > 1.0:
        plate_crop = cv2.resize(
            plate_crop,
            (max(1, int(w * scale)), max(1, int(h * scale))),
            interpolation=cv2.INTER_LANCZOS4,
        )

    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)

    # Bilateral filter to smooth noise while preserving edges
    filtered = cv2.bilateralFilter(gray, d=11, sigmaColor=17, sigmaSpace=17)

    return cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR)


def plate_edge_density(plate_crop: np.ndarray) -> float:
    """
    Computes the proportion of Canny edge pixels in the plate crop.
    Empty bumpers/smooth plastic have near-zero edge density (< 0.02),
    whereas real license plates with characters have edge density > 0.035.
    """
    if plate_crop is None or plate_crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY) if plate_crop.ndim == 3 else plate_crop
    edges = cv2.Canny(gray, 50, 150)
    return float(np.count_nonzero(edges) / max(1, edges.size))


# ── Geometry ───────────────────────────────────────────────────────────────────

def expand_bbox(box: BBox, ratio: float, frame_shape) -> BBox:
    """
    Expand `box` outward by `ratio` * box_height on every side, clamped to
    frame bounds. Used to give the vehicle box some slack for the
    plate-association IoA test, since vehicle-detector boxes are frequently
    a bit too tight to fully contain the plate region (see config.py's
    VEHICLE_BBOX_ASSOC_EXPAND comment). Never used for cropping/storage —
    only for the membership test.
    """
    x1, y1, x2, y2 = box
    h = max(1, y2 - y1)
    pad = int(round(h * ratio))
    fh, fw = frame_shape[:2]
    return (
        max(0, x1 - pad),
        max(0, y1 - pad),
        min(fw, x2 + pad),
        min(fh, y2 + pad),
    )


def intersection_over_area(inner_box: BBox, outer_box: BBox) -> float:
    """
    Intersection-over-Area of `inner_box` (e.g. the plate bbox) against
    `outer_box` (e.g. the vehicle bbox): what fraction of inner_box's own
    area overlaps outer_box. Range [0, 1].

    This is the association rule promised on Slide 5 ("associating plates to
    vehicles mathematically via Intersection-over-Area (IoA >= 50%)"). It's
    deliberately not IoU: a small plate box sitting entirely within a much
    larger vehicle box should score 1.0, not be penalized for the size
    mismatch the way IoU would. It's also deliberately not strict
    containment (px1 > vx1 and ... and px2 < vx2), which was the previous
    stage6 behaviour and fails outright whenever the plate detector's box
    pokes even one pixel outside a (frequently slightly-too-tight) vehicle
    box.
    """
    ix1 = max(inner_box[0], outer_box[0])
    iy1 = max(inner_box[1], outer_box[1])
    ix2 = min(inner_box[2], outer_box[2])
    iy2 = min(inner_box[3], outer_box[3])

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    intersection = iw * ih

    inner_area = max(1, (inner_box[2] - inner_box[0]) * (inner_box[3] - inner_box[1]))
    return intersection / inner_area
