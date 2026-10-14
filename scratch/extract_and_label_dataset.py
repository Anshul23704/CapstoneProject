"""
extract_and_label_dataset.py — Full Pipeline for:
  1. Cropping & Guarded Deskewing across all dataset splits (train/valid/test)
  2. EasyOCR Pseudo-Labeling with Indian plate formatting & validation
  3. Exporting CSV, TXT manifests & Summary metrics
"""
from __future__ import annotations

import os
import sys
import glob
import math
import json
import csv
import string
import re
from typing import List, Tuple, Optional, Dict

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import cv2
import numpy as np
import easyocr

# ── State Codes & OCR Helpers ──────────────────────────────────────────────────
INDIAN_STATE_CODES = {
    "AN", "AP", "AR", "AS", "BR", "CH", "CG", "DD", "DN", "DL",
    "GA", "GJ", "HR", "HP", "JK", "JH", "KA", "KL", "LA", "LD",
    "MP", "MH", "MN", "ML", "MZ", "NL", "OD", "PY", "PB", "RJ",
    "SK", "TN", "TS", "TR", "UP", "UK", "WB"
}

STATE_ALIASES = {
    "UA": "UK", "OR": "OD", "CT": "CG", "DH": "DD",
    "TZ": "TS", "VA": "KA", "LZ": "KL", "FK": "KA", "OA": "AP",
    "BA": "KA", "KK": "KA", "WS": "TS", "ES": "TS", "JS": "TS",
    "RA": "KA", "KO": "KA", "KI": "KA", "TA": "TS", "RO": "KA",
    "RS": "TS", "JJ": "GJ", "JA": "KA", "LU": "KL", "ZA": "KA",
    "AA": "AP", "OS": "OD", "OO": "OD", "IP": "AP", "AJ": "AP",
    "AT": "AP", "PO": "PB", "WO": "WB", "WI": "WB", "MS": "MH",
}

CHAR_TO_INT = {'O': '0', 'D': '0', 'Q': '0', 'I': '1', 'L': '1', 'Z': '2', 'J': '3', 'A': '4', 'S': '5', 'G': '6', 'B': '8'}
INT_TO_CHAR = {'0': 'O', '1': 'I', '2': 'Z', '3': 'J', '4': 'A', '5': 'S', '6': 'G', '7': 'T', '8': 'B'}

_STANDARD_RE = re.compile(r'^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}$')
_BH_RE       = re.compile(r'^[0-9]{2}BH[0-9]{4}[A-Z]{1,2}$')


def levenshtein_dist(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return levenshtein_dist(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def fuzzy_match_state_code(code: str) -> str:
    if not code or len(code) != 2:
        return code
    code = code.upper()
    if code in INDIAN_STATE_CODES:
        return code
    if code in STATE_ALIASES:
        return STATE_ALIASES[code]

    best_match = code
    min_dist = 99
    for valid_code in sorted(INDIAN_STATE_CODES):
        d = levenshtein_dist(code, valid_code)
        if d < min_dist:
            min_dist = d
            best_match = valid_code

    if min_dist <= 1:
        return best_match
    return code


def clean_ocr_text(raw_text: str) -> str:
    cleaned = re.sub(r'[^A-Za-z0-9]', '', raw_text.upper())
    # Strip common IND prefix if detected as separate noise
    if cleaned.startswith("IND") and len(cleaned) >= 11:
        cleaned = cleaned[3:]
    return cleaned


def format_plate_text(text: str) -> str:
    """Applies positional character correction for Indian number plate format."""
    if not text or len(text) < 7:
        return text

    chars = list(text)
    n = len(chars)

    # 1. State code (First 2 chars -> Letters & Fuzzy Match)
    chars[0] = INT_TO_CHAR.get(chars[0], chars[0])
    chars[1] = INT_TO_CHAR.get(chars[1], chars[1])
    snapped = fuzzy_match_state_code("".join(chars[:2]))
    if len(snapped) == 2:
        chars[0] = snapped[0]
        chars[1] = snapped[1]

    # 2. Last 4 digits
    for i in range(max(2, n - 4), n):
        chars[i] = CHAR_TO_INT.get(chars[i], chars[i])

    # 3. Position 2 (RTO first digit) -> Digit
    if n >= 8:
        chars[2] = CHAR_TO_INT.get(chars[2], chars[2])

    # 4. If standard 10 characters: SS DD XX DDDD
    if n == 10:
        chars[3] = CHAR_TO_INT.get(chars[3], chars[3])  # RTO 2nd digit
        chars[4] = INT_TO_CHAR.get(chars[4], chars[4])  # Series 1st letter
        chars[5] = INT_TO_CHAR.get(chars[5], chars[5])  # Series 2nd letter
    elif n == 9:
        if chars[3] in '0123456789' or chars[3] in CHAR_TO_INT:
            chars[3] = CHAR_TO_INT.get(chars[3], chars[3])
            chars[4] = INT_TO_CHAR.get(chars[4], chars[4])
        else:
            chars[3] = INT_TO_CHAR.get(chars[3], chars[3])
            chars[4] = INT_TO_CHAR.get(chars[4], chars[4])

    return "".join(chars)


def complies_format(text: str) -> bool:
    if not text or not (7 <= len(text) <= 10):
        return False
    return bool(_STANDARD_RE.match(text) or _BH_RE.match(text))


# ── Guarded Deskewing Logic ───────────────────────────────────────────────────

def order_quad_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # TL
    rect[2] = pts[np.argmax(s)]   # BR
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # TR
    rect[3] = pts[np.argmax(diff)]  # BL
    return rect


def guarded_deskew_plate(plate_crop: np.ndarray) -> Tuple[np.ndarray, float, bool]:
    """
    Guarded deskewing:
    - If $|angle| < 2.0^\circ$, bypass warping completely (keeps straight plates straight).
    - Validates opposite edge parallelism, interior angles, and aspect ratio.
    - Returns (crop, angle_deg, was_warped).
    """
    if plate_crop is None or plate_crop.size == 0:
        return plate_crop, 0.0, False

    h, w = plate_crop.shape[:2]
    crop_area = w * h

    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY) if plate_crop.ndim == 3 else plate_crop
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

    v = np.median(blurred)
    lower = int(max(0, (1.0 - 0.33) * v))
    upper = int(min(255, (1.0 + 0.33) * v))
    edges = cv2.Canny(blurred, lower, upper)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return plate_crop, 0.0, False

    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for cnt in contours[:6]:
        cnt_area = cv2.contourArea(cnt)
        if cnt_area < (crop_area * 0.35):  # must occupy >35% of crop
            continue

        perimeter = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.035 * perimeter, True)

        if len(approx) == 4 and cv2.isContourConvex(approx):
            pts = approx.reshape(4, 2)
            ordered_pts = order_quad_points(pts)
            tl, tr, br, bl = ordered_pts

            top_v = tr - tl
            bot_v = br - bl
            left_v = bl - tl
            right_v = br - tr

            top_ang = math.degrees(math.atan2(top_v[1], top_v[0]))
            bot_ang = math.degrees(math.atan2(bot_v[1], bot_v[0]))

            # Edge parallelism check
            if abs(top_ang - bot_ang) > 6.0:
                continue

            avg_skew = (top_ang + bot_ang) / 2.0

            # DEADZONE GUARD: If tilt is tiny (< 2.0 degrees), keep original straight crop!
            if abs(avg_skew) < 2.0:
                return plate_crop, avg_skew, False

            # Don't perform extreme wild rotations
            if abs(avg_skew) > 35.0:
                continue

            # Quad aspect ratio check
            width_a = np.linalg.norm(br - bl)
            width_b = np.linalg.norm(tr - tl)
            max_w = max(int(width_a), int(width_b), 160)

            height_a = np.linalg.norm(tr - br)
            height_b = np.linalg.norm(tl - bl)
            max_h = max(int(height_a), int(height_b), 40)

            ar = max_w / float(max_h)
            if not (1.6 <= ar <= 6.0):
                continue

            target_w = max_w
            target_h = int(target_w / ar)

            dst = np.array([
                [0, 0],
                [target_w - 1, 0],
                [target_w - 1, target_h - 1],
                [0, target_h - 1]
            ], dtype=np.float32)

            M = cv2.getPerspectiveTransform(ordered_pts, dst)
            warped = cv2.warpPerspective(
                plate_crop, M, (target_w, target_h),
                flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
            )
            return warped, avg_skew, True

    # Fallback to minAreaRect with deadzone
    largest_cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest_cnt) >= (crop_area * 0.30):
        rect = cv2.minAreaRect(largest_cnt)
        angle = rect[-1]
        if angle < -45:
            angle = -(90 + angle)
        elif angle > 45:
            angle = 90 - angle

        # Only apply affine rotation if tilt is between 2.0 deg and 30.0 deg
        if 2.0 <= abs(angle) <= 30.0:
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            deskewed = cv2.warpAffine(
                plate_crop, M, (w, h),
                flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
            )
            return deskewed, angle, True

    return plate_crop, 0.0, False


def crop_with_padding(image: np.ndarray, bbox: Tuple[int, int, int, int], pad_ratio: float = 0.06) -> np.ndarray:
    ih, iw = image.shape[:2]
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1

    px = int(w * pad_ratio)
    py = int(h * pad_ratio)

    cx1 = max(0, x1 - px)
    cy1 = max(0, y1 - py)
    cx2 = min(iw, x2 + px)
    cy2 = min(ih, y2 + py)

    return image[cy1:cy2, cx1:cx2]


def read_yolo_bboxes(label_path: str, img_w: int, img_h: int) -> List[Tuple[int, int, int, int]]:
    if not os.path.exists(label_path):
        return []
    with open(label_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    bboxes = []
    for line in lines:
        parts = line.split()
        if len(parts) >= 5:
            xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            x1 = int((xc - w / 2.0) * img_w)
            y1 = int((yc - h / 2.0) * img_h)
            x2 = int((xc + w / 2.0) * img_w)
            y2 = int((yc + h / 2.0) * img_h)
            bboxes.append((max(0, x1), max(0, y1), min(img_w, x2), min(img_h, y2)))
    return bboxes


# ── Main Processing Orchestration ──────────────────────────────────────────────

def process_dataset(
    dataset_root: str,
    output_root: str,
    max_images_per_split: Optional[int] = None
):
    print("=" * 65)
    print("STARTING DATASET EXTRACTION & EASYOCR PSEUDO-LABELING")
    print("=" * 65)

    print("[EasyOCR] Initializing reader on GPU/CPU...")
    reader = easyocr.Reader(['en'], gpu=True, verbose=False)

    splits = ["train", "valid", "test"]
    stats = {
        "total_images": 0,
        "total_plates_extracted": 0,
        "deskewed_count": 0,
        "straight_preserved_count": 0,
        "compliant_format_count": 0,
        "split_counts": {}
    }

    all_csv_records = []

    for split in splits:
        img_dir = os.path.join(dataset_root, split, "images")
        lbl_dir = os.path.join(dataset_root, split, "labels")

        if not os.path.exists(img_dir):
            print(f"Skipping {split}: directory not found")
            continue

        split_rect_out = os.path.join(output_root, "crops_rectangular", split)
        split_deskew_out = os.path.join(output_root, "crops_deskewed", split)
        os.makedirs(split_rect_out, exist_ok=True)
        os.makedirs(split_deskew_out, exist_ok=True)

        img_files = sorted(glob.glob(os.path.join(img_dir, "*.*")))
        if max_images_per_split:
            img_files = img_files[:max_images_per_split]

        print(f"\nProcessing split [{split}] ({len(img_files)} images)...")
        stats["split_counts"][split] = len(img_files)

        split_txt_lines = []
        split_csv_rows = []

        for idx, img_path in enumerate(img_files, 1):
            filename = os.path.basename(img_path)
            base_name = os.path.splitext(filename)[0]
            lbl_path = os.path.join(lbl_dir, f"{base_name}.txt")

            img = cv2.imread(img_path)
            if img is None:
                continue

            ih, iw = img.shape[:2]
            bboxes = read_yolo_bboxes(lbl_path, iw, ih)
            if not bboxes:
                continue

            stats["total_images"] += 1

            for p_idx, bbox in enumerate(bboxes):
                stats["total_plates_extracted"] += 1
                crop_name = f"{base_name}_p{p_idx}.jpg"

                # 1. Extract Rectangular Crop
                rect_crop = crop_with_padding(img, bbox, pad_ratio=0.06)
                rect_path = os.path.join(split_rect_out, crop_name)
                cv2.imwrite(rect_path, rect_crop)

                # 2. Extract Guarded Deskewed Crop
                deskew_crop, angle_deg, was_warped = guarded_deskew_plate(rect_crop)
                deskew_path = os.path.join(split_deskew_out, crop_name)
                cv2.imwrite(deskew_path, deskew_crop)

                if was_warped:
                    stats["deskewed_count"] += 1
                else:
                    stats["straight_preserved_count"] += 1

                # 3. EasyOCR Recognition on Deskewed Crop
                ocr_results = reader.readtext(deskew_crop, detail=1, paragraph=False)
                raw_texts = []
                confidences = []
                for _, text, conf in ocr_results:
                    raw_texts.append(text)
                    confidences.append(float(conf))

                raw_combined = " ".join(raw_texts)
                cleaned_text = clean_ocr_text(raw_combined)
                formatted_text = format_plate_text(cleaned_text)
                is_compliant = complies_format(formatted_text)
                avg_conf = float(np.mean(confidences)) if confidences else 0.0

                if is_compliant:
                    stats["compliant_format_count"] += 1

                # Relative path for standard datasets
                rel_crop_path = f"crops_deskewed/{split}/{crop_name}"

                record = {
                    "split": split,
                    "original_image": filename,
                    "crop_filename": crop_name,
                    "crop_relative_path": rel_crop_path,
                    "raw_ocr_text": raw_combined,
                    "cleaned_ocr_text": cleaned_text,
                    "formatted_plate_text": formatted_text,
                    "complies_format": is_compliant,
                    "ocr_confidence": round(avg_conf, 4),
                    "skew_angle_deg": round(angle_deg, 2),
                    "deskew_applied": was_warped,
                }
                split_csv_rows.append(record)
                all_csv_records.append(record)

                if formatted_text:
                    split_txt_lines.append(f"{rel_crop_path}\t{formatted_text}")

            if idx % 100 == 0 or idx == len(img_files):
                print(f"  Processed {idx}/{len(img_files)} images...")

        # Write split txt manifest
        split_txt_file = os.path.join(output_root, f"manifest_{split}.txt")
        with open(split_txt_file, "w", encoding="utf-8") as f:
            f.write("\n".join(split_txt_lines))

        # Write split csv manifest
        split_csv_file = os.path.join(output_root, f"manifest_{split}.csv")
        if split_csv_rows:
            with open(split_csv_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=split_csv_rows[0].keys())
                writer.writeheader()
                writer.writerows(split_csv_rows)

    # Write Master Consolidated CSV
    master_csv = os.path.join(output_root, "master_labeled_dataset.csv")
    if all_csv_records:
        with open(master_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_csv_records[0].keys())
            writer.writeheader()
            writer.writerows(all_csv_records)

    # Write Summary Metadata
    summary_path = os.path.join(output_root, "extraction_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print("\n" + "=" * 65)
    print("DATASET EXTRACTION & LABELING COMPLETE")
    print("=" * 65)
    print(f"Total Source Images Processed : {stats['total_images']}")
    print(f"Total Plate Crops Extracted   : {stats['total_plates_extracted']}")
    print(f"Deskewed (Angle >= 2.0 deg)   : {stats['deskewed_count']}")
    print(f"Straight Preserved (< 2.0 deg): {stats['straight_preserved_count']}")
    print(f"Format-Compliant Plates       : {stats['compliant_format_count']}")
    print(f"Master CSV Manifest           : {master_csv}")
    print("=" * 65)


if __name__ == "__main__":
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    DATASET_PATH = os.path.join(BASE_DIR, "Indian License Plate.v1-v1.yolov11")
    OUTPUT_PATH = os.path.join(BASE_DIR, "extracted_alpr_dataset")

    process_dataset(DATASET_PATH, OUTPUT_PATH)
