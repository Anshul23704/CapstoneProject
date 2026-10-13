"""
compare_crop_methods.py — Evaluates Option A (Crop + Perspective Deskewing)
on a 10-sample batch using:
  1. Ground-Truth YOLO dataset labels
  2. Capstone YOLO detector model
Produces Rectangular crops, Deskewed crops, and a side-by-side visual collage.
"""
from __future__ import annotations

import os
import glob
import math
from typing import List, Tuple, Optional

import cv2
import numpy as np

# ── Configuration ─────────────────────────────────────────────────────────────
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATASET_DIR = os.path.join(BASE_DIR, "Indian License Plate.v1-v1.yolov11")
TRAIN_IMG_DIR = os.path.join(DATASET_DIR, "train", "images")
TRAIN_LBL_DIR = os.path.join(DATASET_DIR, "train", "labels")

OUTPUT_DIR = os.path.join(BASE_DIR, "crop_comparison_results")
GT_RECT_DIR = os.path.join(OUTPUT_DIR, "gt_rectangular")
GT_DESKEW_DIR = os.path.join(OUTPUT_DIR, "gt_deskewed")
MODEL_RECT_DIR = os.path.join(OUTPUT_DIR, "model_rectangular")
MODEL_DESKEW_DIR = os.path.join(OUTPUT_DIR, "model_deskewed")

for d in [GT_RECT_DIR, GT_DESKEW_DIR, MODEL_RECT_DIR, MODEL_DESKEW_DIR]:
    os.makedirs(d, exist_ok=True)


def order_quad_points(pts: np.ndarray) -> np.ndarray:
    """Orders 4 points as: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # TL
    rect[2] = pts[np.argmax(s)]   # BR

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # TR
    rect[3] = pts[np.argmax(diff)]  # BL
    return rect


def deskew_plate_contour(plate_crop: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Robust Option A:
    1. Edge detection + morphological filtering to identify the plate boundary quad.
    2. 4-point perspective warp if quadrilateral found.
    3. Fallback to minAreaRect affine rotation.
    Returns (rectified_image, estimated_skew_angle_deg).
    """
    if plate_crop is None or plate_crop.size == 0:
        return plate_crop, 0.0

    h, w = plate_crop.shape[:2]
    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY) if plate_crop.ndim == 3 else plate_crop

    # Contrast enhancement for sharp edge boundaries
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

    # Edge detection with automatic Canny thresholds
    v = np.median(blurred)
    lower = int(max(0, (1.0 - 0.33) * v))
    upper = int(min(255, (1.0 + 0.33) * v))
    edges = cv2.Canny(blurred, lower, upper)

    # Morphological close to bridge edge gaps along the plate border
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return plate_crop, 0.0

    # Sort contours by area descending
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    crop_area = w * h

    best_quad = None

    for cnt in contours[:8]:
        cnt_area = cv2.contourArea(cnt)
        if cnt_area < (crop_area * 0.25):  # must occupy at least 25% of the crop
            continue

        perimeter = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.03 * perimeter, True)

        if len(approx) == 4 and cv2.isContourConvex(approx):
            pts = approx.reshape(4, 2)
            ordered_pts = order_quad_points(pts)
            # Check aspect ratio of the 4-point quad
            tl, tr, br, bl = ordered_pts
            top_w = np.linalg.norm(tr - tl)
            bot_w = np.linalg.norm(br - bl)
            avg_w = (top_w + bot_w) / 2.0

            left_h = np.linalg.norm(bl - tl)
            right_h = np.linalg.norm(br - tr)
            avg_h = (left_h + right_h) / 2.0

            if avg_h > 0:
                ar = avg_w / avg_h
                if 1.4 <= ar <= 6.5:
                    best_quad = ordered_pts
                    break

    # If clean 4-point quadrilateral found, apply full perspective transform
    if best_quad is not None:
        tl, tr, br, bl = best_quad
        width_a = np.linalg.norm(br - bl)
        width_b = np.linalg.norm(tr - tl)
        max_w = max(int(width_a), int(width_b), 160)

        height_a = np.linalg.norm(tr - br)
        height_b = np.linalg.norm(tl - bl)
        max_h = max(int(height_a), int(height_b), 40)

        aspect_ratio = max_w / float(max_h)
        target_w = max_w
        target_h = int(target_w / aspect_ratio)

        dst = np.array([
            [0, 0],
            [target_w - 1, 0],
            [target_w - 1, target_h - 1],
            [0, target_h - 1]
        ], dtype=np.float32)

        M = cv2.getPerspectiveTransform(best_quad, dst)
        warped = cv2.warpPerspective(
            plate_crop, M, (target_w, target_h),
            flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
        )
        dx = tr[0] - tl[0]
        dy = tr[1] - tl[1]
        angle_deg = math.degrees(math.atan2(dy, dx))
        return warped, angle_deg

    # Fallback: minAreaRect affine deskewing
    largest_cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest_cnt) >= (crop_area * 0.20):
        rect = cv2.minAreaRect(largest_cnt)
        angle = rect[-1]
        if angle < -45:
            angle = -(90 + angle)
        elif angle > 45:
            angle = 90 - angle

        if abs(angle) > 1.0 and abs(angle) < 40.0:
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            deskewed = cv2.warpAffine(
                plate_crop, M, (w, h),
                flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
            )
            return deskewed, angle

    return plate_crop, 0.0


def crop_with_padding(image: np.ndarray, bbox: Tuple[int, int, int, int], pad_ratio: float = 0.06) -> np.ndarray:
    """Crops bounding box (x1, y1, x2, y2) with proportional boundary padding."""
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

    crop = image[cy1:cy2, cx1:cx2]
    return crop


def read_yolo_bbox(label_path: str, img_w: int, img_h: int) -> Optional[Tuple[int, int, int, int]]:
    """Parses standard YOLO bbox [cls, xc, yc, w, h] to (x1, y1, x2, y2)."""
    if not os.path.exists(label_path):
        return None
    with open(label_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    if not lines:
        return None

    parts = lines[0].split()
    if len(parts) >= 5:
        xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        x1 = int((xc - w / 2.0) * img_w)
        y1 = int((yc - h / 2.0) * img_h)
        x2 = int((xc + w / 2.0) * img_w)
        y2 = int((yc + h / 2.0) * img_h)
        return (max(0, x1), max(0, y1), min(img_w, x2), min(img_h, y2))
    return None


def run_comparison(num_samples: int = 10):
    print("=" * 60)
    print("RUNNING LICENSE PLATE CROPPING & DESKEWING EVALUATION")
    print("=" * 60)

    img_files = sorted(glob.glob(os.path.join(TRAIN_IMG_DIR, "*.jpg")) + glob.glob(os.path.join(TRAIN_IMG_DIR, "*.jpeg")))
    if not img_files:
        print(f"Error: No images found in {TRAIN_IMG_DIR}")
        return

    step = max(1, len(img_files) // num_samples)
    sample_files = [img_files[i * step] for i in range(min(num_samples, len(img_files)))]

    detector_model = None
    try:
        from ultralytics import YOLO
        model_candidates = [
            os.path.join(BASE_DIR, "CapstoneProject", "models", "license_plate_yolo26m-3", "weights", "best.pt"),
            os.path.join(BASE_DIR, "CapstoneProject", "models", "best.pt"),
            "yolov8n.pt"
        ]
        for mc in model_candidates:
            if os.path.exists(mc):
                print(f"[Model Detector] Loading weights from: {mc}")
                detector_model = YOLO(mc)
                break
        if detector_model is None:
            print("[Model Detector] Using default YOLO model...")
            detector_model = YOLO("yolov8n.pt")
    except Exception as e:
        print(f"[Warning] Ultralytics YOLO notice: {e}")

    results = []

    for idx, img_path in enumerate(sample_files, 1):
        filename = os.path.basename(img_path)
        base_name = os.path.splitext(filename)[0]
        lbl_path = os.path.join(TRAIN_LBL_DIR, f"{base_name}.txt")

        img = cv2.imread(img_path)
        if img is None:
            continue
        ih, iw = img.shape[:2]

        # 1. Ground Truth YOLO label crop & deskew
        gt_bbox = read_yolo_bbox(lbl_path, iw, ih)
        gt_rect_crop = None
        gt_deskew_crop = None
        gt_angle = 0.0

        if gt_bbox:
            gt_rect_crop = crop_with_padding(img, gt_bbox, pad_ratio=0.06)
            gt_deskew_crop, gt_angle = deskew_plate_contour(gt_rect_crop)

            cv2.imwrite(os.path.join(GT_RECT_DIR, f"sample_{idx:02d}_gt_rect.jpg"), gt_rect_crop)
            cv2.imwrite(os.path.join(GT_DESKEW_DIR, f"sample_{idx:02d}_gt_deskew.jpg"), gt_deskew_crop)

        # 2. Model Detector crop & deskew
        model_bbox = None
        model_rect_crop = None
        model_deskew_crop = None
        model_angle = 0.0

        if detector_model is not None:
            preds = detector_model(img, verbose=False, conf=0.15)[0]
            if preds.boxes is not None and len(preds.boxes) > 0:
                best_box = preds.boxes[0]
                bx1, by1, bx2, by2 = map(int, best_box.xyxy[0])
                model_bbox = (bx1, by1, bx2, by2)
                model_rect_crop = crop_with_padding(img, model_bbox, pad_ratio=0.06)
                model_deskew_crop, model_angle = deskew_plate_contour(model_rect_crop)

                cv2.imwrite(os.path.join(MODEL_RECT_DIR, f"sample_{idx:02d}_model_rect.jpg"), model_rect_crop)
                cv2.imwrite(os.path.join(MODEL_DESKEW_DIR, f"sample_{idx:02d}_model_deskew.jpg"), model_deskew_crop)

        results.append({
            "idx": idx,
            "filename": filename,
            "image": img,
            "gt_bbox": gt_bbox,
            "gt_rect": gt_rect_crop,
            "gt_deskew": gt_deskew_crop,
            "gt_angle": gt_angle,
            "model_bbox": model_bbox,
            "model_rect": model_rect_crop,
            "model_deskew": model_deskew_crop,
            "model_angle": model_angle,
        })
        print(f"Sample {idx:02d}: {filename[:35]}... | GT Angle: {gt_angle:+.1f}° | Model Angle: {model_angle:+.1f}°")

    # 3. Build consolidated visual comparison grid
    row_height = 90
    card_width = 300
    grid_rows = []

    for res in results:
        thumb = res["image"].copy()
        if res["gt_bbox"]:
            gx1, gy1, gx2, gy2 = res["gt_bbox"]
            cv2.rectangle(thumb, (gx1, gy1), (gx2, gy2), (0, 255, 0), 2)
        if res["model_bbox"]:
            mx1, my1, mx2, my2 = res["model_bbox"]
            cv2.rectangle(thumb, (mx1, my1), (mx2, my2), (0, 0, 255), 2)

        thumb_resized = cv2.resize(thumb, (120, row_height))

        def make_cell(crop_img, title):
            cell = np.ones((row_height, card_width, 3), dtype=np.uint8) * 35
            if crop_img is not None and crop_img.size > 0:
                ch, cw = crop_img.shape[:2]
                scale = min((card_width - 16) / float(cw), (row_height - 28) / float(ch))
                nw, nh = max(1, int(cw * scale)), max(1, int(ch * scale))
                resized = cv2.resize(crop_img, (nw, nh))
                ox = (card_width - nw) // 2
                oy = 20 + (row_height - 24 - nh) // 2
                cell[oy:oy+nh, ox:ox+nw] = resized
            cv2.putText(cell, title, (8, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
            return cell

        c1 = thumb_resized
        c2 = make_cell(res["gt_rect"], f"GT Rect Crop (#{res['idx']})")
        c3 = make_cell(res["gt_deskew"], f"GT Deskewed ({res['gt_angle']:+.1f} deg)")
        c4 = make_cell(res["model_rect"], "Model Rect Crop")
        c5 = make_cell(res["model_deskew"], f"Model Deskewed ({res['model_angle']:+.1f} deg)")

        row = np.hstack([c1, c2, c3, c4, c5])
        grid_rows.append(row)

    if grid_rows:
        collage = np.vstack(grid_rows)
        collage_path = os.path.join(OUTPUT_DIR, "cropping_comparison_collage.jpg")
        cv2.imwrite(collage_path, collage)
        print(f"\n[Saved Collage] Comparison collage saved to: {collage_path}")

    print("=" * 60)
    print("EVALUATION COMPLETE. Output saved to folder:", OUTPUT_DIR)
    print("=" * 60)


if __name__ == "__main__":
    run_comparison(10)
