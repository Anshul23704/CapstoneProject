"""
stage10_visualize.py — Annotated Output Video.

ARCHITECTURE CHANGE vs previous version
─────────────────────────────────────────
The previous version rendered from the *interpolated, validated-fusion-only*
CSV and, per car, picked ONE best frame to show an enlarged plate crop
floated above the vehicle box for the rest of that car's track. With only a
handful of vehicles ever reaching full fusion validity, this meant almost
the entire output video had no annotation at all — even on vehicles whose
plate WAS detected and read, just not cleanly enough to pass strict format
validation.

This version renders directly from main_pipeline.py's raw detections CSV
(every frame with a real plate_bbox match, for every track, regardless of
OCR/fusion outcome) and draws the vehicle box + plate box + text label (if
any) on THAT SPECIFIC FRAME, for every row that applies to it — one pass,
no per-car "best frame" selection, no floating crop insert. This is what
actually shows "detect as many visible plates as possible with a tight
bounding box on each", matching the reference full-frame-detection style,
rather than one label per car for its whole track.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd

from plate_utils import get_tilted_bbox

logger = logging.getLogger(__name__)


def _parse_bbox(raw: str):
    """Parses the '[x1 y1 x2 y2]' bbox format used by the pipeline's CSVs."""
    cleaned = str(raw).strip().lstrip("[").rstrip("]").replace(",", " ")
    parts = [p for p in cleaned.split(" ") if p]
    return [float(p) for p in parts]


def render_annotated_video(
    detections_csv_path: str,
    video_source: str,
    output_path: str,
) -> Optional[str]:
    """
    Reads a detections CSV (frame_nmr, car_id, car_bbox, license_plate_bbox,
    license_number, ...), re-reads the source video, and writes an annotated
    copy with the vehicle box, the plate box, and the OCR text label (if any
    was read for that specific frame) drawn directly on every frame that has
    a row. Returns output_path, or None if nothing could be rendered.
    """
    csv_path = Path(detections_csv_path)
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        logger.warning("render_annotated_video: no CSV data at %s — skipping", csv_path)
        return None

    results = pd.read_csv(csv_path)
    if results.empty or "car_id" not in results.columns:
        logger.warning("render_annotated_video: CSV has no usable rows — skipping")
        return None

    # Index rows by frame for O(1) lookup while streaming the video once.
    by_frame: dict = {}
    for _, row in results.iterrows():
        by_frame.setdefault(int(row["frame_nmr"]), []).append(row)

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        logger.error("render_annotated_video: cannot open video source %s", video_source)
        return None

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_nmr = -1
    ret = True
    rows_drawn = 0
    while ret:
        ret, frame = cap.read()
        frame_nmr += 1
        if not ret:
            break

        for row in by_frame.get(frame_nmr, []):
            try:
                cx1, cy1, cx2, cy2 = map(int, _parse_bbox(row["car_bbox"]))
                cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), (0, 255, 0), 3)

                px1, py1, px2, py2 = map(int, _parse_bbox(row["license_plate_bbox"]))
                crop = frame[max(0, py1):py2, max(0, px1):px2]
                tilted_box = get_tilted_bbox(crop, (px1, py1, px2, py2))
                cv2.polylines(frame, [tilted_box], isClosed=True, color=(0, 0, 255), thickness=3)

                text = str(row.get("license_number", "") or "")
                if text and text.lower() != "nan":
                    try:
                        score = float(row.get("license_number_score", 0.0))
                    except (TypeError, ValueError):
                        score = 0.0
                    label = f"{text} ({score*100:.0f}%)" if score else text

                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
                    lx1 = max(0, px1)
                    ly2 = max(th + 8, py1 - 6)
                    ly1 = ly2 - th - 10
                    cv2.rectangle(frame, (lx1, ly1), (lx1 + tw + 10, ly2), (0, 0, 255), -1)
                    cv2.putText(frame, label, (lx1 + 5, ly2 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

                rows_drawn += 1
            except Exception as exc:
                logger.debug(
                    "render_annotated_video: overlay failed frame=%d car=%s (%s)",
                    frame_nmr, row.get("car_id"), exc,
                )

        out.write(frame)

    out.release()
    cap.release()
    logger.info("Annotated video written -> %s (%d boxes drawn)", output_path, rows_drawn)
    return output_path
