from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import List

import numpy as np
from scipy.interpolate import interp1d

logger = logging.getLogger(__name__)

_FIELDS = [
    "frame_nmr", "car_id",
    "car_bbox", "license_plate_bbox",
    "license_plate_bbox_score",
    "license_number", "license_number_score",
]


def _parse_bbox(s: str) -> List[float]:
    """Parse '[x1 y1 x2 y2]' into [x1, y1, x2, y2]."""
    s = str(s).strip().lstrip("[").rstrip("]").replace(",", " ")
    parts = s.split()
    if len(parts) != 4:
        return [0.0, 0.0, 0.0, 0.0]
    return list(map(float, parts))


def _fmt_bbox(coords) -> str:
    return "[{} {} {} {}]".format(*[round(float(v), 4) for v in coords])


def interpolate_csv(src_path: str) -> str:
    """
    Read src_path, interpolate missing frames per car_id, write
    *_interpolated.csv.  Returns the output path.
    """
    src = Path(src_path)
    dst = src.with_name(src.stem + "_interpolated" + src.suffix)

    with open(src, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        data   = list(reader)

    if not data:
        logger.warning("interpolate_csv: empty input")
        dst.write_text("")
        return str(dst)

    cols = set(data[0].keys())
    required = {"frame_nmr", "car_id", "car_bbox", "license_plate_bbox"}
    missing  = required - cols
    if missing:
        logger.warning("interpolate_csv: missing columns %s — cannot interpolate", missing)
        with open(dst, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(cols))
            writer.writeheader()
            writer.writerows(data)
        return str(dst)

    frame_numbers    = np.array([int(row["frame_nmr"])       for row in data])
    car_ids          = np.array([int(float(row["car_id"]))   for row in data])
    car_bboxes       = np.array([_parse_bbox(row["car_bbox"]) for row in data])
    plate_bboxes     = np.array([_parse_bbox(row["license_plate_bbox"]) for row in data])

    interpolated: List[dict] = []
    unique_car_ids = np.unique(car_ids)

    for car_id in unique_car_ids:
        # FIX: this set is now int-typed. Previously it stored the raw CSV
        # string values and was checked via `str(fn) in frame_numbers_`,
        # which is fragile to any formatting difference (e.g. a stray
        # leading/trailing space in the source CSV would silently break the
        # membership test and mark every row as "interpolated" with score 0
        # even for frames that had a real reading).
        real_frame_numbers = {
            int(p["frame_nmr"])
            for p in data
            if int(float(p["car_id"])) == int(float(car_id))
        }

        car_mask              = car_ids == car_id
        car_frame_numbers     = frame_numbers[car_mask]
        car_bboxes_t          = car_bboxes[car_mask]
        plate_bboxes_t        = plate_bboxes[car_mask]

        car_bboxes_interp   = []
        plate_bboxes_interp = []

        first_frame = car_frame_numbers[0]

        for i in range(len(car_bboxes_t)):
            frame_number      = car_frame_numbers[i]
            car_bbox          = car_bboxes_t[i]
            plate_bbox        = plate_bboxes_t[i]

            if i > 0:
                prev_frame      = car_frame_numbers[i - 1]
                prev_car_bbox   = car_bboxes_interp[-1]
                prev_plate_bbox = plate_bboxes_interp[-1]

                frames_gap = frame_number - prev_frame
                if frames_gap > 1:
                    x     = np.array([prev_frame, frame_number], dtype=float)
                    x_new = np.linspace(prev_frame, frame_number,
                                        num=int(frames_gap), endpoint=False)

                    f_car = interp1d(
                        x, np.vstack((prev_car_bbox, car_bbox)), axis=0, kind="linear"
                    )
                    f_plt = interp1d(
                        x, np.vstack((prev_plate_bbox, plate_bbox)), axis=0, kind="linear"
                    )
                    car_bboxes_interp.extend(f_car(x_new)[1:])
                    plate_bboxes_interp.extend(f_plt(x_new)[1:])

            car_bboxes_interp.append(car_bbox)
            plate_bboxes_interp.append(plate_bbox)

        for i, (cb, pb) in enumerate(zip(car_bboxes_interp, plate_bboxes_interp)):
            fn = int(first_frame) + i
            row: dict = {
                "frame_nmr":        str(fn),
                "car_id":           str(car_id),
                "car_bbox":         _fmt_bbox(cb),
                "license_plate_bbox": _fmt_bbox(pb),
            }

            if fn in real_frame_numbers:
                orig = next(
                    p for p in data
                    if int(p["frame_nmr"]) == fn
                    and int(float(p["car_id"])) == int(float(car_id))
                )
                row["license_plate_bbox_score"] = orig.get("license_plate_bbox_score", "0")
                row["license_number"]           = orig.get("license_number",           "0")
                row["license_number_score"]     = orig.get("license_number_score",     "0")
            else:
                row["license_plate_bbox_score"] = "0"
                row["license_number"]           = "0"
                row["license_number_score"]     = "0"

            interpolated.append(row)

    interpolated.sort(key=lambda r: (int(r["frame_nmr"]), int(float(r["car_id"]))))

    with open(dst, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDS)
        writer.writeheader()
        writer.writerows(interpolated)

    logger.info("Interpolation complete: %d rows -> %s", len(interpolated), dst)
    return str(dst)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to results_rich.csv")
    args = parser.parse_args()
    out = interpolate_csv(args.csv)
    print(f"Written -> {out}")
