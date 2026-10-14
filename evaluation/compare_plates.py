#!/usr/bin/env python3
"""
compare_plates.py — Accuracy Evaluator for License Plate Recognition.

Compares ground-truth license plates from 'ActualPlates/<video_name>.txt' (or .json)
against pipeline predictions from the latest output run.

Usage:
    python evaluation/compare_plates.py
    python evaluation/compare_plates.py test3
    python evaluation/compare_plates.py test3.mp4
    python evaluation/compare_plates.py --run output/20261013_204125
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import pandas as pd


# ── Levenshtein Distance & Character Accuracy ─────────────────────────────────

def levenshtein(s1: str, s2: str) -> int:
    """Calculate the Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (c1 != c2)))
        prev = curr
    return prev[-1]


def char_accuracy(actual: str, pred: str) -> float:
    """Character-level accuracy based on edit distance normalized by actual length."""
    if not actual:
        return 1.0 if not pred else 0.0
    dist = levenshtein(actual, pred)
    return max(0.0, 1.0 - (dist / max(len(actual), len(pred))))


# ── Ground Truth Loader ───────────────────────────────────────────────────────

def load_actual_plates(video_name: str, actual_dir: Path) -> Tuple[List[str], Path]:
    """
    Loads ground truth plates for a video.
    Checks:
      - <name>.txt
      - <name>.json
      - stripping extensions (.mp4, .MP4, etc.)
    """
    clean_name = Path(video_name).stem
    candidates = [
        actual_dir / f"{video_name}.txt",
        actual_dir / f"{video_name}.json",
        actual_dir / f"{clean_name}.txt",
        actual_dir / f"{clean_name}.json",
    ]

    found_path = None
    for cand in candidates:
        if cand.exists():
            found_path = cand
            break

    if not found_path:
        available = [p.name for p in actual_dir.glob("*") if p.is_file() and not p.name.startswith(".")]
        raise FileNotFoundError(
            f"Ground truth file not found for '{video_name}' in {actual_dir}.\n"
            f"Available files in ActualPlates: {available if available else 'None'}"
        )

    plates = []
    if found_path.suffix.lower() == ".json":
        with open(found_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                plates = [str(x).strip().upper() for x in data if str(x).strip()]
            elif isinstance(data, dict):
                # Look for a list value under 'plates' or similar key
                for val in data.values():
                    if isinstance(val, list):
                        plates = [str(x).strip().upper() for x in val if str(x).strip()]
                        break
    else:
        # Standard .txt: one plate per line
        with open(found_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                # Ignore empty lines and comment lines starting with # or //
                if line and not line.startswith(("#", "//")):
                    # Take the first whitespace-separated token or full string
                    plate_token = line.split()[0].upper()
                    plates.append(plate_token)

    return plates, found_path


# ── Prediction Loader ─────────────────────────────────────────────────────────

def find_latest_output_dir(project_root: Path) -> Path:
    """Finds the most recent timestamped output folder under output/."""
    out_root = project_root / "output"
    if not out_root.exists():
        raise FileNotFoundError(f"Output directory not found at: {out_root}")

    dirs = [p for p in out_root.iterdir() if p.is_dir() and not p.name.startswith(".")]
    if not dirs:
        raise FileNotFoundError(f"No run folders found inside {out_root}")

    # Sort by directory name (timestamp format YYYYMMDD_HHMMSS) or modification time
    dirs.sort(key=lambda p: (p.name, p.stat().st_mtime), reverse=True)
    return dirs[0]


def load_predictions(run_dir: Path) -> List[Dict[str, Any]]:
    """Loads plate predictions from Final_outputs.csv or results.db."""
    csv_path = run_dir / "Final_outputs.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        if not df.empty and "License_Plate" in df.columns:
            records = []
            for _, row in df.iterrows():
                records.append({
                    "plate": str(row["License_Plate"]).strip().upper(),
                    "conf": float(row.get("Confidence", 0.0)),
                    "track_id": row.get("Track_ID", None),
                    "timestamp": row.get("Timestamp", ""),
                    "source": "Final_outputs.csv",
                })
            return records

    rich_csv = run_dir / "results_rich.csv"
    if rich_csv.exists():
        df = pd.read_csv(rich_csv)
        if not df.empty and "license_number" in df.columns:
            records = []
            grouped = df.groupby(["car_id", "license_number"]).agg({
                "license_number_score": "max"
            }).reset_index()
            for _, row in grouped.iterrows():
                records.append({
                    "plate": str(row["license_number"]).strip().upper(),
                    "conf": float(row.get("license_number_score", 0.0)),
                    "track_id": row.get("car_id", None),
                    "timestamp": "",
                    "source": "results_rich.csv",
                })
            return records

    raise FileNotFoundError(f"No prediction files (Final_outputs.csv / results_rich.csv) found in {run_dir}")


# ── Comparison & Accuracy Scoring ─────────────────────────────────────────────

def compare_plates(actual_plates: List[str], predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
    matched_results = []
    exact_count = 0
    near_count = 0  # 1 char diff
    total_char_acc = 0.0

    # Avoid matching multiple ground-truth plates to the exact same track prediction if possible
    used_indices = set()

    for actual in actual_plates:
        best_cand = None
        best_dist = 999
        best_idx = -1

        for idx, pred in enumerate(predictions):
            d = levenshtein(actual, pred["plate"])
            # Prefer unused candidates, then lowest distance, then higher confidence
            is_used = idx in used_indices
            score = (d, 1 if is_used else 0, -pred["conf"])

            if best_cand is None or score < (best_dist, 1 if best_idx in used_indices else 0, -best_cand["conf"]):
                best_cand = pred
                best_dist = d
                best_idx = idx

        if best_idx != -1:
            used_indices.add(best_idx)

        c_acc = char_accuracy(actual, best_cand["plate"]) if best_cand else 0.0
        total_char_acc += c_acc

        if best_dist == 0:
            status = "Exact Match"
            exact_count += 1
        elif best_dist == 1:
            status = "1 char diff"
            near_count += 1
        elif best_dist == 2:
            status = "2 char diff"
        else:
            status = "Mismatch"

        matched_results.append({
            "actual": actual,
            "predicted": best_cand["plate"] if best_cand else "NO DETECTION",
            "confidence": best_cand["conf"] if best_cand else 0.0,
            "status": status,
            "dist": best_dist if best_cand else len(actual),
            "track_id": best_cand.get("track_id") if best_cand else "-",
            "timestamp": best_cand.get("timestamp") if best_cand else "-",
        })

    total = len(actual_plates)
    exact_accuracy = (exact_count / total * 100.0) if total > 0 else 0.0
    relaxed_accuracy = ((exact_count + near_count) / total * 100.0) if total > 0 else 0.0
    avg_char_accuracy = (total_char_acc / total * 100.0) if total > 0 else 0.0

    return {
        "matches": matched_results,
        "total": total,
        "exact_count": exact_count,
        "near_count": near_count,
        "exact_accuracy": exact_accuracy,
        "relaxed_accuracy": relaxed_accuracy,
        "char_accuracy": avg_char_accuracy,
    }


# ── CLI & Display ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate ALPR pipeline plate accuracy against ground truth.")
    parser.add_argument("video", nargs="?", default="test3", help="Video name or ID (default: 'test3').")
    parser.add_argument("--run", default=None, help="Path to specific run directory (e.g. output/20261013_204125).")
    parser.add_argument("--json", action="store_true", help="Output results in JSON format.")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    actual_dir = script_dir / "ActualPlates"

    if not actual_dir.exists():
        actual_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load actual plates
    try:
        actual_plates, actual_path = load_actual_plates(args.video, actual_dir)
    except FileNotFoundError as err:
        print(f"\n[Error] {err}")
        sys.exit(1)

    # 2. Determine output directory
    if args.run:
        run_dir = Path(args.run)
    else:
        try:
            run_dir = find_latest_output_dir(project_root)
        except FileNotFoundError as err:
            print(f"\n[Error] {err}")
            sys.exit(1)

    # 3. Load predictions
    try:
        predictions = load_predictions(run_dir)
    except Exception as err:
        print(f"\n[Error] {err}")
        sys.exit(1)

    # 4. Run comparison
    results = compare_plates(actual_plates, predictions)

    if args.json:
        print(json.dumps(results, indent=2))
        return

    # Print Clean Formatted Table
    print("\n" + "=" * 68)
    print(f"  ALPR ACCURACY EVALUATION REPORT")
    print("=" * 68)
    print(f"  Ground Truth File : {actual_path.relative_to(project_root)}")
    print(f"  Output Run        : {run_dir.relative_to(project_root)}")
    print(f"  Total Plates      : {results['total']}")
    print("-" * 68)
    print(f"  {'Actual Plate':<14} {'Matched Result':<16} {'Status':<15} {'Confidence'}")
    print("-" * 68)

    for m in results["matches"]:
        conf_str = f"{m['confidence'] * 100:.1f}%" if m['confidence'] > 0 else "-"
        print(f"  {m['actual']:<14} {m['predicted']:<16} {m['status']:<15} {conf_str}")

    print("=" * 68)
    print("  FINAL ACCURACY SCORES:")
    print(f"  • Exact Match Accuracy   : {results['exact_accuracy']:.1f}% ({results['exact_count']}/{results['total']})")
    print(f"  • Character-Level Accuracy: {results['char_accuracy']:.1f}%")
    print(f"  • Relaxed (≤1 char diff) : {results['relaxed_accuracy']:.1f}% ({results['exact_count'] + results['near_count']}/{results['total']})")
    print("=" * 68 + "\n")


if __name__ == "__main__":
    main()
