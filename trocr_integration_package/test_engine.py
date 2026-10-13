"""
Smoke-test and verification script for TrOCR ALPR Inference Engine.
Verifies model loading, FP16 execution, OpenCV array compatibility,
single prediction, and batched inference.
"""

import os
import sys
import time
from pathlib import Path

# Add current package directory to Python path
PACKAGE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE_DIR))

from ocr_engine.trocr_engine import TrOCRInferenceEngine


def run_verification():
    print("=" * 60)
    print("TrOCR Inference Engine Self-Verification Test")
    print("=" * 60)

    model_dir = PACKAGE_DIR / "model_checkpoint"
    sample_dir = PACKAGE_DIR / "sample_crops"

    if not model_dir.exists():
        print(f"Error: Model checkpoint directory not found at: {model_dir}")
        sys.exit(1)

    print(f"[1/4] Loading model from: {model_dir}")
    t0 = time.time()
    engine = TrOCRInferenceEngine(model_path=str(model_dir), use_fp16=True)
    load_time = time.time() - t0
    print(f"      Model loaded successfully on {engine.device} (FP16: {engine.use_fp16}) in {load_time:.2f}s")

    sample_images = list(sample_dir.glob("*.jpg")) + list(sample_dir.glob("*.png"))
    if not sample_images:
        print(f"Warning: No sample images found in {sample_dir}. Skipping image prediction tests.")
        return

    print(f"\n[2/4] Testing Single Prediction on File Paths ({len(sample_images)} samples)")
    for img_path in sample_images:
        t_start = time.time()
        text, avg_conf, char_confs = engine.predict(str(img_path))
        elapsed_ms = (time.time() - t_start) * 1000
        print(f"      File: {img_path.name}")
        print(f"      Prediction: '{text}' | Avg Conf: {avg_conf:.4f} | Latency: {elapsed_ms:.1f}ms")
        print(f"      Per-Char Confidences: {char_confs}")

    print("\n[3/4] Testing OpenCV ndarray Compatibility (Simulating Video Pipeline Crop)")
    try:
        import cv2
        first_img_path = str(sample_images[0])
        cv_img = cv2.imread(first_img_path)
        if cv_img is None:
            print("      Warning: OpenCV failed to read sample image. Skipping OpenCV test.")
        else:
            t_start = time.time()
            text, avg_conf, char_confs = engine.predict(cv_img)
            elapsed_ms = (time.time() - t_start) * 1000
            print(f"      OpenCV input shape: {cv_img.shape} dtype: {cv_img.dtype}")
            print(f"      Prediction: '{text}' | Avg Conf: {avg_conf:.4f} | Latency: {elapsed_ms:.1f}ms")
    except ImportError:
        print("      OpenCV (cv2) not installed in current environment; skipping OpenCV ndarray test.")

    print(f"\n[4/4] Testing Batched Inference (predict_batch with {len(sample_images)} crops)")
    batch_inputs = [str(p) for p in sample_images]
    t_start = time.time()
    batch_results = engine.predict_batch(batch_inputs)
    batch_elapsed_ms = (time.time() - t_start) * 1000

    print(f"      Batch size: {len(batch_inputs)} | Total batch latency: {batch_elapsed_ms:.1f}ms ({batch_elapsed_ms / len(batch_inputs):.1f}ms/crop)")
    for idx, (text, avg_conf, char_confs) in enumerate(batch_results):
        print(f"      Sample {idx + 1}: '{text}' | Conf: {avg_conf:.4f}")

    print("\n" + "=" * 60)
    print("Verification Completed Successfully. All Interfaces Operational.")
    print("=" * 60)


if __name__ == "__main__":
    run_verification()
