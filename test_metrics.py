import cv2
import numpy as np

def _sharpness(crop: np.ndarray) -> float:
    if crop is None or crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

img = np.zeros((100, 100, 3), dtype=np.uint8)
print("Sharpness:", _sharpness(img))
