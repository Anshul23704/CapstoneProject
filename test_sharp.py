import cv2
import numpy as np
def laplacian_sharpness(crop: np.ndarray) -> float:
    if crop is None or crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
print(laplacian_sharpness(img))
