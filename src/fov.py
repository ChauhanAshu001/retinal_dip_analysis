import cv2
import numpy as np

def compute_fov_mask(bgr_image: np.ndarray, threshold_val: int = 15) -> np.ndarray:
    """
    Computes a binary Field-of-View (FOV) mask for the circular fundus lens.
    Erodes slightly to discard border boundary transition artifacts.
    """
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, threshold_val, 255, cv2.THRESH_BINARY)
    
    # Erode border slightly to avoid false positive edges at the lens perimeter
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    fov_mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE, kernel)
    return fov_mask