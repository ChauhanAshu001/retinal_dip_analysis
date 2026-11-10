import cv2
import numpy as np
from src.preprocessing import apply_clahe

def detect_hard_exudates(bgr_image: np.ndarray, fov_mask: np.ndarray, 
                         top_hat_ksize: int = 15, intensity_thresh: int = 45) -> tuple[np.ndarray, list, dict]:
    """
    Detects bright hard-exudate lesions using CIE-Lab space and White Top-Hat transform.
    Filters candidates by connected-component area bounds.
    """
    # 1. CIE-Lab separation
    lab = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2LAB)
    L_channel = lab[:, :, 0]
    
    # 2. Local contrast enhancement on luminance
    L_clahe = apply_clahe(L_channel, clip_limit=2.0)
    
    # 3. White Top-Hat Transform: Original - Opening(Original)
    ksize = top_hat_ksize if top_hat_ksize % 2 != 0 else top_hat_ksize + 1
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    top_hat = cv2.morphologyEx(L_clahe, cv2.MORPH_TOPHAT, se)
    
    # 4. Binarize & apply FOV
    _, thresh = cv2.threshold(top_hat, intensity_thresh, 255, cv2.THRESH_BINARY)
    thresh_fov = cv2.bitwise_and(thresh, thresh, mask=fov_mask)
    
    # 5. Connected Component Analysis (Filter out 1-2 px noise and large optic-disc areas)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh_fov)
    final_exudates = np.zeros_like(thresh_fov)
    candidates = []
    
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        # Valid lesions typically range from 5 px to 500 px
        if 5 <= area <= 500:
            final_exudates[labels == i] = 255
            candidates.append({
                "id": i,
                "area": int(area),
                "centroid_x": int(centroids[i][0]),
                "centroid_y": int(centroids[i][1]),
                "bbox": (int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP]),
                         int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT]))
            })
            
    stages = {
        "Luminance (L Channel)": L_channel,
        "L-CLAHE": L_clahe,
        "White Top-Hat": top_hat,
        "Exudate Candidates": final_exudates
    }
    return final_exudates, candidates, stages